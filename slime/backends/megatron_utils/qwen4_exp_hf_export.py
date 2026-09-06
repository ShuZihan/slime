"""Rebuild the source HF schema from live text weights and immutable assets."""

from __future__ import annotations

import re
from pathlib import Path

import torch
from safetensors import safe_open

from slime_plugins.models.qwen4_exp.lifecycle import ParameterLifecycle, build_lifecycle_manifest

from .hf_to_megatron.common import SafetensorReader


class Qwen4ExpHfExport:
    """Stateful inverse of the source loader, separate from SGLang online layout.

    The direct iterator orders weights by layer/projection, so a grouped expert
    tensor is completed before moving to the next projection. Only incomplete
    groups live on CPU; the large frozen PLE table is never collected from TP.
    """

    _EXPERT = re.compile(r"(model\.language_model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate|up|down)_proj\.weight")

    def __init__(self, source: str | Path):
        self.source = Path(source)
        self.reader = SafetensorReader(source)
        self.records = build_lifecycle_manifest(self.reader.weight_map)
        self.trainable = {
            record.source_name for record in self.records if record.lifecycle is ParameterLifecycle.TRAINABLE_SYNC
        }
        self.emitted = set()
        self.pending = {}

    def _validate_name(self, name):
        if name not in self.trainable:
            raise ValueError(f"Unexpected live Qwen4-Exp HF tensor: {name}")
        if name in self.emitted:
            raise ValueError(f"Duplicate HF tensor while exporting Qwen4-Exp: {name}")

    def convert(self, named_tensors):
        for name, tensor in named_tensors:
            match = self._EXPERT.fullmatch(name)
            if match is None:
                self._validate_name(name)
                if name in self.pending:
                    raise ValueError(f"Duplicate grouped and individual expert tensor: {name}")
                if tuple(tensor.shape) != self.reader.get_shape(name):
                    raise ValueError(f"Qwen4-Exp HF shape mismatch: {name}")
                self.emitted.add(name)
                yield name, tensor
                continue

            prefix, expert, projection = match.groups()
            expert = int(expert)
            grouped_name = f"{prefix}.{'down_proj' if projection == 'down' else 'gate_up_proj'}"
            self._validate_name(grouped_name)
            shape = self.reader.get_shape(grouped_name)
            if len(shape) != 3 or not 0 <= expert < shape[0]:
                raise ValueError(f"Invalid expert shape or ID for {name}: {shape}")
            if projection != "down" and shape[1] % 2:
                raise ValueError(f"Invalid gate/up shape for {grouped_name}: {shape}")
            rows = shape[1] if projection == "down" else shape[1] // 2
            if tuple(tensor.shape) != (rows, shape[2]):
                raise ValueError(f"Qwen4-Exp HF shape mismatch: {name}")
            if grouped_name not in self.pending:
                self.pending[grouped_name] = (torch.empty(shape, dtype=tensor.dtype, device="cpu"), set())
            grouped, received = self.pending[grouped_name]
            part = (expert, projection)
            if part in received:
                raise ValueError(f"Duplicate Qwen4-Exp expert part: {name}")
            if grouped.dtype != tensor.dtype:
                raise ValueError(f"Inconsistent expert dtype: {name}")
            start = rows if projection == "up" else 0
            grouped[expert, start : start + rows].copy_(tensor.detach())
            received.add(part)
            expected_parts = shape[0] * (1 if projection == "down" else 2)
            if len(received) == expected_parts:
                del self.pending[grouped_name]
                self.emitted.add(grouped_name)
                yield grouped_name, grouped

    def static_tensors(self):
        missing = self.trainable - self.emitted
        if missing:
            raise ValueError(f"Qwen4-Exp export is missing live tensors or expert parts: {sorted(missing)}")
        # P0 freezes PLE/Indexer and disables Vision/MTP. Preserve those source
        # tensors, including hash buffers, so the copied HF config stays loadable.
        # Read each original tensor separately; never concatenate PLE shards or
        # copy entire source files that could contain stale trainable weights.
        for record in self.records:
            if record.lifecycle is ParameterLifecycle.TRAINABLE_SYNC:
                continue
            with safe_open(self.source / record.source_file, framework="pt", device="cpu") as tensors:
                yield record.source_name, tensors.get_tensor(record.source_name)
