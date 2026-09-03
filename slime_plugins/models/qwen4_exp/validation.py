"""Structured evidence emitted by the Qwen4-Exp P0 validation run."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from .lifecycle import should_online_update_megatron_parameter


_VALIDATION_DIR_ENV = "SLIME_QWEN4_EXP_VALIDATION_DIR"
_FINGERPRINT_REGEX_ENV = "SLIME_QWEN4_EXP_FINGERPRINT_REGEX"
_DEFAULT_FINGERPRINT_REGEX = re.compile(
    r"(?:embedding\.word_embeddings|output_layer|decoder\.final_layernorm|"
    r"decoder\.layers\.(?:0|1|3)\.(?!mlp\.experts\.local_experts\.(?!0\.))|"
    r"\.self_attention\.indexer\.|\.ple\.ple_embedding\.ngram_embedding\.weight)"
)
_DEFAULT_LOAD_ALIGNMENT_REGEX = re.compile(
    r"(?:embedding\.word_embeddings|output_layer|decoder\.final_layernorm|"
    r"decoder\.layers\.0\.(?:attn_hyper_connection|linear_attn|mlp\.(?:router|shared_experts)|"
    r"mlp\.experts\.linear_fc[12]\.weight0$)|"
    r"decoder\.layers\.1\.ple\.|decoder\.layers\.3\.self_attention\.)"
)
_PLE_BUFFER_SUFFIXES = (
    ".ple.ple_embedding.layer_multipliers",
    ".ple.ple_embedding.ngram_heads_vocab_sizes",
    ".ple.ple_embedding.ngram_heads_offsets",
)


def validation_dir(args: Any = None) -> Path | None:
    configured = getattr(args, "qwen4_exp_validation_dir", None) if args is not None else None
    configured = configured or os.environ.get(_VALIDATION_DIR_ENV)
    return Path(configured).expanduser().resolve() if configured else None


def _dist_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as output:
        output.write(serialized)
        output.write("\n")


def _canonical_name(name: str) -> str:
    while name.startswith("module."):
        name = name.removeprefix("module.")
    return name.removeprefix("language_model.")


def _lifecycle(name: str, tensor: torch.Tensor, is_buffer: bool) -> str:
    canonical = _canonical_name(name)
    if canonical.endswith(_PLE_BUFFER_SUFFIXES):
        return "derived_buffer"
    if not should_online_update_megatron_parameter("qwen4_exp", canonical):
        return "static_shared"
    if is_buffer:
        return "runtime_buffer"
    return "trainable_sync" if tensor.requires_grad else "frozen_parameter"


def _sample_bytes(tensor: torch.Tensor, max_elements: int = 4096) -> bytes:
    flat = tensor.detach().reshape(-1)
    if not flat.numel():
        return b""
    if flat.numel() > max_elements:
        indices = torch.linspace(
            0,
            flat.numel() - 1,
            steps=max_elements,
            dtype=torch.float64,
            device=flat.device,
        ).long()
        flat = flat.index_select(0, indices)
    raw = flat.contiguous().view(torch.uint8).cpu()
    return raw.numpy().tobytes()


def tensor_fingerprint(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(_sample_bytes(tensor))
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "numel": tensor.numel(),
        "sample_sha256": digest.hexdigest(),
    }


class CheckpointLoadAudit:
    """Aggregate exact-copy evidence while HF tensors enter one Megatron rank."""

    def __init__(self, args: Any):
        self.output_dir = validation_dir(args)
        self.rank = _dist_rank()
        self.parameter_count = 0
        self.parameter_numel = 0
        self.lifecycle_counts: Counter[str] = Counter()
        self.sampled = []
        self.streamed = []

    @property
    def enabled(self) -> bool:
        return self.output_dir is not None

    @torch.no_grad()
    def observe(
        self,
        name: str,
        actual: torch.Tensor,
        *,
        expected: torch.Tensor | None,
        load_mode: str,
    ) -> None:
        if not self.enabled:
            return
        lifecycle = _lifecycle(name, actual, is_buffer=False)
        self.parameter_count += 1
        self.parameter_numel += actual.numel()
        self.lifecycle_counts[lifecycle] += 1

        canonical = _canonical_name(name)
        if expected is None:
            self.streamed.append(
                {
                    "name": canonical,
                    "lifecycle": lifecycle,
                    "load_mode": load_mode,
                    **tensor_fingerprint(canonical, actual),
                }
            )
            return
        if not _DEFAULT_LOAD_ALIGNMENT_REGEX.search(canonical):
            return
        if expected.shape != actual.shape:
            raise ValueError(
                f"Qwen4-Exp load audit shape mismatch for {canonical}: "
                f"{tuple(expected.shape)} != {tuple(actual.shape)}"
            )
        expected_bytes = _sample_bytes(expected.to(dtype=actual.dtype))
        actual_bytes = _sample_bytes(actual)
        exact = expected_bytes == actual_bytes
        record = {
            "name": canonical,
            "lifecycle": lifecycle,
            "load_mode": load_mode,
            "exact_sample_match": exact,
            "expected_sample_sha256": hashlib.sha256(expected_bytes).hexdigest(),
            "actual_sample_sha256": hashlib.sha256(actual_bytes).hexdigest(),
            "dtype": str(actual.dtype).removeprefix("torch."),
            "shape": list(actual.shape),
            "numel": actual.numel(),
        }
        self.sampled.append(record)
        if not exact:
            raise ValueError(f"Qwen4-Exp HF to Megatron sampled value mismatch for {canonical}")

    def finish(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        if not self.sampled or not self.streamed:
            raise ValueError("Qwen4-Exp load audit did not cover sampled tensors and streamed PLE")
        payload = {
            "schema": "qwen4-exp-validation/v1",
            "event": "checkpoint_load_alignment",
            "rank": self.rank,
            "time_unix_ns": time.time_ns(),
            "parameter_count": self.parameter_count,
            "parameter_numel": self.parameter_numel,
            "lifecycle_counts": dict(sorted(self.lifecycle_counts.items())),
            "sampled": self.sampled,
            "streamed": self.streamed,
        }
        assert self.output_dir is not None
        _append_jsonl(self.output_dir / "checkpoint" / f"rank-{self.rank:05d}.jsonl", payload)
        return payload


def _iter_state(model: Sequence[torch.nn.Module] | torch.nn.Module):
    modules = list(model) if isinstance(model, (tuple, list)) else [model]
    seen = set()
    for chunk_idx, module in enumerate(modules):
        prefix = f"chunk_{chunk_idx}." if len(modules) > 1 else ""
        for is_buffer, entries in ((False, module.named_parameters()), (True, module.named_buffers())):
            for name, tensor in entries:
                full_name = prefix + name
                if full_name in seen:
                    continue
                seen.add(full_name)
                yield full_name, tensor, is_buffer


def _fingerprint_pattern() -> re.Pattern[str]:
    configured = os.environ.get(_FINGERPRINT_REGEX_ENV)
    return re.compile(configured) if configured else _DEFAULT_FINGERPRINT_REGEX


@torch.no_grad()
def record_model_fingerprints(
    args: Any,
    model: Sequence[torch.nn.Module] | torch.nn.Module,
    event: str,
    *,
    rollout_id: int | None = None,
) -> dict[str, Any] | None:
    output_dir = validation_dir(args)
    if output_dir is None:
        return None

    pattern = _fingerprint_pattern()
    tensors = []
    lifecycle_counts: Counter[str] = Counter()
    lifecycle_numel: Counter[str] = Counter()
    for name, tensor, is_buffer in _iter_state(model):
        lifecycle = _lifecycle(name, tensor, is_buffer)
        lifecycle_counts[lifecycle] += 1
        lifecycle_numel[lifecycle] += tensor.numel()
        if not pattern.search(name):
            continue
        record = {
            "name": name,
            "lifecycle": lifecycle,
            "requires_grad": bool(tensor.requires_grad),
            "tensor_model_parallel": bool(getattr(tensor, "tensor_model_parallel", False)),
            "partition_dim": int(getattr(tensor, "partition_dim", -1)),
            **tensor_fingerprint(name, tensor),
        }
        tensors.append(record)

    aggregate = hashlib.sha256()
    for record in sorted(tensors, key=lambda item: item["name"]):
        aggregate.update(record["name"].encode("utf-8"))
        aggregate.update(record["sample_sha256"].encode("ascii"))
    rank = _dist_rank()
    payload = {
        "schema": "qwen4-exp-validation/v1",
        "event": event,
        "rollout_id": rollout_id,
        "rank": rank,
        "time_unix_ns": time.time_ns(),
        "selected_tensor_count": len(tensors),
        "selected_aggregate_sha256": aggregate.hexdigest(),
        "lifecycle_counts": dict(sorted(lifecycle_counts.items())),
        "lifecycle_numel": dict(sorted(lifecycle_numel.items())),
        "tensors": tensors,
    }
    _append_jsonl(output_dir / "model" / f"rank-{rank:05d}.jsonl", payload)
    return payload


def record_engine_checksums(
    args: Any,
    rollout_engines: Iterable[Any],
    event: str,
    *,
    rollout_id: int | None = None,
) -> dict[str, Any] | None:
    output_dir = validation_dir(args)
    if output_dir is None or _dist_rank() != 0:
        return None

    import ray

    engines = list(rollout_engines)
    responses = ray.get([engine.check_weights.remote(action="checksum") for engine in engines])
    payload = {
        "schema": "qwen4-exp-validation/v1",
        "event": event,
        "rollout_id": rollout_id,
        "rank": 0,
        "time_unix_ns": time.time_ns(),
        "engine_count": len(engines),
        "responses": responses,
    }
    _append_jsonl(output_dir / "rollout" / "engine-checksums.jsonl", payload)
    return payload


def write_driver_event(
    output_dir: str | Path,
    event: str,
    **fields: Any,
) -> None:
    payload = {
        "schema": "qwen4-exp-validation/v1",
        "event": event,
        "time_unix_ns": time.time_ns(),
        **fields,
    }
    _append_jsonl(Path(output_dir) / "driver.jsonl", payload)
