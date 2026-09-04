"""Qwen4-Exp checkpoint and online-update parameter lifecycle."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable, Mapping

from .config import Qwen4ExpP0Config


class ParameterLifecycle(str, Enum):
    """Ownership of one public-checkpoint tensor in the P0 RL graph."""

    TRAINABLE_SYNC = "trainable_sync"
    STATIC_SHARED = "static_shared"
    DERIVED_BUFFER = "derived_buffer"
    DISABLED = "disabled"


@dataclass(frozen=True)
class ParameterLifecycleRecord:
    source_name: str
    source_file: str
    lifecycle: ParameterLifecycle
    reason: str


_PLE_BUFFER_SUFFIXES = (
    ".ple.ple_embedding.layer_multipliers",
    ".ple.ple_embedding.ngram_heads_vocab_sizes",
    ".ple.ple_embedding.ngram_heads_offsets",
)


def _classify_parameter(source_name: str, source_file: str) -> ParameterLifecycleRecord:
    if source_name.startswith("model.visual."):
        return ParameterLifecycleRecord(
            source_name, source_file, ParameterLifecycle.DISABLED, "vision tower is outside the P0 text graph"
        )
    if source_name.startswith("mtp."):
        return ParameterLifecycleRecord(
            source_name, source_file, ParameterLifecycle.DISABLED, "MTP and speculative decoding are outside P0"
        )
    if ".ple.ple_embedding.ngram_embedding." in source_name:
        return ParameterLifecycleRecord(
            source_name,
            source_file,
            ParameterLifecycle.STATIC_SHARED,
            "PLE table is loaded on both engines and remains frozen during RL",
        )
    if ".self_attn.indexer." in source_name:
        return ParameterLifecycleRecord(
            source_name,
            source_file,
            ParameterLifecycle.STATIC_SHARED,
            "QSA indexer is loaded on both engines and remains frozen during RL",
        )
    if source_name.endswith(_PLE_BUFFER_SUFFIXES):
        return ParameterLifecycleRecord(
            source_name,
            source_file,
            ParameterLifecycle.DERIVED_BUFFER,
            "PLE hash metadata is reproducible from config and verified against the checkpoint",
        )
    if source_name == "lm_head.weight" or source_name.startswith("model.language_model."):
        return ParameterLifecycleRecord(
            source_name,
            source_file,
            ParameterLifecycle.TRAINABLE_SYNC,
            "parameter participates in training and is transferred after optimizer steps",
        )
    raise ValueError(f"unclassified Qwen4-Exp checkpoint tensor: {source_name}")


def build_lifecycle_manifest(
    weight_map: Mapping[str, str],
    config: Qwen4ExpP0Config | None = None,
) -> list[ParameterLifecycleRecord]:
    records = [_classify_parameter(name, weight_map[name]) for name in sorted(weight_map)]
    _validate_manifest(records, config)
    return records


def _validate_manifest(
    records: Iterable[ParameterLifecycleRecord],
    config: Qwen4ExpP0Config | None,
) -> None:
    records = list(records)
    counts = Counter(record.lifecycle for record in records)
    if not counts[ParameterLifecycle.TRAINABLE_SYNC]:
        raise ValueError("Qwen4-Exp manifest contains no trainable text parameters")
    if config is None:
        return

    ple_table_shards = [
        record
        for record in records
        if record.lifecycle is ParameterLifecycle.STATIC_SHARED
        and ".ple.ple_embedding.ngram_embedding.shard_" in record.source_name
    ]
    expected_ple_table_shards = len(config.ple_layer_ids) * config.split_ngram_parts
    if len(ple_table_shards) != expected_ple_table_shards:
        raise ValueError(
            f"checkpoint has {len(ple_table_shards)} PLE table shards; expected {expected_ple_table_shards}"
        )

    indexer_records = [
        record
        for record in records
        if record.lifecycle is ParameterLifecycle.STATIC_SHARED and ".self_attn.indexer." in record.source_name
    ]
    qsa_layer_count = sum(layer_type == "qwen_sparse_attention" for layer_type in config.layer_types)
    if len(indexer_records) != qsa_layer_count * 3:
        raise ValueError(
            f"checkpoint has {len(indexer_records)} QSA indexer tensors; expected {qsa_layer_count * 3}"
        )


def lifecycle_summary(records: Iterable[ParameterLifecycleRecord]) -> dict[str, int]:
    counts = Counter(record.lifecycle.value for record in records)
    return {lifecycle.value: counts[lifecycle.value] for lifecycle in ParameterLifecycle}


def is_qwen4_exp_model_name(model_name: str) -> bool:
    normalized = model_name.lower().replace("_", "").replace("-", "")
    return "qwen4exp" in normalized


def should_online_update_megatron_parameter(model_name: str, megatron_name: str) -> bool:
    """Apply the P0 static-parameter contract before TP/EP collection."""

    if not is_qwen4_exp_model_name(model_name):
        return True
    canonical = megatron_name.lower()
    if ".ple.ple_embedding." in canonical:
        return False
    if ".self_attention.indexer." in canonical or ".self_attn.indexer." in canonical:
        return False
    return True


def manifest_sha256(records: Iterable[ParameterLifecycleRecord]) -> str:
    payload = [
        {
            **asdict(record),
            "lifecycle": record.lifecycle.value,
        }
        for record in records
    ]
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
