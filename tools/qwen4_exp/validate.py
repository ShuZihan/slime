#!/usr/bin/env python3
"""One-command staged validation for the Qwen4-Exp slime P0 path."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import subprocess
import sys
import tarfile
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TESTS = (
    "tests/test_qwen4_exp_config_lifecycle.py",
    "tests/test_qwen4_exp_hf_config.py",
    "tests/test_qwen4_exp_reference.py",
    "tests/test_qwen4_exp_checkpoint_mapping.py",
    "tests/test_qwen4_exp_hf_export.py",
    "tests/utils/test_hf_checkpoint_saver.py",
    "tests/test_qwen4_exp_distributed_cpu.py",
    "tests/test_qwen4_exp_megatron_layer.py",
    "tests/test_qwen4_exp_megatron_model.py",
    "tests/test_qwen4_exp_packed_data.py",
    "tests/test_qwen4_exp_validation.py",
    "tests/test_update_weight_factory.py",
    "tests/test_qwen3_linear_attention_cu_seqlens.py",
    "tests/utils/test_loss_mask_type_qwen35.py",
)
STAGE_CODES = {
    "preflight": "Q4E-100",
    "manifest": "Q4E-200",
    "unit": "Q4E-300",
    "e2e": "Q4E-400",
    "checkpoint_alignment": "Q4E-410",
    "model_transition": "Q4E-500",
    "engine_sync": "Q4E-510",
    "rl_closure": "Q4E-600",
    "archive": "Q4E-900",
}
PUBLIC_CHECKPOINT_CONTRACT = {
    "tensor_count": 1_658,
    "file_count": 131,
    "metadata_total_size": 359_999_963_128,
    "lifecycle": {
        "trainable_sync": 1_127,
        "static_shared": 164,
        "derived_buffer": 3,
        "disabled": 364,
    },
    "manifest_sha256": "00ae2f76f403da35de3f9cb05514c7720affa1bac907e7a90f0f4ae2f3c9e56d",
}
_CRITICAL_SLIME_SOURCES = (
    "slime/backends/megatron_utils/actor.py",
    "slime/backends/megatron_utils/arguments.py",
    "slime/backends/megatron_utils/data.py",
    "slime/backends/megatron_utils/hf_to_megatron/qwen4_exp.py",
    "slime/backends/megatron_utils/hf_checkpoint_saver.py",
    "slime/backends/megatron_utils/qwen4_exp_hf_export.py",
    "slime/backends/megatron_utils/megatron_to_hf/qwen4_exp.py",
    "slime/backends/megatron_utils/update_weight/__init__.py",
    "slime/backends/megatron_utils/update_weight/hf_weight_iterator_direct.py",
    "slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py",
    "slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py",
    "slime/utils/arguments.py",
    "slime/utils/hf_config.py",
    "slime/utils/mask_utils.py",
    "slime_plugins/models/hf_attention.py",
    "slime_plugins/models/qwen3_5.py",
    "slime_plugins/models/qwen4_exp/config.py",
    "slime_plugins/models/qwen4_exp/lifecycle.py",
    "slime_plugins/models/qwen4_exp/model.py",
    "slime_plugins/models/qwen4_exp/reference.py",
    "slime_plugins/models/qwen4_exp/validation.py",
    "slime_plugins/models/qwen4_exp/validation_reward.py",
)


class ValidationFailure(RuntimeError):
    pass


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    return repr(value)


class RunLog:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.path = run_dir / "stages.jsonl"
        self.last_failed_stage: str | None = None
        self.last_failed_code: str | None = None
        run_dir.mkdir(parents=True, exist_ok=False)

    def write(self, event: str, **fields: Any) -> None:
        payload = {
            "schema": "qwen4-exp-validation/v1",
            "event": event,
            "time_unix_ns": time.time_ns(),
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default))
            output.write("\n")

    def stage(self, name: str, function: Callable[[], dict[str, Any] | None]) -> dict[str, Any]:
        code = STAGE_CODES[name]
        started = time.time_ns()
        self.write("stage_started", stage=name, code=code)
        try:
            details = function() or {}
        except Exception as error:
            self.last_failed_stage = name
            self.last_failed_code = code
            self.write(
                "stage_failed",
                stage=name,
                code=code,
                duration_seconds=(time.time_ns() - started) / 1e9,
                error_type=type(error).__name__,
                error=str(error),
                traceback=traceback.format_exc(),
            )
            raise
        self.write(
            "stage_passed",
            stage=name,
            code=code,
            duration_seconds=(time.time_ns() - started) / 1e9,
            details=details,
        )
        return details


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> None:
    with log_path.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            output.write(line)
            output.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        returncode = process.wait()
    if returncode:
        raise ValidationFailure(f"command exited {returncode}: {' '.join(command)}")


def _command_output(command: list[str], cwd: Path | None = None) -> str | None:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _module_git_state(module_name: str, *, commit_env: str | None = None) -> dict[str, Any]:
    module = importlib.import_module(module_name)
    module_path = Path(module.__file__).resolve()
    current = module_path.parent
    while current != current.parent and not (current / ".git").exists():
        current = current.parent
    git_root = current if (current / ".git").exists() else None
    git_commit = _command_output(["git", "rev-parse", "HEAD"], git_root) if git_root else None
    build_commit = os.environ.get(commit_env) if commit_env else None
    return {
        "module_path": str(module_path),
        "git_root": str(git_root) if git_root else None,
        "commit": git_commit or build_commit,
        "commit_source": "git" if git_commit else commit_env if build_commit else None,
        "dirty": bool(_command_output(["git", "status", "--porcelain"], git_root)) if git_root else None,
    }


def _critical_source_hashes(repo_root: Path) -> dict[str, str]:
    hashes = {}
    for relative_path in _CRITICAL_SLIME_SOURCES:
        path = repo_root / relative_path
        if not path.is_file():
            raise ValidationFailure(f"critical Qwen4-Exp source is absent: {path}")
        hashes[relative_path] = _sha256_file(path)
    return hashes


def _ray_node_probe(
    checkpoint_dir: str,
    marker_path: str,
    pins: dict[str, Any],
    expected_source_hashes: dict[str, str],
) -> dict[str, Any]:
    """Run inside one Ray node and verify its immutable Qwen4-Exp inputs."""

    try:
        import socket

        import megatron.core as megatron
        import sglang
        import slime
        import transformers

        def module_commit(module: Any, commit_env: str | None = None) -> tuple[str, str | None, str | None]:
            module_path = Path(module.__file__).resolve()
            current = module_path.parent
            while current != current.parent and not (current / ".git").exists():
                current = current.parent
            git_commit = (
                _command_output(["git", "rev-parse", "HEAD"], current)
                if (current / ".git").exists()
                else None
            )
            build_commit = os.environ.get(commit_env) if commit_env else None
            commit_source = "git" if git_commit else commit_env if build_commit else None
            return str(module_path), git_commit or build_commit, commit_source

        checkpoint = Path(checkpoint_dir)
        index_path = checkpoint / "model.safetensors.index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = sorted(set(index["weight_map"].values()))
        missing = [name for name in shard_names if not (checkpoint / name).is_file()]
        sglang_path, sglang_commit, sglang_commit_source = module_commit(sglang, "SGLANG_BUILD_COMMIT")
        megatron_path, megatron_commit, megatron_commit_source = module_commit(megatron)
        slime_path = Path(slime.__file__).resolve()
        slime_root = slime_path.parents[1]
        source_hashes = _critical_source_hashes(slime_root)
        qwen4_path = Path(sglang_path).parents[0] / "srt" / "models" / "qwen4_exp.py"
        checker_path = Path(sglang_path).parents[0] / "srt" / "utils" / "weight_checker.py"
        qsa_metadata_path = Path(sglang_path).parents[0] / "srt" / "layers" / "attention" / "qsa" / "metadata.py"
        qsa_indexer_path = Path(sglang_path).parents[0] / "srt" / "layers" / "attention" / "qsa" / "qsa_indexer.py"
        qsa_backend_path = (
            Path(sglang_path).parents[0] / "srt" / "layers" / "attention" / "qwen_sparse_attn_backend.py"
        )
        qwen4_source = qwen4_path.read_text(encoding="utf-8")
        checker_source = checker_path.read_text(encoding="utf-8")
        qsa_metadata_source = qsa_metadata_path.read_text(encoding="utf-8")
        qsa_indexer_source = qsa_indexer_path.read_text(encoding="utf-8")
        qsa_backend_source = qsa_backend_path.read_text(encoding="utf-8")
        failures = []
        if missing:
            failures.append(f"{len(missing)} checkpoint shards missing")
        if not Path(marker_path).is_file():
            failures.append("shared validation marker is invisible")
        if transformers.__version__ != pins["transformers_version"]:
            failures.append(f"transformers={transformers.__version__}")
        if sglang_commit != pins["sglang"]["commit"]:
            failures.append(f"sglang={sglang_commit}")
        if megatron_commit != pins["megatron_commit"]:
            failures.append(f"megatron={megatron_commit}")
        mismatched_sources = sorted(
            name for name, digest in source_hashes.items() if expected_source_hashes.get(name) != digest
        )
        if mismatched_sources:
            failures.append(f"Qwen4-Exp source hash mismatch: {mismatched_sources}")
        if "_preserve_on_weight_reset = True" not in qwen4_source:
            failures.append("Qwen4 static-weight marker patch missing")
        if "getattr(self.config, \"seed\", None) or 1234" not in qwen4_source:
            failures.append("Qwen4 null-seed patch missing")
        if "_preserve_on_weight_reset" not in checker_source:
            failures.append("WeightChecker static reset patch missing")
        if "compress_plan_valid" not in qsa_metadata_source:
            failures.append("QSA plan-valid metadata patch missing")
        if "metadata.compress_plan_valid[:, None]" not in qsa_indexer_source:
            failures.append("QSA short-extend source-row patch missing")
        if "group_plan_valid" not in qsa_backend_source:
            failures.append("QSA write-plan validity propagation patch missing")
        return {
            "hostname": socket.gethostname(),
            "checkpoint_shard_count": len(shard_names),
            "missing_checkpoint_shards": missing[:16],
            "transformers_version": transformers.__version__,
            "sglang_module": sglang_path,
            "sglang_commit": sglang_commit,
            "sglang_commit_source": sglang_commit_source,
            "megatron_module": megatron_path,
            "megatron_commit": megatron_commit,
            "megatron_commit_source": megatron_commit_source,
            "slime_module": str(slime_path),
            "source_hashes": source_hashes,
            "failures": failures,
        }
    except Exception as error:
        return {"hostname": None, "failures": [f"{type(error).__name__}: {error}"]}


def _ray_cluster_preflight(args: argparse.Namespace, run_dir: Path, pins: dict[str, Any]) -> dict[str, Any]:
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    ray.init(address=args.ray_address, ignore_reinit_error=True, logging_level="ERROR")
    try:
        gpu_nodes = [
            node
            for node in ray.nodes()
            if node.get("Alive") and float((node.get("Resources") or {}).get("GPU", 0)) > 0
        ]
        if args.expected_gpus % args.expected_nodes:
            raise ValidationFailure("--expected-gpus must be divisible by --expected-nodes")
        expected_gpus_per_node = args.expected_gpus // args.expected_nodes
        eligible_nodes = [
            node
            for node in gpu_nodes
            if int(float(node["Resources"].get("GPU", 0))) >= expected_gpus_per_node
        ]
        total_gpus = int(sum(float(node["Resources"].get("GPU", 0)) for node in gpu_nodes))
        details: dict[str, Any] = {
            "address": args.ray_address,
            "gpu_node_count": len(gpu_nodes),
            "eligible_gpu_node_count": len(eligible_nodes),
            "gpu_count": total_gpus,
            "expected_nodes": args.expected_nodes,
            "expected_gpus": args.expected_gpus,
            "expected_gpus_per_node": expected_gpus_per_node,
            "nodes": [],
        }
        evidence_path = run_dir / "ray-preflight.json"

        def write_evidence() -> None:
            evidence_path.write_text(
                json.dumps(details, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
                encoding="utf-8",
            )

        write_evidence()
        if len(eligible_nodes) < args.expected_nodes:
            raise ValidationFailure(
                f"Ray exposes {len(eligible_nodes)} nodes with at least {expected_gpus_per_node} GPUs; "
                f"expected at least {args.expected_nodes}"
            )
        if total_gpus < args.expected_gpus:
            raise ValidationFailure(f"Ray exposes {total_gpus} GPUs; expected at least {args.expected_gpus}")

        marker_path = run_dir / ".shared-filesystem-probe"
        marker_path.write_text(str(time.time_ns()), encoding="ascii")
        expected_source_hashes = _critical_source_hashes(REPO_ROOT)
        try:
            remote_probe = ray.remote(num_cpus=0)(_ray_node_probe)
            refs = [
                remote_probe.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)
                ).remote(
                    str(args.hf_checkpoint.resolve()),
                    str(marker_path),
                    pins,
                    expected_source_hashes,
                )
                for node in gpu_nodes
            ]
            reports = ray.get(refs, timeout=args.ray_probe_timeout)
        except Exception as error:
            details["probe_error"] = f"{type(error).__name__}: {error}"
            write_evidence()
            raise
        finally:
            marker_path.unlink(missing_ok=True)
        failures = [
            {"node_id": node["NodeID"], "report": report}
            for node, report in zip(gpu_nodes, reports, strict=True)
            if report.get("failures")
        ]
        details["nodes"] = [
            {
                "node_id": node["NodeID"],
                "node_manager_address": node.get("NodeManagerAddress"),
                "gpu_count": int(float(node["Resources"].get("GPU", 0))),
                "probe": report,
            }
            for node, report in zip(gpu_nodes, reports, strict=True)
        ]
        details["failures"] = failures
        write_evidence()
        if failures:
            raise ValidationFailure(f"Ray node preflight failed: {failures}")
        return details
    finally:
        ray.shutdown()


def _version_pins() -> dict[str, Any]:
    path = REPO_ROOT / "tools" / "qwen4_exp" / "versions.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _preflight(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    pins = _version_pins()
    packages = {}
    for name in ("torch", "transformers", "safetensors"):
        module = importlib.import_module(name)
        packages[name] = getattr(module, "__version__", "unknown")

    import torch

    details: dict[str, Any] = {
        "mode": args.mode,
        "python": sys.version,
        "executable": sys.executable,
        "repo_commit": _command_output(["git", "rev-parse", "HEAD"], REPO_ROOT),
        "repo_dirty": bool(_command_output(["git", "status", "--porcelain"], REPO_ROOT)),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "version_pins": pins,
    }
    if torch.cuda.is_available():
        details["cuda_devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]

    if args.mode == "intranet":
        if not torch.cuda.is_available():
            raise ValidationFailure("CUDA is unavailable")
        if not args.hf_checkpoint or not args.hf_checkpoint.is_dir():
            raise ValidationFailure("--hf-checkpoint must point to the complete Qwen4-Exp checkpoint")
        details["sglang"] = _module_git_state("sglang", commit_env="SGLANG_BUILD_COMMIT")
        details["megatron"] = _module_git_state("megatron.core")
        expected_sglang = pins["sglang"]["commit"]
        expected_megatron = pins["megatron_commit"]
        if details["sglang"]["commit"] != expected_sglang:
            raise ValidationFailure(
                f"SGLang commit mismatch: {details['sglang']['commit']} != {expected_sglang}"
            )
        if details["megatron"]["commit"] != expected_megatron:
            raise ValidationFailure(
                f"Megatron commit mismatch: {details['megatron']['commit']} != {expected_megatron}"
            )
        expected_transformers = pins["transformers_version"]
        if packages["transformers"] != expected_transformers:
            raise ValidationFailure(
                f"Transformers version mismatch: {packages['transformers']} != {expected_transformers}"
            )
        from sglang.srt.managers.io_struct import (
            CheckWeightsReqInput,
            UpdateWeightsFromDistributedReqInput,
            UpdateWeightsFromTensorReqInput,
        )
        from sglang.srt.server_args import ServerArgs

        server_fields = {field.name for field in __import__("dataclasses").fields(ServerArgs)}
        required_fields = {
            "disable_radix_cache",
            "dp_size",
            "enable_deterministic_inference",
            "ep_size",
            "language_model_only",
            "moe_dense_tp_size",
            "ple_offload_embedding",
        }
        missing_fields = sorted(required_fields - server_fields)
        if missing_fields:
            raise ValidationFailure(f"SGLang ServerArgs lacks fields: {missing_fields}")
        details["sglang_contracts"] = {
            "check_weights": CheckWeightsReqInput.__name__,
            "distributed_update": UpdateWeightsFromDistributedReqInput.__name__,
            "tensor_update": UpdateWeightsFromTensorReqInput.__name__,
            "server_fields": sorted(required_fields),
        }
        import sglang.srt.models.qwen4_exp as qwen4_sglang
        import sglang.srt.layers.attention.qsa.metadata as qsa_metadata
        import sglang.srt.layers.attention.qsa.qsa_indexer as qsa_indexer
        import sglang.srt.layers.attention.qwen_sparse_attn_backend as qsa_backend
        import sglang.srt.utils.weight_checker as weight_checker

        qwen4_source = Path(qwen4_sglang.__file__).read_text(encoding="utf-8")
        checker_source = Path(weight_checker.__file__).read_text(encoding="utf-8")
        qsa_metadata_source = Path(qsa_metadata.__file__).read_text(encoding="utf-8")
        qsa_indexer_source = Path(qsa_indexer.__file__).read_text(encoding="utf-8")
        qsa_backend_source = Path(qsa_backend.__file__).read_text(encoding="utf-8")
        if "_preserve_on_weight_reset = True" not in qwen4_source:
            raise ValidationFailure("SGLang Qwen4-Exp static-weight marker patch is absent")
        if "_preserve_on_weight_reset" not in checker_source or "getattr(" not in checker_source:
            raise ValidationFailure("SGLang WeightChecker static-weight reset patch is absent")
        if 'getattr(self.config, "seed", None) or 1234' not in qwen4_source:
            raise ValidationFailure("SGLang Qwen4-Exp null-seed patch is absent")
        if "compress_plan_valid" not in qsa_metadata_source:
            raise ValidationFailure("SGLang QSA plan-valid metadata patch is absent")
        if "metadata.compress_plan_valid[:, None]" not in qsa_indexer_source:
            raise ValidationFailure("SGLang QSA short-extend source-row patch is absent")
        if "group_plan_valid" not in qsa_backend_source:
            raise ValidationFailure("SGLang QSA write-plan validity propagation patch is absent")
        details["sglang_patch_contract"] = {
            "static_weight_reset": True,
            "null_seed": True,
            "qsa_short_extend": True,
        }
        details["ray_cluster"] = _ray_cluster_preflight(args, run_dir, pins)

    (run_dir / "preflight.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return details


def _resolve_checkpoint_metadata(args: argparse.Namespace) -> tuple[Path, Path]:
    config_path = args.config_json
    index_path = args.index_json
    if args.hf_checkpoint:
        config_path = config_path or args.hf_checkpoint / "config.json"
        index_path = index_path or args.hf_checkpoint / "model.safetensors.index.json"
    if config_path is None or index_path is None:
        raise ValidationFailure("manifest stage needs --hf-checkpoint or both --config-json and --index-json")
    if not config_path.is_file() or not index_path.is_file():
        raise ValidationFailure(f"missing checkpoint metadata: config={config_path}, index={index_path}")
    return config_path, index_path


def _validate_public_checkpoint_contract(details: dict[str, Any]) -> None:
    actual = {
        key: details[key]
        for key in ("tensor_count", "file_count", "metadata_total_size", "lifecycle", "manifest_sha256")
    }
    if actual == PUBLIC_CHECKPOINT_CONTRACT:
        return
    mismatches = {
        key: {"actual": actual[key], "expected": expected}
        for key, expected in PUBLIC_CHECKPOINT_CONTRACT.items()
        if actual[key] != expected
    }
    raise ValidationFailure(f"public checkpoint manifest mismatch: {mismatches}")


def _manifest(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    from slime_plugins.models.qwen4_exp.config import Qwen4ExpP0Config
    from slime_plugins.models.qwen4_exp.lifecycle import (
        ParameterLifecycle,
        build_lifecycle_manifest,
        lifecycle_summary,
        manifest_sha256,
    )

    config_path, index_path = _resolve_checkpoint_metadata(args)
    config = Qwen4ExpP0Config.from_hf_config(json.loads(config_path.read_text(encoding="utf-8")))
    if args.mode == "intranet":
        config.validate_public_release_contract()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValidationFailure("checkpoint index has no weight_map")
    records = build_lifecycle_manifest(weight_map, config)
    summary = lifecycle_summary(records)
    digest = manifest_sha256(records)

    referenced_files = sorted(set(weight_map.values()))
    missing_files = []
    total_file_bytes = 0
    checkpoint_dir = index_path.parent
    for filename in referenced_files:
        path = checkpoint_dir / filename
        if not path.is_file():
            missing_files.append(filename)
        else:
            total_file_bytes += path.stat().st_size
    if args.mode == "intranet" and missing_files:
        raise ValidationFailure(f"checkpoint index references {len(missing_files)} missing files")

    manifest_path = run_dir / "parameter-lifecycle.jsonl"
    with manifest_path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(
                json.dumps(
                    {
                        "source_name": record.source_name,
                        "source_file": record.source_file,
                        "lifecycle": record.lifecycle.value,
                        "reason": record.reason,
                        "optimizer": record.lifecycle is ParameterLifecycle.TRAINABLE_SYNC,
                        "online_update": record.lifecycle is ParameterLifecycle.TRAINABLE_SYNC,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            output.write("\n")

    details = {
        "config": str(config_path),
        "index": str(index_path),
        "tensor_count": len(records),
        "file_count": len(referenced_files),
        "metadata_total_size": (index.get("metadata") or {}).get("total_size"),
        "checkpoint_bytes_present": total_file_bytes,
        "missing_file_count": len(missing_files),
        "missing_files": missing_files[:32],
        "lifecycle": summary,
        "manifest_sha256": digest,
        "p0": {
            "layers": config.num_hidden_layers,
            "qsa_layers": sum(item == "qwen_sparse_attention" for item in config.layer_types),
            "ple_layers": list(config.ple_layer_ids),
            "experts": config.num_experts,
            "topk": config.num_experts_per_tok,
            "norm_topk_prob": config.norm_topk_prob,
            "gdn_output_gate": config.output_gate_type,
            "pad_token_id": config.pad_token_id,
            "eos_token_id": config.eos_token_id,
        },
    }
    summary_path = run_dir / "manifest-summary.json"
    summary_path.write_text(
        json.dumps(details, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.mode == "intranet":
        _validate_public_checkpoint_contract(details)
        details["public_checkpoint_contract"] = "passed"
        summary_path.write_text(
            json.dumps(details, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return details


def _unit(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    command = [args.python, "-m", "pytest", "-q", *DEFAULT_TESTS]
    _run(command, cwd=REPO_ROOT, env=dict(os.environ), log_path=run_dir / "unit.log")
    return {"tests": list(DEFAULT_TESTS), "command": command}


def _e2e(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    if args.hf_checkpoint is None:
        raise ValidationFailure("intranet E2E stage needs --hf-checkpoint")
    script = args.e2e_script.resolve()
    if not script.is_file():
        raise ValidationFailure(f"E2E script does not exist: {script}")
    env = dict(os.environ)
    if args.expected_gpus % args.expected_nodes:
        raise ValidationFailure("--expected-gpus must be divisible by --expected-nodes")
    gpus_per_node = args.expected_gpus // args.expected_nodes
    env.update(
        {
            "ACTOR_NUM_NODES": str(args.expected_nodes),
            "ACTOR_NUM_GPUS_PER_NODE": str(gpus_per_node),
            "HF_CHECKPOINT": str(args.hf_checkpoint.resolve()),
            "QWEN4_EXP_VALIDATION_DIR": str(run_dir),
            "ROLLOUT_DP_SIZE": str(args.expected_gpus),
            "ROLLOUT_EP_SIZE": str(args.expected_gpus),
            "ROLLOUT_NUM_GPUS": str(args.expected_gpus),
            "ROLLOUT_NUM_GPUS_PER_ENGINE": str(args.expected_gpus),
            "MAX_TRAIN_ROLLOUT_DIFF": str(args.logprob_threshold),
        }
    )
    _run([str(script)], cwd=REPO_ROOT, env=env, log_path=run_dir / "e2e.log")
    return {"script": str(script), "log": str(run_dir / "e2e.log")}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValidationFailure(f"missing evidence file: {path}")
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValidationFailure(f"invalid JSONL at {path}:{line_number}: {error}") from error
    return records


def _checkpoint_alignment(run_dir: Path, expected_ranks: int | None = None) -> dict[str, Any]:
    files = sorted((run_dir / "checkpoint").glob("rank-*.jsonl"))
    if not files:
        raise ValidationFailure("checkpoint load-alignment files are absent")

    rank_summaries = []
    sampled_count = 0
    streamed_count = 0
    saw_indexer = False
    saw_trainable = False
    saw_ple_table = False
    for path in files:
        records = _read_jsonl(path)
        record = next((item for item in records if item.get("event") == "checkpoint_load_alignment"), None)
        if record is None:
            raise ValidationFailure(f"checkpoint load-alignment record is absent in {path.name}")
        sampled = record.get("sampled") or []
        streamed = record.get("streamed") or []
        mismatches = [item["name"] for item in sampled if not item.get("exact_sample_match")]
        if mismatches:
            raise ValidationFailure(f"HF to Megatron sampled value mismatch on {path.name}: {mismatches[:8]}")
        if not sampled or not streamed:
            raise ValidationFailure(f"checkpoint audit lacks mapped or streamed coverage on {path.name}")
        sampled_count += len(sampled)
        streamed_count += len(streamed)
        saw_indexer |= any(".self_attention.indexer." in item["name"] for item in sampled)
        saw_trainable |= any(item.get("lifecycle") == "trainable_sync" for item in sampled)
        saw_ple_table |= any("ple_embedding.ngram_embedding.weight" in item["name"] for item in streamed)
        rank_summaries.append(
            {
                "file": path.name,
                "rank": record.get("rank"),
                "parameter_count": record.get("parameter_count"),
                "parameter_numel": record.get("parameter_numel"),
                "lifecycle_counts": record.get("lifecycle_counts"),
                "sampled_count": len(sampled),
                "streamed_count": len(streamed),
            }
        )
    rank_ids = [item["rank"] for item in rank_summaries]
    if len(rank_ids) != len(set(rank_ids)):
        raise ValidationFailure(f"checkpoint audit contains duplicate rank ids: {rank_ids}")
    if expected_ranks is not None and set(rank_ids) != set(range(expected_ranks)):
        missing = sorted(set(range(expected_ranks)) - set(rank_ids))
        extra = sorted(set(rank_ids) - set(range(expected_ranks)))
        raise ValidationFailure(f"checkpoint audit rank coverage mismatch; missing={missing}, extra={extra}")
    if not saw_trainable or not saw_indexer or not saw_ple_table:
        raise ValidationFailure(
            "checkpoint audit must cover trainable tensors, the frozen QSA indexer, and the streamed PLE table"
        )

    details = {
        "rank_count": len(files),
        "sampled_exact_match_count": sampled_count,
        "streamed_ple_count": streamed_count,
        "ranks": rank_summaries,
    }
    (run_dir / "checkpoint-alignment-summary.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return details


def _model_transition(run_dir: Path, expected_ranks: int | None = None) -> dict[str, Any]:
    files = sorted((run_dir / "model").glob("rank-*.jsonl"))
    if not files:
        raise ValidationFailure("model fingerprint files are absent")
    records_by_path = {path: _read_jsonl(path) for path in files}
    observed_ranks = {
        item.get("rank")
        for records in records_by_path.values()
        for item in records
        if item.get("event") in {"before_train", "after_train"}
    }
    observed_ranks.discard(None)
    if expected_ranks is not None and observed_ranks != set(range(expected_ranks)):
        missing = sorted(set(range(expected_ranks)) - observed_ranks)
        extra = sorted(observed_ranks - set(range(expected_ranks)))
        raise ValidationFailure(f"model fingerprint rank coverage mismatch; missing={missing}, extra={extra}")
    common_rollout_ids = None
    for records in records_by_path.values():
        before_ids = {
            item.get("rollout_id") for item in records if item.get("event") == "before_train"
        }
        after_ids = {
            item.get("rollout_id") for item in records if item.get("event") == "after_train"
        }
        paired = (before_ids & after_ids) - {None}
        common_rollout_ids = paired if common_rollout_ids is None else common_rollout_ids & paired
    if not common_rollout_ids:
        raise ValidationFailure("no common before/after training rollout exists across model ranks")
    rollout_id = min(common_rollout_ids)

    changed_trainable = []
    unchanged_static = []
    missing_pairs = []
    for path in files:
        records = records_by_path[path]
        before = next(
            (
                item
                for item in records
                if item["event"] == "before_train" and item.get("rollout_id") == rollout_id
            ),
            None,
        )
        after = next(
            (
                item
                for item in records
                if item["event"] == "after_train" and item.get("rollout_id") == rollout_id
            ),
            None,
        )
        if before is None or after is None:
            missing_pairs.append(path.name)
            continue
        before_map = {item["name"]: item for item in before["tensors"]}
        after_map = {item["name"]: item for item in after["tensors"]}
        if before_map.keys() != after_map.keys():
            raise ValidationFailure(f"selected model tensors changed membership on {path.name}")
        for name in sorted(before_map):
            left, right = before_map[name], after_map[name]
            changed = left["sample_sha256"] != right["sample_sha256"]
            if left["lifecycle"] in {"static_shared", "derived_buffer"}:
                if changed:
                    raise ValidationFailure(f"static tensor changed during optimizer step: {path.name}:{name}")
                unchanged_static.append(f"{path.name}:{name}")
            elif left["lifecycle"] == "trainable_sync" and changed:
                changed_trainable.append(f"{path.name}:{name}")
    if missing_pairs:
        raise ValidationFailure(f"before/after training fingerprint pair missing on ranks: {missing_pairs}")
    if not changed_trainable:
        raise ValidationFailure("optimizer step changed no sampled trainable tensor")
    if not unchanged_static:
        raise ValidationFailure("no static QSA/PLE tensor was fingerprinted")
    details = {
        "rollout_id": rollout_id,
        "rank_count": len(files),
        "changed_trainable_count": len(changed_trainable),
        "unchanged_static_count": len(unchanged_static),
        "changed_trainable_examples": changed_trainable[:24],
        "unchanged_static_examples": unchanged_static[:24],
    }
    (run_dir / "model-transition-summary.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return details


def _flatten_engine_checksums(record: dict[str, Any]) -> dict[tuple[int, int], dict[str, str]]:
    flattened = {}
    for engine_index, response in enumerate(record.get("responses", [])):
        if response is None:
            continue
        if not isinstance(response, dict) or not response.get("success"):
            raise ValidationFailure(f"engine {engine_index} checksum request failed: {response!r}")
        payload = response.get("ranks", response.get("payload")) or []
        if isinstance(payload, dict):
            payload = [payload]
        for rank_payload in payload:
            parallel = rank_payload.get("parallelism_info") or {}
            tp_rank = int(parallel.get("tp_rank", parallel.get("rank", 0)))
            flattened[(engine_index, tp_rank)] = rank_payload.get("checksums") or {}
    return flattened


def _is_static_sglang_name(name: str) -> bool:
    return ".ple.ple_embedding." in name or ".indexer." in name


def _engine_sync(run_dir: Path, expected_ranks: int | None = None) -> dict[str, Any]:
    records = _read_jsonl(run_dir / "rollout" / "engine-checksums.jsonl")
    before_ids = {
        item.get("rollout_id") for item in records if item.get("event") == "before_weight_sync"
    }
    after_ids = {
        item.get("rollout_id") for item in records if item.get("event") == "after_weight_sync"
    }
    paired_ids = (before_ids & after_ids) - {None}
    if not paired_ids:
        raise ValidationFailure("post-optimizer rollout checksum pair is absent")
    rollout_id = min(paired_ids)
    before = next(
        (
            item
            for item in records
            if item["event"] == "before_weight_sync" and item.get("rollout_id") == rollout_id
        ),
        None,
    )
    after = next(
        (
            item
            for item in records
            if item["event"] == "after_weight_sync" and item.get("rollout_id") == rollout_id
        ),
        None,
    )
    if before is None or after is None:
        raise ValidationFailure("post-optimizer rollout checksum pair is absent")
    before_map = _flatten_engine_checksums(before)
    after_map = _flatten_engine_checksums(after)
    if before_map.keys() != after_map.keys() or not before_map:
        raise ValidationFailure("rollout checksum rank membership changed")
    if expected_ranks is not None and len(before_map) != expected_ranks:
        raise ValidationFailure(
            f"rollout checksum covers {len(before_map)} engine ranks; expected {expected_ranks}"
        )

    static_checked = 0
    trainable_changed = 0
    changed_names = []
    for rank_key in sorted(before_map):
        left, right = before_map[rank_key], after_map[rank_key]
        if left.keys() != right.keys():
            raise ValidationFailure(f"rollout tensor membership changed on engine rank {rank_key}")
        common_names = left.keys() & right.keys()
        for name in common_names:
            changed = left[name] != right[name]
            if _is_static_sglang_name(name):
                static_checked += 1
                if changed:
                    raise ValidationFailure(f"rollout static tensor changed after sync: {rank_key}:{name}")
            elif changed:
                trainable_changed += 1
                if len(changed_names) < 24:
                    changed_names.append(f"engine={rank_key[0]},tp={rank_key[1]}:{name}")
    if static_checked == 0:
        raise ValidationFailure("rollout checksum exposes no QSA indexer or PLE table")
    if trainable_changed == 0:
        raise ValidationFailure("rollout checksum shows no trainable weight update")
    details = {
        "rollout_id": rollout_id,
        "engine_tp_ranks": len(before_map),
        "static_tensor_count": static_checked,
        "changed_trainable_count": trainable_changed,
        "changed_examples": changed_names,
    }
    (run_dir / "engine-sync-summary.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return details


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _flatten_samples(value: Any):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_samples(item)
    elif isinstance(value, dict):
        if "reward" in value and "response_length" in value:
            yield value
        elif "samples" in value:
            yield from _flatten_samples(value["samples"])
    elif hasattr(value, "reward") and hasattr(value, "response_length"):
        yield value


def _sample_field(sample: Any, name: str, default: Any = None) -> Any:
    if isinstance(sample, dict):
        return sample.get(name, default)
    return getattr(sample, name, default)


def _rl_closure(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    log_path = run_dir / "e2e.log"
    log = log_path.read_text(encoding="utf-8", errors="replace")
    diff_values = [
        float(value)
        for value in re.findall(r"train/train_rollout_logprob_abs_diff['\"]?\s*[:=]\s*([0-9.eE+-]+)", log)
    ]
    if not diff_values:
        raise ValidationFailure("train/rollout logprob difference is absent from E2E log")
    if not all(math.isfinite(value) and value <= args.logprob_threshold for value in diff_values):
        raise ValidationFailure(
            f"train/rollout logprob difference exceeds {args.logprob_threshold}: {diff_values}"
        )
    grad_values = [
        float(value)
        for value in re.findall(
            r"train/(?:actor-)?grad_norm['\"]?\s*[:=]\s*([0-9.eE+-]+)",
            log,
        )
    ]
    if not any(math.isfinite(value) and value > 0 for value in grad_values):
        raise ValidationFailure(f"positive finite gradient norm is absent: {grad_values}")

    rollout_dir = run_dir / "rollout_data"
    rollout_paths = sorted(
        (path for path in rollout_dir.glob("*.pt") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    if len(rollout_paths) < 2:
        raise ValidationFailure(f"two numeric rollout dumps are required under {rollout_dir}")
    rollout_paths = rollout_paths[:2]

    import torch

    rollout_summaries = []
    version_sets = []
    for path in rollout_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        samples = list(_flatten_samples(payload.get("samples", [])))
        rewards = [
            float(reward)
            for sample in samples
            if isinstance((reward := _sample_field(sample, "reward")), (int, float))
        ]
        response_lengths = [int(_sample_field(sample, "response_length", 0)) for sample in samples]
        versions = sorted(
            {
                str(version)
                for sample in samples
                for version in (_sample_field(sample, "weight_versions", []) or [])
            }
        )
        if not samples or not any(length > 0 for length in response_lengths):
            raise ValidationFailure(f"rollout dump has no generated response: {path}")
        if path == rollout_paths[0] and len(set(rewards)) < 2:
            raise ValidationFailure(f"training rollout reward has no within-group variance: {rewards}")
        version_sets.append(versions)
        rollout_summaries.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "sample_count": len(samples),
                "rewards": rewards,
                "response_lengths": response_lengths,
                "weight_versions": versions,
            }
        )
    if not version_sets[0] or not version_sets[1] or version_sets[0] == version_sets[1]:
        raise ValidationFailure(f"regeneration did not observe a new rollout weight version: {version_sets}")

    details = {
        "max_train_rollout_logprob_abs_diff": max(diff_values),
        "logprob_threshold": args.logprob_threshold,
        "positive_grad_norms": [value for value in grad_values if value > 0],
        "rollouts": rollout_summaries,
    }
    (run_dir / "rl-closure-summary.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return details


def _archive(run_dir: Path) -> dict[str, Any]:
    archive_path = run_dir.with_suffix(".tar.gz")
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(run_dir, arcname=run_dir.name)
    digest = _sha256_file(archive_path)
    checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    checksum_path.write_text(f"{digest}  {archive_path.name}\n", encoding="utf-8")
    return {"archive": str(archive_path), "sha256": digest, "checksum_file": str(checksum_path)}


def _default_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local", "intranet"), default="local")
    parser.add_argument("--hf-checkpoint", type=Path)
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--index-json", type=Path)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "qwen4-exp-validation-results")
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--e2e-script", type=Path, default=REPO_ROOT / "scripts" / "run-qwen4-exp-p0-validation.sh"
    )
    parser.add_argument("--logprob-threshold", type=float, default=0.1)
    parser.add_argument("--ray-address", default=os.environ.get("RAY_ADDRESS", "auto"))
    parser.add_argument("--expected-nodes", type=int, default=8)
    parser.add_argument("--expected-gpus", type=int, default=64)
    parser.add_argument("--ray-probe-timeout", type=float, default=300.0)
    parser.add_argument("--skip-unit", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.output_root.expanduser().resolve() / args.run_id
    run_log = RunLog(run_dir)
    failed = None
    try:
        run_log.stage("preflight", lambda: _preflight(args, run_dir))
        if args.hf_checkpoint or (args.config_json and args.index_json):
            run_log.stage("manifest", lambda: _manifest(args, run_dir))
        elif args.mode == "intranet":
            raise ValidationFailure("intranet validation requires checkpoint metadata")
        if not args.skip_unit:
            run_log.stage("unit", lambda: _unit(args, run_dir))
        if args.mode == "intranet":
            run_log.stage("e2e", lambda: _e2e(args, run_dir))
            run_log.stage(
                "checkpoint_alignment",
                lambda: _checkpoint_alignment(run_dir, expected_ranks=args.expected_gpus),
            )
            run_log.stage(
                "model_transition",
                lambda: _model_transition(run_dir, expected_ranks=args.expected_gpus),
            )
            run_log.stage(
                "engine_sync",
                lambda: _engine_sync(run_dir, expected_ranks=args.expected_gpus),
            )
            run_log.stage("rl_closure", lambda: _rl_closure(args, run_dir))
    except Exception as error:
        failed = error
    finally:
        result = {
            "schema": "qwen4-exp-validation/v1",
            "status": "failed" if failed is not None else "passed",
            "failed_stage": run_log.last_failed_stage,
            "error_code": run_log.last_failed_code,
            "error_type": type(failed).__name__ if failed is not None else None,
            "error": str(failed) if failed is not None else None,
        }
        (run_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        run_log.write("validation_finished", **result)
        try:
            archive_details = run_log.stage("archive", lambda: _archive(run_dir))
        except Exception as archive_error:
            if failed is None:
                failed = archive_error
                result.update(
                    {
                        "status": "failed",
                        "failed_stage": run_log.last_failed_stage,
                        "error_code": run_log.last_failed_code,
                        "error_type": type(archive_error).__name__,
                        "error": str(archive_error),
                    }
                )
                (run_dir / "result.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            archive_details = {}

    if failed is not None:
        failure_code = run_log.last_failed_code or "Q4E-000"
        print(f"Qwen4-Exp validation FAILED [{failure_code}]: {failed}", file=sys.stderr)
        print(f"evidence: {run_dir}", file=sys.stderr)
        return 1
    print("Qwen4-Exp validation PASSED")
    print(f"evidence: {run_dir}")
    print(f"archive: {archive_details['archive']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
