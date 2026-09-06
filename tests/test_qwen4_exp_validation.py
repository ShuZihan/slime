from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from slime_plugins.models.qwen4_exp.validation import CheckpointLoadAudit, tensor_fingerprint
from slime_plugins.models.qwen4_exp.validation_reward import alternating_group_reward

_VALIDATE_PATH = Path(__file__).parents[1] / "tools" / "qwen4_exp" / "validate.py"
_SPEC = importlib.util.spec_from_file_location("qwen4_exp_validate", _VALIDATE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_VALIDATE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _VALIDATE
_SPEC.loader.exec_module(_VALIDATE)

ValidationFailure = _VALIDATE.ValidationFailure
PUBLIC_CHECKPOINT_CONTRACT = _VALIDATE.PUBLIC_CHECKPOINT_CONTRACT
_checkpoint_alignment = _VALIDATE._checkpoint_alignment
_engine_sync = _VALIDATE._engine_sync
_flatten_engine_checksums = _VALIDATE._flatten_engine_checksums
_model_transition = _VALIDATE._model_transition
_rl_closure = _VALIDATE._rl_closure
_validate_public_checkpoint_contract = _VALIDATE._validate_public_checkpoint_contract


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


@pytest.mark.unit
def test_node_preflight_locates_megatron_core_under_a_namespace_package(tmp_path, monkeypatch):
    # Match upstream's layout: megatron/ has no __init__.py; core/ does.
    sources = {
        "megatron/core/__init__.py": "",
        "sglang/__init__.py": "",
        "sglang/srt/models/qwen4_exp.py": (
            '_preserve_on_weight_reset = True\ngetattr(self.config, "seed", None) or 1234'
        ),
        "sglang/srt/utils/weight_checker.py": "_preserve_on_weight_reset",
        "sglang/srt/layers/attention/qsa/metadata.py": "compress_plan_valid",
        "sglang/srt/layers/attention/qsa/qsa_indexer.py": "metadata.compress_plan_valid[:, None]",
        "sglang/srt/layers/attention/qwen_sparse_attn_backend.py": "group_plan_valid",
    }
    for name, body in sources.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"],
        cwd=tmp_path,
        check=True,
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    namespace = ModuleType("megatron")
    namespace.__file__ = None
    namespace.__path__ = [str(tmp_path / "megatron")]
    core = ModuleType("megatron.core")
    core.__file__ = str(tmp_path / "megatron/core/__init__.py")
    namespace.core = core
    monkeypatch.setitem(sys.modules, "megatron", namespace)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    for name, path in (
        ("sglang", tmp_path / "sglang/__init__.py"),
        ("slime", _VALIDATE.REPO_ROOT / "slime/__init__.py"),
    ):
        module = ModuleType(name)
        module.__file__ = str(path)
        monkeypatch.setitem(sys.modules, name, module)
    transformers = ModuleType("transformers")
    transformers.__version__ = "fixture"
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map": {"w": "shard"}}')
    (tmp_path / "shard").touch()
    (tmp_path / "marker").touch()

    local = _VALIDATE._module_git_state("megatron.core")
    remote = _VALIDATE._ray_node_probe(
        str(tmp_path),
        str(tmp_path / "marker"),
        {"sglang": {"commit": commit}, "megatron_commit": commit, "transformers_version": "fixture"},
        _VALIDATE._critical_source_hashes(_VALIDATE.REPO_ROOT),
    )
    assert local["commit"] == remote["megatron_commit"] == commit
    assert Path(remote["megatron_module"]) == Path(core.__file__)
    assert not remote["failures"]


@pytest.mark.unit
def test_tensor_fingerprint_is_deterministic_and_detects_value_change():
    left = torch.arange(32, dtype=torch.float32)
    right = left.clone()

    first = tensor_fingerprint("weight", left)
    second = tensor_fingerprint("weight", right)
    right[17] += 1
    changed = tensor_fingerprint("weight", right)

    assert first == second
    assert first["sample_sha256"] != changed["sample_sha256"]


@pytest.mark.unit
def test_checkpoint_load_audit_records_mapped_and_streamed_evidence(tmp_path):
    audit = CheckpointLoadAudit(SimpleNamespace(qwen4_exp_validation_dir=str(tmp_path)))
    trainable = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    indexer = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    ple = torch.arange(12, dtype=torch.float32).reshape(6, 2)

    audit.observe(
        "decoder.layers.0.linear_attn.in_proj_qkv.weight",
        trainable,
        expected=trainable.clone(),
        load_mode="mapped_exact_copy",
    )
    audit.observe(
        "decoder.layers.3.self_attention.indexer.index_qk_proj.weight",
        indexer,
        expected=indexer.clone(),
        load_mode="mapped_exact_copy",
    )
    audit.observe(
        "decoder.layers.1.ple.ple_embedding.ngram_embedding.weight",
        ple,
        expected=None,
        load_mode="streamed_source_slices",
    )
    payload = audit.finish()

    assert payload is not None
    assert payload["parameter_count"] == 3
    assert all(item["exact_sample_match"] for item in payload["sampled"])
    records = [
        json.loads(line)
        for line in (tmp_path / "checkpoint" / "rank-00000.jsonl").read_text().splitlines()
    ]
    assert records == [payload]


@pytest.mark.unit
def test_checkpoint_alignment_gate_requires_all_lifecycle_classes(tmp_path):
    _write_jsonl(
        tmp_path / "checkpoint" / "rank-00000.jsonl",
        [
            {
                "event": "checkpoint_load_alignment",
                "rank": 0,
                "parameter_count": 3,
                "parameter_numel": 36,
                "lifecycle_counts": {"trainable_sync": 1, "static_shared": 2},
                "sampled": [
                    {
                        "name": "decoder.layers.0.linear_attn.in_proj_qkv.weight",
                        "lifecycle": "trainable_sync",
                        "exact_sample_match": True,
                    },
                    {
                        "name": "decoder.layers.3.self_attention.indexer.index_qk_proj.weight",
                        "lifecycle": "static_shared",
                        "exact_sample_match": True,
                    },
                ],
                "streamed": [
                    {
                        "name": "decoder.layers.1.ple.ple_embedding.ngram_embedding.weight",
                        "lifecycle": "static_shared",
                    }
                ],
            }
        ],
    )

    details = _checkpoint_alignment(tmp_path)

    assert details["rank_count"] == 1
    assert details["sampled_exact_match_count"] == 2
    assert details["streamed_ple_count"] == 1


@pytest.mark.unit
def test_checkpoint_alignment_gate_rejects_incomplete_rank_coverage(tmp_path):
    _write_jsonl(
        tmp_path / "checkpoint" / "rank-00000.jsonl",
        [
            {
                "event": "checkpoint_load_alignment",
                "rank": 0,
                "sampled": [
                    {
                        "name": "decoder.layers.0.linear_attn.in_proj_qkv.weight",
                        "lifecycle": "trainable_sync",
                        "exact_sample_match": True,
                    },
                    {
                        "name": "decoder.layers.3.self_attention.indexer.index_qk_proj.weight",
                        "lifecycle": "static_shared",
                        "exact_sample_match": True,
                    },
                ],
                "streamed": [
                    {"name": "decoder.layers.1.ple.ple_embedding.ngram_embedding.weight"}
                ],
            }
        ],
    )

    with pytest.raises(ValidationFailure, match="rank coverage mismatch"):
        _checkpoint_alignment(tmp_path, expected_ranks=2)


@pytest.mark.unit
def test_model_transition_gate_pairs_rollout_and_checks_static_state(tmp_path):
    trainable_before = {
        "name": "decoder.layers.0.linear_attn.in_proj_qkv.weight",
        "lifecycle": "trainable_sync",
        "sample_sha256": "before",
    }
    static_before = {
        "name": "decoder.layers.3.self_attention.indexer.index_qk_proj.weight",
        "lifecycle": "static_shared",
        "sample_sha256": "static",
    }
    _write_jsonl(
        tmp_path / "model" / "rank-00000.jsonl",
        [
            {"event": "before_train", "rollout_id": 7, "tensors": [trainable_before, static_before]},
            {
                "event": "after_train",
                "rollout_id": 7,
                "tensors": [
                    {**trainable_before, "sample_sha256": "after"},
                    dict(static_before),
                ],
            },
        ],
    )

    details = _model_transition(tmp_path)

    assert details["rollout_id"] == 7
    assert details["changed_trainable_count"] == 1
    assert details["unchanged_static_count"] == 1


def _checksum_record(event, trainable, static):
    return {
        "event": event,
        "rollout_id": 3,
        "responses": [
            {
                "success": True,
                "ranks": [
                    {
                        "parallelism_info": {"tp_rank": 0},
                        "checksums": {
                            "model.layers.0.linear_attn.in_proj_qkv.weight": trainable,
                            "model.layers.3.self_attn.indexer.index_qk_proj.weight": static,
                        },
                    }
                ],
            }
        ],
    }


@pytest.mark.unit
def test_engine_sync_gate_accepts_trainable_change_and_static_identity(tmp_path):
    before = _checksum_record("before_weight_sync", "old", "fixed")
    after = _checksum_record("after_weight_sync", "new", "fixed")
    _write_jsonl(tmp_path / "rollout" / "engine-checksums.jsonl", [before, after])

    assert _flatten_engine_checksums(before)[(0, 0)][
        "model.layers.3.self_attn.indexer.index_qk_proj.weight"
    ] == "fixed"
    details = _engine_sync(tmp_path)

    assert details["rollout_id"] == 3
    assert details["changed_trainable_count"] == 1
    assert details["static_tensor_count"] == 1


@pytest.mark.unit
def test_engine_sync_gate_rejects_static_change(tmp_path):
    before = _checksum_record("before_weight_sync", "old", "fixed")
    after = _checksum_record("after_weight_sync", "new", "changed")
    _write_jsonl(tmp_path / "rollout" / "engine-checksums.jsonl", [before, after])

    with pytest.raises(ValidationFailure, match="static tensor changed"):
        _engine_sync(tmp_path)


@pytest.mark.unit
def test_public_checkpoint_contract_checks_exact_release_identity():
    details = copy.deepcopy(PUBLIC_CHECKPOINT_CONTRACT)
    _validate_public_checkpoint_contract(details)

    details["tensor_count"] -= 1
    with pytest.raises(ValidationFailure, match="tensor_count"):
        _validate_public_checkpoint_contract(details)


@pytest.mark.unit
def test_rl_closure_gate_checks_logprob_gradient_reward_and_weight_version(tmp_path):
    (tmp_path / "e2e.log").write_text(
        "step 0: {'train/train_rollout_logprob_abs_diff': 0.02, 'train/grad_norm': 1.25}\n",
        encoding="utf-8",
    )
    rollout_dir = tmp_path / "rollout_data"
    rollout_dir.mkdir()
    first = [
        {"reward": 0.0, "response_length": 4, "weight_versions": ["0"]},
        {"reward": 1.0, "response_length": 5, "weight_versions": ["0"]},
    ]
    second = [
        {"reward": 0.0, "response_length": 3, "weight_versions": ["1"]},
        {"reward": 1.0, "response_length": 6, "weight_versions": ["1"]},
    ]
    torch.save({"rollout_id": 0, "samples": first}, rollout_dir / "0.pt")
    torch.save({"rollout_id": 1, "samples": second}, rollout_dir / "1.pt")

    details = _rl_closure(SimpleNamespace(logprob_threshold=0.1), tmp_path)

    assert details["max_train_rollout_logprob_abs_diff"] == pytest.approx(0.02)
    assert details["positive_grad_norms"] == [pytest.approx(1.25)]
    assert [item["weight_versions"] for item in details["rollouts"]] == [["0"], ["1"]]


@pytest.mark.unit
def test_alternating_group_reward_has_variance_inside_each_group():
    samples = [
        SimpleNamespace(group_index=4),
        SimpleNamespace(group_index=4),
        SimpleNamespace(group_index=8),
        SimpleNamespace(group_index=8),
    ]

    rewards = asyncio.run(alternating_group_reward(SimpleNamespace(), samples))

    assert rewards == [0.0, 1.0, 0.0, 1.0]
