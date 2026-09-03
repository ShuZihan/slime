#!/usr/bin/env python3
"""Execute the patched QSA short-extend write-plan contract on CPU."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import torch


def _load_write_plan(backend_path: Path):
    tree = ast.parse(backend_path.read_text(encoding="utf-8"), filename=str(backend_path))
    target = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_qsa_write_plan"
        ),
        None,
    )
    if target is None:
        raise RuntimeError(f"_qsa_write_plan is absent from {backend_path}")
    target.decorator_list = []
    module = ast.Module(body=[target], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, str(backend_path), "exec"), namespace)
    return namespace["_qsa_write_plan"]


def _safe_group_locs(member_rows: torch.Tensor, valid: torch.Tensor, ratio: int) -> torch.Tensor:
    group_locs = member_rows[:, None] + torch.arange(ratio, dtype=torch.long)
    return torch.where(valid[:, None], group_locs, torch.zeros_like(group_locs))


def check_sglang_patch(sglang_root: Path) -> dict[str, object]:
    source_root = sglang_root / "python" / "sglang" / "srt"
    backend_path = source_root / "layers" / "attention" / "qwen_sparse_attn_backend.py"
    indexer_path = source_root / "layers" / "attention" / "qsa" / "qsa_indexer.py"
    metadata_path = source_root / "layers" / "attention" / "qsa" / "metadata.py"
    for path in (backend_path, indexer_path, metadata_path):
        if not path.is_file():
            raise RuntimeError(f"SGLang source is absent: {path}")

    indexer_source = indexer_path.read_text(encoding="utf-8")
    metadata_source = metadata_path.read_text(encoding="utf-8")
    if "metadata.compress_plan_valid[:, None]" not in indexer_source:
        raise RuntimeError("QSA indexer does not mask padded source rows")
    if "compress_plan_valid: Optional[torch.Tensor]" not in metadata_source:
        raise RuntimeError("QSA indexer metadata lacks compress_plan_valid")

    write_plan = _load_write_plan(backend_path)
    token_slots = torch.tensor([[4, 5, 6, 7]], dtype=torch.int64)
    empty_plan = write_plan(
        token_slot_table=token_slots,
        start_blocks=torch.tensor([0]),
        end_blocks=torch.tensor([0]),
        capacity=1,
        compress_ratio=4,
        row_token_starts=torch.tensor([0]),
        prefix_lens=torch.tensor([0]),
    )
    empty_locs = _safe_group_locs(empty_plan[3], empty_plan[4], ratio=4)
    if empty_plan[4].tolist() != [False] or empty_locs.tolist() != [[0, 0, 0, 0]]:
        raise RuntimeError("one-token QSA plan did not fold its padded row to source row zero")

    mixed_plan = write_plan(
        token_slot_table=token_slots,
        start_blocks=torch.tensor([0]),
        end_blocks=torch.tensor([1]),
        capacity=2,
        compress_ratio=4,
        row_token_starts=torch.tensor([0]),
        prefix_lens=torch.tensor([0]),
    )
    mixed_locs = _safe_group_locs(mixed_plan[3], mixed_plan[4], ratio=4)
    if mixed_plan[4].tolist() != [True, False]:
        raise RuntimeError("QSA write-plan validity mask is incorrect")
    if mixed_locs.tolist() != [[0, 1, 2, 3], [0, 0, 0, 0]]:
        raise RuntimeError("QSA plan changed a valid group or exposed an invalid source row")

    return {
        "one_token_validity": empty_plan[4].tolist(),
        "one_token_source_rows": empty_locs.tolist(),
        "mixed_validity": mixed_plan[4].tolist(),
        "mixed_source_rows": mixed_locs.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-root", required=True, type=Path)
    args = parser.parse_args()
    print(check_sglang_patch(args.sglang_root.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
