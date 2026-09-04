from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron")

from megatron.core import mpu

from slime.backends.megatron_utils.data import DataIterator, get_batch
from slime.utils import accelerator
from slime_plugins.models.qwen4_exp.reference import Qwen4ExpPackedLayout


@pytest.mark.unit
def test_qwen4_exp_batch_preserves_boundaries_positions_mask_and_pad_token(monkeypatch):
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(accelerator, "current_device", lambda: "cpu")
    monkeypatch.setattr(accelerator, "device", lambda *_args, **_kwargs: torch.device("cpu"))

    rollout_data = {
        "tokens": [torch.tensor([1, 2, 3]), torch.tensor([4, 5])],
        "loss_masks": [torch.tensor([1]), torch.tensor([1])],
        "total_lengths": [3, 2],
        "response_lengths": [1, 1],
    }
    iterator = DataIterator(rollout_data, micro_batch_indices=[[0, 1]])

    batch = get_batch(
        iterator,
        keys=("tokens", "loss_masks", "total_lengths", "response_lengths"),
        pad_multiplier=1,
        pad_token_id=63,
    )

    assert batch["tokens"].tolist() == [[1, 2, 3, 4, 5, 63]]
    assert [tokens.tolist() for tokens in batch["unconcat_tokens"]] == [[1, 2, 3], [4, 5]]
    assert batch["packed_seq_params"].cu_seqlens_q.tolist() == [0, 3, 5, 6]
    assert batch["full_loss_masks"].tolist() == [[0, 1, 0, 1, 0, 0]]

    layout = Qwen4ExpPackedLayout.from_cu_seqlens(
        batch["tokens"].numel(),
        batch["packed_seq_params"].cu_seqlens_q,
    )
    assert layout.boundaries == (0, 3, 5, 6)
    assert layout.positions.tolist() == [0, 1, 2, 0, 1, 0]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("pad_token_id", "expected"),
    [(None, 248044), (248046, 248046), ([248044, 248046], 248044)],
)
def test_qwen4_exp_pad_token_falls_back_to_eos(pad_token_id, expected):
    from slime.utils.hf_config import resolve_pad_token_id

    config = SimpleNamespace(pad_token_id=pad_token_id, eos_token_id=248044)

    assert resolve_pad_token_id(config) == expected
