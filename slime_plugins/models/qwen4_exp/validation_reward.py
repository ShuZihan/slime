"""Deterministic non-zero-variance reward for the Qwen4-Exp RL gate."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from slime.utils.types import Sample


async def alternating_group_reward(args: Any, samples: list["Sample"], **kwargs: Any) -> list[float]:
    del args, kwargs
    group_offsets: dict[int, int] = {}
    rewards = []
    for sample in samples:
        group_index = int(sample.group_index if sample.group_index is not None else -1)
        offset = group_offsets.get(group_index, 0)
        group_offsets[group_index] = offset + 1
        rewards.append(float(offset % 2))
    return rewards
