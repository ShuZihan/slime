"""Qwen4-Exp model integration for slime.

The package keeps model-independent semantics in small modules so checkpoint
conversion, Megatron training, and SGLang rollout can share one contract.
"""

from .config import Qwen4ExpP0Config
from .lifecycle import ParameterLifecycle, build_lifecycle_manifest

__all__ = ["ParameterLifecycle", "Qwen4ExpP0Config", "build_lifecycle_manifest"]
