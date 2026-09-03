"""Hugging Face configuration loading with runtime-owned model registrations."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


_QWEN4_EXP_MODEL_TYPES = {"qwen4_exp", "qwen4_exp_text"}


def resolve_pad_token_id(hf_config: Any, *, default: int = 0) -> int:
    """Resolve the training pad token, using EOS for checkpoints without one."""

    if hf_config is None:
        return default
    text_config = getattr(hf_config, "text_config", hf_config)
    pad_token_id = getattr(text_config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(text_config, "eos_token_id", None)
    if isinstance(pad_token_id, (list, tuple)):
        pad_token_id = pad_token_id[0] if pad_token_id else None
    return default if pad_token_id is None else int(pad_token_id)


def _read_local_config(name_or_path: str | Path) -> dict[str, Any] | None:
    try:
        path = Path(name_or_path).expanduser()
    except TypeError:
        return None
    config_path = path / "config.json" if path.is_dir() else path
    if config_path.name != "config.json" or not config_path.is_file():
        return None
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    return config if isinstance(config, dict) else None


def _register_tokenizer(config_class) -> None:
    """Bind the Qwen2 tokenizer shared by the Qwen4-Exp checkpoint."""

    from transformers import AutoTokenizer, Qwen2Tokenizer, Qwen2TokenizerFast

    try:
        AutoTokenizer.register(
            config_class,
            slow_tokenizer_class=Qwen2Tokenizer,
            fast_tokenizer_class=Qwen2TokenizerFast,
            exist_ok=True,
        )
    except TypeError:
        # Transformers releases predating ``exist_ok`` still accept repeated
        # registration when the same class tuple is supplied.
        try:
            AutoTokenizer.register(
                config_class,
                slow_tokenizer_class=Qwen2Tokenizer,
                fast_tokenizer_class=Qwen2TokenizerFast,
            )
        except ValueError as error:
            if "already" not in str(error).lower():
                raise


@lru_cache(maxsize=1)
def _register_qwen4_exp_fallback():
    """Register a config-only Qwen4-Exp type when SGLang is unavailable."""

    import torch
    from transformers import AutoConfig, PretrainedConfig

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    class Qwen4ExpTextConfig(PretrainedConfig):
        model_type = "qwen4_exp_text"
        base_config_key = "text_config"

        def __init__(self, **kwargs):
            for name in ("dtype", "torch_dtype"):
                if kwargs.get(name) in dtype_map:
                    kwargs[name] = dtype_map[kwargs[name]]
            super().__init__(**kwargs)

        def to_dict(self):
            result = super().to_dict()
            for name in ("dtype", "torch_dtype"):
                if isinstance(result.get(name), torch.dtype):
                    result[name] = str(result[name]).removeprefix("torch.")
            return result

    class Qwen4ExpConfig(PretrainedConfig):
        model_type = "qwen4_exp"
        sub_configs = {"text_config": Qwen4ExpTextConfig}

        def __init__(self, text_config=None, vision_config=None, **kwargs):
            if isinstance(text_config, dict):
                text_config = Qwen4ExpTextConfig(**text_config)
            self.text_config = text_config
            self.vision_config = vision_config
            super().__init__(**kwargs)

    for model_type, config_class in (
        (Qwen4ExpTextConfig.model_type, Qwen4ExpTextConfig),
        (Qwen4ExpConfig.model_type, Qwen4ExpConfig),
    ):
        try:
            AutoConfig.register(model_type, config_class)
        except ValueError as error:
            if "already" not in str(error).lower() and "used" not in str(error).lower():
                raise
    _register_tokenizer(Qwen4ExpConfig)
    return Qwen4ExpConfig


def ensure_hf_auto_classes(name_or_path: str | Path) -> bool:
    """Register Qwen4-Exp config/tokenizer classes for a local checkpoint.

    SGLang owns the serving-time config implementation pinned by the P0 build.
    The local fallback carries the same public config fields for CPU tools that
    run without an SGLang installation.
    """

    config = _read_local_config(name_or_path)
    if config is None or config.get("model_type") not in _QWEN4_EXP_MODEL_TYPES:
        return False

    try:
        from sglang.srt.configs.qwen4_exp import Qwen4ExpConfig, Qwen4ExpTextConfig
    except ModuleNotFoundError as error:
        missing_module = error.name or ""
        if missing_module != "sglang" and not missing_module.startswith("sglang."):
            raise
        _register_qwen4_exp_fallback()
        return True

    from transformers import AutoConfig

    for model_type, config_class in (
        (Qwen4ExpTextConfig.model_type, Qwen4ExpTextConfig),
        (Qwen4ExpConfig.model_type, Qwen4ExpConfig),
    ):
        try:
            AutoConfig.register(model_type, config_class)
        except ValueError as error:
            if "already" not in str(error).lower() and "used" not in str(error).lower():
                raise
    _register_tokenizer(Qwen4ExpConfig)
    return True


def load_hf_config(name_or_path: str | Path, *, trust_remote_code: bool = True, **kwargs):
    """Load a config after installing model registrations required by its runtime."""

    from transformers import AutoConfig

    ensure_hf_auto_classes(name_or_path)
    return AutoConfig.from_pretrained(name_or_path, trust_remote_code=trust_remote_code, **kwargs)
