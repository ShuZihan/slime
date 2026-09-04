from __future__ import annotations

import json

import pytest
import torch

from slime.utils.hf_config import ensure_hf_auto_classes, load_hf_config


@pytest.mark.unit
def test_qwen4_exp_config_loads_without_transformers_native_model(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": ["Qwen4ExpForConditionalGeneration"],
                "model_type": "qwen4_exp",
                "text_config": {
                    "model_type": "qwen4_exp_text",
                    "dtype": "bfloat16",
                    "hidden_size": 32,
                    "num_hidden_layers": 4,
                },
            }
        ),
        encoding="utf-8",
    )

    ensure_hf_auto_classes(tmp_path)
    config = load_hf_config(tmp_path)

    assert type(config).__name__ == "Qwen4ExpConfig"
    assert config.model_type == "qwen4_exp"
    assert config.architectures == ["Qwen4ExpForConditionalGeneration"]
    assert config.text_config.model_type == "qwen4_exp_text"
    assert config.text_config.hidden_size == 32
    assert config.text_config.dtype == torch.bfloat16


@pytest.mark.unit
def test_unrelated_config_keeps_transformers_default_loader(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "llama", "hidden_size": 32}),
        encoding="utf-8",
    )

    ensure_hf_auto_classes(tmp_path)
    config = load_hf_config(tmp_path)

    assert config.model_type == "llama"
    assert config.hidden_size == 32
