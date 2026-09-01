# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config.model import _normalize_config_dtype


def test_string_config_dtype_is_normalized_after_hf_override():
    assert _normalize_config_dtype("float16") is torch.float16
    assert _normalize_config_dtype("torch.bfloat16") is torch.bfloat16
