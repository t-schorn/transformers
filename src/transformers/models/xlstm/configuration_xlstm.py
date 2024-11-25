# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""XLSTM configuration"""

import math

from ...configuration_utils import PretrainedConfig
from ...utils import logging

try:
    from mlstm_simple.model import mLSTMConfig, round_up_to_next_multiple_of
except ImportError:
    # TODO This was only added for testing, since mlstm_torch_simple is not installable yet
    import sys
    import os

    sys.path.append(os.path.split(os.path.abspath(__file__))[0] + "/../../../../mlstm_simple_torch")
    from mlstm_simple.model import mLSTMConfig, round_up_to_next_multiple_of
    # raise ImportError("Need mlstm_simple_torch to be installed")


logger = logging.get_logger(__name__)


class xLSTMConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`xLSTM`]. It is used to instantiate a xLSTM
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the XLSTM

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:



    Example:

    ```python
    >>> from transformers import xLSTMConfig, xLSTMModel

    >>> # Initializing a xLSTM configuration
    >>> configuration = xLSTMConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = xLSTMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "xlstm"

    def __init__(
        self,
        vocab_size: int = 50304,
        embedding_dim: int = 4096,
        num_blocks: int = 32,
        num_heads: int = 8,
        use_bias: bool = False,
        tie_word_embeddings: bool = False,
        add_embedding_dropout: bool = False,
        add_post_blocks_norm: bool = True,
        norm_eps: float = 1e-6,
        # init_distribution: "normal",
        # init_distribution_embed: "normal",
        # output_init_fn: str = "wang",
        # lm_head_dtype: "bfloat16",
        add_post_norm: bool = False,
        qk_dim_factor: float = 0.5,
        v_dim_factor: float = 1.0,
        mlstm_round_up_to_multiple_of: int = 64,
        # gate_dtype: float32,
        # backend: triton_kernels,
        # backend_name: "max_triton_noslice",
        igate_bias_init_range: float = -10.0,
        add_qk_norm: bool = False,
        # cell_norm_type: rmsnorm,
        cell_norm_eps: float = 1e-6,
        gate_soft_cap: float = 15.0,
        norm_reduction_force_float32: bool = True,
        output_logit_soft_cap: float = 30,
        # reset_at_document_boundaries: False,
        forward_backend_name: str = "chunkwise--triton_xl_chunk",
        step_backend_name: str = "triton_fused",
        add_forward_backend_padding: bool = False,
        ffn_proj_factor: float = 2.667,
        ffn_round_up_to_multiple_of: int = 64,
        return_last_states: bool = True,
        use_cache: bool = True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.use_bias = use_bias
        self.head_dim = self.embedding_dim // self.num_heads
        self.tie_word_embeddings = tie_word_embeddings
        self.add_embedding_dropout = add_embedding_dropout
        self.add_post_blocks_norm = add_post_blocks_norm
        self.norm_eps = norm_eps
        self.output_logit_soft_cap = output_logit_soft_cap
        self.add_post_norm = add_post_norm
        self.qk_dim_factor = qk_dim_factor
        self.v_dim_factor = v_dim_factor
        self.mlstm_round_up_to_multiple_of = mlstm_round_up_to_multiple_of
        self.igate_bias_init_range = igate_bias_init_range
        self.add_qk_norm = add_qk_norm
        self.cell_norm_eps = cell_norm_eps
        self.gate_soft_cap = gate_soft_cap
        self.norm_reduction_force_float32 = norm_reduction_force_float32
        self.forward_backend_name = forward_backend_name
        self.step_backend_name = step_backend_name
        self.add_forward_backend_padding = add_forward_backend_padding
        self.ffn_proj_factor = ffn_proj_factor
        self.ffn_round_up_to_multiple_of = ffn_round_up_to_multiple_of
        # adapted as it is a runtime config
        self.return_last_states = True
        self.use_cache = use_cache

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def qk_dim(self):
        return round_up_to_next_multiple_of(
            self.embedding_dim * self.qk_dim_factor,
            multiple_of=self.mlstm_round_up_to_multiple_of,
        )

    @property
    def v_dim(self):
        return round_up_to_next_multiple_of(
            self.embedding_dim * self.v_dim_factor,
            multiple_of=self.mlstm_round_up_to_multiple_of,
        )

    @property
    def qk_head_dim(self):
        return self.qk_dim // self.num_heads

    @property
    def v_head_dim(self):
        return self.v_dim // self.num_heads

    def to_mlstm_block_config(self):
        return mLSTMConfig(
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
            vocab_size=self.vocab_size,
            use_bias=self.use_bias,
            norm_eps=self.norm_eps,
            norm_reduction_force_float32=self.norm_reduction_force_float32,
            qk_dim_factor=self.qk_dim_factor,
            v_dim_factor=self.v_dim_factor,
            mlstm_round_up_to_multiple_of=self.mlstm_round_up_to_multiple_of,
            # this was changed as it is a runtime config
            return_last_states=True,
            forward_backend_name=self.forward_backend_name,
            step_backend_name=self.step_backend_name,
            add_forward_backend_padding=self.add_forward_backend_padding,
            ffn_proj_factor=self.ffn_proj_factor,
            ffn_round_up_to_multiple_of=self.ffn_round_up_to_multiple_of,
            gate_soft_cap=self.gate_soft_cap,
            output_logit_soft_cap=self.output_logit_soft_cap,
        )
