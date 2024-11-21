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
    from mlstm_simple_torch.mlstm_simple.model import mLSTMConfig, round_up_to_next_multiple_of
except:
    raise ImportError("Need mlstm_simple_torch to be installed")


logger = logging.get_logger(__name__)


class xLSTMConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`XLSTM`]. It is used to instantiate a xLSTM
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the XLSTM

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        num_heads (`int`, *optional*, defaults to 128):
            Number of heads for the evolution matrices of mamba 2.
        head_dim (`int`, *optional*, defaults to 64):
            Dimension of each head.
        vocab_size (`int`, *optional*, defaults to 32768):
            Vocabulary size of the xLSTM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`xLSTMModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the embeddings and hidden states.
        state_size (`int`, *optional*, defaults to 128): shape of the state space latents.
        num_hidden_layers (`int`, *optional*, defaults to 64):
            Number of hidden layers in the model.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon to use in the layer normalization layers.
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 0):
            The id of the beginning of sentence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the end of sentence token in the vocabulary.
        expand (`int`, *optional*, defaults to 2): Expanding factor used to determine the intermediate size.
        conv_kernel (`int`, *optional*, defaults to 4): Size of the convolution kernel.
        n_groups (`int`, *optional*, defaults to 8):
            Number of groups for the evolution matrices of mamba 2.
        use_bias (`bool`, *optional*, defaults to `False`):
            Whether or not to use bias in ["in_proj", "out_proj"] of the mixer block
        use_conv_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use bias in the convolution layer of the mixer block.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        initializer_range (`float`, *optional*, defaults to 0.1):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        residual_in_fp32 (`bool`, *optional*, defaults to `True`):
            Whether or not residuals should be in `float32`. If set to `False` residuals will keep the same `dtype` as the rest of the model
        time_step_rank (`Union[int,str]`, *optional*, defaults to `"auto"`):
            Rank of the discretization projection matrix. `"auto"` means that it will default to `math.ceil(self.hidden_size / 16)`
        time_step_min (`float`, *optional*, defaults to 0.001):
            Minimum `time_step` used to bound `dt_proj.bias`.
        time_step_max (`float`, *optional*, defaults to 0.1):
            Maximum `time_step` used to bound `dt_proj.bias`.
        time_step_floor (`float`, *optional*, defaults to 0.0001):
            Minimum clamping value of the `dt_proj.bias` layer initialization.
        time_step_limit (`tuple`, *optional*, defaults to `(0.0, inf)`):
            Accepted range of time step values.
        rescale_prenorm_residual (`bool`, *optional*, defaults to `False`):
            Whether or not to rescale `out_proj` weights when initializing.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the cache should be used.
        rms_norm (`bool`, *optional*, defaults to `True`):
            Whether to use RMS norm or not.
        chunk_size (`int`, *optional*, defaults to 256):
            Size of the chunks that will comprise the sequence.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie word embeddings or not.


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

    model_type = "xLSTM"

    def __init__(
        self,
        vocab_size: int = 50304,
        embedding_dim: int = 4096,
        num_blocks: int = 32,
        num_heads: int = 8,
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
        output_logit_soft_cap:float = 30,
        # reset_at_document_boundaries: False,
        forward_backend_name: str = "chunkwise--triton_xl_chunk",
        step_backend_name: str = "triton_fused",
        add_forward_backend_padding: bool = False,
        ffn_proj_factor: float = 2.667,
        ffn_round_up_to_multiple_of: int = 64,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.head_dim = self.embedding_dim // self.num_heads
        self.tie_word_embeddings = tie_word_embeddings
        self.add_embedding_dropout = add_embedding_dropout
        self.add_post_blocks_norm = add_post_blocks_norm
        self.norm_eps = norm_eps
        self.init_distribution = init_distribution
        self.init_distribution_embed = init_distribution_embed
        self.output_init_fn = output_init_fn
        self.logits_soft_cap = logits_soft_cap
        self.add_post_norm = add_post_norm
        self.qk_dim_factor = qk_dim_factor
        self.v_dim_factor = v_dim_factor
        self.igate_bias_init_range = igate_bias_init_range
        self.add_qk_norm = add_qk_norm
        self.cell_norm_eps = cell_norm_eps
        self.gate_soft_cap = gate_soft_cap
        self.output_logit_soft_cap = output_logit_soft_cap
        self.forward_backend_name = forward_backend_name
        self.step_backend_name = step_backend_name
        self.add_forward_backend_padding = add_forward_backend_padding
        self.ffn_proj_factor = ffn_proj_factor
        self.ffn_round_up_to_multiple_of = ffn_round_up_to_multiple_of

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
            return_last_states=self.return_last_states,
            forward_backend_name=self.forward_backend_name,
            step_backend_name=self.step_backend_name,
            add_forward_backend_padding=self.add_forward_backend_padding,
            ffn_proj_factor=self.ffn_proj_factor,
            ffn_round_up_to_multiple_of=self.ffn_round_up_to_multiple_of,
            gate_soft_cap=self.gate_soft_cap,
            output_logit_soft_cap=self.output_logit_soft_cap,
        )
