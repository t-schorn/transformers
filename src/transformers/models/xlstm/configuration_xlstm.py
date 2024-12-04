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
    from mlstm_simple.model import (
        mLSTMConfig,
        round_up_to_next_multiple_of,
        ChunkwiseKernelType,
        SequenceKernelType,
        StepKernelType,
        BackendModeType,
        DtypeType,
        WeightModeType,
    )
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
        norm_reduction_force_float32: bool = True,
        tie_word_embeddings: bool = False,
        add_out_norm: bool = True,
        norm_eps: float = 1e-6,
        # mlstm_layer
        qk_dim_factor: float = 0.5,
        v_dim_factor: float = 1.0,
        mlstm_round_up_to_multiple_of: int = 64,
        # mlstm backend
        chunkwise_kernel: ChunkwiseKernelType = "chunkwise--native_autograd",
        sequence_kernel: SequenceKernelType = "native_sequence__native",
        step_kernel: StepKernelType = "native",
        # nedded to enable generation
        mode: BackendModeType = "inference",
        chunk_size: int = 64,
        # needed to be true for generation
        return_last_states: bool = True,
        autocast_kernel_dtype: DtypeType = "bfloat16",
        eps: float = 1e-6,
        inference_state_dtype: DtypeType = "float32",
        # feedforward
        ffn_proj_factor: float = 2.667,
        ffn_round_up_to_multiple_of: int = 64,
        # capping
        gate_soft_cap: float = 15.0,
        output_logit_soft_cap: float = 30.0,
        # weights
        weight_mode: WeightModeType = "single",
        # HF interface
        use_cache: bool = True,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        force_bos_token_insert: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.use_bias = use_bias
        self.tie_word_embeddings = tie_word_embeddings
        self.add_out_norm = add_out_norm
        self.norm_eps = norm_eps
        self.norm_reduction_force_float32 = norm_reduction_force_float32
        # mlstm_layer
        self.qk_dim_factor = qk_dim_factor
        self.v_dim_factor = v_dim_factor
        self.mlstm_round_up_to_multiple_of = mlstm_round_up_to_multiple_of
        # mlstm backend
        self.chunkwise_kernel = chunkwise_kernel
        self.sequence_kernel = sequence_kernel
        self.step_kernel = step_kernel
        self.mode = mode
        self.chunk_size = chunk_size
        self.return_last_states = return_last_states
        self.autocast_kernel_dtype = autocast_kernel_dtype
        self.eps = eps
        self.inference_state_dtype = inference_state_dtype
        # feedforward
        self.ffn_proj_factor = ffn_proj_factor
        self.ffn_round_up_to_multiple_of = ffn_round_up_to_multiple_of
        # capping
        self.gate_soft_cap = gate_soft_cap
        self.output_logit_soft_cap = output_logit_soft_cap
        self.weight_mode = weight_mode

        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.force_bos_token_insert = force_bos_token_insert

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
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            use_bias=self.use_bias,
            add_out_norm=self.add_out_norm,
            norm_eps=self.norm_eps,
            norm_reduction_force_float32=self.norm_reduction_force_float32,
            # mlstm_layer
            qk_dim_factor=self.qk_dim_factor,
            v_dim_factor=self.v_dim_factor,
            mlstm_round_up_to_multiple_of=self.mlstm_round_up_to_multiple_of,
            # mlstm backend
            chunkwise_kernel=self.chunkwise_kernel,
            sequence_kernel=self.sequence_kernel,
            step_kernel=self.step_kernel,
            mode=self.mode,
            chunk_size=self.chunk_size,
            return_last_states=self.return_last_states,
            autocast_kernel_dtype=self.autocast_kernel_dtype,
            eps=self.eps,
            inference_state_dtype=self.inference_state_dtype,
            # feedforward
            ffn_proj_factor=self.ffn_proj_factor,
            ffn_round_up_to_multiple_of=self.ffn_round_up_to_multiple_of,
            # capping
            gate_soft_cap=self.gate_soft_cap,
            output_logit_soft_cap=self.output_logit_soft_cap,
            weight_mode=self.weight_mode,
        )
