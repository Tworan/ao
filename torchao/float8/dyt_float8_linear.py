from typing import Optional 

import torch 
import torch.nn as nn

from torchao.float8.config import ScalingType, Float8LinearConfig
from torchao.float8.stateful_float8_linear import StatefulFloat8Linear
from torchao.float8.float8_scaling_utils import hp_tensor_to_float8_static

from torchao.float8.fsdp_utils import (
    WeightWithDelayedFloat8CastTensor,
    WeightWithDynamicFloat8CastTensor,
    WeightWithStaticFloat8CastTensor,
)

"""
class LayerNormDyT(nn.Module):
    def __init__(self, normalized_shape, alpha_init_value):
        super(LayerNormDyT, self).__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value

        self.alpha = nn.Parameter(torch.empty(1))
        self.weight = nn.Parameter(torch.empty(normalized_shape))

    def forward(self, x):
        return self.weight * torch.tanh(self.alpha * x)
"""

class DyTFloat8Linear(StatefulFloat8Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.DyT = None 
    
    def initialize_dyt(self, dyt: nn.Module):
        self.DyT = dyt 

    def cast_input_to_float8(self, input: torch.Tensor) -> torch.Tensor:
        assert self.DyT is not None, "DyT module is not initialized."

        input_fp8 = hp_tensor_to_float8_static(
            input,
            self.DyT.weight.abs().max(), 
            self.config.cast_config_input.target_dtype,
            self.linear_mm_config,
        )

        return input_fp8

    @classmethod
    def from_float(
        cls,
        mod,
        config: Optional[Float8LinearConfig] = None,
    ):
        """
        Create an nn.Linear with fp8 compute from a regular nn.Linear

        Args:
            mod (torch.nn.Linear): nn.Linear to convert
            config (Optional[Float8LinearConfig]): configuration for conversion to float8
        """
        if config is None:
            config = Float8LinearConfig()
        with torch.device("meta"):
            new_mod = cls(
                mod.in_features,
                mod.out_features,
                bias=False,
                config=config,
            )
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        # need to create buffers again when moving from meta device to
        # real device
        new_mod.create_buffers()

        # If FSDP float8 all-gather is on, wrap the weight in a float8-aware
        # tensor subclass. This must happen last because:
        # 1. weight needs to be on the correct device to create the buffers
        # 2. buffers need to be already created for the delayed scaling version
        #    of the weight wrapper to be initialized
        if config.enable_fsdp_float8_all_gather:
            if config.cast_config_weight.scaling_type is ScalingType.DYNAMIC:
                new_mod.weight = torch.nn.Parameter(
                    WeightWithDynamicFloat8CastTensor(
                        new_mod.weight,
                        new_mod.linear_mm_config,
                        new_mod.config.cast_config_weight.target_dtype,
                    )
                )
            elif config.cast_config_weight.scaling_type is ScalingType.DELAYED:
                new_mod.weight = torch.nn.Parameter(
                    WeightWithDelayedFloat8CastTensor(
                        new_mod.weight,
                        new_mod.fp8_amax_weight,
                        new_mod.fp8_amax_history_weight,
                        new_mod.fp8_scale_weight,
                        new_mod.linear_mm_config,
                        new_mod.config.cast_config_weight.target_dtype,
                        new_mod.is_amax_initialized,
                    )
                )
            else:
                assert config.cast_config_weight.scaling_type is ScalingType.STATIC
                new_mod.weight = torch.nn.Parameter(
                    WeightWithStaticFloat8CastTensor(
                        new_mod.weight,
                        new_mod.fp8_static_scale_weight,
                        new_mod.linear_mm_config,
                        new_mod.config.cast_config_weight.target_dtype,
                    )
                )

        return new_mod