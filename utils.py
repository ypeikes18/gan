from dataclasses import dataclass, asdict
from typing import Literal
import torch.nn as nn

@dataclass
class ConvNdKwargs:
    in_channels: int=3
    out_channels: int=3
    kernel_size: int=3
    stride: int=1
    padding: int=1

def create_nd_conv(conv_dims: Literal[1,2,3]=2, **kwargs: ConvNdKwargs) -> nn.Module:
    # populates kwargs with defaults for any ommited parameters
    kwargs = asdict(ConvNdKwargs(**kwargs))
    if conv_dims == 1:
        return nn.Conv1d(**kwargs)
    elif conv_dims == 2:
        return nn.Conv2d(**kwargs)
    elif conv_dims == 3:
        return nn.Conv3d(**kwargs)
    else:
        raise ValueError(f"Unsupported number of dimensions: {conv_dims}")
    
def create_nd_conv_transpose(conv_dims: Literal[1,2,3]=2, **kwargs: ConvNdKwargs) -> nn.Module:
    if conv_dims == 1:
        return nn.ConvTranspose1d(**kwargs)
    elif conv_dims == 2:
        return nn.ConvTranspose2d(**kwargs)
    elif conv_dims == 3:
        return nn.ConvTranspose3d(**kwargs)
    else:
        raise ValueError(f"Unsupported number of dimensions: {conv_dims}")
