# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Custom replacement for `torch.nn.functional.grid_sample` that
supports arbitrarily high order gradients between the input and output.
Only works on 2D images and assumes
`mode='bilinear'`, `padding_mode='zeros'`, `align_corners=False`."""

import warnings
from torch import autograd
import torch
from pkg_resources import parse_version

# pylint: disable=redefined-builtin
# pylint: disable=arguments-differ
# pylint: disable=protected-access

#----------------------------------------------------------------------------

enabled = False  # Enable the custom op by setting this to true.

#check API version and use correct function
_use_pytorch_1_11_api = parse_version(torch.__version__) >= parse_version('1.11.0a') # Allow

#----------------------------------------------------------------------------

def grid_sample(input, grid):
    if _should_use_custom_op():
        return _GridSample2dForward.apply(input, grid)
    return torch.nn.functional.grid_sample(input=input, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=False)

#----------------------------------------------------------------------------

def _should_use_custom_op():
    if not enabled:
        return False
    if any(torch.__version__.startswith(x) for x in ['1.7.', '1.8.', '1.9','1.11']):
        return True
    warnings.warn(f'grid_sample_gradfix not supported on PyTorch {torch.__version__}. Falling back to torch.nn.functional.grid_sample().')
    return False

#----------------------------------------------------------------------------

class _GridSample2dForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid):
        assert input.ndim == 4
        assert grid.ndim == 4
        output = torch.nn.functional.grid_sample(input=input, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        ctx.save_for_backward(input, grid)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        grad_input, grad_grid = _GridSample2dBackward.apply(grad_output, input, grid)
        return grad_input, grad_grid

#----------------------------------------------------------------------------

class _GridSample2dBackward(autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, input, grid):
        op = torch._C._jit_get_operation("aten::grid_sampler_2d_backward")
        grad_input, grad_grid = op[0](grad_output, input, grid, 0, 0, False, [True, True])
        ctx.save_for_backward(grid)

        return grad_input, grad_grid

    @staticmethod
    def backward(ctx, grad_grad_input, grad_grad_grid):
        (grid,) = ctx.saved_tensors
        grad_grad_output = None

        if ctx.needs_input_grad[0]:
            grad_grad_output = GridSampleForward.apply(grad_grad_input, grid)

        return grad_grad_output, None, None

#----------------------------------------------------------------------------
