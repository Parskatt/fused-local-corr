import torch
from torch import Tensor
from typing import Literal

__all__ = ["local_corr"]

def to_pixel_coords(x, H, W):
    return torch.stack((W*(x[...,0]+1)/2, H*(x[...,1]+1)/2),dim = -1)

def local_corr(
    im_A: Tensor, 
    im_B: Tensor, 
    warp: Tensor, 
    mode: Literal["nearest", "bilinear"] = "nearest",
    normalized_coords: bool = False,
) -> Tensor:
    """Performs a correlation of a by b at coordinates c
    
    Args:
        im_A: Input tensor (B x HW_A x D)
        im_B: Input tensor (B x H_B x W_B x D)
        warp: Coordinates tensor (B x HW_A x N x 2)
        mode: Interpolation mode for correlation ("nearest" or "bilinear")
        normalized_coords: if True, warp is between [-1,1], else H,W of im_B assumed
    """
    H,W = im_B.shape[-3:-1]
    if normalized_coords:
        warp = to_pixel_coords(warp, H, W)
    return torch.ops.fused_local_corr.corr(im_A, im_B, warp, mode)

@torch.library.register_fake("fused_local_corr::corr")
def _(a, b, c, mode="nearest"):
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    # warp dtype depends on interpolation mode: int for nearest, float for bilinear
    if mode == "bilinear":
        torch._check(c.dtype == torch.float)
    else:  # nearest
        torch._check(c.dtype in (torch.int, torch.long, torch.float))
    torch._check(a.device == b.device)
    torch._check(c.device == a.device)
    torch._check(
        mode in ["nearest", "bilinear"], 
        f"Interpolation mode must be 'nearest' or 'bilinear', got: {mode}"
    )
    B, HW, N = c.shape[0], c.shape[1], c.shape[2]
    return torch.zeros((B, HW, N)).to(a)

@torch.library.register_fake("fused_local_corr::corr_backward_A")
def _(grad, im_B, warp, mode="nearest"):
    # grad: (B, HW, N), im_B: (B, H, W, C), warp: (B, HW, N, 2)
    torch._check(grad.dtype == torch.float)
    torch._check(im_B.dtype == torch.float)
    if mode == "bilinear":
        torch._check(warp.dtype == torch.float)
    else:
        torch._check(warp.dtype in (torch.int, torch.long, torch.float))
    torch._check(grad.device == im_B.device)
    torch._check(warp.device == grad.device)
    torch._check(
        mode in ["nearest", "bilinear"],
        f"Interpolation mode must be 'nearest' or 'bilinear', got: {mode}"
    )
    B = grad.shape[0]
    HW = grad.shape[1]
    C = im_B.shape[-1]
    return torch.zeros((B, HW, C)).to(im_B)

def _backward(ctx, grad):
    a, b, c = ctx.saved_tensors
    mode = ctx.mode  # Retrieve saved interpolation mode
    grad_a, grad_b, grad_c = None, None, None
    if ctx.needs_input_grad[0]:
        grad_a = torch.ops.fused_local_corr.corr_backward_A(grad, b, c, mode)
    
    if ctx.needs_input_grad[1]:
        raise NotImplementedError("No backward impl. for im_B yet")
    
    if ctx.needs_input_grad[2]:
        raise NotImplementedError("No backward impl. for the coords")
    
    # Return None for the mode parameter
    return grad_a, grad_b, grad_c, None

def _setup_context(ctx, inputs, output):
    a, b, c, mode = inputs
    
    # Save tensors needed for backward
    saved_a = a if ctx.needs_input_grad[0] else None
    saved_b = b if ctx.needs_input_grad[0] else None
    saved_c = c if ctx.needs_input_grad[0] else None
    ctx.save_for_backward(saved_a, saved_b, saved_c)
    
    # Save interpolation mode
    ctx.mode = mode

# Register the autograd function
torch.library.register_autograd(
    "fused_local_corr::corr",
    _backward,
    setup_context=_setup_context
)