
T = 10
import time




def baseline_corr_bilinear(im_A, im_B, warp):
    B, H, W, C = im_B.shape
    # warp_n = to_normalized(warp, H, W)
    # print(warp_n)
    im_B_warped = F.grid_sample(
        im_B.permute(0, 3, 1, 2),
        warp,
        mode="bilinear",
        align_corners=False,
    )
    out = torch.einsum("bnc, bcnm -> bnm", im_A, im_B_warped)
    return out

from matplotlib import pyplot as plt
from local_corr import corr
import torch
import torch.nn.functional as F
def to_normalized(x, H, W):
    return torch.stack((2 * x[..., 0] / W - 1, 2 * x[..., 1] / H - 1), dim=-1)

devices = ["cpu","cuda"]
for device in devices:
    B, HW, H, W, C, N = 2, 32 * 32, 32, 32, 512, 64
    im_A = torch.randn(B, C, H, W).to(device).requires_grad_()
    im_B = torch.randn(B, C, H, W).to(device)
    cols = torch.randint(-10, W + 10, (B, HW, N)).to(device) + torch.randn(B, HW, N).to(device)
    rows = torch.randint(-10, H + 10, (B, HW, N)).to(device) + torch.randn(B, HW, N).to(device)
    warp = torch.stack((cols, rows), dim=-1)
    warp = to_normalized(warp, H, W)
    x_baseline = baseline_corr_bilinear(    
        im_A.reshape(B, C, H * W).permute(0, 2, 1)/ (C**0.5),
        im_B.detach().permute(0, 2, 3, 1),
        warp.detach(),
    )
    baseline_grad = torch.autograd.grad(x_baseline.mean(), im_A)[0]
    x = corr(
        im_A.reshape(B, C, H * W).permute(0, 2, 1)/ (C**0.5),
        im_B.detach().permute(0, 2, 3, 1),
        warp.detach(),
        mode="bilinear",
        normalized_coords=True,
    )
    mygrad = torch.autograd.grad(x.mean(), im_A)[0]
    # print((x_baseline-x).abs().mean())
    assert (x_baseline-x).abs().max() < 1e-5, (x_baseline-x).abs().max()
    assert (mygrad-baseline_grad).abs().max() < 1e-8, (mygrad-baseline_grad).abs().max()
    # print(stuff[1])
