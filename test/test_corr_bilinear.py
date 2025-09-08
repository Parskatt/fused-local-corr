from matplotlib import pyplot as plt
from local_corr import corr
import torch
import torch.nn.functional as F
T = 10
import time

def to_normalized(x, H, W):
    return torch.stack((2*x[...,0]/W - 1, 2*x[...,1]/H - 1), dim = -1)

def baseline_corr_bilinear(im_A, im_B, warp):
    B, H, W, C = im_B.shape
    warp_n = to_normalized(warp, H, W)
    # print(warp_n)
    im_B_warped = F.grid_sample(
        im_B.permute(0,3,1,2),
        warp_n,
        mode = "bilinear",
        align_corners=False,
    )
    out = torch.einsum("bnc, bcnm -> bnm", im_A, im_B_warped)
    return out

device = "cpu"
B, HW, H, W, C, N = 8, 32*32, 32, 32, 512, 64
im_A = torch.randn(B, HW, C).to(device).requires_grad_()
im_B = torch.randn(B, H, W, C).to(device)
cols = torch.randint(0, W, (B, HW, N)).to(device)+ .6
rows = torch.randint(0, H, (B, HW, N)).to(device)+ .6
warp = torch.stack((cols, rows), dim = -1)

x_baseline = baseline_corr_bilinear(im_A, im_B, warp)
print(torch.autograd.grad(x_baseline, im_A))
x = corr(im_A, im_B, warp, mode = "bilinear")
print(torch.autograd.grad(x_baseline, im_A))

diff = (x-x_baseline)
print(diff)
big_diff = (diff.abs() > 1e-1)
print(diff.abs().max())
print(big_diff.float().mean())
print(warp[big_diff])
