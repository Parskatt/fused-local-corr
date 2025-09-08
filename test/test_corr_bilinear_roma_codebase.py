from matplotlib import pyplot as plt
from local_corr import corr
import torch
import torch.nn.functional as F
T = 10
import time
from roma.utils.local_correlation import local_correlation

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

devices = ["cuda"]
for device in devices:
    B, HW, H, W, C, N = 2, 32*32, 32, 32, 512, 64
    im_A = torch.randn(B, C, H, W).to(device).requires_grad_()
    im_B = torch.randn(B, C, H, W).to(device)
    warp = torch.randint(0, W, (B, H, W, 2)).to(device)+ .6
    warp = to_normalized(warp, H, W).permute(0,3,1,2)

    r = 5
    
    native = local_correlation(im_A, im_B, r, flow = warp,use_custom_corr=False)
    custom = local_correlation(im_A, im_B, r, flow = warp,use_custom_corr=True)

    native_grad = torch.autograd.grad(native.sum(), im_A)[0]
    custom_grad = torch.autograd.grad(custom.sum(), im_A)[0]
    # print(native_grad)

    # print(native, custom)
    # print(custom.shape, native.shape)
    error = (custom-native).abs()
    error_grad = (custom_grad-native_grad).abs()
    ftol, gradtol = 2e-5, 2e-5
    # print(error)
    # print(native_grad)
    # print(custom_grad)


    assert error.abs().max() < ftol, error.abs().max()
    assert error_grad.abs().max() < gradtol, error_grad.abs().max()
