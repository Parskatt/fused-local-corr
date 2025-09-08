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

def perf(f, name, use_custom_corr, device):
    torch.cuda.reset_max_memory_allocated()
    f(im_A,im_B, r, flow = warp, use_custom_corr=use_custom_corr, sample_mode = "bilinear")
    t0 = time.perf_counter()
    for t in range(T):
        y = f(im_A,im_B, r, flow = warp, use_custom_corr=use_custom_corr, sample_mode = "bilinear")
        torch.autograd.grad(y.mean(), im_A)
        # `print(y.mean())
        # print(y.max())
        torch.cuda.synchronize()
    print(f"{name} performance on {device=}: \n time: {(time.perf_counter()-t0)/T} \n max GPU mem: {torch.cuda.max_memory_allocated()/10**6} MB")

devices = ["cuda"]#"cpu", 
for device in devices:
    r = 7
    res = 32
    B, HW, H, W, C = 8, res*res, res, res, 512
    im_A = torch.randn(B, C, H, W).to(device).requires_grad_()
    im_B = torch.randn(B, C, H, W).to(device)
    warp = torch.randint(0, W, (B, H, W, 2)).to(device)+ .6
    warp = to_normalized(warp, H, W).permute(0,3,1,2)
    for use_custom in [False, True]:
        perf(local_correlation, "custom" if use_custom else "native", use_custom_corr=use_custom, device = device)

    # native = local_correlation(im_A, im_B, r, flow = warp,use_custom_corr=False)

    # native_grad = torch.autograd.grad(native.mean(), im_A)[0]

    # # print(native, custom)
    # # print(custom.shape, native.shape)
    # error = (custom-native).abs()
    # error_grad = (custom_grad-native_grad).abs()
    # ftol, gradtol = 1e-5, 1e-10
    # assert error.abs().max() < ftol, error.abs().max()
    # assert error_grad.abs().max() < gradtol, error_grad.abs().max()
