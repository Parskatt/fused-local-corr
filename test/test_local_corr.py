import os
import pytest
import torch
import torch.nn.functional as F

from local_corr import local_corr

# Ensure best matmul precision for fair perf baselines
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("highest")

compile_backend = 'inductor'

def _available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def _baseline_nearest(im_A: torch.Tensor, im_B: torch.Tensor, warp: torch.Tensor) -> torch.Tensor:
    # im_A: (B, HW, C), im_B: (B, H, W, C), warp: (B, HW, N, 2) [pixel coords]
    B, H, W, C = im_B.shape
    inds = (warp[..., 1] * W + warp[..., 0]).long()
    im_B_warped = torch.take_along_dim(
        im_B.reshape(B, 1, H * W, C),
        inds[..., None].long(),
        dim=2,
    )
    out = torch.einsum("bnc, bnmc -> bnm", im_A, im_B_warped)
    return out


def _to_normalized(warp_xy: torch.Tensor, H: int, W: int) -> torch.Tensor:
    # Convert pixel coords (x, y) to normalized grid_sample coords in [-1, 1]
    return torch.stack(
        (
            2 * warp_xy[..., 0] / W - 1,
            2 * warp_xy[..., 1] / H - 1,
        ),
        dim=-1,
    )


def _baseline_bilinear(im_A: torch.Tensor, im_B: torch.Tensor, warp_norm: torch.Tensor) -> torch.Tensor:
    # im_A: (B, HW, C), im_B: (B, H, W, C), warp_norm: (B, HW, N, 2) in [-1, 1]
    im_B_warped = F.grid_sample(
        im_B.permute(0, 3, 1, 2),
        warp_norm,
        mode="bilinear",
        align_corners=False,
    )  # (B, C, HW, N)
    out = torch.einsum("bnc, bcnm -> bnm", im_A, im_B_warped)
    return out


@pytest.mark.parametrize("device", _available_devices())
def test_forward_nearest_matches_baseline(device):
    torch.manual_seed(0)
    B, H, W, C, N = 2, 16, 20, 32, 7
    HW = H * W
    im_A = torch.randn(B, HW, C, device=device, dtype=torch.float32)
    im_B = torch.randn(B, H, W, C, device=device, dtype=torch.float32)
    cols = torch.randint(0, W, (B, HW, N), device=device)
    rows = torch.randint(0, H, (B, HW, N), device=device)
    warp = torch.stack((cols, rows), dim=-1).int()

    y_ref = _baseline_nearest(im_A, im_B, warp)
    y = local_corr(im_A, im_B, warp, mode="nearest", normalized_coords=False)
    torch.testing.assert_close(y, y_ref, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("device", _available_devices())
def test_backward_wrt_A_nearest_matches_baseline(device):
    torch.manual_seed(1)
    B, H, W, C, N = 2, 12, 10, 16, 5
    HW = H * W
    im_A_ref = torch.randn(B, HW, C, device=device, dtype=torch.float32, requires_grad=True)
    im_A_op = im_A_ref.detach().clone().requires_grad_()
    im_B = torch.randn(B, H, W, C, device=device, dtype=torch.float32)
    cols = torch.randint(0, W, (B, HW, N), device=device)
    rows = torch.randint(0, H, (B, HW, N), device=device)
    warp = torch.stack((cols, rows), dim=-1).int()

    y_ref = _baseline_nearest(im_A_ref, im_B, warp)
    (y_ref.mean()).backward()
    grad_ref = im_A_ref.grad.detach().clone()

    y = local_corr(im_A_op, im_B, warp, mode="nearest", normalized_coords=False)
    (y.mean()).backward()
    grad_op = im_A_op.grad.detach().clone()

    torch.testing.assert_close(grad_op, grad_ref, atol=1e-6, rtol=1e-5)


def _maybe_sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def _time_ms(fn, iters: int, device: str) -> float:
    # Warmup
    with torch.inference_mode():
        fn()
        _maybe_sync(device)
    import time
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iters):
            fn()
        _maybe_sync(device)
    return (time.perf_counter() - t0) * 1e3 / max(iters, 1)


@pytest.mark.skipif(os.environ.get("FUSED_LOCAL_CORR_PERF") != "1", reason="Set FUSED_LOCAL_CORR_PERF=1 to run perf tests")
@pytest.mark.parametrize("device", _available_devices())
def test_perf_nearest_vs_baseline(device):
    torch.manual_seed(4)
    B, H, W, C, N = 4, 64, 64, 512, 64
    HW = H * W
    im_A = torch.randn(B, HW, C, device=device)
    im_B = torch.randn(B, H, W, C, device=device)
    cols = torch.randint(0, W, (B, HW, N), device=device)
    rows = torch.randint(0, H, (B, HW, N), device=device)
    warp = torch.stack((cols, rows), dim=-1).int()

    def run_baseline():
        return _baseline_nearest(im_A, im_B, warp)

    def run_op():
        return local_corr(im_A, im_B, warp, mode="nearest", normalized_coords=False)

    iters = 20 if device == "cuda" else 5
    t_base = _time_ms(run_baseline, iters, device)
    t_op = _time_ms(run_op, iters, device)

    print(f"[nearest][{device}] baseline: {t_base:.2f} ms/iter, op: {t_op:.2f} ms/iter")


@pytest.mark.skipif(os.environ.get("FUSED_LOCAL_CORR_PERF") != "1", reason="Set FUSED_LOCAL_CORR_PERF=1 to run perf tests")
@pytest.mark.parametrize("device", _available_devices())
def test_perf_bilinear_vs_baseline(device):
    torch.manual_seed(5)
    B, H, W, C, N = 4, 64, 64, 256, 64
    HW = H * W
    im_A = torch.randn(B, HW, C, device=device)
    im_B = torch.randn(B, H, W, C, device=device)
    cols = torch.rand(B, HW, N, device=device) * W
    rows = torch.rand(B, HW, N, device=device) * H
    warp_pix = torch.stack((cols, rows), dim=-1)
    warp_norm = _to_normalized(warp_pix, H, W)

    def run_baseline():
        return _baseline_bilinear(im_A, im_B, warp_norm)

    def run_op():
        return local_corr(im_A, im_B, warp_norm, mode="bilinear", normalized_coords=True)

    iters = 20 if device == "cuda" else 5
    t_base = _time_ms(run_baseline, iters, device)
    t_op = _time_ms(run_op, iters, device)

    print(f"[bilinear][{device}] baseline: {t_base:.2f} ms/iter, op: {t_op:.2f} ms/iter")


@pytest.mark.skipif(os.environ.get("FUSED_LOCAL_CORR_PERF") != "1", reason="Set FUSED_LOCAL_CORR_PERF=1 to run perf tests")
@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
@pytest.mark.parametrize("device", _available_devices())
def test_perf_nearest_compiled_baseline(device):
    torch.manual_seed(6)
    B, H, W, C, N = 4, 64, 64, 512, 64
    HW = H * W
    im_A = torch.randn(B, HW, C, device=device)
    im_B = torch.randn(B, H, W, C, device=device)
    cols = torch.randint(0, W, (B, HW, N), device=device)
    rows = torch.randint(0, H, (B, HW, N), device=device)
    warp = torch.stack((cols, rows), dim=-1).int()

    compiled = torch.compile(_baseline_nearest, backend = compile_backend)

    def run_base():
        return _baseline_nearest(im_A, im_B, warp)

    def run_compiled():
        return compiled(im_A, im_B, warp)

    iters = 20 if device == "cuda" else 5
    t_base = _time_ms(run_base, iters, device)
    t_comp = _time_ms(run_compiled, iters, device)
    print(f"[nearest][{device}] baseline: {t_base:.2f} ms/iter, compiled: {t_comp:.2f} ms/iter")


@pytest.mark.skipif(os.environ.get("FUSED_LOCAL_CORR_PERF") != "1", reason="Set FUSED_LOCAL_CORR_PERF=1 to run perf tests")
@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile not available")
@pytest.mark.parametrize("device", _available_devices())
def test_perf_bilinear_compiled_baseline(device):
    torch.manual_seed(7)
    B, H, W, C, N = 4, 64, 64, 256, 64
    HW = H * W
    im_A = torch.randn(B, HW, C, device=device)
    im_B = torch.randn(B, H, W, C, device=device)
    cols = torch.rand(B, HW, N, device=device) * W
    rows = torch.rand(B, HW, N, device=device) * H
    warp_pix = torch.stack((cols, rows), dim=-1)
    warp_norm = _to_normalized(warp_pix, H, W)

    compiled = torch.compile(_baseline_bilinear, backend = compile_backend)

    def run_base():
        return _baseline_bilinear(im_A, im_B, warp_norm)

    def run_compiled():
        return compiled(im_A, im_B, warp_norm)

    iters = 20 if device == "cuda" else 5
    t_base = _time_ms(run_base, iters, device)
    t_comp = _time_ms(run_compiled, iters, device)
    print(f"[bilinear][{device}] baseline: {t_base:.2f} ms/iter, compiled: {t_comp:.2f} ms/iter")

@pytest.mark.parametrize("device", _available_devices())
def test_forward_bilinear_matches_baseline(device):
    torch.manual_seed(2)
    B, H, W, C, N = 2, 14, 11, 24, 6
    HW = H * W
    im_A = torch.randn(B, HW, C, device=device, dtype=torch.float32)
    im_B = torch.randn(B, H, W, C, device=device, dtype=torch.float32)
    # sample warp in pixel space within image then normalize for baseline and op
    cols = torch.rand(B, HW, N, device=device) * (W - 1)
    rows = torch.rand(B, HW, N, device=device) * (H - 1)
    warp_pix = torch.stack((cols, rows), dim=-1)
    warp_norm = _to_normalized(warp_pix, H, W)

    y_ref = _baseline_bilinear(im_A, im_B, warp_norm)
    y = local_corr(im_A, im_B, warp_norm, mode="bilinear", normalized_coords=True)
    torch.testing.assert_close(y, y_ref, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("device", _available_devices())
def test_backward_wrt_A_bilinear_matches_baseline(device):
    torch.manual_seed(3)
    B, H, W, C, N = 2, 10, 13, 20, 4
    HW = H * W
    im_A_ref = torch.randn(B, HW, C, device=device, dtype=torch.float32, requires_grad=True)
    im_A_op = im_A_ref.detach().clone().requires_grad_()
    im_B = torch.randn(B, H, W, C, device=device, dtype=torch.float32)
    cols = torch.rand(B, HW, N, device=device) * (W - 1)
    rows = torch.rand(B, HW, N, device=device) * (H - 1)
    warp_pix = torch.stack((cols, rows), dim=-1)
    warp_norm = _to_normalized(warp_pix, H, W)

    y_ref = _baseline_bilinear(im_A_ref, im_B, warp_norm)
    (y_ref.mean()).backward()
    grad_ref = im_A_ref.grad.detach().clone()

    y = local_corr(im_A_op, im_B, warp_norm, mode="bilinear", normalized_coords=True)
    (y.mean()).backward()
    grad_op = im_A_op.grad.detach().clone()

    torch.testing.assert_close(grad_op, grad_ref, atol=1e-6, rtol=1e-5)


