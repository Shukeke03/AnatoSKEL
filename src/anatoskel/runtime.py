"""Check the installed runtime with the operations this model actually uses."""
import platform

import torch
import torch.nn.functional as F


def choose_device(value="auto"):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use CPU locally or CUDA on the training server; 3D MPS is not enabled")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def doctor(require_cuda=False):
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
              "cuda_available": torch.cuda.is_available()}
    if str(torch.__version__).split("+")[0] != "2.8.0":
        raise RuntimeError("AnatoSKEL currently pins PyTorch 2.8.0")
    if require_cuda and (not torch.cuda.is_available() or torch.version.cuda != "12.8"):
        raise RuntimeError("AutoDL target requires torch 2.8.0+cu128 and an available CUDA device")
    device = choose_device("cuda" if require_cuda else "auto")
    if device.type == "cuda":
        report.update(gpu=torch.cuda.get_device_name(),
                      capability=list(torch.cuda.get_device_capability()),
                      compiled_architectures=torch.cuda.get_arch_list())
    x = torch.randn(1, 2, 8, 8, 8, device=device, requires_grad=True)
    weight = torch.randn(4, 2, 3, 3, 3, device=device, requires_grad=True)
    y = F.conv3d(x, weight, padding=1)
    grid = torch.zeros(1, 2, 2, 2, 3, device=device, requires_grad=True)
    sampled = F.grid_sample(y, grid, align_corners=False)
    q = torch.randn(1, 2, 8, 16, device=device, requires_grad=True)
    attention = F.scaled_dot_product_attention(q, q, q)
    (sampled.square().mean() + attention.square().mean()).backward()
    if not all(torch.isfinite(t.grad).all() for t in (x, weight, grid, q)):
        raise RuntimeError("Nonfinite gradients in device operation checks")
    if device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            F.conv3d(x.detach(), weight.detach(), padding=1).sum().item()
        torch.cuda.synchronize()
        report["bf16_conv3d"] = "passed"
    report["conv3d_attention_grid_sample_backward"] = "passed"
    report["device_tested"] = str(device)
    return report
