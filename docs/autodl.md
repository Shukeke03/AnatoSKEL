# AutoDL 兼容性约定

本次只准备本地代码与配置，没有在服务器创建环境或下载数据。

| 项目 | 固定版本 / 约定 |
|---|---|
| Python | 3.11（本地实际安装 3.11.16） |
| PyTorch | 2.8.0 |
| Linux CUDA wheel | 2.8.0+cu128，官方 CUDA 12.8 wheel |
| 本地 macOS | 同版本 macOS arm64 wheel，CPU 运行 3D 验证 |
| 租机 | RTX 4090 24GB 或 RTX 5090；核对实际显存，标准 5090 为 32GB |
| 精度 | CUDA BF16 autocast；坐标、网格采样与距离损失保持 FP32 |
| 扩展 | 仅原生 PyTorch，无 FlashAttention/xFormers/PyTorch3D 自编译依赖 |
| Linux 系统 | x86_64，glibc ≥2.28；优先 Ubuntu 22.04/24.04 |

[PyTorch 官方安装表](https://pytorch.org/get-started/previous-versions/)
明确提供 2.8.0/cu128；[PyTorch 维护者说明](https://discuss.pytorch.org/t/request-for-sm-120-rtx-50xx-cuda-12-8-support-in-pytorch/222207/2)
确认 CUDA 12.8 构建支持 Blackwell。
[AutoDL 镜像表](https://www.autodl.com/docs/base_config/)提供 PyTorch 2.8.0 / Python 3.12 / CUDA 12.8。
届时选择该镜像，在独立 conda 环境安装 Python 3.11，避免复用镜像的 Python 3.12 包。
还要核对镜像的 Ubuntu/glibc，不能仅按 CUDA 版本选旧 Ubuntu 18.04：
[官方 cu128 wheel 索引](https://download.pytorch.org/whl/cu128/torch/)
中 Python3.11/Linux 的2.8.0包为 `manylinux_2_28_x86_64`。

后续服务器命令（目前尚未执行）：

```bash
conda env create -f environment-gpu.yml
conda activate anatoskel
python -m pip install -r requirements-cuda.txt
python -m pip install -e '.[dev]'
nvidia-smi
ldd --version
anatoskel doctor --require-cuda
pytest -q
```

保留官方 cu128 index；不要换成未核实的默认镜像包或把本地 macOS 的 freeze 文件用于 Linux。
PyTorch wheel 带运行库，`nvcc --version`、`nvidia-smi` 显示的 CUDA 能力、
`torch.version.cuda` 是三个不同信息。没有 CUDA 扩展时不需要另装系统 Toolkit。

对于 5090，宿主驱动必须支持该 GPU；建议 R570 或更新的受支持驱动，
最终以 `doctor` 的实际 Conv3d、attention、5D grid_sample 前向/反向通过为准。
[NVIDIA CUDA 12.8 GA 对应 Toolkit 驱动表](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)
列出 Linux 570.26；这不是所有 CUDA 12.x wheel 的统一最低驱动要求，
也不能代替 AutoDL 对具体 GPU 的驱动支持。

4090 compute capability 是 8.9，5090 是 12.0。`doctor` 会记录设备、架构列表、运行库和运算结果。
尚未在这两种 GPU 上运行，不能承诺显存或吞吐。研究配置用 B=1、每步最多两个局部骨块、
梯度检查点、全局 token 上限与粗体素预算；租机后先测峰值，再调整局部批量/采样点数。
不缩放患者真实骨尺寸，也不将完整 CT 强行拉伸成固定立方体。
