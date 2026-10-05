# 本地验证记录

日期：2026-10-05。环境：macOS arm64、Miniforge `anatoskel`、Python3.11.16、PyTorch2.8.0。

已执行：

| 检查 | 结果 |
|---|---|
| `pip check` | 无依赖冲突 |
| `ruff check .` | 通过 |
| `python -m compileall -q src scripts` | 通过 |
| `pytest -q` | 39项通过 |
| `anatoskel doctor` | CPU Conv3d、attention、5D grid_sample前向/反向通过 |
| `python scripts/smoke.py` | 合成NIfTI→prior优化→主模型优化→checkpoint→预测ROI推理/验证/参考OBJ导出通过 |
| `build-atlas` / 带atlas的`train-prior` | 合成训练集构建/弱配准/安全checkpoint加载通过 |

测试覆盖非循环移位窗口、物理位置/affine与stride、低精度累加、
BF16→FP32场查询反向、多骨ROI独立affine与关系attention、零旋转更新梯度、
数据泄漏检查、CT覆盖域与标注有效域分离、TSDF边界censor、反射网格绕序、
开放扫描边界、弱配准歧义、OBJ多对象，以及续训与连续训练权重一致。

非空预测网格集成测试使用合成椭球prior拟合及测试用单类别路由，以确保模型场→OBJ通路被实际执行。
它不是临床重建结果。默认两步prior/一步主模型smoke的实际预测可以没有零等值面，
参考网格明确来自GT，不能冒充预测或医学准确率。

尚未执行：真实VerSe下载/去重/训练、真实atlas质量检查、4090/5090 CUDA运行与显存峰值、
外部数据集验证。CUDA兼容性采用官方版本/平台信息选择；实际服务器核验留到租机后执行。
当前研究代码的局部query在粗/细训练边界detach；没有宣称完整端到端优化或已验证解剖轴。
