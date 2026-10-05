# 首轮真实 CT 实验

设备：RTX 5090，32GB 显存；CPU 配额25核，内存配额90GiB。
Python3.11.16、PyTorch2.8.0+cu128、CUDA运行库12.8。
配置 `configs/autodl_5090.yaml`：全扫描B=1，局部骨块B≤6，128³/1mm，8192点/骨，BF16。
先验batch32，100轮；主模型100轮（141×100=14,100个扫描步骤）。
粗分支保留所有26种椎体身份；细分支与首轮先验限定腰椎20–25。
两个CPU扫描worker预取，六个ROI线程准备监督；真实速度在status.json中持续更新。

训练缓存只包含train患者。完整FOV按1mm物理间距重采样，CT以FP16存盘，标签以uint8存盘；
加载时CT恢复FP32。该缓存会引入一次插值与FP16存储量化，避免重复解压原始CT；
它不会把扫描变成立方体或缩放真实骨尺寸。粗体积单独从原CT生成，并校验配置/manifest匹配。
验证和测试使用原始CT坐标，GT缺标区域仍被屏蔽。

CT覆盖域截断与标注有效域边界分别记录。
先验仅使用CT中完整可见的训练骨；VerSe对完整可见骨提供完整标签的约定
见[原论文](https://arxiv.org/pdf/2103.06360)。本配置对这些骨按CT覆盖域拟合完整形状，
主模型仍沿用保守标注有效域。PCA初始化及atlas配准给出弱姿态监督，不代表解剖轴已验证。
冻结先验前，检查每个实际训练骨的采样场正负号与形状代码更新覆盖率。

后台入口：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python -u scripts/run_autodl.py --publish-demo > outputs/first5090/logs/run.log 2>&1
```

依次执行cache→CUDA自检→训练atlas→先验→先验审计→主训练→checkpoint审计→全量val/test→固定CT示例。
主模型每100步保存可继续训练的checkpoint。失败写`JOB_FAILED`并停止；
所有步骤及Git上传成功后写`JOB_COMPLETE`，不会自动关机。
重跑入口可复用缓存、已完成的先验与主模型checkpoint。

cache、atlas、prior_samples为CPU数据准备阶段，GPU利用率可以为0%。
prior_samples会先为全部141个训练扫描生成骨块、距离场监督与配准坐标；
之后进入prior_train的GPU先验优化，再启动主模型train。
监控脚本会显示先验样本准备进度。主训练启动需确认step持续增加、loss有限，
以及有有效腰椎骨块的步骤中fine_bones大于0。

监控：

```bash
watch -n 5 /root/autodl-tmp/conda-envs/anatoskel/bin/python scripts/monitor_run.py
tail -F outputs/first5090/logs/run.log
```

`train_eta_hours`只包含剩余训练，之后还有全量验证、测试与导出时间。
运行前测量GPU图和真实ROI准备时间用于粗估，进入主训练后按实际步骤更新；不能保证固定关机时间。

示例病例在训练前固定：官方test集中按scan_id排序，首个有至少四个公开腰椎质心的病例。
推理只接收CT与affine，不使用GT质心/ROI。示例若为空，会原样报告，不用GT网格替换预测。
Git仅上传 `examples/first5090/{bones.obj,bones.json,metrics.json,README.md}`。
权重、CT、缓存、全部病例的输出和训练日志留在服务器。
服务器使用仅限本私有仓库的deploy key上传，不保存Mac的GitHub访问令牌。

结果完成后在Mac拉取：

```bash
git -C /Users/shuxing/AIlearning/AnatoSKEL pull --ff-only
```
