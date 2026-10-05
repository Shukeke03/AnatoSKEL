# 数据集决定

第一阶段采用 **VerSe 2019/2020 完整重组版**，重建腰椎，粗定位保留完整可见脊柱上下文。
[官方说明与下载入口](https://github.com/anjany/verse)报告 374 CT / 355 名患者，
提供 CT、逐椎体 mask、编号及质心，数据许可 CC BY-SA 4.0，工具代码 MIT。
使用官网推荐的 complete/restructured 版本，不混用原挑战标注格式。
本轮没有下载真实数据；数据许可不因本仓库 MIT 许可而改变。

遵守发布 split，以患者为单位核验。2019 和 2020 存在重叠，不能简单拼接；
根据共同患者标识及必要的 CT 指纹生成统一 manifest。不能从测试集创建 atlas、形状先验或标准化统计。
当前 manifest 读取器检查患者跨 split、重复 scan 与 CT 路径；未来导入时仍须解决跨发布的患者别名，
相同内容不同文件名需要额外指纹核验，不能依赖路径去重。

编号映射：1–7 C1–C7，8–19 T1–T12，20–25 L1–L6，28 T13；
26 骶骨和 27 尾骨没有该数据集的标注。模型的 26 个实体类别加 no-object 共 27 类。
第一版输出腰椎 20–25，允许部分视野，不强迫每例有五个椎体。
未标注边界骨/原 CT 外部必须在 `valid_mask` 标为无效。
如果不能确认背景标注完整，应先修正有效域再训练，不把缺标骨当负样本。

JSONL manifest 每行：

```json
{"patient_id":"verse001", "scan_id":"verse001_series1", "split":"train", "ct":"raw/ct.nii.gz", "mask":"labels/vertebrae.nii.gz", "valid_mask":"labels/valid.nii.gz"}
```

路径相对 manifest 文件；`valid_mask` 是与 CT 同 affine 的二值**标注有效域** NIfTI，可省略时表示整个扫描域均已确认标注有效。
可另提供 `ct_valid_mask` 表示**实际 CT 覆盖域**（默认整个 CT 体积），如扫描内部存在已知无效区域。
训练的标注有效域与 CT 覆盖域取交集；推理只使用 CT 覆盖域，避免 GT 缺标区域影响预测。
`mask` 可以在推理时省略，训练必须有标注。质心文件可保留作 provenance，但中心监督从实际有效 mask 计算。
准备好文件后运行 `anatoskel inspect-data --manifest ...`；这只校验本地数据。

后续候选：

| 目的 | 数据集 | 当前决定 |
|---|---|---|
| 扩展脊柱与外部验证 | [CTSpine1K](https://github.com/MIRACLE-Center/CTSpine1K) | 后续审核患者重叠和 CC BY-NC-SA 4.0 条件后引入 |
| 骨盆阶段 | [CTPelvic1K](https://github.com/MIRACLE-Center/CTPelvic1K) | 后续独立标签/许可审核，不与腰椎直接混训 |
| 骨折碎片 | [PENGWIN](https://pengwin.grand-challenge.org/) | 独立碎片实例任务；当前隐式骨模型不等同碎片检测 |

首期不把上述数据作为已确定的训练输入，先把 VerSe 的患者去重、有效域与 canonical frame 流程跑通。
