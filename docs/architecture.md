# 模型实现规格

张量统一 `B,C,D,H,W`，NIfTI affine 映射 `x=W,y=H,z=D` 体素中心到 RAS mm。
重采样保持物理视野和比例；各向异性/带 shear 输入先显式 regrid 为正交网格，当前卷积采用各向同性核。
不预测独立 scale，不从完整人体 SKEL 的 q46/β10 或运动树继承不适用的脊柱约束。

## 1. 粗定位

| stage | 默认 channels | Res blocks | 下采样 |
|---|---:|---:|---|
| E0 | 16 | 1 | stem Conv3³，stride1 |
| E1 | 32 | 2 | 首 block stride2 |
| E2 | 64 | 2 | 首 block stride2 |
| E3 | 128 | 2 | 首 block stride2 |
| E4 | 256 | 2 | 首 block stride2 |

Res block：Conv3³ → InstanceNorm affine → LeakyReLU(.01) → Conv3³ → IN → residual → LeakyReLU。
E4 两个 Pre-LN 全局 attention block，8 heads×32，FFN 256→1024→256，GELU。
四级 U-Net skip decoder，channel128/64/32/16，每级一个 residual block。
32 个 learned object slots，4 层 cross-attention → query self-attention → FFN，d256、8 heads；
依次读取 E4/E3/E2/E3，context 上限1024 token。池化 context 后投影，避免高分辨率密集256维特征。
位置编码使用相对 FOV 中心的毫米坐标，保留方向/spacing并保持患者坐标平移等变性。

mask embedding 与64维 decoded features做点积，产生真实逐 query 体素 mask。
27类输出包含26个 VerSe ID和no-object；中心由可见 mask 软质心加无边界位移预测；
可见性两个 logit 的含义是实例存在与原扫描截断。
Hungarian 一对一匹配综合分类、mask Dice/BCE和物理中心。query不绑定固定椎体序号。
全扫描 dense mask 仍受体素预算限制：默认最大维192、总粗体素4M；必要时提高统一毫米 spacing。

## 2. 局部 CT 编码

五阶段 channels32/64/128/256/320，各 stage 两个 residual block，共10个；
默认128³mm块、1mm体素。四级解码 channels256/128/64/32，每级一个 residual block。
E3两层4³ window attention，offset0/2；使用非循环偏移与 padding mask，禁止对侧边界穿越。
E4两层全局 attention。E3/E4 均8 heads，单头32/40；FFN4×宽度，GELU，Pre-LN，dropout0。
当前没有额外 DropPath。研究配置使用非重入式梯度 checkpoint。
局部 query-mask 与 voxel BCE 接受监督，并作为 point refinement 的概率输入。

## 3. 参数与隐式几何

`q256 → Linear256 → GELU → Linear128 → GELU`，分别输出R6、delta-t3、β64、visibility2。
旋转6维向量转换为SO(3)，identity初始化；translation、β初始零。
`t=ROI anchor+delta-t`，anchor 是可见质心/检测中心，不宣称为关节中心。
每块骨独立放置 `x=Ry+t`，不在椎体间执行LBS/FK。

先验输入：canonical y3/100mm + β64 + class16 =83。
MLP4×256，Softplus20；第三隐藏层拼回原83维输入，末层线性输出归一化TSDF。
目标 `clip(distance_mm/8,-1,1)`。网络输出仍线性，不保证全域为数学上的严格SDF。

残差输入：E1–E4各32维采样，共128；y3+β64+class16+prior1+HU2+valid1，总215。
同样4×256 Softplus20、第三层输入skip，末层零初始化，最终场为prior+residual。
HU2是归一化HU及其平方。没有硬幅度截断。
canonical点先通过R/t映回world，再经inverse affine映回CT；
feature采样考虑stride及体素中心，`align_corners=False`。
无效观测处 residual=0，训练/网格提取同时忽略该域。
`ct_valid_mask` 控制实际观测与提取，`valid_mask` 控制标注监督；GT 标注缺失不用于限制推理表面。

## 4. 两轮共享细化

每骨从固定Sobol候选中按低|prior|选择2048点，top-k不反传；它是近表面候选，
质量依赖已训练prior，不能保证早期候选就是实际零面。
当前 canonical候选覆盖默认±65mm，首期针对椎体；骨盆阶段必须重新配置范围。
各轮重新变换点并采CT；feature缓存，编码器不重跑。
point输入128+canonical3+prior1+finalfield1+maskprob1+valid1+HU2=137，投影256。
Pre-LN point cross-attention8heads → 骨间self-attention8heads → FFN256→1024→256 GELU。
骨间attention显式接收相对R9/t3和两骨可见性4维，通过MLP生成head bias。
共享head256→128GELU→70，输出delta-t3、delta-axis-angle3、delta-β64，零初始化。
更新 `R←R exp(δω)`、`t←t+δt_world`、`β←β+δβ`。
每骨ROI可不同affine；编码用BN批维，关系attention恢复B,N，避免变成单骨自注意力。

## 5. 监督与流程约束

Atlas exemplar和prior只能由training患者建立。PCA不是姿态GT；多起点trimmed ICP是弱标签拟合，
partial、几何对称或多解关闭pose监督。高质量弱配准的R/t采用小权重，β可以使用prior拟合代码作弱约束。
没有可信配准时，主监督是mask、类别、中心、有效域TSDF与占据，R不强行套用伪真值。
冻结prior前必须加载带训练步骤与患者provenance的priorcheckpoint；不能默默冻结随机prior。

粗/细loss按阶段反传，fine入口query detach用于降低峰值，因此不是全部损失联合更新粗decoder。
ROI GT+jitter→预测curriculum；训练记录漏ROI，验证包含漏检。后续若要真正端到端，
需要服务器显存实测后调整图保存/重计算策略，不能只删除detach而宣称24GB必然可用。
AMP时3D采样、affine/rigid变换与损失采用FP32，避免BF16不支持或跨autocast类型冲突。
checkpoint包含optimizer、下一scan/epoch、Torch/Python/NumPy随机状态。

## 6. 导出

常规marching cubes只在推理/参考导出执行，不通过它训练。
逐骨患者网格与 `T_i`；关系 `Q_ij=inv(T_i)T_j` 派生，允许平移/间隙/滑移，不使用人体固定运动树。
局部块表面触边时扩大物理采样范围，原CT域依然限制提取；仍触边则记录人工截断。
反射affine修正面绕序。仅保存可观测表面，不padding封口；JSON记录开放边界与未验证轴。
静态ROI截断/visibility不等价于真实结构缺失，骨折碎片需要另建任务与标签。
