"""Detailed Chinese diagram of ResUNet at v0.1 (2d732ae).

Run: python docs/figures/draw_v01_resunet.py
Documentation only; uses Matplotlib and a local CJK font.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from draw_architectures import INK, MUTED, PALETTE, chinese_font_family


OUT = Path(__file__).resolve().parent
FONT = chinese_font_family()


class Diagram:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(20, 24))
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.ax.set(xlim=(0, 20), ylim=(0, 24), aspect="equal")
        self.ax.axis("off")
        self.fig.patch.set_facecolor("white")
        self.checked = []

    def text(self, x, y, value, fs=10.5, color=INK, **kwargs):
        return self.ax.text(x, y, value, fontsize=fs, color=color,
                            fontfamily=FONT, zorder=5, **kwargs)

    def panel(self, x, y, w, h, title):
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=.10",
            facecolor="#fafbfd", edgecolor="#cbd5df", linewidth=1, zorder=0))
        self.text(x + .18, y + h - .22, title, fs=14, fontweight="bold", va="top")

    def box(self, x, y, w, h, lines, kind="encoder", fs=10.5):
        fill, edge = PALETTE[kind]
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=.07",
            facecolor=fill, edgecolor=edge, linewidth=1.3, zorder=3))
        value = self.text(x + w / 2, y + h / 2, "\n".join(lines), fs=fs,
                          ha="center", va="center", linespacing=1.4)
        self.checked.append((lines[0], (x, y, w, h), value))

    def node(self, x, y, value="⊕", color=INK):
        self.ax.add_patch(Circle((x, y), .17, facecolor="white",
                                 edgecolor=color, linewidth=1.3, zorder=3))
        self.text(x, y, value, ha="center", va="center", fs=11.5, color=color)

    def junction(self, x, y, color=INK):
        self.ax.add_patch(Circle((x, y), .035, facecolor=color, edgecolor=color, zorder=4))

    def arrow(self, points, color=INK, label=None, at=None):
        self.ax.plot(*zip(*points), color="white", linewidth=4.5, zorder=1)
        if len(points) > 2:
            self.ax.plot(*zip(*points[:-1]), color=color, linewidth=1.3, zorder=2)
        self.ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                         mutation_scale=12, linewidth=1.3, color=color,
                                         shrinkA=0, shrinkB=0, zorder=2))
        if label:
            self.text(*at, label, fs=8.8, color=color, ha="center", va="center",
                      bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.3})

    def save(self):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        for name, (x, y, w, h), value in self.checked:
            lo = self.ax.transData.transform((x, y))
            hi = self.ax.transData.transform((x + w, y + h))
            bounds = value.get_window_extent(renderer)
            if (bounds.x0 < lo[0] or bounds.x1 > hi[0]
                    or bounds.y0 < lo[1] or bounds.y1 > hi[1]):
                raise ValueError(f"Text outside box: {name}")
        stem = "anatoskel-v0.1-resunet-detail-zh"
        for suffix in ("svg", "png"):
            path = OUT / f"{stem}.{suffix}"
            metadata = {"Creator": "AnatoSKEL"}
            if suffix == "svg":
                metadata["Date"] = None
            self.fig.savefig(path, dpi=180, metadata=metadata, facecolor="white")
            if suffix == "svg":
                path.write_text("\n".join(s.rstrip() for s in path.read_text().splitlines())
                                + "\n", encoding="utf-8")
        plt.close(self.fig)


def unet(d, offset, fine=False):
    channels = (32, 64, 128, 256, 320) if fine else (16, 32, 64, 128, 256)
    counts = (2, 2, 2, 2, 2) if fine else (1, 2, 2, 2, 2)
    ys = (18.5, 16.9, 15.3, 13.7, 12.1)
    ex, dx, cx = offset + .55, offset + 5.5, offset + 5.05
    ew, dw = 3.55, 3.55
    ec, dc = ex + ew / 2, dx + dw / 2
    d.panel(offset + .35, 11.55, 9.15, 10.7,
            "A2  细尺度 ResUNet（独立参数）" if fine else "A1  粗尺度 ResUNet")
    d.box(ex, 20.15, ew, 1.0,
          ["CT → Stem Conv3d 3³", f"1 → {channels[0]} 通道；步幅 1；无附加归一化 / 激活"],
          "input", fs=9.8)
    d.box(dx, 20.15, dw, 1.0,
          [f"decoded：{channels[0]} 通道，全输入分辨率", "送往后续实例掩膜 / 几何特征分支"], "output", fs=10)
    d.arrow([(ec, 20.15), (ec, 19.85)])
    for i, (width, count, y) in enumerate(zip(channels, counts, ys, strict=True)):
        attention = i == 4 or (fine and i == 3)
        info = ("残差块后：全局 SA ×2；8 头" if i == 4 else
                "残差块后：窗口 SA ×2；8 头" if attention else "该阶段不加自注意力")
        d.box(ex, y, ew, 1.35, [
            f"E{i}：{width} 通道；累计步幅 {2**i}",
            f"RB ×{count}；首块 s={1 if i == 0 else 2}，其余 s=1", info,
        ], "shape" if attention else "encoder", fs=10.1)
        if i:
            d.arrow([(ec, ys[i - 1]), (ec, y + 1.35)])
        if i < 4:
            next_width = channels[i + 1]
            d.box(dx, y, dw, 1.35, [
                f"D{i}：RB ×1，s=1 → {width} 通道",
                f"拼接输入 {next_width} + {width} = {next_width + width} 通道",
                "RB 内部投影残差：1³ Conv，s=1",
            ], "field", fs=10.1)
            mid = y + .675
            d.node(cx, mid, "C", PALETTE["query"][1])
            d.arrow([(ex + ew, mid), (cx - .17, mid)], color=PALETTE["query"][1])
            d.arrow([(cx + .17, mid), (dx, mid)])
            # Main decoder stream is resized before entering the concat node.
            if i < 3:
                lower_top = ys[i + 1] + 1.35
                d.arrow([(dc, lower_top), (dc, lower_top + .12),
                         (cx, lower_top + .12), (cx, mid - .17)],
                        label="↑ 三线性插值", at=(cx + 1.2, lower_top + .12))
    # E4 already includes global attention; it initializes the decoder stream.
    d.arrow([(ex + ew, ys[4] + .675), (cx, ys[4] + .675),
             (cx, ys[3] + .675 - .17)],
            label="↑ 三线性插值", at=(cx + .3, 13.37))
    d.arrow([(dc, 19.85), (dc, 20.15)])
    d.text(offset + .55, 11.85,
           "紫框：自注意力位置；绿色跳连送入 C；橙框：U-Net 解码器残差块。",
           fs=10.1, color=MUTED)


def residual(d):
    d.panel(.35, 7.75, 9.15, 3.5, "B  单个卷积残差块 RB：相加后才激活")
    y = 9.6
    d.text(.7, y, "x", fs=12, va="center")
    for x, w, lines in [
        (1.15, 1.4, ["Conv3d 3³", "Cin→Cout, s"]),
        (2.85, 1.5, ["IN", "+ LeakyReLU"]),
        (4.65, 1.4, ["Conv3d 3³", "Cout→Cout, s=1"]),
        (6.35, .65, ["IN"]),
        (7.9, 1.3, ["LeakyReLU", "斜率 0.01"]),
    ]:
        d.box(x, y - .45, w, .9, lines, "encoder", fs=9.5)
    d.node(7.45, y)
    for a, b in [(.95, 1.15), (2.55, 2.85), (4.35, 4.65), (6.05, 6.35),
                 (7, 7.28), (7.62, 7.9)]:
        d.arrow([(a, y), (b, y)])
    d.box(2.75, 8.2, 3.6, .65,
          ["Cin=Cout 且 s=1：Identity", "否则：1³ Conv3d，步幅 s"], "query", fs=9.5)
    d.arrow([(1, y), (1, 8.525), (2.75, 8.525)], color=PALETTE["query"][1])
    d.arrow([(6.35, 8.525), (7.45, 8.525), (7.45, y - .17)],
            color=PALETTE["query"][1])
    d.junction(1, y, PALETTE["query"][1])
    d.text(.55, 10.5, "主支路：Conv → IN → 激活 → Conv → IN；shortcut 在第二个 IN 后相加。",
           fs=10.2, color=MUTED)
    d.text(.55, 7.95, "IN = InstanceNorm3d（affine=True）；上采样在 RB 外部完成。", fs=10)


def self_attention(d):
    d.panel(10.05, 7.75, 9.5, 3.5, "C  单个 AttentionBlock：8 头自注意力 + 两次残差相加")
    y = 9.6
    d.text(10.3, y, "T", fs=12, va="center")
    d.box(10.75, y - .45, .65, .9, ["LN"], "shape", fs=10)
    d.box(11.75, y - .45, 1.85, .9, ["多头自注意力", "MHA：8 个头"], "shape", fs=10)
    d.node(14.05, y, color=PALETTE["shape"][1])
    d.box(14.5, y - .45, .65, .9, ["LN"], "shape", fs=10)
    d.box(15.5, y - .45, 2.35, .9, ["FFN：C→4C→C", "Linear → GELU → Linear"], "shape", fs=9.8)
    d.node(18.3, y, color=PALETTE["shape"][1])
    for a, b in [(10.55, 10.75), (11.4, 11.75), (13.6, 13.88), (14.22, 14.5),
                 (15.15, 15.5), (17.85, 18.13), (18.47, 19.0)]:
        d.arrow([(a, y), (b, y)])
    d.text(19.05, y, "T′", fs=12, va="center")
    d.arrow([(10.6, y), (10.6, 10.45), (14.05, 10.45), (14.05, y + .17)],
            color=PALETTE["shape"][1])
    d.arrow([(14.3, y), (14.3, 10.45), (18.3, 10.45), (18.3, y + .17)],
            color=PALETTE["shape"][1])
    d.junction(10.6, y, PALETTE["shape"][1])
    d.junction(14.3, y, PALETTE["shape"][1])
    d.text(10.25, 8.8, "Q = K = LN(T) + P；V = LN(T)；P 为物理位置编码，仅加到 Q / K。", fs=10.1)
    d.text(10.25, 8.43, "每头 d=C/8：C=256 时 d=32；细网络 E4 的 C=320 时 d=40。", fs=10.1)
    d.text(10.25, 8.06, "各头 Softmax(Qh Khᵀ / √d) Vh → 拼接 8 头 → 输出投影；dropout=0。", fs=10)


def global_wrapper(d):
    d.panel(.35, 3.1, 9.15, 4.25, "D  E4 全局注意力：内层残差 + 外层特征增量回注")
    d.text(.7, 5.75, "F", fs=12, va="center")
    d.box(1.2, 5.3, 2.5, .9, ["按需自适应平均池化", "展平 → T0；tokens ≤1024"], "input", fs=9.8)
    d.box(4.1, 5.3, 2.25, .9, ["AttentionBlock ×2", "每块均采用面板 C"], "shape", fs=10)
    d.box(6.8, 5.3, 2.4, .9, ["输出 T2", "ΔT = T2 − T0"], "field", fs=10)
    for a, b in [(1, 1.2), (3.7, 4.1), (6.35, 6.8)]:
        d.arrow([(a, 5.75), (b, 5.75)])
    d.arrow([(3.85, 5.75), (3.85, 4.99), (8, 4.99), (8, 5.3)],
            label="T0", at=(5.2, 4.99), color=PALETTE["query"][1])
    d.box(5.15, 3.95, 2.85, .8, ["ΔT 还原为 3D 特征增量", "按需插值到原 E4 空间尺寸 → ΔF"], "field", fs=9.6)
    d.arrow([(8.7, 5.3), (8.7, 4.35), (8, 4.35)])
    d.node(3.75, 4.35)
    d.arrow([(5.15, 4.35), (3.92, 4.35)])
    d.arrow([(1, 5.75), (1, 4.35), (3.58, 4.35)],
            label="原始 F", at=(2.2, 4.35), color=PALETTE["query"][1])
    d.junction(1, 5.75, PALETTE["query"][1])
    d.junction(3.85, 5.75, PALETTE["query"][1])
    d.arrow([(3.75, 4.18), (3.75, 3.62)])
    d.text(2.9, 3.35, "F′ = F + ΔF", fs=11.2, color=PALETTE["field"][1])
    d.text(.55, 6.53, "位置编码来自相对 CT 中心的 RAS 毫米坐标；池化前后保留物理坐标对应。",
           fs=10.2, color=MUTED)


def decoder(d):
    d.panel(10.05, 3.1, 9.5, 4.25, "E  单级 U-Net 解码：拼接不是相加；RB 内部再做残差相加")
    d.text(10.3, 5.75, "D(i+1)", fs=10.5, va="center")
    d.box(11.2, 5.3, 2.05, .9, ["三线性上采样", "size = Ei 的空间尺寸"], "input", fs=9.9)
    d.node(13.8, 5.75, "C", PALETTE["query"][1])
    d.box(14.3, 5.3, 3.15, .9, ["ResidualBlock ×1，s=1", "Cin=C(i+1)+Ci → Cout=Ci", "内部 shortcut：1³ Conv 投影"],
          "field", fs=9.4)
    for a, b in [(11, 11.2), (13.25, 13.63), (13.97, 14.3), (17.45, 18.5)]:
        d.arrow([(a, 5.75), (b, 5.75)])
    d.text(18.6, 5.75, "Di", fs=12, va="center")
    d.box(11.4, 4.05, 2.4, .8, ["编码器特征 Ei", "Ei 已包含该层的注意力更新"], "encoder", fs=9.8)
    d.arrow([(13.8, 4.45), (13.8, 5.58)], color=PALETTE["query"][1])
    d.text(14.2, 4.7, "C = torch.cat(..., dim=1)", fs=10.2, color=PALETTE["query"][1])
    d.text(14.2, 4.3, "首次：D4 := E4；依次 D3→D2→D1→D0", fs=10.1)
    d.text(10.25, 6.53, "每一级均为：上采样 → 与对应 Ei 通道拼接 → 单个 RB；不使用转置卷积。",
           fs=10.2, color=MUTED)
    d.text(10.25, 3.48, "解码器不另设自注意力层；其残差投影和相加位置见面板 B。", fs=10.2)


def main():
    matplotlib.rcParams.update({"svg.fonttype": "path", "svg.hashsalt": "anatoskel-v01-detail"})
    d = Diagram()
    d.text(.45, 23.43, "AnatoSKEL v0.1｜ResUNet 编码器、解码器与残差 / 注意力细节",
           fs=23, fontweight="bold")
    d.text(.45, 22.97, "源码标签 v0.1 · 2d732ae · 默认配置；粗、细网络权重独立；E0–E4 表示五个编码阶段。",
           fs=11, color=MUTED)
    d.text(.45, 22.56, "图例：⊕ = 逐元素相加；C = 通道拼接；紫色 = 自注意力；绿色 = shortcut / 编码器跳连；实心点 = 分支节点。",
           fs=11, color=MUTED)
    unet(d, 0)
    unet(d, 10, fine=True)
    residual(d)
    self_attention(d)
    global_wrapper(d)
    decoder(d)
    d.panel(.35, .7, 19.2, 1.8, "F  窗口自注意力及连接规则")
    d.text(.55, 1.72,
           "仅细网络 E3：F → 4³ 窗口划分 → 注意力块（偏移 0）→ 还原 → 重新划分（偏移 2）→ 注意力块 → 还原；每窗口 64 tokens，8 个头。",
           fs=11)
    d.text(.55, 1.31,
           "偏移通过边界 padding 实现，不做循环 roll；padding token 在注意力键和输出中屏蔽。窗口模块使用 C 中的两条残差支路。", fs=11)
    d.text(.55, .9,
           "编码阶段的首个 RB 在 E1–E4 用 stride=2 下采样；其余 RB 的 stride=1。此处画 ResUNet 的特征解码器；实例查询解码器属于独立检测模块。",
           fs=11)
    d.text(.45, .29, "依据：src/anatoskel/models/encoder.py 的 ResidualBlock / AttentionBlock / WindowAttention / GlobalAttention / ResUNet；尺寸取 v0.1 ModelConfig。",
           fs=9.9, color=MUTED)
    d.save()
    print("Rendered v0.1 ResUNet detail diagram (Chinese SVG + PNG).")


if __name__ == "__main__":
    main()
