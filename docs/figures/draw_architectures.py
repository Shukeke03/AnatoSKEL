"""Render the architectures implemented at the immutable v0.1 / v1.0 tags.

Documentation-only utility; requires matplotlib, separate from model dependencies.
Run: python docs/figures/draw_architectures.py --language all
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent
INK = "#192b3c"
MUTED = "#536578"
PALETTE = {
    "input": ("#edf2f7", "#5e748a"),
    "encoder": ("#eaf2fe", "#3466a0"),
    "query": ("#e6f4f1", "#24796d"),
    "shape": ("#f0ecfa", "#7053a0"),
    "field": ("#fff2e4", "#ba7430"),
    "output": ("#edf5e8", "#60853e"),
    "train": ("#f3f4f6", "#7b8490"),
}


def chinese_font_family():
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC",
                 "STHeiti", "Heiti TC", "Arial Unicode MS", "PingFang HK"):
        if name in available:
            return ["DejaVu Sans", name]
    raise RuntimeError("Chinese rendering requires a CJK font, e.g. Noto Sans CJK SC.")


class Plate:
    def __init__(self, height, title, subtitle, language="en"):
        self.height = height
        self.language = language
        self.translations = (json.loads((OUT / "architecture_zh.json").read_text(encoding="utf-8"))
                             if language == "zh" else {})
        self.font_family = chinese_font_family() if language == "zh" else ["DejaVu Sans"]
        self.fig, self.ax = plt.subplots(figsize=(20, height))
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.ax.set(xlim=(0, 20), ylim=(0, height), aspect="equal")
        self.ax.axis("off")
        self.fig.patch.set_facecolor("white")
        self.ax.text(.45, height - .48, self.translate(title), fontsize=24,
                     fontweight="bold", color=INK, fontfamily=self.font_family)
        self.ax.text(.45, height - .88, self.translate(subtitle), fontsize=10.5,
                     color=MUTED, fontfamily=self.font_family)
        self.bounds = []

    def translate(self, value):
        # Require complete localization; formulas and identifiers are explicit entries too.
        return self.translations[value] if self.language == "zh" and value else value

    def box(self, x, y, w, h, title, lines, kind="encoder", fs=10.4, gap=.26,
            dashed=False, title_fs=13):
        fill, edge = PALETTE[kind]
        self.ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.09",
            facecolor=fill, edgecolor=edge, linewidth=1.4,
            linestyle=(0, (5, 3)) if dashed else "solid", zorder=2,
        ))
        title_artist = self.ax.text(
            x + .18, y + h - .23, self.translate(title), va="top", fontsize=title_fs,
            fontweight="bold", color=edge, zorder=3, fontfamily=self.font_family,
        )
        self.ax.plot([x + .16, x + w - .16], [y + h - .59] * 2,
                     color=edge, alpha=.25, linewidth=1, zorder=3)
        artists = [title_artist]
        for index, line in enumerate(lines):
            artists.append(self.ax.text(
                x + .18, y + h - .77 - index * gap, self.translate(line),
                va="top", fontsize=fs, color=INK, zorder=3, fontfamily=self.font_family,
            ))
        self.bounds.append((title, (x, y, w, h), artists))

    def arrow(self, points, label=None, label_at=None, dashed=False, color=INK):
        style = (0, (5, 3)) if dashed else "solid"
        # White underlay separates crossed routes; crossings are not junctions.
        self.ax.plot(*zip(*points), color="white", linewidth=5, zorder=1)
        if len(points) > 2:
            xs, ys = zip(*points[:-1])
            self.ax.plot(xs, ys, color=color, linewidth=1.4,
                         linestyle=style, zorder=1)
        self.ax.add_patch(FancyArrowPatch(
            points[-2], points[-1], arrowstyle="-|>", mutation_scale=13,
            linewidth=1.4, color=color, linestyle=style, zorder=1,
            shrinkA=0, shrinkB=0,
        ))
        if label:
            if label_at is None:
                a, b = points[0], points[-1]
                label_at = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + .12)
            self.ax.text(*label_at, self.translate(label), ha="center", va="center",
                         fontsize=8.8, color=color, zorder=4, fontfamily=self.font_family,
                         bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.8})

    def note(self, x, y, text, fs=10, color=MUTED, **kwargs):
        self.ax.text(x, y, self.translate(text), fontsize=fs, color=color,
                     fontfamily=self.font_family, **kwargs)

    def save(self, stem):
        if self.language == "zh":
            stem += "-zh"
        # Detect box text overflow before publishing any render.
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        for name, (x, y, w, h), artists in self.bounds:
            lo = self.ax.transData.transform((x, y))
            hi = self.ax.transData.transform((x + w, y + h))
            for artist in artists:
                bounds = artist.get_window_extent(renderer)
                if (bounds.x0 < lo[0] or bounds.x1 > hi[0]
                        or bounds.y0 < lo[1] or bounds.y1 > hi[1]):
                    raise ValueError(f"Text outside {name}: {artist.get_text()}")
        for suffix in ("svg", "png"):
            metadata = {"Creator": "AnatoSKEL"}
            if suffix == "svg":
                metadata["Date"] = None
            path = OUT / f"{stem}.{suffix}"
            self.fig.savefig(path, dpi=180, facecolor="white", metadata=metadata)
            if suffix == "svg":
                # Matplotlib's multiline SVG path attributes contain trailing spaces.
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())
                                + "\n", encoding="utf-8")
        plt.close(self.fig)


def v01(language="en"):
    p = Plate(14, "AnatoSKEL v0.1 | CT-conditioned implicit bone reconstruction",
              "Code tag v0.1 · 2d732ae · default model / first5090 configuration · solid: inference; dashed: training-only",
              language=language)
    p.box(.45, 10.55, 2.2, 2.2, "CT input", [
        "[B, 1, D, H, W]", "Affine → RAS mm", "Acquisition support", "Full-FOV resampling",
        "Adaptive voxel budget",
    ], "input", fs=10, gap=.25)
    p.box(3.15, 10.55, 4.15, 2.2, "Coarse ResUNet", [
        "E0–E4: 16 / 32 / 64 / 128 / 256", "ResBlocks: 1 / 2 / 2 / 2 / 2",
        "E4 global attention ×2; 8 heads", "4-level skip decoder → mask features",
        "Conv: IN + LeakyReLU(0.01)",
    ], "encoder", gap=.25)
    p.box(7.8, 10.55, 4.35, 2.2, "Object query decoder ×4", [
        "32 learned queries ×256; 8 heads", "CT cross-attn → self-attn → FFN",
        "FFN: 256 → 1024 → 256, GELU", "Memory levels: E4 / E3 / E2 / E3",
        "≤1024 context tokens per level",
    ], "query", gap=.25)
    p.box(12.65, 10.55, 3.35, 2.2, "Instance heads", [
        "27 IDs (26 + no-object)", "Query-conditioned masks",
        "Mask centroid + Δcentre", "Visibility ×2; carried q256", "Select one query per ID",
    ], "query", fs=10, gap=.25)
    p.box(16.5, 10.55, 3.05, 2.2, "Per-bone CT ROI", [
        "Base: 128³ @ 1 mm", "Predicted centre anchor", "Boundary retry: ≤3 passes",
        "Spacing grows ×1.5", "Fixed grid; no tile fusion",
    ], "input", fs=10, gap=.25)
    for a, b in [(2.65, 3.15), (7.3, 7.8), (12.15, 12.65), (16, 16.5)]:
        p.arrow([(a, 11.65), (b, 11.65)])

    p.box(.45, 3.65, 4.85, 6.1, "Fine ResUNet | separate weights", [
        "E0–E4: 32 / 64 / 128 / 256 / 320",
        "ResBlocks: 2 / 2 / 2 / 2 / 2",
        "Strides: 1 / 2 / 4 / 8 / 16",
        "",
        "E3: window attention ×2",
        "4³ windows; shifts 0 / 2; 8 heads",
        "E4: global attention ×2; 8 heads",
        "Attention FFN: 4× width, GELU",
        "",
        "4-level U-Net skip decoder",
        "Carried q256 → local instance mask",
        "E1–E4 point features: 4×32 =128",
        "",
        "Cache features for both refinements",
        "Re-encode only on ROI retry",
        "Coarse/fine encoders do not share θ",
    ], "encoder", gap=.31)
    p.arrow([(18.02, 10.55), (18.02, 10.15), (2.88, 10.15), (2.88, 9.75)],
            "ROI + carried query", (7.0, 10.15))
    p.box(5.85, 7.05, 3.65, 2.7, "Initial pose / shape", [
        "Input: carried coarse q256", "MLP: 256 → 256 → 128, GELU",
        "R6 → SO(3); Δt3; β64; vis2", "t = ROI anchor + Δt",
        "y = Rᵀ(x_RAS − t)", "Independent rigid pose per bone",
        "No SKEL joint tree / FK / LBS",
    ], "shape", fs=10.2, gap=.26)
    p.arrow([(14.32, 10.55), (14.32, 10.35), (7.67, 10.35), (7.67, 9.75)],
            "coarse q256", (10.2, 10.35), color=PALETTE["query"][1])
    p.box(10.05, 7.05, 5.15, 2.7, "Shared geometric refinement ×2", [
        "2048 near-prior points / 8192 Sobol candidates",
        "Point tokens: 137 → 256; sample cached CT",
        "Point cross-attn → inter-bone self-attn",
        "8 heads; relative R / t / visibility bias",
        "FFN: 4× width, GELU; delta head →70",
        "Update R, t, β, q (same weights both rounds)",
        "R ← R exp(Δω); t += Δt; β += Δβ",
    ], "query", fs=10.2, gap=.26)
    p.arrow([(9.5, 8.4), (10.05, 8.4)])
    p.arrow([(5.3, 6.7), (11.0, 6.7), (11.0, 7.05)],
            "cached CT features + local mask", (7.9, 6.7), color=PALETTE["encoder"][1])
    p.arrow([(10.7, 6.7), (10.7, 6.3)], color=PALETTE["encoder"][1])
    p.box(15.75, 7.05, 3.8, 2.7, "Observed-domain export", [
        "Marching cubes on valid CT", "Open boundary at scan truncation",
        "RAS mm; affine-aware winding", "bones.obj + bones.json",
        "Per-bone transform Ti + β", "Qij = inv(Ti) Tj (derived)",
        "Pose axes remain unvalidated",
    ], "output", fs=10.2, gap=.26)

    p.box(5.85, 3.65, 3.65, 2.65, "Canonical shape prior", [
        "Inputs: y3/100 + β64 + ID16 =83", "MLP: 4×256; input skip at layer 3",
        "Softplus(β=20); linear output", "Learned auto-decoder, not an AE",
        "Normalized TSDF (8 mm scale)", "Prior parameters frozen for CT stage",
        "Gradient to β / R / t remains",
    ], "shape", fs=9.9, gap=.25)
    p.box(10.05, 3.65, 5.15, 2.65, "CT-conditioned residual field", [
        "128 CT + y3 + β64 + ID16 + prior1",
        "+ HU2 + valid1 =215 inputs",
        "MLP: 4×256; input skip at layer 3",
        "Softplus(β=20); zero-initialized output",
        "Evaluate with refined R / t / β",
        "Residual vanishes outside acquisition support",
        "Prior and field also drive near-surface refinement",
    ], "field", fs=10, gap=.25)
    p.box(15.75, 3.65, 3.8, 2.65, "Continuous field evaluator", [
        "F(x) = prior(y, β, ID)", "+ valid(x) · residual(x)",
        "Current / refined R, t, β", "After round 2 → final mesh",
        "Unclamped normalized TSDF", "One mesh per selected bone",
        "No diffusion in v0.1",
    ], "field", fs=10.1, gap=.25)
    p.arrow([(7.67, 7.05), (7.67, 6.3)])
    p.arrow([(9.5, 4.98), (10.05, 4.98)])
    p.arrow([(15.2, 4.98), (15.75, 4.98)])
    p.arrow([(12.9, 7.05), (12.9, 6.52), (9.2, 6.52), (9.2, 6.3)],
            "updated R / t / β", (11.1, 6.52), color=PALETTE["shape"][1])
    p.arrow([(12.9, 6.52), (12.9, 6.3)], color=PALETTE["shape"][1])
    p.arrow([(15.75, 5.55), (15.48, 5.55), (15.48, 6.84), (14.85, 6.84),
             (14.85, 7.05)], "field samples", (14.1, 6.84), color=PALETTE["field"][1])
    p.arrow([(17.65, 6.3), (17.65, 7.05)])

    p.box(.45, .95, 19.1, 1.95, "Training-only routes and gradient boundaries", [
        "A  Complete training masks → provisional PCA frame / atlas + trimmed ICP weak pose → train auto-decoder + per-instance β codes.",
        "B  CT detection + local mask / occupancy / TSDF + weak pose / shape losses; load and freeze the learned prior parameters.",
        "    Main CT trainer detaches coarse q and centre at the fine entry; local losses do not update coarse detection through that route.",
        "    GT identities / masks / centres / atlas registration are supervision; inference starts from CT predictions.",
    ], "train", fs=11.1, gap=.28, dashed=True)
    p.arrow([(7.67, 2.9), (7.67, 3.65)], "prior weights", (8.5, 3.27), dashed=True,
            color=PALETTE["train"][1])
    p.note(.45, .42,
           "Scope: tagged implementation, not a clinical performance claim. Attention counts describe the default model, not training epochs.",
           fs=10.3)
    p.save("anatoskel-v0.1-architecture")


def v10(language="en"):
    p = Plate(17, "AnatoSKEL v1.0 | shared CT Transformer + residual latent diffusion",
              "Code tag v1.0 · 90fe833 · v1_research / v1_autodl_4090 dimensions · solid: inference; dashed: training-only",
              language=language)
    p.box(.45, 13.25, 2.25, 2.45, "CT + support", [
        "CT + affine → RAS mm", "Acquisition-valid mask", "Soft-window [0, 1]",
        "Wide HU [−1, 1]", "Global longest side ≤128", "Pad to multiples of 32",
    ], "input", fs=9.8, gap=.25)
    p.box(3.25, 13.25, 4.1, 2.45, "Shared CT encoder θ | global", [
        "SwinUNETR V2; patch 2³; window 7³", "Stage blocks: 2 / 2 / 2 / 2; GELU",
        "Heads: 3 / 6 / 12 / 24; base width 48",
        "S2–S32: 48 / 96 / 192 / 384 / 768",
        "Wide-HU + valid: GN / SiLU branch", "5-level detail decoder; detail32 + aux heads",
    ], "encoder", fs=9.7, gap=.25)
    p.box(7.9, 13.25, 6.05, 2.45, "Global context + typed geometry queries", [
        "S8 / S16 / S32 + physical PE → 256 slots; context blocks ×2",
        "64 instances ×3 query types: localization / identity / shape",
        "Global GeometryBlock ×3; width384; 6 heads; SwiGLU1024",
        "Self-attn → memory cross-attn → 5-scale point attn → FFN",
        "Proposal seeds + centre updates → select predicted instances",
        "27 classes; bounds / visibility / masks; fixed seed anchors",
    ], "query", fs=10.1, gap=.25)
    p.box(14.5, 13.25, 5.05, 2.45, "Predicted bounds → overlap tiles", [
        "Per-bone 128³ tiles @1 mm; overlap32 (stride96)",
        "If ROI cuts covered bone: add tiles at the same spacing",
        "All tiles share the original fixed proposal anchor",
        "u = (x_RAS − anchor) / 64 mm",
        "One latent / diffusion noise per instance across all tiles",
        "Inference reads CT support, without GT annotations",
    ], "input", fs=10.1, gap=.25)
    for a, b in [(2.7, 3.25), (7.35, 7.9), (13.95, 14.5)]:
        p.arrow([(a, 14.47), (b, 14.47)])

    p.box(.45, 9.6, 4.55, 2.75, "Same encoder θ | each local tile", [
        "Separate local forward; shared global/local weights",
        "Swin + wide-HU detail branch + detail decoder",
        "Cache five scales + decoded32 per tile",
        "Mask logits + auxiliary TSDF supervision",
        "Fuse overlapping features in physical coordinates",
        "Feature values are sampled at common world points",
        "VoCo initialization is optional; default is random",
    ], "encoder", fs=10, gap=.26)
    p.arrow([(17.02, 13.25), (17.02, 12.85), (2.72, 12.85), (2.72, 12.35)],
            "local CT tiles", (7.0, 12.85))
    p.box(5.55, 9.6, 4.85, 2.75, "Local GeometryBlock ×3", [
        "Same typed queries; width384; 6 heads",
        "Self-attn → memory cross-attn",
        "→ 5-scale point attn → SwiGLU1024",
        "16 query points; all covering tiles contribute",
        "Update centres / bounds; anchor stays fixed",
        "ID head: identity query; bounds: localization query",
        "Latent / mask / visibility use LN(mean of 3 types)",
    ], "query", fs=9.9, gap=.26)
    p.arrow([(5, 10.97), (5.55, 10.97)])
    p.box(10.95, 9.6, 4.0, 2.75, "Instance conditioning bundle", [
        "Mglobal: 256 learned CT context slots",
        "Mlocal: ≤256; fused S8 + PE; blocks ×2",
        "M = concat(Mglobal, Mlocal), width384",
        "CT memory + token-valid mask → ≤512 tokens",
        "Carry refined mean query to latent predictor",
        "The same M conditions latent prediction and DiT",
        "CT memory differs from shape decoder memory",
    ], "query", fs=9.7, gap=.26)
    p.arrow([(10.4, 10.97), (10.95, 10.97)])
    p.arrow([(8.25, 13.25), (8.25, 12.35)],
            "global q + Mglobal", (9.22, 12.53), color=PALETTE["query"][1])
    p.arrow([(12.4, 13.25), (12.4, 12.35)],
            "Mglobal", (13.05, 12.55), color=PALETTE["query"][1])
    p.box(15.5, 9.6, 4.05, 2.75, "Initial latent predictor ×2", [
        "256 learned slots ×384 + refined mean(q)",
        "CT cross-attn / self-attn blocks ×2",
        "6 heads; FFN: SwiGLU hidden1024",
        "LN + Linear → z_anchor [256, 128]",
        "Keep one shared instance physical frame",
        "Output anchors diffusion; no per-tile latent",
        "No hard ID → shape feedback loop",
    ], "shape", fs=9.9, gap=.26)
    p.arrow([(14.95, 10.97), (15.5, 10.97)])

    p.box(.45, 5.95, 6.45, 2.85, "Conditional residual DiT ×6", [
        "Inputs: noisy r_t + z_anchor + time; CT cross-attn uses M",
        "Width384; 6 heads; AdaLN-Zero; SwiGLU hidden1024",
        "Each block: self-attn → CT cross-attn → adapter → FFN",
        "Adapter: LN → 384 → 96 → SiLU → 384 (zero up-projection)",
        "Train: cosine schedule, 1000 steps, velocity prediction",
        "Infer: Gaussian r_T →20 DDIM steps (η=0); last 4 adapter-on*",
        "z = z_anchor + μ + σ · r̂0; stats calibrated on complete GT",
        "*Adapters enabled only when an adapter-trained checkpoint is loaded",
    ], "shape", fs=10.1, gap=.255)
    p.arrow([(17.52, 9.6), (17.52, 9.15), (3.67, 9.15), (3.67, 8.8)],
            "z_anchor", (15.65, 9.15))
    p.arrow([(11.4, 9.6), (11.4, 9.0), (5.9, 9.0), (5.9, 8.8)],
            "shared CT memory M", (9.2, 9.0), color=PALETTE["query"][1])
    p.box(7.45, 5.95, 5.1, 2.85, "Shared shape decoder | frozen θ", [
        "z [256,128] → width256 + indexed slots",
        "Latent self-attn / SwiGLU blocks ×4; 8 heads",
        "SwiGLU hidden704 → shape memory [256,256]",
        "Query Fourier PE(u) → point cross-attn",
        "Field head: 2 hidden layers ×128, Softplus(20)",
        "Linear output: shape SDF in mm (unclamped)",
        "Freeze parameters; keep ∂S / ∂z for geometry losses",
        "Not a SKEL articulated skeleton or joint decoder",
    ], "shape", fs=10, gap=.255)
    p.arrow([(6.9, 7.37), (7.45, 7.37)], "z", (7.17, 7.58))
    p.box(13.1, 5.95, 6.45, 2.85, "Per-tile CT residual field δj(x)", [
        "Inputs at world point x: detail32 + u3 + mean(z)128",
        "+ shape SDF1 + wideHU1 + valid1 + instance mask1 =167",
        "Cached local CT features; same field MLP across all tiles",
        "SkipFieldMLP: 4 hidden layers ×128; Softplus(20)",
        "Linear zero-initialized residual output in mm",
        "Auxiliary TSDF head supervises detail; not an MLP input",
        "Blend CT residuals using overlap weights × valid support",
        "All tiles use the same shape SDF, anchor, latent and noise",
    ], "field", fs=10.1, gap=.255)
    p.arrow([(12.55, 7.37), (13.1, 7.37)], "prior", (12.82, 7.58))
    p.arrow([(4.7, 9.6), (4.7, 9.45), (19.8, 9.45), (19.8, 7.37), (19.55, 7.37)],
            "cached local detail + HU + valid", (7.2, 9.45),
            color=PALETTE["encoder"][1])

    p.box(.45, 3.0, 8.35, 2.15, "Complete-shape AE encoder | training-only GT branch", [
        "4096 complete surface points + normals; Fourier PE(u) → width256",
        "256 learned slots: surface cross-attn ×1 → latent self-attn / SwiGLU ×4",
        "8 heads; SwiGLU hidden704 → z_GT [256,128]; shares the decoder above",
        "Only complete GT supplies latent / v targets; synthetic CT crops retain full GT",
        "Real scan truncation: observed geometry only; no fabricated complete latent target",
    ], "train", fs=10.4, gap=.255, dashed=True)
    p.arrow([(8.15, 5.15), (8.15, 5.6), (8.55, 5.6), (8.55, 5.95)],
            "z_GT", (7.85, 5.6), dashed=True, color=PALETTE["train"][1])
    p.arrow([(1.8, 5.15), (1.8, 5.95)],
            "z_GT: residual target / μ, σ", (3.8, 5.5), dashed=True,
            color=PALETTE["train"][1])
    p.box(9.35, 3.0, 4.85, 2.15, "Fuse fields before extracting mesh", [
        "δfused(x) = Σ wj validj δj / Σ wj validj",
        "Sfinal(x) = Sshape(x) + δfused(x)",
        "Add the single shared prior exactly once",
        "Internal coverage holes → explicit error",
        "Fuse continuous fields, then mesh per instance",
    ], "field", fs=10.4, gap=.255)
    p.arrow([(10.0, 5.95), (10.0, 5.15)], "Sshape", (9.4, 5.55))
    p.arrow([(16.32, 5.95), (16.32, 5.6), (13.5, 5.6), (13.5, 5.15)],
            "δj + weights", (14.1, 5.6))
    p.box(14.75, 3.0, 4.8, 2.15, "Observed-domain mesh / OBJ", [
        "Marching cubes once per selected instance",
        "Clip triangles to original physical CT FOV",
        "Real scan truncation → open boundary",
        "RAS mm: bones_observed.obj + bones.json",
        "Observed patient structure is the default export",
    ], "output", fs=10.4, gap=.255)
    p.arrow([(14.2, 4.07), (14.75, 4.07)])

    p.box(.45, .8, 19.1, 1.6, "Four training stages | separate supervision domains", [
        "1 Shape AE  →  2 CT/query/latent/detail (AE frozen)  →  3 Base DiT (AE + CT frozen)  →  4 Partial adapters only (all base modules frozen).",
        "Complete + synthetic missing CT: full latent / v supervision. Real partial: observed-domain geometry; 16 detached DDIM steps +4 differentiable adapter steps.",
        "Acquisition / annotation / condition / tile masks stay distinct; synthetic masking precedes CT encoding; frozen decoder still backpropagates to latent.",
    ], "train", fs=10.7, gap=.27, dashed=True)
    p.note(.45, .3,
           "Scope: implemented architecture; v1.0 has software / synthetic checks, without real-CT accuracy results. Reliability gating and calibrated uncertainty are not implemented.",
           fs=10)
    p.save("anatoskel-v1.0-architecture")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("en", "zh", "all"), default="all")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path",
                         "svg.hashsalt": "anatoskel-architectures"})
    for language in (("en", "zh") if args.language == "all" else (args.language,)):
        v01(language)
        v10(language)
    print(f"Rendered v0.1 and v1.0 architecture diagrams: {args.language} (SVG + PNG).")
