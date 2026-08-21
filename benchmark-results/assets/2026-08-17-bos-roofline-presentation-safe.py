#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

plt.rcParams["svg.hashsalt"] = "bos-roofline-2026-08-17"
plt.rcParams.update(
    {
        "font.size": 12.0,
        "axes.titlesize": 12.0,
        "axes.labelsize": 12.0,
        "xtick.labelsize": 12.0,
        "ytick.labelsize": 12.0,
    }
)

# The memory line is an empirical synthetic transport reference, not a
# physical-bandwidth counter.  The bounds come from two independently reported
# all-bank measurements: 8 KiB request/64 KiB batch and the bank-scaling sweep.
DRAM_GBPS_LOW = 95.262
DRAM_GBPS_HIGH = 96.139
DRAM_GBPS_CENTER = (DRAM_GBPS_LOW + DRAM_GBPS_HIGH) / 2.0

# Measured directly with the 3x4 (12-core) large-GEMM benchmark:
# warmup 5, 100 measured iterations, host non-trace steady state.
HIFI2_12_CORE_MEASURED_TFLOPS = 14.3573
LOFI_12_CORE_MEASURED_TFLOPS = 27.6356
SDPA_ACTIVE_COMPUTE_CORES = 16
SDPA_HIFI2_PROJECTED_TFLOPS = HIFI2_12_CORE_MEASURED_TFLOPS * SDPA_ACTIVE_COMPUTE_CORES / 12.0

TILE = 32
BFP8_TILE_BYTES = 1088
BFP4_TILE_BYTES = 576
DECODE_LOGICAL_M = 1
DECODE_ISSUED_M = 32


def matmul_ops(m: int, k: int, n: int) -> int:
    return 2 * m * k * n


def tiled_weight_bytes(k: int, n: int, tile_bytes: int) -> int:
    assert k % TILE == 0 and n % TILE == 0
    return (k // TILE) * (n // TILE) * tile_bytes


def tflops(ops: int, duration_ns: int) -> float:
    return ops / duration_ns / 1000.0


# label, K, N, encoded tile bytes, vanilla/baseline duration ns,
# stable duration ns, color, marker, source tag.
#
# QKV/Wo durations: controlled projection A/B, 2026-08-09.
# W1/W3/W2 vanilla durations: Jul-25 actual-prefill profile.
# W1/W3/W2 stable durations: Aug-09 stable layer-0 profile.
matmul_specs = [
    ("QKV BFP8", 3072, 5120, BFP8_TILE_BYTES, 433448, 278943, "#2563eb", "o", "controlled A/B"),
    ("Wo BFP8", 3072, 3072, BFP8_TILE_BYTES, 236042, 165835, "#2563eb", "s", "controlled A/B"),
    ("W2 BFP8", 8192, 3072, BFP8_TILE_BYTES, 682583, 428812, "#2563eb", "^", "profile"),
    ("W1 BFP4", 3072, 8192, BFP4_TILE_BYTES, 553245, 233389, "#ea580c", "o", "profile"),
    ("W3 BFP4", 3072, 8192, BFP4_TILE_BYTES, 552743, 235877, "#ea580c", "s", "profile"),
]

matmul_points = []
vanilla_matmul_tflops = {}
issued_matmul_tflops = {}
useful_matmul_tflops = {}
effective_weight_gbps = {}
for label, k, n, tile_bytes, vanilla_ns, stable_ns, color, marker, _source in matmul_specs:
    weight_bytes = tiled_weight_bytes(k, n, tile_bytes)
    useful_ops = matmul_ops(DECODE_LOGICAL_M, k, n)
    issued_ops = matmul_ops(DECODE_ISSUED_M, k, n)
    issued_oi = issued_ops / weight_bytes
    issued_vanilla = tflops(issued_ops, vanilla_ns)
    issued_stable = tflops(issued_ops, stable_ns)
    matmul_points.append((label, issued_oi, issued_stable, color, marker))
    vanilla_matmul_tflops[label] = issued_vanilla
    issued_matmul_tflops[label] = (issued_vanilla, issued_stable)
    useful_matmul_tflops[label] = (
        tflops(useful_ops, vanilla_ns),
        tflops(useful_ops, stable_ns),
    )
    effective_weight_gbps[label] = (
        weight_bytes / vanilla_ns,
        weight_bytes / stable_ns,
    )

# SDPA useful numerator counts 24 query heads for QK and PV.  The encoded-byte
# denominator counts K+V for 8 KV heads.  The issued diagnostic packs each
# 3-query GQA group into a padded M=32 tile, hence issued/useful = 32/3.
SDPA_CONTEXT = 65536
SDPA_HEAD_DIM = 128
SDPA_Q_HEADS = 24
SDPA_KV_HEADS = 8
SDPA_GQA_GROUP = SDPA_Q_HEADS // SDPA_KV_HEADS
SDPA_VANILLA_NS = 3484977
SDPA_STABLE_NS = 2039225
sdpa_kv_tiles = (
    (SDPA_CONTEXT // TILE)
    * (SDPA_HEAD_DIM // TILE)
    * SDPA_KV_HEADS
    * 2
)
sdpa_encoded_kv_bytes = sdpa_kv_tiles * BFP8_TILE_BYTES
sdpa_useful_ops = 2 * SDPA_Q_HEADS * SDPA_CONTEXT * SDPA_HEAD_DIM * 2
sdpa_issued_ops = sdpa_useful_ops * DECODE_ISSUED_M // SDPA_GQA_GROUP
SDPA_USEFUL_OI = sdpa_useful_ops / sdpa_encoded_kv_bytes
SDPA_ISSUED_OI = sdpa_issued_ops / sdpa_encoded_kv_bytes
VANILLA_SDPA_TFLOPS = tflops(sdpa_useful_ops, SDPA_VANILLA_NS)
STABLE_SDPA_TFLOPS = tflops(sdpa_useful_ops, SDPA_STABLE_NS)
VANILLA_SDPA_ISSUED_TFLOPS = tflops(sdpa_issued_ops, SDPA_VANILLA_NS)
STABLE_SDPA_ISSUED_TFLOPS = tflops(sdpa_issued_ops, SDPA_STABLE_NS)
sdpa_points = [
    ("SDPA issued QK+PV", SDPA_ISSUED_OI, STABLE_SDPA_ISSUED_TFLOPS, "#0891b2", "D"),
]

output_dir = Path(__file__).resolve().parent


def normalize_svg(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")


x = np.logspace(-1, 3.4, 1200)
memory_roof_low = DRAM_GBPS_LOW * x / 1000.0
memory_roof_high = DRAM_GBPS_HIGH * x / 1000.0
memory_roof_center = DRAM_GBPS_CENTER * x / 1000.0
hifi2_roof = np.minimum(memory_roof_center, HIFI2_12_CORE_MEASURED_TFLOPS)
lofi_roof = np.minimum(memory_roof_center, LOFI_12_CORE_MEASURED_TFLOPS)


def draw_memory_reference(axis, linewidth=2.0):
    axis.fill_between(
        x,
        memory_roof_low,
        memory_roof_high,
        color="#94a3b8",
        alpha=0.24,
        linewidth=0,
        label="≈96 GB/s",
    )
    axis.loglog(x, memory_roof_center, linestyle=":", linewidth=linewidth, color="#475569")


def annotate_memory_slope(axis, oi):
    performance = DRAM_GBPS_CENTER * oi / 1000.0
    start = axis.transData.transform((oi, performance))
    end = axis.transData.transform((oi * 2.0, performance * 2.0))
    display_angle = np.degrees(np.arctan2(end[1] - start[1], end[0] - start[0]))
    axis.annotate(
        "≈96 GB/s",
        (oi, performance),
        xytext=(8, 8),
        textcoords="offset points",
        rotation=display_angle,
        rotation_mode="anchor",
        color="#334155",
        fontsize=12.0,
        fontweight="bold",
    )


POINT_SYMBOLS = {"o": "●", "s": "■", "^": "▲", "D": "◆"}


def assert_label_boxes_clear_points(fig, axis, annotations, points, clearance_px=8.0):
    """Fail generation if a text box obscures a plotted data marker."""
    renderer = fig.canvas.get_renderer()
    label_boxes = []
    for label, annotation in annotations:
        bbox_patch = annotation.get_bbox_patch()
        if bbox_patch is None:
            continue
        bbox = bbox_patch.get_window_extent(renderer)
        padded = Bbox.from_extents(
            bbox.x0 - clearance_px,
            bbox.y0 - clearance_px,
            bbox.x1 + clearance_px,
            bbox.y1 + clearance_px,
        )
        for point_label, point_x, point_y in points:
            point_px, point_py = axis.transData.transform((point_x, point_y))
            if padded.contains(point_px, point_py):
                raise RuntimeError(f"label {label!r} overlaps point {point_label!r}")
        label_boxes.append((label, bbox))

    for index, (label_a, bbox_a) in enumerate(label_boxes):
        for label_b, bbox_b in label_boxes[index + 1 :]:
            if bbox_a.overlaps(bbox_b):
                raise RuntimeError(f"label boxes overlap: {label_a!r} and {label_b!r}")


def assert_legend_clear_points_and_labels(fig, axis, annotations, points, clearance_px=6.0):
    renderer = fig.canvas.get_renderer()
    legend = axis.get_legend()
    if legend is None:
        return
    legend_bbox = legend.get_window_extent(renderer)
    padded = Bbox.from_extents(
        legend_bbox.x0 - clearance_px,
        legend_bbox.y0 - clearance_px,
        legend_bbox.x1 + clearance_px,
        legend_bbox.y1 + clearance_px,
    )
    for point_label, point_x, point_y in points:
        point_px, point_py = axis.transData.transform((point_x, point_y))
        if padded.contains(point_px, point_py):
            raise RuntimeError(f"legend overlaps point {point_label!r}")
    for label, annotation in annotations:
        bbox_patch = annotation.get_bbox_patch()
        if bbox_patch is not None and padded.overlaps(bbox_patch.get_window_extent(renderer)):
            raise RuntimeError(f"legend overlaps label {label!r}")


fig, ax = plt.subplots(figsize=(14.0, 7.2))
draw_memory_reference(ax)
ax.loglog(
    x,
    hifi2_roof,
    linewidth=2.1,
    color="#2563eb",
    label=f"12-core HiFi2 GEMM: {HIFI2_12_CORE_MEASURED_TFLOPS:.3f} TFLOP/s",
)
ax.loglog(
    x,
    lofi_roof,
    linewidth=2.1,
    color="#ea580c",
    label=f"12-core LoFi GEMM: {LOFI_12_CORE_MEASURED_TFLOPS:.3f} TFLOP/s",
)

hifi2_ridge = HIFI2_12_CORE_MEASURED_TFLOPS * 1000.0 / DRAM_GBPS_CENTER
lofi_ridge = LOFI_12_CORE_MEASURED_TFLOPS * 1000.0 / DRAM_GBPS_CENTER
for ridge, ceiling, color, label, offset, horizontal_alignment, vertical_alignment in [
    (
        hifi2_ridge,
        HIFI2_12_CORE_MEASURED_TFLOPS,
        "#2563eb",
        f"HiFi2 measured ridge\n{hifi2_ridge:.1f} OP/byte",
        (-8, 9),
        "right",
        "bottom",
    ),
    (
        lofi_ridge,
        LOFI_12_CORE_MEASURED_TFLOPS,
        "#ea580c",
        f"LoFi measured ridge\n{lofi_ridge:.1f} OP/byte",
        (8, -9),
        "left",
        "top",
    ),
]:
    ax.scatter([ridge], [ceiling], s=48, color=color, zorder=5)
    ax.annotate(
        label,
        (ridge, ceiling),
        xytext=offset,
        textcoords="offset points",
        ha=horizontal_alignment,
        va=vertical_alignment,
        color=color,
        fontsize=12.0,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": color, "alpha": 0.92},
    )

for oi, color in [
    (matmul_ops(DECODE_ISSUED_M, TILE, TILE) / BFP8_TILE_BYTES, "#2563eb"),
    (matmul_ops(DECODE_ISSUED_M, TILE, TILE) / BFP4_TILE_BYTES, "#ea580c"),
]:
    ax.axvline(oi, color=color, linestyle="-.", linewidth=0.9, alpha=0.28)

label_layout = {
    "QKV BFP8": {"offset": (-24, -76), "ha": "right", "va": "top"},
    "Wo BFP8": {"offset": (-24, -8), "ha": "right", "va": "center"},
    "W2 BFP8": {"offset": (-24, 58), "ha": "right", "va": "bottom"},
    "W1 BFP4": {"offset": (24, 12), "ha": "left", "va": "bottom"},
    "W3 BFP4": {"offset": (24, -36), "ha": "left", "va": "top"},
}
matmul_annotations = []
matmul_data_points = []
for label, oi, performance, color, marker in matmul_points:
    vanilla_performance = vanilla_matmul_tflops[label]
    ax.plot([oi, oi], [vanilla_performance, performance], color="#94a3b8", linewidth=1.2, zorder=6)
    ax.scatter([oi], [vanilla_performance], s=80, facecolor="none", edgecolor="#475569", marker=marker, linewidth=1.3, zorder=7)
    ax.scatter([oi], [performance], s=82, color=color, marker=marker, edgecolor="white", linewidth=0.8, zorder=8)
    matmul_data_points.extend(
        [
            (f"{label} vanilla", oi, vanilla_performance),
            (f"{label} stable", oi, performance),
        ]
    )
    stable_bw = effective_weight_gbps[label][1]
    layout = label_layout[label]
    annotation = ax.annotate(
        f"{POINT_SYMBOLS[marker]} {label}\n{performance:.3f} issued TFLOP/s\n"
        f"{stable_bw:.2f} GB/s encoded-weight rate",
        (oi, performance),
        xytext=layout["offset"],
        textcoords="offset points",
        ha=layout["ha"],
        va=layout["va"],
        color=color,
        fontsize=12.0,
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.94},
        arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.8, "shrinkA": 5, "shrinkB": 7},
        annotation_clip=True,
    )
    matmul_annotations.append((label, annotation))

ax.set_xlim(5, 1000)
ax.set_ylim(0.5, 35)
ax.set_xlabel("Issued operational intensity (padded M=32 OP / encoded weight byte)")
ax.set_ylabel("Issued-work performance (TFLOP/s)")
ax.set_title("Custom BOS NPU — Issued-Work Static-Weight Matmul Roofline")
fig.subplots_adjust(left=0.10, right=0.97, bottom=0.16, top=0.88)
ax.grid(True, which="both", linewidth=0.55, alpha=0.28)
ax.scatter([], [], s=70, facecolor="none", edgecolor="#475569", label="Vanilla")
ax.scatter([], [], s=70, color="#0f172a", edgecolor="white", label="Stable")
ax.legend(loc="upper left", ncol=2, borderaxespad=0.8, framealpha=0.95, fontsize=12.0)
annotate_memory_slope(ax, oi=5.5)
fig.canvas.draw()
assert_label_boxes_clear_points(fig, ax, matmul_annotations, matmul_data_points)
assert_legend_clear_points_and_labels(fig, ax, matmul_annotations, matmul_data_points)
fig.savefig(output_dir / "2026-08-17-bos-roofline-matmul-reference.png", dpi=180)
matmul_svg = output_dir / "2026-08-17-bos-roofline-matmul-reference.svg"
fig.savefig(matmul_svg, metadata={"Date": None})
normalize_svg(matmul_svg)

sdpa_fig, sdpa_ax = plt.subplots(figsize=(14.0, 7.2))
draw_memory_reference(sdpa_ax, linewidth=2.2)
sdpa_projected_hifi2_roof = np.minimum(memory_roof_center, SDPA_HIFI2_PROJECTED_TFLOPS)
sdpa_ax.loglog(
    x,
    sdpa_projected_hifi2_roof,
    linewidth=2.1,
    linestyle="--",
    color="#7c3aed",
    label=(
        f"16-core HiFi2 projection: {SDPA_HIFI2_PROJECTED_TFLOPS:.3f} TFLOP/s "
        "(12-core measured × 16/12)"
    ),
)

sdpa_annotations = []
sdpa_data_points = []
for label, oi, performance, color, marker in sdpa_points:
    gain = (performance / VANILLA_SDPA_ISSUED_TFLOPS - 1.0) * 100.0
    vanilla_bw = sdpa_encoded_kv_bytes / SDPA_VANILLA_NS
    stable_bw = sdpa_encoded_kv_bytes / SDPA_STABLE_NS
    sdpa_ax.plot([oi, oi], [VANILLA_SDPA_ISSUED_TFLOPS, performance], color="#94a3b8", linewidth=1.2, zorder=6)
    sdpa_ax.scatter([oi], [VANILLA_SDPA_ISSUED_TFLOPS], s=90, facecolor="none", edgecolor="#475569", marker=marker, linewidth=1.3, zorder=7)
    sdpa_ax.scatter([oi], [performance], s=90, color=color, marker=marker, edgecolor="white", linewidth=0.8, zorder=8)
    sdpa_data_points.extend(
        [
            ("SDPA vanilla", oi, VANILLA_SDPA_ISSUED_TFLOPS),
            ("SDPA stable", oi, performance),
        ]
    )
    annotation = sdpa_ax.annotate(
        f"{POINT_SYMBOLS[marker]} {label}\n"
        f"{VANILLA_SDPA_ISSUED_TFLOPS:.3f} → {performance:.3f} issued TFLOP/s (+{gain:.1f}%)\n"
        f"effective K+V rate: {vanilla_bw:.2f} → {stable_bw:.2f} GB/s",
        (oi, performance),
        xytext=(24, 38),
        textcoords="offset points",
        ha="left",
        va="bottom",
        color=color,
        fontsize=12.0,
        bbox={"boxstyle": "round,pad=0.27", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.94},
        arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.9, "shrinkA": 5, "shrinkB": 8},
        annotation_clip=True,
    )
    sdpa_annotations.append((label, annotation))

sdpa_ax.axvline(SDPA_ISSUED_OI, color="#0891b2", linestyle="-.", linewidth=0.9, alpha=0.35)
sdpa_ax.set_xlim(5, 500)
sdpa_ax.set_ylim(0.5, 25)
sdpa_ax.set_xlabel("Issued QK+PV intensity (padded GQA-group OP / encoded K+V byte)")
sdpa_ax.set_ylabel("Issued QK+PV-equivalent performance (TFLOP/s)")
sdpa_ax.set_title("Custom BOS NPU — SDPA Issued-Work Roofline-Style Reference")
sdpa_fig.subplots_adjust(left=0.10, right=0.97, bottom=0.16, top=0.88)
sdpa_ax.grid(True, which="both", linewidth=0.55, alpha=0.28)
sdpa_ax.scatter([], [], s=70, facecolor="none", edgecolor="#475569", marker="D", label="Vanilla Jul-25")
sdpa_ax.scatter([], [], s=70, color="#0891b2", edgecolor="white", marker="D", label="Stable Aug-09")
sdpa_ax.legend(loc="upper left", ncol=2, borderaxespad=0.8, framealpha=0.95, fontsize=12.0)
annotate_memory_slope(sdpa_ax, oi=15.0)
sdpa_fig.canvas.draw()
assert_label_boxes_clear_points(sdpa_fig, sdpa_ax, sdpa_annotations, sdpa_data_points)
assert_legend_clear_points_and_labels(sdpa_fig, sdpa_ax, sdpa_annotations, sdpa_data_points)

sdpa_fig.savefig(output_dir / "2026-08-17-bos-roofline-sdpa-reference.png", dpi=180)
sdpa_svg = output_dir / "2026-08-17-bos-roofline-sdpa-reference.svg"
sdpa_fig.savefig(sdpa_svg, metadata={"Date": None})
normalize_svg(sdpa_svg)

print(
    f"memory_band_gbps={DRAM_GBPS_LOW:.3f}..{DRAM_GBPS_HIGH:.3f}; "
    f"measured_12core_gemm_hifi2={HIFI2_12_CORE_MEASURED_TFLOPS:.4f}; "
    f"measured_12core_gemm_lofi={LOFI_12_CORE_MEASURED_TFLOPS:.4f}; "
    f"projected_16core_sdpa_hifi2={SDPA_HIFI2_PROJECTED_TFLOPS:.4f}"
)
for label, oi, performance, _color, _marker in matmul_points:
    useful_vanilla, useful_stable = useful_matmul_tflops[label]
    print(
        f"{label}: issued_oi={oi:.6f}, useful_oi={oi / DECODE_ISSUED_M:.6f}, "
        f"issued_tflops={vanilla_matmul_tflops[label]:.6f}->{performance:.6f}, "
        f"useful_tflops={useful_vanilla:.6f}->{useful_stable:.6f}, "
        f"encoded_weight_gbps={effective_weight_gbps[label][0]:.6f}->{effective_weight_gbps[label][1]:.6f}"
    )
print(
    f"SDPA: useful_oi={SDPA_USEFUL_OI:.6f}, issued_oi={SDPA_ISSUED_OI:.6f}, "
    f"useful_tflops={VANILLA_SDPA_TFLOPS:.6f}->{STABLE_SDPA_TFLOPS:.6f}, "
    f"issued_tflops={VANILLA_SDPA_ISSUED_TFLOPS:.6f}->{STABLE_SDPA_ISSUED_TFLOPS:.6f}"
)
