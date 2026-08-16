#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DRAM_GBPS = 96.0
HIFI2_TFLOPS = 26.624
LOFI_TFLOPS = 53.248
ACTIVE_MATMUL_CORE_FRACTION = 12.0 / 20.0
HIFI2_12_CORE_TFLOPS = HIFI2_TFLOPS * ACTIVE_MATMUL_CORE_FRACTION
LOFI_12_CORE_TFLOPS = LOFI_TFLOPS * ACTIVE_MATMUL_CORE_FRACTION

sdpa_points = [
    # label, OI [OP/byte], measured TFLOP/s, color, marker
    ("SDPA K/V padded", 60.24, 4.21, "#0891b2", "D"),
]

matmul_points = [
    # label, OI [OP/byte], measured TFLOP/s, color, marker
    ("QKV BFP8", 60.24, 3.61, "#2563eb", "o"),
    ("Wo BFP8", 60.24, 3.64, "#2563eb", "s"),
    ("W2 BFP8", 60.24, 3.76, "#2563eb", "^"),
    ("W1 BFP4", 113.78, 7.12, "#ea580c", "o"),
    ("W3 BFP4", 113.78, 7.04, "#ea580c", "s"),
]

vanilla_matmul_tflops = {
    "QKV BFP8": 2.329964,
    "Wo BFP8": 2.565945,
    "W2 BFP8": 2.359585,
    "W1 BFP4": 2.911211,
    "W3 BFP4": 2.913855,
}
VANILLA_SDPA_TFLOPS = 2.464847

output_dir = Path(__file__).resolve().parent
x = np.logspace(-1, 3.4, 1200)
memory_roof = DRAM_GBPS * x / 1000.0
hifi2_roof = np.minimum(memory_roof, HIFI2_TFLOPS)
lofi_roof = np.minimum(memory_roof, LOFI_TFLOPS)


def annotate_memory_slope(axis, oi=2.5):
    performance = DRAM_GBPS * oi / 1000.0
    axis.annotate(
        "96 GB/s memory-bandwidth slope",
        (oi, performance),
        xytext=(8, 8),
        textcoords="offset points",
        rotation=27,
        rotation_mode="anchor",
        color="#334155",
        fontsize=9.0,
        fontweight="bold",
    )


fig, ax = plt.subplots(figsize=(10.5, 6.3), constrained_layout=True)
ax.loglog(x, hifi2_roof, linewidth=2.4, color="#2563eb", label="20-core HiFi2 roof: 26.624 TFLOP/s")
ax.loglog(x, lofi_roof, linewidth=2.4, color="#ea580c", label="20-core LoFi roof: 53.248 TFLOP/s")
ax.axhline(
    HIFI2_12_CORE_TFLOPS,
    color="#2563eb",
    linestyle="--",
    linewidth=1.3,
    alpha=0.65,
    label="12-core HiFi2 ceiling: 15.974 TFLOP/s",
)
ax.axhline(
    LOFI_12_CORE_TFLOPS,
    color="#ea580c",
    linestyle="--",
    linewidth=1.3,
    alpha=0.65,
    label="12-core LoFi ceiling: 31.949 TFLOP/s",
)
ax.loglog(
    x,
    memory_roof,
    linestyle=":",
    linewidth=2.0,
    color="#475569",
    label="Global read-only upper roof: 96 GB/s × OI",
)
annotate_memory_slope(ax)

hifi2_ridge = HIFI2_TFLOPS * 1000.0 / DRAM_GBPS
lofi_ridge = LOFI_TFLOPS * 1000.0 / DRAM_GBPS
for ridge, ceiling, color, label in [
    (hifi2_ridge, HIFI2_TFLOPS, "#2563eb", "HiFi2 ridge\n277.3 OP/byte"),
    (lofi_ridge, LOFI_TFLOPS, "#ea580c", "LoFi ridge\n554.7 OP/byte"),
]:
    ax.scatter([ridge], [ceiling], s=54, color=color, zorder=5)
    ax.annotate(label, (ridge, ceiling), xytext=(8, -32), textcoords="offset points", color=color, fontsize=9)

for oi, label, color in [
    (60.24, "BFP8 padded M=32 OI\n60.24 OP/byte", "#2563eb"),
    (113.78, "BFP4 padded M=32 OI\n113.78 OP/byte", "#ea580c"),
]:
    ax.axvline(oi, color=color, linestyle="-.", linewidth=1.0, alpha=0.45)
    ax.annotate(
        label,
        (oi, DRAM_GBPS * oi / 1000.0),
        xytext=(7, 8),
        textcoords="offset points",
        color=color,
        fontsize=8.5,
    )

annotation_offsets = {
    "SDPA K/V padded": (8, 25),
    "QKV BFP8": (-88, -42),
    "Wo BFP8": (-88, -16),
    "W2 BFP8": (8, -30),
    "W1 BFP4": (8, 6),
    "W3 BFP4": (8, -18),
}
for label, oi, performance, color, marker in matmul_points:
    roof = DRAM_GBPS * oi / 1000.0
    utilization = performance / roof * 100.0
    vanilla_performance = vanilla_matmul_tflops[label]
    gain = (performance / vanilla_performance - 1.0) * 100.0
    ax.plot([oi, oi], [vanilla_performance, performance], color="#94a3b8", linewidth=1.2, zorder=6)
    ax.scatter(
        [oi],
        [vanilla_performance],
        s=82,
        facecolor="none",
        edgecolor="#475569",
        marker=marker,
        linewidth=1.3,
        zorder=7,
    )
    ax.scatter(
        [oi],
        [performance],
        s=82,
        color=color,
        marker=marker,
        edgecolor="white",
        linewidth=0.8,
        zorder=8,
    )
    ax.annotate(
        f"{label}: {vanilla_performance:.2f} → {performance:.2f} TFLOP/s\n+{gain:.1f}%; optimized {utilization:.1f}% of memory roof",
        (oi, performance),
        xytext=annotation_offsets[label],
        textcoords="offset points",
        color=color,
        fontsize=8.2,
    )

ax.set_xlim(0.1, 2500)
ax.set_ylim(0.008, 90)
ax.set_xlabel("Operational intensity (OP/byte)")
ax.set_ylabel("Performance (TFLOP/s)")
ax.set_title("Custom 20-core BOS NPU — Matmul Roofline")
ax.text(
    0.01,
    0.98,
    "96 GB/s: optimal read-only microbenchmark upper bound\n"
    "Hollow: Jul 25 vanilla actual-prefill profile; filled: later best-stable profiles\n"
    "Cross-run comparison; device-kernel duration and padded issued OPs\n"
    "QKV/Wo/W2: BFP8; W1/W3: BFP4",
    transform=ax.transAxes,
    va="top",
    ha="left",
    fontsize=9.0,
    bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92},
)
ax.grid(True, which="both", linewidth=0.55, alpha=0.28)
ax.scatter([], [], s=70, facecolor="none", edgecolor="#475569", label="Vanilla: Jul 25 actual prefill")
ax.scatter([], [], s=70, color="#0f172a", edgecolor="white", label="Best-stable optimized")
ax.legend(loc="lower right", framealpha=0.95)
fig.savefig(output_dir / "2026-08-16-bos-roofline-96gbps-matmul.png", dpi=180)
fig.savefig(output_dir / "2026-08-16-bos-roofline-96gbps-matmul.svg")

sdpa_fig, sdpa_ax = plt.subplots(figsize=(10.5, 6.3), constrained_layout=True)
sdpa_ax.loglog(
    x,
    memory_roof,
    linestyle=":",
    linewidth=2.4,
    color="#475569",
    label="Read-only upper roof: 96 GB/s × OI",
)
annotate_memory_slope(sdpa_ax)

for label, oi, performance, color, marker in sdpa_points:
    roof = DRAM_GBPS * oi / 1000.0
    utilization = performance / roof * 100.0
    gain = (performance / VANILLA_SDPA_TFLOPS - 1.0) * 100.0
    sdpa_ax.plot(
        [oi, oi], [VANILLA_SDPA_TFLOPS, performance], color="#94a3b8", linewidth=1.2, zorder=6
    )
    sdpa_ax.scatter(
        [oi],
        [VANILLA_SDPA_TFLOPS],
        s=90,
        facecolor="none",
        edgecolor="#475569",
        marker=marker,
        linewidth=1.3,
        zorder=7,
    )
    sdpa_ax.scatter(
        [oi],
        [performance],
        s=90,
        color=color,
        marker=marker,
        edgecolor="white",
        linewidth=0.8,
        zorder=8,
    )
    sdpa_ax.annotate(
        f"{label}: {VANILLA_SDPA_TFLOPS:.2f} → {performance:.2f} TFLOP/s\n+{gain:.1f}%; optimized {utilization:.1f}% of memory roof",
        (oi, performance),
        xytext=(10, -34),
        textcoords="offset points",
        color=color,
        fontsize=9.0,
    )

sdpa_ax.axvline(60.24, color="#0891b2", linestyle="-.", linewidth=1.0, alpha=0.45)
sdpa_ax.set_xlim(1, 1000)
sdpa_ax.set_ylim(0.08, 20)
sdpa_ax.set_xlabel("Issued QK/PV operational intensity (OP/byte)")
sdpa_ax.set_ylabel("Issued QK/PV-equivalent performance (TFLOP/s)")
sdpa_ax.set_title("Custom 20-core BOS NPU — SDPA Memory Roof")
sdpa_ax.text(
    0.01,
    0.98,
    "Hollow: Jul 25 vanilla actual-prefill; filled: later best-stable\n"
    "Cross-run device-kernel comparison; padded QK/PV MACs only\n"
    "Softmax/reducer OPs excluded; no single matmul compute ceiling",
    transform=sdpa_ax.transAxes,
    va="top",
    ha="left",
    fontsize=9.0,
    bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92},
)
sdpa_ax.grid(True, which="both", linewidth=0.55, alpha=0.28)
sdpa_ax.scatter([], [], s=70, facecolor="none", edgecolor="#475569", marker="D", label="Vanilla: Jul 25 actual prefill")
sdpa_ax.scatter([], [], s=70, color="#0891b2", edgecolor="white", marker="D", label="Best-stable optimized")
sdpa_ax.legend(loc="lower right", framealpha=0.95)

sdpa_fig.savefig(output_dir / "2026-08-16-bos-roofline-96gbps-sdpa.png", dpi=180)
sdpa_fig.savefig(output_dir / "2026-08-16-bos-roofline-96gbps-sdpa.svg")
