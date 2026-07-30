#!/usr/bin/env python3
"""Draw a compact D=1 versus D=2 sliding-window comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def draw_case(
    ax: plt.Axes,
    *,
    title: str,
    window_points: int,
    fs_hz: int,
    color: str,
) -> None:
    overlap_points = window_points // 2
    hop_points = window_points - overlap_points
    point_interval_ms = 1000.0 / fs_hz
    window_ms = window_points * point_interval_ms
    overlap_ms = overlap_points * point_interval_ms
    hop_ms = hop_points * point_interval_ms
    total_points = window_points + hop_points

    time_ms = np.arange(total_points, dtype=np.float64) * point_interval_ms
    signal = (
        0.72 * np.sin(2.0 * np.pi * time_ms / 78.0)
        + 0.22 * np.sin(2.0 * np.pi * time_ms / 27.0)
    )

    # Draw both complete windows and emphasize their shared 50% region.
    ax.axvspan(
        0.0,
        window_ms,
        color=color,
        alpha=0.09,
        ec=color,
        lw=1.8,
    )
    ax.axvspan(
        hop_ms,
        hop_ms + window_ms,
        color=color,
        alpha=0.09,
        ec=color,
        lw=1.8,
    )
    ax.axvspan(
        hop_ms,
        window_ms,
        color=color,
        alpha=0.23,
        ec="none",
    )

    ax.text(
        window_ms / 2.0,
        1.15,
        "Window 1",
        color=color,
        fontsize=10.5,
        fontweight="bold",
        ha="center",
        va="center",
    )
    ax.annotate(
        "",
        xy=(window_ms, 1.07),
        xytext=(0.0, 1.07),
        arrowprops=dict(arrowstyle="<->", color=color, lw=1.3),
    )
    ax.text(
        hop_ms + window_ms / 2.0,
        0.95,
        "Window 2",
        color=color,
        fontsize=10.5,
        fontweight="bold",
        ha="center",
        va="center",
    )
    ax.annotate(
        "",
        xy=(hop_ms + window_ms, 0.87),
        xytext=(hop_ms, 0.87),
        arrowprops=dict(arrowstyle="<->", color=color, lw=1.3),
    )

    marker_size = 3.0 if window_points == 128 else 4.2
    ax.plot(
        time_ms,
        signal,
        color="#17202a",
        lw=1.15,
        marker="o",
        markersize=marker_size,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.35,
    )
    ax.text(
        hop_ms + overlap_ms / 2.0,
        -1.03,
        f"Overlap\n{overlap_points} points = {overlap_ms:g} ms",
        ha="center",
        va="center",
        fontsize=9.5,
        color=color,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.92),
    )
    ax.axhline(0.0, color="#7f8c8d", lw=0.7)
    ax.set_xlim(-5, hop_ms + window_ms + 5)
    ax.set_ylim(-1.30, 1.32)
    ax.set_yticks([])
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    ax.text(
        hop_ms + window_ms - 3,
        -0.12,
        f"Window: {window_points} points = {window_ms:g} ms\n"
        f"Hop: {hop_points} points = {hop_ms:g} ms",
        ha="right",
        va="center",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#aab7b8"),
    )
    ax.text(
        hop_ms + window_ms - 3,
        -1.17,
        f"Effective sampling rate: {fs_hz} Hz",
        ha="right",
        fontsize=9.5,
        color="#566573",
    )
    ax.grid(axis="x", color="#d5d8dc", linestyle="--", alpha=0.75)
    ax.spines[["top", "right", "left"]].set_visible(False)


def main() -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 6.6),
        sharex=True,
        constrained_layout=False,
    )
    draw_case(
        axes[0],
        title="D=1: no downsampling",
        window_points=128,
        fs_hz=500,
        color="#e74c3c",
    )
    draw_case(
        axes[1],
        title="D=2: downsampled by 2",
        window_points=64,
        fs_hz=250,
        color="#2980b9",
    )

    axes[1].set_xlabel("Time (ms)", fontsize=12)
    axes[1].set_xticks([0, 64, 128, 192, 256, 320, 384])
    fig.suptitle(
        "50% Overlapping Windows: D=1 vs D=2",
        fontsize=19,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(
        left=0.04,
        right=0.99,
        top=0.89,
        bottom=0.15,
        hspace=0.16,
    )
    fig.text(
        0.5,
        0.035,
        "Both cases use a 256 ms window and a 128 ms hop; "
        "D=2 contains half as many sampled points.",
        ha="center",
        fontsize=11,
        color="#34495e",
    )

    output_root = Path(__file__).with_name("sliding_window_d1_d2")
    fig.savefig(output_root.with_suffix(".png"), dpi=180, facecolor="white")
    fig.savefig(output_root.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    print(output_root.with_suffix(".png").resolve())
    print(output_root.with_suffix(".svg").resolve())


if __name__ == "__main__":
    main()
