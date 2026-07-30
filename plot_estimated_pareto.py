"""Generate an SVG with two estimated Pareto-front plots (standard library only)."""

from __future__ import annotations

from html import escape
from pathlib import Path


OUTPUT = Path("estimated_pareto_fronts.svg")
WIDTH, HEIGHT = 1600, 760

# Estimated placeholders only; replace with measured on-device values.
CONFIGS = [
    {"label": "TD3-D1", "family": "TD3", "d": 1, "time": 0.32, "ram": 1.2, "ba": 84.5},
    {"label": "TD5-D1", "family": "TD5", "d": 1, "time": 0.48, "ram": 1.3, "ba": 87.5},
    {"label": "FD4-D1", "family": "FD4", "d": 1, "time": 1.13, "ram": 3.0, "ba": 81.5},
    {"label": "TFD9-D1", "family": "TFD9", "d": 1, "time": 1.40, "ram": 3.2, "ba": 89.5},
    {"label": "TD3-D2", "family": "TD3", "d": 2, "time": 0.40, "ram": 0.9, "ba": 82.5},
    {"label": "TD5-D2", "family": "TD5", "d": 2, "time": 0.53, "ram": 1.0, "ba": 85.5},
    {"label": "FD4-D2", "family": "FD4", "d": 2, "time": 0.88, "ram": 2.0, "ba": 78.5},
    {"label": "TFD9-D2", "family": "TFD9", "d": 2, "time": 1.10, "ram": 2.2, "ba": 87.5},
]

COLORS = {
    "TD3": "#2878B5",
    "TD5": "#2A9D8F",
    "FD4": "#E9A23B",
    "TFD9": "#D1495B",
}


def pareto_front(x_key: str) -> list[dict]:
    front = []
    for candidate in CONFIGS:
        dominated = any(
            other[x_key] <= candidate[x_key]
            and other["ba"] >= candidate["ba"]
            and (
                other[x_key] < candidate[x_key]
                or other["ba"] > candidate["ba"]
            )
            for other in CONFIGS
        )
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda row: row[x_key])


def text(x: float, y: float, value: str, **attrs: object) -> str:
    attributes = " ".join(
        f'{key.replace("_", "-")}="{escape(str(val))}"'
        for key, val in attrs.items()
    )
    return f'<text x="{x:.1f}" y="{y:.1f}" {attributes}>{escape(value)}</text>'


def panel(
    x0: int,
    title: str,
    x_key: str,
    x_label: str,
    x_min: float,
    x_max: float,
    ticks: list[float],
) -> list[str]:
    items: list[str] = []
    left, right = x0 + 92, x0 + 700
    top, bottom = 178, 596
    y_min, y_max = 76.0, 91.0

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (right - left)

    def sy(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    items.append(
        f'<rect x="{x0 + 25}" y="135" width="720" height="510" '
        'rx="12" fill="#ffffff" stroke="#d9dee7"/>'
    )
    items.append(text(x0 + 385, 164, title, text_anchor="middle", font_size="21", font_weight="700"))

    for y_value in (75, 80, 85, 90):
        y = sy(y_value)
        items.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
            'stroke="#d7dde6" stroke-dasharray="3 5"/>'
        )
        items.append(text(left - 14, y + 5, f"{y_value}%", text_anchor="end", font_size="14", fill="#4b5563"))

    for tick in ticks:
        x = sx(tick)
        items.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" '
            'stroke="#eef1f5"/>'
        )
        items.append(text(x, bottom + 25, f"{tick:g}", text_anchor="middle", font_size="14", fill="#4b5563"))

    items.extend(
        [
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#252a32" stroke-width="1.8"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#252a32" stroke-width="1.8"/>',
            # Zigzag axis-break marks indicate that neither axis begins at zero.
            f'<path d="M {left + 12} {bottom + 4} l 5 -8 l 5 8 l 5 -8" fill="none" stroke="#252a32" stroke-width="1.8"/>',
            f'<path d="M {left - 4} {bottom - 15} l 8 -5 l -8 -5 l 8 -5" fill="none" stroke="#252a32" stroke-width="1.8"/>',
            text((left + right) / 2, bottom + 62, x_label, text_anchor="middle", font_size="16", font_weight="600"),
            (
                f'<text x="{left - 65}" y="{(top + bottom) / 2:.1f}" '
                'text-anchor="middle" font-size="16" font-weight="600" '
                f'transform="rotate(-90 {left - 65} {(top + bottom) / 2:.1f})">'
                'Classification Accuracy (%)</text>'
            ),
            text(left + 12, top + 24, "Better ↖", font_size="15", font_weight="700", fill="#374151"),
        ]
    )

    front = pareto_front(x_key)
    points = " ".join(f"{sx(row[x_key]):.1f},{sy(row['ba']):.1f}" for row in front)
    items.append(
        f'<polyline points="{points}" fill="none" stroke="#20242b" '
        'stroke-width="2.5" stroke-dasharray="8 6"/>'
    )

    label_offsets = {
        "TD3-D1": (10, -12),
        "TD5-D1": (10, -12),
        "FD4-D1": (10, -12),
        "TFD9-D1": (-94, -12),
        "TD3-D2": (10, 23),
        "TD5-D2": (10, 23),
        "FD4-D2": (10, 23),
        "TFD9-D2": (10, 23),
    }
    for row in CONFIGS:
        x, y = sx(row[x_key]), sy(row["ba"])
        is_front = row in front
        radius = 8.5 if is_front else 7
        opacity = 1 if is_front else 0.68
        if row["d"] == 1:
            shape = (
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" '
                f'fill="{COLORS[row["family"]]}" fill-opacity="{opacity}" '
                f'stroke="{"#111827" if is_front else "#ffffff"}" stroke-width="2"/>'
            )
        else:
            shape = (
                f'<polygon points="{x:.1f},{y - radius - 1:.1f} '
                f'{x - radius:.1f},{y + radius:.1f} {x + radius:.1f},{y + radius:.1f}" '
                f'fill="{COLORS[row["family"]]}" fill-opacity="{opacity}" '
                f'stroke="{"#111827" if is_front else "#ffffff"}" stroke-width="2"/>'
            )
        items.append(shape)
        dx, dy = label_offsets[row["label"]]
        items.append(text(x + dx, y + dy, row["label"], font_size="14", font_weight="600", fill="#252a32"))

    return items


def main() -> None:
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="#f5f7fa"/>',
        '<g font-family="Arial, Helvetica, sans-serif">',
        text(WIDTH / 2, 48, "Estimated Pareto Fronts: Accuracy–Cost Trade-off", text_anchor="middle", font_size="30", font_weight="700", fill="#172033"),
        text(WIDTH / 2, 78, "Higher accuracy, lower processing time, and lower memory footprint are better.", text_anchor="middle", font_size="17", font_weight="600", fill="#374151"),
        text(WIDTH / 2, 105, "Illustrative placeholders only — replace with on-device measurements", text_anchor="middle", font_size="15", font_weight="600", fill="#a23b3b"),
    ]
    svg += panel(
        35,
        "(a) Accuracy vs. Processing Time",
        "time",
        "Processing Time (ms)",
        0.2,
        1.5,
        [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4],
    )
    svg += panel(
        815,
        "(b) Accuracy vs. Peak RAM",
        "ram",
        "Memory Usage (KB)",
        0.6,
        3.4,
        [0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2],
    )

    legend_y = 700
    legend_x = 340
    for family in ("TD3", "TD5", "FD4", "TFD9"):
        svg.append(f'<circle cx="{legend_x}" cy="{legend_y}" r="7" fill="{COLORS[family]}"/>')
        svg.append(text(legend_x + 13, legend_y + 5, family, font_size="15", fill="#252a32"))
        legend_x += 100
    svg.extend(
        [
            f'<circle cx="{legend_x + 10}" cy="{legend_y}" r="7" fill="#6b7280"/>',
            text(legend_x + 24, legend_y + 5, "D=1", font_size="15", fill="#252a32"),
            (
                f'<polygon points="{legend_x + 94},{legend_y - 8} '
                f'{legend_x + 86},{legend_y + 7} {legend_x + 102},{legend_y + 7}" fill="#6b7280"/>'
            ),
            text(legend_x + 108, legend_y + 5, "D=2", font_size="15", fill="#252a32"),
            f'<line x1="{legend_x + 170}" y1="{legend_y}" x2="{legend_x + 215}" y2="{legend_y}" stroke="#20242b" stroke-width="2.5" stroke-dasharray="8 6"/>',
            text(legend_x + 225, legend_y + 5, "Pareto front", font_size="15", fill="#252a32"),
            "</g></svg>",
        ]
    )
    OUTPUT.write_text("\n".join(svg), encoding="utf-8")
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
