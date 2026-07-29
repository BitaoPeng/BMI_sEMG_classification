"""Generate an estimated real-time sEMG prediction timeline as an SVG."""

from __future__ import annotations

from html import escape
from pathlib import Path


OUTPUT = Path("estimated_realtime_prediction_timeline.svg")
WIDTH, HEIGHT = 1600, 820

# Estimated placeholder event times in seconds.
GROUND_TRUTH = [
    (0.0, 0),
    (3.0, 1),
    (7.5, 0),
    (12.0, 0),
]

PREDICTION = [
    (0.0, 0),
    (3.26, 1),   # 260 ms response latency
    (5.20, 0),   # brief false switch
    (5.36, 1),
    (7.72, 0),   # 220 ms response latency
    (9.40, 1),   # brief false switch
    (9.54, 0),
    (12.0, 0),
]

CORRECT_REGIONS = [
    (0.0, 3.0),
    (3.26, 5.20),
    (5.36, 7.50),
    (7.72, 9.40),
    (9.54, 12.0),
]


def text(x: float, y: float, value: str, **attrs: object) -> str:
    attributes = " ".join(
        f'{key.replace("_", "-")}="{escape(str(val))}"'
        for key, val in attrs.items()
    )
    return f'<text x="{x:.1f}" y="{y:.1f}" {attributes}>{escape(value)}</text>'


def step_path(events: list[tuple[float, int]], sx, y_relax: float, y_clench: float) -> str:
    def sy(state: int) -> float:
        return y_clench if state else y_relax

    commands = [f"M {sx(events[0][0]):.1f} {sy(events[0][1]):.1f}"]
    previous_state = events[0][1]
    for time_s, state in events[1:]:
        x = sx(time_s)
        commands.append(f"H {x:.1f}")
        if state != previous_state:
            commands.append(f"V {sy(state):.1f}")
        previous_state = state
    return " ".join(commands)


def main() -> None:
    left, right = 190, 1510
    plot_top, plot_bottom = 205, 650
    gt_clench, gt_relax = 270, 355
    pred_clench, pred_relax = 480, 565

    def sx(time_s: float) -> float:
        return left + time_s / 12.0 * (right - left)

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<defs>',
        '<marker id="arrow-amber" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">'
        '<path d="M0,0 L0,6 L9,3 z" fill="#B66A00"/></marker>',
        '<marker id="arrow-red" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">'
        '<path d="M0,0 L0,6 L9,3 z" fill="#C83E4D"/></marker>',
        '</defs>',
        '<rect width="100%" height="100%" fill="#f5f7fa"/>',
        '<g font-family="Arial, Helvetica, sans-serif">',
        text(
            WIDTH / 2,
            48,
            "Estimated Real-time Prediction Example — TD5-D1",
            text_anchor="middle",
            font_size="30",
            font_weight="700",
            fill="#172033",
        ),
        text(
            WIDTH / 2,
            82,
            "Illustrative placeholders only — replace with continuous on-device test data",
            text_anchor="middle",
            font_size="17",
            font_weight="600",
            fill="#a23b3b",
        ),
        (
            f'<rect x="{left - 60}" y="{plot_top - 35}" width="{right - left + 85}" '
            f'height="{plot_bottom - plot_top + 70}" rx="14" '
            'fill="#ffffff" stroke="#d9dee7" stroke-width="1.5"/>'
        ),
    ]

    # Correct stable prediction regions span both lanes.
    for start, stop in CORRECT_REGIONS:
        svg.append(
            f'<rect x="{sx(start):.1f}" y="{plot_top}" '
            f'width="{sx(stop) - sx(start):.1f}" height="{plot_bottom - plot_top}" '
            'fill="#CFEBDD" fill-opacity="0.62"/>'
        )

    # Alternating lane backgrounds and horizontal state guides.
    svg.extend(
        [
            f'<rect x="{left}" y="230" width="{right-left}" height="150" fill="#f8fafc" fill-opacity="0.70"/>',
            f'<rect x="{left}" y="440" width="{right-left}" height="150" fill="#f8fafc" fill-opacity="0.70"/>',
        ]
    )
    for y in (gt_clench, gt_relax, pred_clench, pred_relax):
        svg.append(
            f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" '
            'stroke="#cfd6df" stroke-width="1" stroke-dasharray="4 6"/>'
        )

    # Time grid and shared x-axis.
    for second in range(13):
        x = sx(float(second))
        svg.append(
            f'<line x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" y2="{plot_bottom}" '
            'stroke="#e4e8ee" stroke-width="1"/>'
        )
        svg.append(
            text(x, plot_bottom + 27, str(second), text_anchor="middle", font_size="14", fill="#4b5563")
        )
    svg.extend(
        [
            f'<line x1="{left}" y1="{plot_bottom}" x2="{right}" y2="{plot_bottom}" '
            'stroke="#252a32" stroke-width="1.8"/>',
            text((left + right) / 2, plot_bottom + 65, "Time (s)", text_anchor="middle", font_size="17", font_weight="600"),
            text(left - 25, 305, "Ground-truth", text_anchor="end", font_size="17", font_weight="700", fill="#252a32"),
            text(left - 25, 515, "STM32 prediction", text_anchor="end", font_size="17", font_weight="700", fill="#2878B5"),
            text(left - 3, gt_clench + 5, "Clench (1)", text_anchor="end", font_size="13", fill="#56606e"),
            text(left - 3, gt_relax + 5, "Relax (0)", text_anchor="end", font_size="13", fill="#56606e"),
            text(left - 3, pred_clench + 5, "Clench (1)", text_anchor="end", font_size="13", fill="#56606e"),
            text(left - 3, pred_relax + 5, "Relax (0)", text_anchor="end", font_size="13", fill="#56606e"),
        ]
    )

    # Ground truth and prediction step traces.
    svg.append(
        f'<path d="{step_path(GROUND_TRUTH, sx, gt_relax, gt_clench)}" '
        'fill="none" stroke="#20242b" stroke-width="5" stroke-linejoin="round"/>'
    )
    svg.append(
        f'<path d="{step_path(PREDICTION, sx, pred_relax, pred_clench)}" '
        'fill="none" stroke="#2878B5" stroke-width="5" stroke-linejoin="round"/>'
    )

    # Ground-truth transitions.
    transitions = [
        (3.0, "Relax → Clench", -100),
        (7.5, "Clench → Relax", -100),
    ]
    for time_s, label, label_dx in transitions:
        x = sx(time_s)
        svg.extend(
            [
                f'<line x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" y2="{plot_bottom}" '
                'stroke="#5B6574" stroke-width="2" stroke-dasharray="7 6"/>',
                text(x + label_dx, plot_top - 13, label, font_size="15", font_weight="700", fill="#394150"),
            ]
        )

    # Response-latency arrows between ground-truth and prediction transitions.
    latency_specs = [
        (3.0, 3.26, 420, "Response latency ≈ 260 ms"),
        (7.5, 7.72, 420, "Response latency ≈ 220 ms"),
    ]
    for truth_t, prediction_t, y, label in latency_specs:
        x1, x2 = sx(truth_t), sx(prediction_t)
        svg.extend(
            [
                f'<line x1="{x1 + 3:.1f}" y1="{y}" x2="{x2 - 4:.1f}" y2="{y}" '
                'stroke="#B66A00" stroke-width="3" marker-end="url(#arrow-amber)"/>',
                text((x1 + x2) / 2, y - 12, label, text_anchor="middle", font_size="14", font_weight="700", fill="#9A5A00"),
            ]
        )

    # False-switch callouts.
    false_switches = [
        (5.28, pred_relax, "False switch ≈ 160 ms", -72, 74),
        (9.47, pred_clench, "False switch ≈ 140 ms", -72, -55),
    ]
    for time_s, y, label, dx, dy in false_switches:
        x = sx(time_s)
        label_y = y + dy
        svg.extend(
            [
                f'<ellipse cx="{x:.1f}" cy="{y}" rx="13" ry="14" '
                'fill="none" stroke="#C83E4D" stroke-width="3"/>',
                f'<line x1="{x + dx / 2:.1f}" y1="{label_y + (8 if dy < 0 else -8):.1f}" '
                f'x2="{x - 5:.1f}" y2="{y + (-12 if dy < 0 else 12):.1f}" '
                'stroke="#C83E4D" stroke-width="2.5" marker-end="url(#arrow-red)"/>',
                text(x + dx, label_y, label, font_size="14", font_weight="700", fill="#B83242"),
            ]
        )

    # Legend and takeaway.
    legend_y = 758
    svg.extend(
        [
            f'<rect x="260" y="{legend_y - 14}" width="28" height="18" fill="#CFEBDD" fill-opacity="0.9"/>',
            text(298, legend_y, "Correct stable prediction", font_size="15", fill="#252a32"),
            f'<line x1="530" y1="{legend_y - 5}" x2="575" y2="{legend_y - 5}" stroke="#20242b" stroke-width="5"/>',
            text(587, legend_y, "Ground truth", font_size="15", fill="#252a32"),
            f'<line x1="735" y1="{legend_y - 5}" x2="780" y2="{legend_y - 5}" stroke="#2878B5" stroke-width="5"/>',
            text(792, legend_y, "STM32 prediction", font_size="15", fill="#252a32"),
            f'<line x1="1010" y1="{legend_y - 5}" x2="1055" y2="{legend_y - 5}" stroke="#B66A00" stroke-width="3"/>',
            text(1067, legend_y, "Response latency", font_size="15", fill="#252a32"),
            f'<circle cx="1265" cy="{legend_y - 5}" r="8" fill="none" stroke="#C83E4D" stroke-width="3"/>',
            text(1280, legend_y, "False switch", font_size="15", fill="#252a32"),
            "</g></svg>",
        ]
    )

    OUTPUT.write_text("\n".join(svg), encoding="utf-8")
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
