"""Inline SVG for the two diagrams that carry the argument.

Drawn on the server as plain SVG rather than by a charting library in the
browser. A deployed demo that renders a blank panel because a CDN was slow or
blocked scores as a demo that does not work, and the whole reason the dashboard
exists is that judges have to be able to load it.

Two charts:

  the cost waterfall   gross edge, then a bar down for each component of
                       execution cost, then what is left. When the net bar is
                       below zero the reader is looking at the finding this
                       project is built on: real gross alpha, no net alpha.

  the payoff diagram   the structure's value at expiry across prices, with its
                       breakevens, its worst case, and where spot was at entry.
                       For a broken-wing butterfly entered for a credit, the
                       flat positive tail above every strike is visible, which
                       is the thing that makes it worth trading over a ratio
                       spread with an open downside.
"""

from __future__ import annotations

from html import escape
from typing import Sequence

# The order the waterfall is drawn in, and how each component is labelled.
COMPONENTS = (
    ("gross_edge", "gross edge"),
    ("half_spread", "half-spread"),
    ("slippage", "slippage"),
    ("fees", "fees"),
    ("exit_reserve", "exit reserve"),
    ("net_edge", "net edge"),
)


def _money(value: float) -> str:
    return f"{value:+,.2f}"


def waterfall_svg(waterfall: dict[str, float], width: int = 720, height: int = 300) -> str:
    """Gross edge, less each cost, to net. Totals are drawn from the baseline."""
    missing = [key for key, _ in COMPONENTS if key not in waterfall]
    if missing:
        raise ValueError(f"the waterfall is missing {missing}")

    gross = float(waterfall["gross_edge"])
    net = float(waterfall["net_edge"])
    steps = [(key, label, float(waterfall[key])) for key, label in COMPONENTS]

    # The vertical scale has to cover the running total as it walks down from
    # gross to net, not just the endpoints, or a large cost clips off the chart.
    running, levels = 0.0, [0.0]
    for key, _, value in steps:
        running = value if key in ("gross_edge", "net_edge") else running + value
        levels.append(running)
    top, bottom = max(levels + [0.0]), min(levels + [0.0])
    span = (top - bottom) or 1.0

    pad_left, pad_top, pad_bottom = 70, 26, 46
    plot_height = height - pad_top - pad_bottom
    plot_width = width - pad_left - 20
    slot = plot_width / len(steps)
    bar_width = min(slot * 0.56, 74)

    def y_of(value: float) -> float:
        return pad_top + (top - value) / span * plot_height

    baseline = y_of(0.0)
    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="execution cost against gross edge" class="chart">',
        f'<line x1="{pad_left - 8}" y1="{baseline:.1f}" x2="{width - 14}" y2="{baseline:.1f}" '
        f'class="axis"/>',
    ]

    running = 0.0
    for index, (key, label, value) in enumerate(steps):
        x = pad_left + slot * index + (slot - bar_width) / 2
        is_total = key in ("gross_edge", "net_edge")
        if is_total:
            start, end = 0.0, value
            running = value
        else:
            start, end = running, running + value
            running = end

        y_top, y_bottom = y_of(max(start, end)), y_of(min(start, end))
        bar_height = max(abs(y_bottom - y_top), 1.6)
        if is_total:
            css = "bar-total" if value >= 0 else "bar-negative"
        else:
            css = "bar-cost"
        parts.append(
            f'<rect x="{x:.1f}" y="{y_top:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" rx="2" class="{css}"/>'
        )
        centre = x + bar_width / 2
        label_y = y_top - 7 if end >= start else y_bottom + 15
        parts.append(
            f'<text x="{centre:.1f}" y="{label_y:.1f}" class="bar-value">{_money(value)}</text>'
        )
        parts.append(
            f'<text x="{centre:.1f}" y="{height - 24}" class="bar-label">{escape(label)}</text>'
        )

    verdict = (
        "cost exceeded the edge" if net < 0 else "edge survived the cost"
    )
    parts.append(
        f'<text x="{pad_left - 12}" y="{pad_top - 10}" class="chart-note" '
        f'text-anchor="start">gross {_money(gross)} → net {_money(net)} · {verdict}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def payoff_svg(
    curve: Sequence[tuple[float, float]],
    breakevens: Sequence[float] = (),
    spot: float | None = None,
    width: int = 720,
    height: int = 300,
) -> str:
    """Value at expiry across underlying prices, with the kinks kept exact."""
    if len(curve) < 2:
        raise ValueError("a payoff diagram needs at least two points")

    prices = [point[0] for point in curve]
    values = [point[1] for point in curve]
    low_price, high_price = min(prices), max(prices)
    low_value, high_value = min(values + [0.0]), max(values + [0.0])
    price_span = (high_price - low_price) or 1.0
    value_span = (high_value - low_value) or 1.0

    pad_left, pad_right, pad_top, pad_bottom = 58, 18, 24, 40
    plot_width = width - pad_left - pad_right
    plot_height = height - pad_top - pad_bottom

    def x_of(price: float) -> float:
        return pad_left + (price - low_price) / price_span * plot_width

    def y_of(value: float) -> float:
        return pad_top + (high_value - value) / value_span * plot_height

    zero_y = y_of(0.0)
    points = " ".join(f"{x_of(p):.1f},{y_of(v):.1f}" for p, v in curve)
    # Two fills, clipped at the zero line, so profit and loss read apart at a
    # glance without needing a legend.
    area_up = f"{pad_left},{zero_y:.1f} " + points + f" {pad_left + plot_width},{zero_y:.1f}"

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="payoff at expiry" class="chart">',
        f'<defs><clipPath id="above"><rect x="0" y="0" width="{width}" '
        f'height="{zero_y:.1f}"/></clipPath>'
        f'<clipPath id="below"><rect x="0" y="{zero_y:.1f}" width="{width}" '
        f'height="{max(height - zero_y, 0):.1f}"/></clipPath></defs>',
        f'<polygon points="{area_up}" class="pay-fill-up" clip-path="url(#above)"/>',
        f'<polygon points="{area_up}" class="pay-fill-down" clip-path="url(#below)"/>',
        f'<line x1="{pad_left}" y1="{zero_y:.1f}" x2="{width - pad_right}" '
        f'y2="{zero_y:.1f}" class="axis"/>',
        f'<polyline points="{points}" class="pay-line"/>',
    ]

    for level in breakevens:
        if low_price <= level <= high_price:
            x = x_of(level)
            parts.append(
                f'<line x1="{x:.1f}" y1="{pad_top}" x2="{x:.1f}" '
                f'y2="{height - pad_bottom}" class="marker-breakeven"/>'
            )
            parts.append(
                f'<text x="{x:.1f}" y="{height - 24}" class="bar-label">{level:g}</text>'
            )

    if spot is not None and low_price <= spot <= high_price:
        x = x_of(spot)
        parts.append(
            f'<line x1="{x:.1f}" y1="{pad_top}" x2="{x:.1f}" '
            f'y2="{height - pad_bottom}" class="marker-spot"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{pad_top - 8}" class="bar-label">spot {spot:,.2f}</text>'
        )

    parts.append(
        f'<text x="{pad_left - 10}" y="{y_of(high_value):.1f}" class="axis-label" '
        f'text-anchor="end">{high_value:+,.0f}</text>'
    )
    parts.append(
        f'<text x="{pad_left - 10}" y="{y_of(low_value):.1f}" class="axis-label" '
        f'text-anchor="end">{low_value:+,.0f}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)
