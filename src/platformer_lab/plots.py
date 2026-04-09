"""Figure builders and plotting helpers for experiment reports."""

from collections.abc import Mapping, Sequence
from importlib.resources import as_file, files
import os
from pathlib import Path
from typing import Any

from .settings import (
    ABLATION_COLORS,
    AUTO_TICK_MARKER,
    CACHE_OUTPUT_DIR,
    COLOR_PALETTE,
    CONTROLLER_COMPACT_NAMES,
    CONTROLLER_DISPLAY_ORDER,
    CONTROLLER_FILL_COLORS,
    CONTROLLER_LINE_COLORS,
    CONTROLLER_MARKERS,
    CONTROLLER_SHORT_NAMES,
    NOISE_ROBUSTNESS_LEVELS,
    PAPER_AXES_LABEL_SIZE,
    PAPER_AXES_TITLE_SIZE,
    PAPER_CALLOUT_LABEL_SIZE,
    PAPER_DENSE_TICK_LABEL_SIZE,
    PAPER_FIGURE_TITLE_SIZE,
    PAPER_INLINE_LABEL_SIZE,
    PAPER_TICK_LABEL_SIZE,
    PLOT_DPI,
    PLOT_FONT_FILENAME,
    PLOT_FONT_NAME,
    PLOT_OUTPUT_DIR,
    PLOT_SPECS,
    PLOT_WIDTH,
    SHORT_PLOT_HEIGHT,
    TALL_PLOT_HEIGHT,
    sort_controller_names,
)

# Configure Matplotlib cache locations before importing pyplot.
XDG_CACHE_DIR = CACHE_OUTPUT_DIR / "xdg-cache"
MPL_CONFIG_DIR = CACHE_OUTPUT_DIR / "mplconfig"
for cache_dir in (CACHE_OUTPUT_DIR, XDG_CACHE_DIR, MPL_CONFIG_DIR):
    cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter

from .environment import GridPosition, LevelScenario, PatrolEnemy

PlotRow = Mapping[str, Any]
SeriesMap = Mapping[str, Sequence[float]]
TraceMap = Mapping[str, Sequence[GridPosition]]


def load_plot_font_family() -> str:
    """Registers the bundled plot font and returns its family name."""
    font_resource = files("platformer_lab").joinpath(
        "assets", "fonts", PLOT_FONT_FILENAME
    )
    with as_file(font_resource) as font_path:
        font_manager.fontManager.addfont(str(font_path))
        font_family = font_manager.FontProperties(fname=str(font_path)).get_name()
    if font_family != PLOT_FONT_NAME:
        raise RuntimeError(
            f"Expected bundled plot font {PLOT_FONT_NAME!r}, got {font_family!r}."
        )
    return font_family


plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "path"
PLOT_FONT_FAMILY = load_plot_font_family()
plt.rcParams["font.family"] = PLOT_FONT_FAMILY
plt.rcParams["font.monospace"] = [PLOT_FONT_FAMILY]
plt.rcParams["font.size"] = PAPER_TICK_LABEL_SIZE
plt.rcParams["figure.titlesize"] = PAPER_FIGURE_TITLE_SIZE
plt.rcParams["axes.titlesize"] = PAPER_AXES_TITLE_SIZE
plt.rcParams["axes.labelsize"] = PAPER_AXES_LABEL_SIZE
plt.rcParams["xtick.labelsize"] = PAPER_TICK_LABEL_SIZE
plt.rcParams["ytick.labelsize"] = PAPER_TICK_LABEL_SIZE


def resolve_metric_summary(
    row: PlotRow,
    metric_key: str,
) -> tuple[float, float, float]:
    """Resolves the preferred point estimate and interval columns."""
    value_key = next(
        (candidate for candidate in (
            f"{metric_key}_iqm",
            f"{metric_key}_mean",
            metric_key) if candidate in row),
        metric_key,
    )
    low_key = next(
        (
            candidate
            for candidate in (
                value_key.replace("_iqm", "_iqm_ci_low"),
                value_key.replace("_mean", "_ci_low"),
                f"{metric_key}_ci_low",
            )
            if candidate in row
        ),
        value_key,
    )
    high_key = next(
        (
            candidate
            for candidate in (
                value_key.replace("_iqm", "_iqm_ci_high"),
                value_key.replace("_mean", "_ci_high"),
                f"{metric_key}_ci_high",
            )
            if candidate in row
        ),
        value_key,
    )
    return float(row[value_key]), float(row[low_key]), float(row[high_key])


def style_plot_title(fig: Any, title: str) -> None:
    """Applies the shared figure title styling used across report graphics."""
    fig.set_facecolor(COLOR_PALETTE["sky"])
    title_artist = fig.suptitle(
        title,
        fontsize=PAPER_FIGURE_TITLE_SIZE,
        fontweight="bold",
        fontfamily=PLOT_FONT_FAMILY,
        color=COLOR_PALETTE["ink"],
        y=0.976,
    )
    title_artist.set_bbox(
        dict(
            facecolor=COLOR_PALETTE["cloud"],
            edgecolor=COLOR_PALETTE["ink"],
            boxstyle="square,pad=.29",
            linewidth=1.6,
        )
    )


def identity_line_handle(artist: Any) -> Any:
    """Returns a plotted artist unchanged to keep call sites readable."""
    return artist


def style_axes(
    ax: Any,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    facecolor: str | None = None,
    grid_axis: str = "y",
) -> None:
    """Applies the project's common axis styling and label formatting."""
    ax.set_facecolor(facecolor or COLOR_PALETTE["panel"])
    for spine in ax.spines.values():
        spine.set_color(COLOR_PALETTE["ink"])
        spine.set_linewidth(1.85)
    ax.tick_params(
        axis="both",
        colors=COLOR_PALETTE["ink"],
        labelsize=PAPER_TICK_LABEL_SIZE,
        width=1.28,
        length=3.8)
    for axis in (ax.xaxis, ax.yaxis):
        formatter = ScalarFormatter(useMathText=False)
        formatter.set_scientific(False)
        formatter.set_useOffset(False)
        axis.set_major_formatter(formatter)
        axis.get_offset_text().set_visible(False)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontfamily(PLOT_FONT_FAMILY)
    title_artist = ax.set_title(
        title,
        fontsize=PAPER_AXES_TITLE_SIZE,
        fontweight="bold",
        fontfamily=PLOT_FONT_FAMILY,
        color=COLOR_PALETTE["ink"],
        pad=8.0,
    )
    title_artist.set_bbox(
        dict(
            facecolor=COLOR_PALETTE["cloud"],
            edgecolor=COLOR_PALETTE["ink"],
            boxstyle="square,pad=.16",
            linewidth=1.15,
        )
    )
    if xlabel is not None:
        ax.set_xlabel(
            xlabel,
            fontsize=PAPER_AXES_LABEL_SIZE,
            fontweight="bold",
            fontfamily=PLOT_FONT_FAMILY,
            color=COLOR_PALETTE["ink"],
        )
    if ylabel is not None:
        ax.set_ylabel(
            ylabel,
            fontsize=PAPER_AXES_LABEL_SIZE,
            fontweight="bold",
            fontfamily=PLOT_FONT_FAMILY,
            color=COLOR_PALETTE["ink"],
        )
    if grid_axis:
        ax.grid(
            axis=grid_axis,
            color=COLOR_PALETTE["grid"],
            linewidth=0.82,
            linestyle=(0, (1, 2)),
            alpha=0.68,
        )
    ax.set_axisbelow(True)


def save_plot(fig: Any, output_path: Path) -> None:
    """Saves a figure to disk and closes it to release resources."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=PLOT_DPI, facecolor=fig.get_facecolor())
    plt.close(fig)


def resolve_plot_path(name: str) -> Path:
    """Resolves the output path for a named report figure."""
    return PLOT_OUTPUT_DIR / PLOT_SPECS[name]["output_file"]


def resolve_plot_title(name: str) -> str:
    """Resolves the display title for a named report figure."""
    return str(PLOT_SPECS[name]["title"])


def flatten_axes(ax: Any) -> list[Any]:
    """Flattens Matplotlib axis containers into a simple list."""
    return list(np.ravel(ax if isinstance(ax, np.ndarray) else [ax]))


def reshape_axes_grid(ax: Any) -> np.ndarray:
    """Normalizes arbitrary axis containers into a 2D object array."""
    g = np.array(ax, dtype=object)
    return g.reshape(
        1, 1) if g.ndim == 0 else g.reshape(
        1, -1) if g.ndim == 1 else g


def figure_bbox(fig: Any, artist: Any) -> Any | None:
    """Measures a figure-level artist in figure coordinates."""
    if artist is None:
        return None
    fig.canvas.draw()
    return artist.get_window_extent(fig.canvas.get_renderer()).transformed(
        fig.transFigure.inverted()
    )


def axis_span_margins(
    fig: Any,
    axis: Any,
    renderer: Any,
) -> dict[str, float]:
    """Estimates the occupied margins around one axis' labels and title."""
    position = axis.get_position()

    def measure_boxes(artists: Sequence[Any]) -> list[Any]:
        boxes: list[Any] = []
        for artist in artists:
            if artist is None or not getattr(
                artist,
                "get_visible",
                lambda: True,
            )():
                continue
            try:
                bbox = artist.get_window_extent(renderer).transformed(
                    fig.transFigure.inverted()
                )
            except (AttributeError, RuntimeError, ValueError):
                continue
            if bbox.width > 0 and bbox.height > 0:
                boxes.append(bbox)
        return boxes

    x_boxes = measure_boxes([*axis.get_xticklabels(), axis.xaxis.label])
    y_boxes = measure_boxes([*axis.get_yticklabels(), axis.yaxis.label])
    title_boxes = measure_boxes([axis.title])
    return {
        "top": max([bbox.height for bbox in title_boxes], default=0.0),
        "bottom": max(
            [
                bbox.height
                for bbox in x_boxes
                if bbox.y1 <= position.y0 + 0.002
            ],
            default=0.0,
        ),
        "left": max(
            [bbox.width for bbox in y_boxes if bbox.x1 <= position.x0 + 0.002],
            default=0.0,
        ),
        "right": max(
            [bbox.width for bbox in y_boxes if bbox.x0 >= position.x1 - 0.002],
            default=0.0,
        ),
    }


def axis_pair_gap(
    fig: Any,
    renderer: Any,
    span_margins: Mapping[tuple[int, int], Mapping[str, float]],
    axis_a: Any,
    axis_b: Any,
    key_a: tuple[int, int],
    key_b: tuple[int, int],
    axis: str,
) -> tuple[float, list[float], float] | None:
    """Measures the visible gap between two neighboring axes."""
    if axis_a is None or axis_b is None:
        return None
    bbox_a = axis_a.get_tightbbox(renderer).transformed(
        fig.transFigure.inverted())
    bbox_b = axis_b.get_tightbbox(renderer).transformed(
        fig.transFigure.inverted())
    if axis == "w":
        return (
            bbox_b.x0 - bbox_a.x1,
            [axis_a.get_position().width, axis_b.get_position().width],
            max(span_margins[key_a]["right"], span_margins[key_b]["left"]),
        )
    return (
        bbox_a.y0 - bbox_b.y1,
        [axis_a.get_position().height, axis_b.get_position().height],
        max(span_margins[key_a]["bottom"], span_margins[key_b]["top"]),
    )


def optimize_gap(
    fig: Any,
    grid: np.ndarray,
    space: float,
    axis: str,
    target: float,
    min_gap: float = 0.012,
    gain: float = 0.85,
    comfort: float = 0.45,
) -> float:
    """Adjusts subplot spacing without wasting room or clipping labels."""
    renderer = fig.canvas.get_renderer()
    gap_values = []
    size_values = []
    span_margins = {
        (i, j): axis_span_margins(fig, grid[i, j], renderer)
        for i in range(grid.shape[0])
        for j in range(grid.shape[1])
        if grid[i, j] is not None
    }
    pref = []
    if axis == "w":
        if grid.shape[1] < 2:
            return space
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1] - 1):
                pair_gap = axis_pair_gap(
                    fig,
                    renderer,
                    span_margins,
                    grid[i, j],
                    grid[i, j + 1],
                    (i, j),
                    (i, j + 1),
                    "w",
                )
                if pair_gap is None:
                    continue
                gap, pair_sizes, preferred_gap = pair_gap
                gap_values.append(gap)
                size_values.extend(pair_sizes)
                pref.append(preferred_gap)
    else:
        if grid.shape[0] < 2:
            return space
        for i in range(grid.shape[0] - 1):
            for j in range(grid.shape[1]):
                pair_gap = axis_pair_gap(
                    fig,
                    renderer,
                    span_margins,
                    grid[i, j],
                    grid[i + 1, j],
                    (i, j),
                    (i + 1, j),
                    "h",
                )
                if pair_gap is None:
                    continue
                gap, pair_sizes, preferred_gap = pair_gap
                gap_values.append(gap)
                size_values.extend(pair_sizes)
                pref.append(preferred_gap)
    if not gap_values:
        return space
    target = max(target, min_gap + comfort *
                 float(np.median(pref))) if pref else target
    base = (
        min(gap_values)
        if min(gap_values) < min_gap
        else float(np.median(gap_values))
    )
    return float(
        np.clip(
            space
            + gain
            * ((min_gap if min(gap_values) < min_gap else target) - base)
            / max(float(np.mean(size_values)), 1e-6),
            0.0,
            0.7,
        )
    )


def apply_auto_layout(
    fig: Any,
    ax: Any,
    legend: Any | None = None,
    left: float = 0.07,
    right: float = 0.98,
    bottom: float = 0.06,
    wspace: float | None = None,
    hspace: float | None = None,
    title_gap: float = 0.018,
    legend_gap: float = 0.018,
    loops: int = 4,
) -> None:
    """Iteratively refines subplot spacing for the current figure contents."""
    axes_list = flatten_axes(ax)
    grid = reshape_axes_grid(ax)
    suptitle_bbox = figure_bbox(fig, getattr(fig, "_suptitle", None))
    top = max(
        bottom + 0.28,
        (suptitle_bbox.y0 - title_gap) if suptitle_bbox is not None else 0.965,
    )
    legend_height = (
        max(0.0, figure_bbox(fig, legend).height + legend_gap)
        if legend is not None
        else 0.0
    )
    bottom = max(bottom, legend_height)
    ws = (0.18 if grid.shape[1] > 1 else 0.0) if wspace is None else wspace
    hs = (0.10 if grid.shape[0] > 1 else 0.0) if hspace is None else hspace
    fig.subplots_adjust(
        left=left, right=right, bottom=bottom, top=top, wspace=ws, hspace=hs
    )
    for _ in range(loops):
        if legend is not None:
            legend.set_bbox_to_anchor(
                (0.5, fig.subplotpars.bottom / 2), transform=fig.transFigure
            )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axis_bboxes = [
            axis.get_tightbbox(renderer).transformed(
                fig.transFigure.inverted())
            for axis in axes_list
        ]
        x0 = min(bbox.x0 for bbox in axis_bboxes)
        y0 = min(bbox.y0 for bbox in axis_bboxes)
        x1 = max(bbox.x1 for bbox in axis_bboxes)
        y1 = max(bbox.y1 for bbox in axis_bboxes)
        left_margin = np.clip(
            fig.subplotpars.left + 0.85 * (left - x0), 0.035, 0.22
        )
        right_margin = np.clip(
            fig.subplotpars.right + 0.85 * (right - x1), 0.80, 0.992
        )
        bottom_margin = np.clip(
            fig.subplotpars.bottom + 0.85 * (bottom - y0), 0.05, 0.32
        )
        top_margin = np.clip(
            fig.subplotpars.top + 0.85 * (top - y1),
            max(bottom_margin + 0.28, 0.62),
            0.965,
        )
        if right_margin - left_margin < 0.36:
            left_margin = max(0.03, right_margin - 0.36)
        if top_margin - bottom_margin < 0.34:
            bottom_margin = max(0.03, top_margin - 0.34)
        nw = optimize_gap(fig, grid, ws, "w", 0.022)
        nh = optimize_gap(fig, grid, hs, "h", 0.014)
        delta = max(
            abs(left_margin - fig.subplotpars.left),
            abs(right_margin - fig.subplotpars.right),
            abs(bottom_margin - fig.subplotpars.bottom),
            abs(top_margin - fig.subplotpars.top),
            abs(nw - ws),
            abs(nh - hs),
        )
        ws, hs = nw, nh
        fig.subplots_adjust(
            left=left_margin,
            right=right_margin,
            bottom=bottom_margin,
            top=top_margin,
            wspace=ws,
            hspace=hs,
        )
        if delta < 5e-4:
            break
    if legend is not None:
        legend.set_bbox_to_anchor(
            (0.5, fig.subplotpars.bottom / 2), transform=fig.transFigure
        )


def apply_named_layout(
    name: str,
    fig: Any,
    ax: Any,
    legend: Any | None = None,
) -> None:
    """Applies the stored layout preset for a named report figure."""
    apply_auto_layout(fig, ax, legend, **PLOT_SPECS[name].get("layout", {}))


def sample_marker_indices(
    ax: Any,
    point_count: int,
    pixels_per_marker: int = 86,
    lower_count: int = 4,
    upper_count: int = 12,
) -> list[int]:
    """Chooses a readable subset of marker positions for dense line plots."""
    if point_count <= 1:
        return [0]
    pixel_width = max(
        ax.get_position().width *
        ax.figure.get_size_inches()[0] * ax.figure.dpi, 1.0
    )
    marker_count = int(
        np.clip(round(pixel_width / pixels_per_marker),
                lower_count, upper_count)
    )
    if point_count <= marker_count:
        return list(range(point_count))
    indices = np.linspace(0, point_count - 1, marker_count, dtype=int).tolist()
    return sorted(dict.fromkeys(indices + [point_count - 1]))


def plot_series_line(
    ax: Any,
    x_values: Sequence[float],
    y_values: Sequence[float],
    color: str,
    marker: str,
    label: str | None = None,
    line_style: str = "solid",
    draw_style: str = "default",
    zorder: int = 4,
) -> None:
    """Plots a styled line with sparsified markers for one data series."""
    identity_line_handle(
        ax.plot(
            x_values,
            y_values,
            color=color,
            linewidth=2.6,
            linestyle=line_style,
            label=label,
            drawstyle=draw_style,
            zorder=zorder,
        )[0]
    )
    marker_indices = sample_marker_indices(ax, len(x_values))
    identity_line_handle(
        ax.plot(
            np.array(x_values)[marker_indices],
            np.array(y_values)[marker_indices],
            linestyle="none",
            marker=marker,
            markersize=8 if len(x_values) < 25 else 6.8,
            color=color,
            markeredgecolor=COLOR_PALETTE["ink"],
            markeredgewidth=1.0,
            zorder=zorder + 1,
        )[0]
    )


def nudge_text_labels(
    ax: Any,
    text_labels: Sequence[Any],
    step: int = 5,
    loops: int = 30,
) -> None:
    """Pushes overlapping text labels upward until they no longer collide."""
    if len(text_labels) < 2:
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    y0, y1 = ax.get_ylim()
    cap = y1 - 0.03 * max(y1 - y0, 1e-6)
    for _ in range(loops):
        boxes = [
            label.get_window_extent(renderer).expanded(1.02, 1.05)
            for label in text_labels
        ]
        moved = False
        for i, box in enumerate(boxes):
            for j in range(i):
                if not box.overlaps(boxes[j]):
                    continue
                x_pos, y_pos = text_labels[i].get_position()
                x_px, y_px = ax.transData.transform((x_pos, y_pos))
                _, next_y = ax.transData.inverted().transform(
                    (x_px, y_px + step)
                )
                text_labels[i].set_position((x_pos, min(next_y, cap)))
                moved = True
                break
        if not moved:
            break
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()


def configure_stat_axis(
    ax: Any,
    title: str,
    ylabel: str,
    x_positions: np.ndarray,
    labels: Sequence[str],
) -> None:
    """Applies shared styling for bar, dot, and error summary panels."""
    style_axes(ax, title, None, ylabel, None, "y")
    ax.set_xticks(x_positions, labels)
    ax.set_xlim(
        -0.42,
        max(0.42, float(x_positions[-1]) + 0.42) if len(x_positions) else 0.42,
    )
    ax.tick_params(axis="x", labelsize=PAPER_DENSE_TICK_LABEL_SIZE, pad=3.1)


def flat_value_limits(
    values: Sequence[float],
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> tuple[float, float]:
    """Builds stable y-limits when all plotted values are nearly identical."""
    if not values:
        return 0.0, 1.0
    if max(values) - min(values) > 1e-12:
        return compute_y_limits(
            values,
            False,
            lower_bound,
            upper_bound,
            0.10,
            0.14,
        )
    value = float(values[0])
    pad = max(
        0.06 if upper_bound is not None and upper_bound <= 1.0 else 0.18,
        abs(value) * 0.12,
        1e-6,
    )
    min_value = value - pad
    max_value = (min(upper_bound, value + pad)
                 if upper_bound is not None else value + pad)
    if value == 0.0 and lower_bound is not None and lower_bound >= 0:
        max_value = min(upper_bound, 0.06) if upper_bound is not None else 0.06
    min_value = max(
        lower_bound, min_value) if lower_bound is not None else min_value
    return min_value, (max_value if max_value > min_value else min_value + 1.0)


def hide_axis_labels(
    ax: Any,
    hide_x: bool = False,
    hide_y: bool = False,
) -> None:
    """Hides selected tick labels for multi-panel figure layouts."""
    if hide_x:
        ax.tick_params(axis="x", labelbottom=False, bottom=False)
    if hide_y:
        ax.tick_params(axis="y", labelleft=False, left=False)


def annotate_stat_values(
    ax: Any,
    x_positions: Sequence[float],
    y_values: Sequence[float],
    text_labels: Sequence[str],
    y_min: float,
    y_max: float,
    mode: str = "above",
    color: str = COLOR_PALETTE["ink"],
    unique: bool = False,
    step: int = 4,
    loops: int = 24,
) -> None:
    """Annotates summary panels with readable numeric value labels."""
    y_span = max(y_max - y_min, 1e-6)
    seen = set()
    rendered_labels = []
    for x_pos, y_value, text_value in zip(x_positions, y_values, text_labels):
        if unique and text_value in seen:
            continue
        seen.add(text_value)
        if mode == "inside":
            if y_min < 0 < y_max and y_value < 0:
                in_bar = abs(y_value) >= 0.16 * abs(y_min)
                text_y = (
                    min(y_value + 0.05 * y_span, -0.04 * y_span)
                    if in_bar
                    else max(y_value - 0.024 * y_span, y_min + 0.05 * y_span)
                )
                vertical_alignment = "bottom" if in_bar else "top"
                text_color = COLOR_PALETTE["cloud"] if in_bar else color
            else:
                base_value = max(y_min, 0.0) if y_min < 0 < y_max else y_min
                in_bar = y_value - base_value >= 0.18 * y_span
                text_y = (
                    max(y_value - 0.05 * y_span, base_value + 0.04 * y_span)
                    if in_bar
                    else min(y_value + 0.022 * y_span, y_max - 0.05 * y_span)
                )
                vertical_alignment = "top" if in_bar else "bottom"
                text_color = COLOR_PALETTE["cloud"] if in_bar else color
        elif mode == "point":
            text_y = min(y_value + 0.045 * y_span, y_max - 0.06 * y_span)
            vertical_alignment = "bottom"
            text_color = color
        else:
            text_y = min(
                max(y_value + 0.03 * y_span, y_min + 0.05 * y_span),
                y_max - 0.06 * y_span,
            )
            vertical_alignment = "bottom"
            text_color = color
        text_artist = ax.text(
            x_pos,
            text_y,
            text_value,
            ha="center",
            va=vertical_alignment,
            fontsize=PAPER_INLINE_LABEL_SIZE,
            fontfamily=PLOT_FONT_FAMILY,
            color=text_color,
            clip_on=True,
            zorder=5,
        )
        rendered_labels.append(text_artist)
    nudge_text_labels(ax, rendered_labels, step, loops)


def spread_positions(
    y_positions: Sequence[float],
    lower_bound: float,
    upper_bound: float,
    gap: float,
) -> list[float]:
    """Spreads label positions apart while keeping them in bounds."""
    y_array = np.array(y_positions, float)
    if len(y_array) < 2:
        return y_array.tolist()
    sorted_indices = np.argsort(y_array)
    adjusted = y_array[sorted_indices].copy()
    adjusted[0] = max(adjusted[0], lower_bound)
    for index in range(1, len(adjusted)):
        adjusted[index] = max(adjusted[index], adjusted[index - 1] + gap)
    if adjusted[-1] > upper_bound:
        adjusted[-1] = upper_bound
        for index in range(len(adjusted) - 2, -1, -1):
            adjusted[index] = min(adjusted[index], adjusted[index + 1] - gap)
        if adjusted[0] < lower_bound:
            delta = lower_bound - adjusted[0]
            adjusted += delta
            if adjusted[-1] > upper_bound:
                adjusted -= adjusted[-1] - upper_bound
    output = np.empty_like(adjusted)
    output[sorted_indices] = adjusted
    return output.tolist()


def annotate_line_end_labels(
    ax: Any,
    labels: Sequence[str],
    x_positions: Sequence[float],
    y_positions: Sequence[float],
    colors: Sequence[str],
) -> None:
    """Places direct labels near the final point of each plotted line."""
    if not labels:
        return
    text_labels = []
    for label, x_pos, y_pos, color in zip(
            labels, x_positions, y_positions, colors):
        text_artist = ax.annotate(
            label,
            xy=(x_pos, y_pos),
            xytext=(8, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=PAPER_INLINE_LABEL_SIZE,
            fontfamily=PLOT_FONT_FAMILY,
            fontweight="bold",
            color=(
                COLOR_PALETTE["coin_dark"]
                if color in {COLOR_PALETTE["coin"], COLOR_PALETTE["risk_line"]}
                else color
            ),
            annotation_clip=False,
            zorder=7,
        )
        text_labels.append(text_artist)
    if len(text_labels) < 2:
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axis_bbox = ax.get_window_extent(renderer)
    label_boxes = [text.get_window_extent(renderer) for text in text_labels]
    half_height = max((bbox.height for bbox in label_boxes), default=0.0) / 2
    center_positions = [(bbox.y0 + bbox.y1) / 2 for bbox in label_boxes]
    adjusted_positions = spread_positions(
        center_positions,
        axis_bbox.y0 + half_height + 2,
        axis_bbox.y1 - half_height - 2,
        max((bbox.height for bbox in label_boxes), default=0.0) + 2,
    )
    points_per_pixel = 72 / fig.dpi
    for text_artist, center_y, adjusted_y in zip(
        text_labels, center_positions, adjusted_positions
    ):
        x_pos, y_pos = text_artist.get_position()
        text_artist.set_position(
            (x_pos, y_pos + (adjusted_y - center_y) * points_per_pixel)
        )


def annotate_object_tag(
    ax: Any,
    label: str,
    xy: tuple[float, float],
    color: str,
    y_offset: int = 9,
) -> None:
    """Draws a highlighted text tag for an important scene object."""
    ax.annotate(
        label,
        xy=xy,
        xytext=(0, y_offset),
        textcoords="offset points",
        ha="center",
        va="bottom" if y_offset >= 0 else "top",
        fontsize=PAPER_CALLOUT_LABEL_SIZE,
        fontfamily=PLOT_FONT_FAMILY,
        fontweight="bold",
        color=color,
        annotation_clip=True,
        zorder=7,
    )


def offset_series_positions(
    x_positions: Sequence[float],
    series: SeriesMap,
) -> dict[str, np.ndarray]:
    """Offsets series so overlapping markers remain distinguishable."""
    controller_names = sort_controller_names(series)
    base_positions = np.array(x_positions, float)
    offset_positions = {
        controller_name: base_positions.copy()
        for controller_name in controller_names
    }
    if len(controller_names) < 2 or len(base_positions) < 2:
        return offset_positions
    step = max(float(np.min(np.diff(base_positions))), 1e-6)
    offsets = (
        np.arange(len(controller_names)) - (len(controller_names) - 1) / 2
    ) * (0.07 * step)
    for controller_name, offset in zip(controller_names, offsets):
        offset_positions[controller_name] += offset
    return offset_positions


def compute_y_limits(
    values: Sequence[float],
    zero: bool = False,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    lower_pad: float = 0.06,
    upper_pad: float = 0.08,
) -> tuple[float, float]:
    """Computes padded y-axis bounds for a collection of plotted values."""
    min_value = float(min(values)) if values else 0.0
    max_value = float(max(values)) if values else 1.0
    if zero:
        min_value = min(min_value, 0.0)
        max_value = max(max_value, 0.0)
    if min_value < 0 < max_value:
        negative_span = max(-min_value, 1e-6)
        positive_span = max(max_value, 1e-6)
        min_value = -negative_span * (1 + 1.2 * lower_pad)
        max_value = positive_span * (1 + 1.25 * upper_pad)
    else:
        span = max(max_value - min_value, 1e-6)
        min_value -= lower_pad * span
        max_value += upper_pad * span
    if lower_bound is not None:
        min_value = max(min_value, lower_bound)
    if upper_bound is not None:
        max_value = min(max_value, upper_bound)
    return (
        (min_value, max_value)
        if max_value > min_value
        else (min_value, min_value + 1)
    )


def draw_zero_reference_line(
    ax: Any,
    y_min: float,
    y_max: float,
    line_width: float = 1.18,
    alpha: float = 0.48,
    values: Sequence[float] | None = None,
) -> None:
    """Draws a zero baseline when the displayed data crosses zero."""
    if values is not None:
        plotted_values = np.array(values, float)
        if len(plotted_values) == 0 or not (
            float(plotted_values.min()) < 0.0 < float(plotted_values.max())
        ):
            return
    if y_min < 0 < y_max:
        ax.axhline(
            0.0,
            color=COLOR_PALETTE["ink"],
            linewidth=line_width,
            alpha=alpha,
            zorder=1.35,
        )


def metric_bounds(metric_key: str) -> tuple[float | None, float | None]:
    """Returns preferred axis bounds for well-known summary metrics."""
    return {
        "success_rate": (0.0, 1.0),
        "avg_path_efficiency": (0.0, 1.0),
        "avg_hazard_failures": (0.0, 1.0),
        "avg_decision_time_ms": (0.0, None),
        "avg_steps": (0.0, None),
    }.get(metric_key, (None, None))


def render_dual_curve_plot(
    rows: Sequence[PlotRow],
    output_path: Path,
    title: str,
    specs: Sequence[tuple[str, str, str, str]],
    plot_name: str,
) -> None:
    """Renders the common two-panel training-curve plot layout."""
    if not rows:
        return
    x = np.array([float(r["epoch"]) for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(PLOT_WIDTH, SHORT_PLOT_HEIGHT))
    style_plot_title(fig, title)
    for a, (u, v, t, y) in zip(ax, specs):
        render_line_panel(
            a,
            x,
            {
                "Train": np.array([float(r[u]) for r in rows]),
                "Validation": np.array([float(r[v]) for r in rows]),
            },
            t,
            "Epoch",
            y,
            colors={
                "Train": COLOR_PALETTE["pipe"],
                "Validation": COLOR_PALETTE["brick"],
            },
            markers={"Train": "s", "Validation": "D"},
            labels={"Train": "Train", "Validation": "Validation"},
            direct=True,
            order=["Train", "Validation"],
            xticks=None,
            offset=False,
        )
    apply_named_layout(plot_name, fig, ax)
    save_plot(fig, output_path)


def render_value_training_plot(
    rows: Sequence[PlotRow],
    output_path: Path,
    plot_name: str = "primary_value_training",
    title: str | None = None,
) -> None:
    """Renders a two-panel value-model training history plot."""
    p = "_median" if rows and "train_loss_median" in rows[0] else ""
    render_dual_curve_plot(
        rows,
        output_path,
        title or resolve_plot_title(plot_name),
        [
            (f"train_loss{p}", f"val_loss{p}", "Regression Loss", "Loss"),
            (f"train_mae{p}", f"val_mae{p}", "Prediction MAE", "MAE"),
        ],
        plot_name,
    )


def render_risk_training_plot(
    rows: Sequence[PlotRow],
    output_path: Path,
) -> None:
    """Renders the risk-model training history plot."""
    render_dual_curve_plot(
        rows,
        output_path,
        resolve_plot_title("primary_risk_training"),
        [
            ("train_bce", "val_bce", "Prediction BCE", "BCE"),
            ("train_mae", "val_mae", "Prediction MAE", "MAE"),
        ],
        "primary_risk_training",
    )


def render_stat_panel(
    ax: Any,
    rows: Sequence[PlotRow],
    key: str,
    title: str,
    ylabel: str,
    labels: Sequence[str],
    mode: str,
    colors: Sequence[str] | None = None,
    dec: int = 2,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    base0: bool = False,
) -> None:
    """Renders one summary subplot as bars, dots, or interval markers."""
    x_positions = np.arange(len(rows), dtype=float)
    configure_stat_axis(ax, title, ylabel, x_positions, labels)
    if mode == "error":
        plotted_values = []
        for index, row in enumerate(rows):
            mean_value, low_value, high_value = resolve_metric_summary(
                row, key)
            plotted_values += [low_value, high_value]
            color = CONTROLLER_FILL_COLORS.get(
                str(row["controller_name"]),
                COLOR_PALETTE["brick"],
            )
            marker = CONTROLLER_MARKERS.get(str(row["controller_name"]), "s")
            cap_width = 0.085
            ax.vlines(
                x_positions[index],
                low_value,
                high_value,
                color=COLOR_PALETTE["err"],
                linewidth=1.45,
                zorder=2,
            )
            ax.hlines(
                [low_value, high_value],
                x_positions[index] - cap_width,
                x_positions[index] + cap_width,
                color=COLOR_PALETTE["err"],
                linewidth=1.45,
                zorder=2,
            )
            ax.scatter(
                [x_positions[index]],
                [mean_value],
                s=92,
                marker=marker,
                c=[color],
                edgecolors=COLOR_PALETTE["ink"],
                linewidths=1.3,
                zorder=4,
            )
        y_min, y_max = compute_y_limits(
            plotted_values, False, lower_bound, upper_bound, 0.04, 0.08
        )
        ax.set_ylim(y_min, y_max)
        draw_zero_reference_line(ax, y_min, y_max, values=plotted_values)
        return
    y_values = [float(row[key]) for row in rows]
    if mode == "dot":
        y_min, y_max = compute_y_limits(
            y_values, False, lower_bound, upper_bound, 0.24, 0.24
        )
        y_span = max(y_max - y_min, 1e-6)
        ax.set_ylim(y_min, y_max)
        draw_zero_reference_line(
            ax, y_min, y_max, line_width=0.98, alpha=0.32, values=y_values
        )
        ax.vlines(
            x_positions,
            y_min + 0.08 * y_span,
            y_values,
            color=colors,
            linewidth=1.65,
            alpha=0.9,
            zorder=3,
        )
        ax.scatter(
            x_positions,
            y_values,
            s=84,
            marker="o",
            c=colors,
            edgecolors=COLOR_PALETTE["ink"],
            linewidths=1.25,
            zorder=4,
        )
        annotate_stat_values(
            ax,
            x_positions,
            y_values,
            [f"{value:.{dec}f}" for value in y_values],
            y_min,
            y_max,
            mode="point",
            color=COLOR_PALETTE["ink"],
            unique=True,
            step=4,
            loops=24,
        )
        return
    bars = ax.bar(
        x_positions,
        y_values,
        width=0.62,
        color=colors,
        edgecolor=COLOR_PALETTE["ink"],
        linewidth=1.35,
        zorder=2,
    )
    if max(y_values) - min(y_values) <= 1e-12:
        y_min, y_max = flat_value_limits(y_values, lower_bound, upper_bound)
    else:
        y_min, y_max = compute_y_limits(
            y_values, False, lower_bound, upper_bound, 0.08, 0.12
        )
        if base0 and min(y_values) >= 0:
            y_min = 0.0
            y_max = (
                min(upper_bound, y_max)
                if upper_bound is not None
                else max(
                    y_max,
                    max(y_values)
                    + max(
                        0.12 * (max(y_values) - min(y_values)),
                        0.06 * max(y_values),
                        1e-6,
                    ),
                )
            )
    ax.set_ylim(y_min, y_max)
    draw_zero_reference_line(
        ax,
        y_min,
        y_max,
        line_width=1.22,
        alpha=(
            0.52
            if any(value < 0 for value in y_values)
            and any(value > 0 for value in y_values)
            else 0.34
        ),
        values=y_values,
    )
    annotate_stat_values(
        ax,
        [bar.get_x() + bar.get_width() / 2 for bar in bars],
        y_values,
        [f"{value:.{dec}f}" for value in y_values],
        y_min,
        y_max,
        mode="inside",
        color=COLOR_PALETTE["ink"],
        unique=False,
        step=5,
        loops=30,
    )


def render_error_panel(
    ax: Any,
    rows: Sequence[PlotRow],
    key: str,
    title: str,
    ylabel: str,
    labels: Sequence[str],
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> None:
    """Renders a summary panel with estimates and confidence intervals."""
    render_stat_panel(
        ax,
        rows,
        key,
        title,
        ylabel,
        labels,
        "error",
        None,
        2,
        lower_bound,
        upper_bound,
    )


def render_bar_panel(
    ax: Any,
    rows: Sequence[PlotRow],
    key: str,
    title: str,
    ylabel: str,
    labels: Sequence[str],
    colors: Sequence[str],
    dec: int = 2,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    base0: bool = False,
) -> None:
    """Renders a standard filled-bar summary panel."""
    render_stat_panel(
        ax,
        rows,
        key,
        title,
        ylabel,
        labels,
        "bar",
        colors,
        dec,
        lower_bound,
        upper_bound,
        base0,
    )


def render_dot_panel(
    ax: Any,
    rows: Sequence[PlotRow],
    key: str,
    title: str,
    ylabel: str,
    labels: Sequence[str],
    colors: Sequence[str],
    dec: int = 3,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    base0: bool = False,
) -> None:
    """Renders a lollipop-style summary panel."""
    render_stat_panel(
        ax,
        rows,
        key,
        title,
        ylabel,
        labels,
        "dot",
        colors,
        dec,
        lower_bound,
        upper_bound,
        base0,
    )


def render_line_panel(
    ax: Any,
    x_positions: Sequence[float],
    series: SeriesMap,
    title: str,
    xlabel: str,
    ylabel: str,
    xlabels: Sequence[str] | None = None,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    direct: bool = False,
    colors: Mapping[str, str] | None = None,
    markers: Mapping[str, str] | None = None,
    labels: Mapping[str, str] | None = None,
    order: Sequence[str] | None = None,
    xticks: Sequence[float] | object | None = AUTO_TICK_MARKER,
    offset: bool = True,
) -> None:
    """Renders one multi-series line chart using the shared visual style."""
    x_positions = np.array(x_positions, float)
    plotted_values = []
    controller_names = (
        [
            controller_name
            for controller_name in order
            if controller_name in series
        ]
        if order is not None
        else sort_controller_names(series)
    )
    raw_series = {
        controller_name: np.array(series[controller_name], float)
        for controller_name in controller_names
    }
    for controller_name in controller_names:
        plotted_values += raw_series[controller_name].tolist()
    y_min, y_max = compute_y_limits(
        plotted_values, False, lower_bound, upper_bound, 0.10, 0.12
    )
    displayed_positions = (
        offset_series_positions(x_positions, raw_series)
        if offset
        else {
            controller_name: x_positions.copy()
            for controller_name in controller_names
        }
    )
    direct = bool(direct and len(controller_names) > 1)
    span = max(float(x_positions[-1] - x_positions[0]), 1.0)
    step = x_positions[1] - x_positions[0] if len(x_positions) > 1 else span
    left_shift = max(
        [
            float(x_positions[0] - displayed_positions[controller_name][0])
            for controller_name in controller_names
        ],
        default=0.0,
    )
    right_shift = max(
        [
            float(displayed_positions[controller_name][-1] - x_positions[-1])
            for controller_name in controller_names
        ],
        default=0.0,
    )

    def resolve_color(controller_name): return (
        colors.get(controller_name)
        if colors and controller_name in colors
        else CONTROLLER_LINE_COLORS.get(
            controller_name,
            CONTROLLER_FILL_COLORS.get(
                controller_name, COLOR_PALETTE["brick"]),
        )
    )

    def resolve_marker(controller_name): return (
        markers.get(controller_name)
        if markers and controller_name in markers
        else CONTROLLER_MARKERS.get(controller_name, "s")
    )

    def resolve_label(controller_name): return (
        labels.get(controller_name)
        if labels and controller_name in labels
        else CONTROLLER_SHORT_NAMES.get(controller_name, controller_name)
    )
    for controller_name in controller_names:
        plot_series_line(
            ax,
            displayed_positions[controller_name],
            raw_series[controller_name],
            resolve_color(controller_name),
            resolve_marker(controller_name),
            resolve_label(controller_name),
        )
    left_padding = max(0.2 * step, 0.03 * span) + left_shift
    right_padding = (0.42 if direct else 0.3) * step + right_shift
    style_axes(ax, title, xlabel, ylabel, None, "both")
    ax.set_ylim(y_min, y_max)
    draw_zero_reference_line(
        ax, y_min, y_max, line_width=0.98, alpha=0.34, values=plotted_values
    )
    if direct:
        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axis_width = max(ax.get_window_extent(renderer).width, 1.0)
        font_props = font_manager.FontProperties(
            family=PLOT_FONT_FAMILY,
            weight="bold",
            size=PAPER_INLINE_LABEL_SIZE,
        )
        text_width = max(
            [
                renderer.get_text_width_height_descent(
                    resolve_label(controller_name), font_props, False
                )[0]
                for controller_name in controller_names
            ],
            default=0.0,
        )
        label_offset = 8 * fig.dpi / 72
        needed_padding = (
            (
                float(
                    max(
                        displayed_positions[controller_name][-1]
                        for controller_name in controller_names
                    )
                )
                - (
                    float(
                        min(
                            displayed_positions[controller_name][0]
                            for controller_name in controller_names
                        )
                    )
                    - left_padding
                )
            )
            * (label_offset + text_width + 6)
            / max(axis_width - (label_offset + text_width + 6), 1.0)
        )
        right_padding = max(right_padding, needed_padding)
    ax.set_xlim(
        float(
            min(
                displayed_positions[controller_name][0]
                for controller_name in controller_names
            )
        )
        - left_padding,
        float(
            max(
                displayed_positions[controller_name][-1]
                for controller_name in controller_names
            )
        )
        + right_padding,
    )
    if xticks is AUTO_TICK_MARKER:
        xticks = x_positions
    if xticks is not None or xlabels is not None:
        ax.set_xticks(
            x_positions if xticks is None else xticks,
            xlabels
            if xlabels is not None
            else (x_positions if xticks is None else xticks),
        )
    ax.tick_params(axis="x", labelsize=PAPER_DENSE_TICK_LABEL_SIZE, pad=2)
    if direct:
        annotate_line_end_labels(
            ax,
            [resolve_label(controller_name)
             for controller_name in controller_names],
            [
                displayed_positions[controller_name][-1]
                for controller_name in controller_names
            ],
            [raw_series[controller_name][-1]
                for controller_name in controller_names],
            [resolve_color(controller_name)
             for controller_name in controller_names],
        )


def render_benchmark_overview_plot(
    rows: Sequence[PlotRow],
    output_path: Path,
    plot_name: str = "primary_benchmark_overview",
    title: str | None = None,
) -> None:
    """Builds a benchmark comparison plot across controllers."""
    keep = {"Risk-Aware Value MPC", "Value MPC", "Dynamic A*"}
    rows = [row for row in rows if str(row["controller_name"]) in keep] or rows
    rows = sorted(
        rows,
        key=lambda row: (
            (
                CONTROLLER_DISPLAY_ORDER.index(str(row["controller_name"]))
                if str(row["controller_name"]) in CONTROLLER_DISPLAY_ORDER
                else 99
            ),
            str(row["controller_name"]),
        ),
    )
    labels = [
        CONTROLLER_COMPACT_NAMES.get(
            str(row["controller_name"]),
            str(row["controller_name"]),
        )
        for row in rows
    ]
    fill_colors = [
        CONTROLLER_FILL_COLORS.get(
            str(row["controller_name"]), COLOR_PALETTE["brick"])
        for row in rows
    ]
    aggregated_rows = bool(
        rows
        and any(
            key.endswith(("_mean", "_iqm", "_ci_low", "_ci_high"))
            for key in rows[0]
        )
    )
    fig, ax = plt.subplots(2, 3, figsize=(PLOT_WIDTH, TALL_PLOT_HEIGHT))
    style_plot_title(fig, title or resolve_plot_title(plot_name))
    panel_specs = [
        ("success_rate", "Success Rate", "Rate"),
        ("avg_steps", "Average Steps", "Steps"),
        ("avg_total_reward", "Average Return", "Return"),
        ("avg_path_efficiency", "Path Efficiency", "Efficiency"),
        ("avg_decision_time_ms", "Decision Time", "Time (ms)"),
        ("avg_hazard_failures", "Hazard Failure Rate", "Failure Rate"),
    ]
    for index, (axis, (metric_key, panel_title, ylabel)) in enumerate(
        zip(ax.flat, panel_specs)
    ):
        lower_bound, upper_bound = metric_bounds(metric_key)
        (
            render_error_panel(
                axis,
                rows,
                metric_key,
                panel_title,
                ylabel,
                labels,
                lower_bound,
                upper_bound,
            )
            if aggregated_rows
            else render_bar_panel(
                axis,
                rows,
                metric_key,
                panel_title,
                ylabel,
                labels,
                fill_colors,
                2 if metric_key != "avg_path_efficiency" else 3,
                lower_bound,
                upper_bound,
            )
        )
        hide_axis_labels(axis, hide_x=index < 3)
    apply_named_layout(plot_name, fig, ax)
    save_plot(fig, output_path)


def render_sensitivity_plot(
    rows: Sequence[PlotRow],
    output_path: Path,
) -> None:
    """Builds the figure summarizing horizon and enemy-count sensitivity."""
    fig, ax = plt.subplots(2, 2, figsize=(PLOT_WIDTH, TALL_PLOT_HEIGHT))
    style_plot_title(fig, resolve_plot_title("sensitivity_study"))
    horizon_rows = [row for row in rows if row["study"] == "planning_horizon"]
    horizon_values = [float(int(row["setting"])) for row in horizon_rows]
    horizon_returns = [float(row["avg_total_reward"]) for row in horizon_rows]
    horizon_times = [
        float(row["avg_decision_time_ms"]) for row in horizon_rows
    ]
    render_line_panel(
        ax[0, 0],
        horizon_values,
        {"Risk-Aware Value MPC": horizon_returns},
        "Horizon Return",
        "Planning Horizon",
        "Average Return",
        [f"H={int(value)}" for value in horizon_values],
        direct=True,
    )
    render_line_panel(
        ax[0, 1],
        horizon_values,
        {"Risk-Aware Value MPC": horizon_times},
        "Horizon Time",
        "Planning Horizon",
        "Decision Time (ms)",
        [f"H={int(value)}" for value in horizon_values],
        direct=True,
    )
    enemy_count_rows = [
        row
        for row in rows
        if row["study"] == "enemy_count"
        and str(row["controller_name"]) in {
            "Dynamic A*",
            "Risk-Aware Value MPC",
        }
    ]
    enemy_counts = sorted({int(row["setting"]) for row in enemy_count_rows})
    x_positions = [float(value) for value in enemy_counts]
    controller_names = sort_controller_names(
        str(row["controller_name"]) for row in enemy_count_rows
    )

    def build_enemy_count_series(metric_key: str) -> dict[str, list[float]]:
        return {
            controller_name: [
                float(
                    next(
                        row[metric_key]
                        for row in enemy_count_rows
                        if str(row["controller_name"]) == controller_name
                        and int(row["setting"]) == enemy_count
                    )
                )
                for enemy_count in enemy_counts
            ]
            for controller_name in controller_names
        }

    render_line_panel(
        ax[1, 0],
        x_positions,
        build_enemy_count_series("avg_total_reward"),
        "Enemy Return",
        "Number of Enemies",
        "Average Return",
        [str(value) for value in enemy_counts],
        direct=True,
    )
    render_line_panel(
        ax[1, 1],
        x_positions,
        build_enemy_count_series("avg_decision_time_ms"),
        "Enemy Time",
        "Number of Enemies",
        "Decision Time (ms)",
        [str(value) for value in enemy_counts],
        direct=True,
    )
    apply_named_layout("sensitivity_study", fig, ax)
    save_plot(fig, output_path)


def render_ablation_plot(rows: Sequence[PlotRow], output_path: Path) -> None:
    """Builds the figure comparing risk-aware controller ablations."""
    label_map = {
        "Full Risk-Aware MPC": "Full\nRisk-Aware MPC",
        "No Learned Terminal": "No Learned\nTerminal",
        "No Heuristic Terminal": "No Heuristic\nTerminal",
        "No Risk Predictor": "No Risk\nPredictor",
        "No Predictive Hazard": "No Predictive\nHazard",
    }
    labels = [
        label_map.get(
            str(row["controller_name"]),
            str(row["controller_name"]).replace("-", "\n"),
        )
        for row in rows
    ]
    colors = [
        ABLATION_COLORS.get(
            str(row["controller_name"]), COLOR_PALETTE["brick"])
        for row in rows
    ]
    fig, ax = plt.subplots(2, 2, figsize=(PLOT_WIDTH, TALL_PLOT_HEIGHT))
    style_plot_title(fig, resolve_plot_title("ablation_study"))
    panel_specs = [
        (
            "avg_total_reward",
            "Average Return",
            2,
            None,
            None,
            render_bar_panel,
            True,
        ),
        (
            "avg_path_efficiency",
            "Path Efficiency",
            3,
            None,
            None,
            render_dot_panel,
            False,
        ),
        (
            "avg_decision_time_ms",
            "Decision Time (ms)",
            2,
            0.0,
            None,
            render_bar_panel,
            True,
        ),
        ("avg_steps", "Average Steps", 2, None, None, render_dot_panel, False),
    ]
    for index, (
        axis,
        (
            metric_key,
            panel_title,
            decimals,
            lower_bound,
            upper_bound,
            render_panel,
            base0,
        ),
    ) in enumerate(
        zip(
            ax.flat,
            panel_specs,
        )
    ):
        render_panel(
            axis,
            rows,
            metric_key,
            panel_title,
            panel_title,
            labels,
            colors,
            decimals,
            lower_bound,
            upper_bound,
            base0,
        )
        hide_axis_labels(axis, hide_x=index < 2)
    apply_named_layout("ablation_study", fig, ax)
    save_plot(fig, output_path)


def render_seed_sweep_plot(rows: Sequence[PlotRow], output_path: Path) -> None:
    """Builds the figure summarizing cross-seed stability statistics."""
    rows = [
        r
        for r in sorted(
            rows,
            key=lambda r: (
                (
                    CONTROLLER_DISPLAY_ORDER.index(str(r["controller_name"]))
                    if str(r["controller_name"]) in CONTROLLER_DISPLAY_ORDER
                    else 99
                ),
                str(r["controller_name"]),
            ),
        )
        if str(r["controller_name"])
        in {"Risk-Aware Value MPC", "Value MPC", "Dynamic A*"}
    ] or rows
    labs = [
        CONTROLLER_COMPACT_NAMES.get(
            str(r["controller_name"]),
            CONTROLLER_SHORT_NAMES.get(
                str(r["controller_name"]),
                str(r["controller_name"]),
            ),
        )
        for r in rows
    ]
    fig, ax = plt.subplots(1, 2, figsize=(PLOT_WIDTH, SHORT_PLOT_HEIGHT))
    style_plot_title(fig, resolve_plot_title("seed_sweep_summary"))
    render_error_panel(
        ax[0],
        rows,
        "avg_total_reward",
        "IQM Return",
        "IQM Return",
        labs,
        *metric_bounds("avg_total_reward"),
    )
    render_error_panel(
        ax[1],
        rows,
        "success_rate",
        "IQM Success Rate",
        "IQM Success Rate",
        labs,
        *metric_bounds("success_rate"),
    )
    apply_named_layout("seed_sweep_summary", fig, ax)
    save_plot(fig, output_path)


def render_noise_robustness_plot(
    rows: Sequence[PlotRow],
    output_path: Path,
) -> None:
    """Builds the robustness figure for injected action-noise experiments."""
    controller_names = sort_controller_names(
        str(row["controller_name"])
        for row in rows
        if str(row["controller_name"])
        in {"Dynamic A*", "Value MPC", "Risk-Aware Value MPC"}
    )
    x_positions = [float(value) for value in NOISE_ROBUSTNESS_LEVELS]

    def build_noise_series(metric_key: str) -> dict[str, list[float]]:
        return {
            controller_name: [
                float(
                    next(
                        row[metric_key]
                        for row in rows
                        if str(row["controller_name"]) == controller_name
                        and abs(
                            float(row["action_noise"]) - noise_level
                        ) < 1e-9
                    )
                )
                for noise_level in x_positions
            ]
            for controller_name in controller_names
        }

    fig, ax = plt.subplots(1, 2, figsize=(PLOT_WIDTH, SHORT_PLOT_HEIGHT))
    style_plot_title(fig, resolve_plot_title("noise_robustness"))
    render_line_panel(
        ax[0],
        x_positions,
        build_noise_series("success_rate"),
        "Noise Success Rate",
        "Action Noise Probability",
        "Success Rate",
        lower_bound=0.0,
        upper_bound=1.0,
        direct=True,
    )
    render_line_panel(
        ax[1],
        x_positions,
        build_noise_series("avg_hazard_failures"),
        "Noise Hazard Rate",
        "Action Noise Probability",
        "Hazard Failure Rate",
        lower_bound=0.0,
        direct=True,
    )
    apply_named_layout("noise_robustness", fig, ax)
    save_plot(fig, output_path)


def compress_path_trace(
    trace: Sequence[GridPosition],
) -> tuple[list[GridPosition], list[tuple[int, int, int]]]:
    """Collapses repeated path positions and records dwell counts."""
    u = []
    r = []
    p = None
    n = 0
    for z in trace:
        z = (int(z[0]), int(z[1]))
        if z == p:
            n += 1
            continue
        if p is not None:
            u.append(p)
            n > 1 and r.append((p[0], p[1], n))
        p = z
        n = 1
    if p is not None:
        u.append(p)
        n > 1 and r.append((p[0], p[1], n))
    return u, r


def path_trace_penalty(trace: Sequence[GridPosition], goal_col: int) -> int:
    """Scores a trajectory for showcase selection with smoothness penalties."""
    t, r = compress_path_trace(trace)
    h = [t[i + 1][1] - t[i][1]
         for i in range(len(t) - 1)] if len(t) > 1 else []
    v = [t[i + 1][0] - t[i][0]
         for i in range(len(t) - 1)] if len(t) > 1 else []
    g = 1 if goal_col > t[0][1] else -1 if goal_col < t[0][1] else 0
    return (
        sum(max(0, n - 1) for _, _, n in r) * 10
        + (len(t) - len(set(t))) * 12
        + sum(abs(s) for s in h if g and s and s * g < 0) * 2
        + sum(1 for a, b in zip(h, h[1:]) if a and b and (a > 0) != (b > 0))
        + sum(1 for a, b in zip(v, v[1:]) if a and b and (a > 0) != (b > 0))
    )


def showcase_sort_key(
    level: LevelScenario,
    traces: TraceMap,
) -> tuple[int, int, int, int]:
    """Builds the ranking key used to pick the showcase level."""
    ps = [path_trace_penalty(t, level.goal[1]) for t in traces.values()]
    return (
        path_trace_penalty(traces["Risk-Aware Value MPC"], level.goal[1]),
        sum(ps),
        max(ps),
        sum(len(compress_path_trace(t)[0]) for t in traces.values()),
    )


def normalize_showcase_orientation(
    level: LevelScenario,
    traces: TraceMap,
) -> tuple[LevelScenario, TraceMap]:
    """Mirrors showcase data so paths consistently progress left to right."""
    if level.start[1] <= level.goal[1]:
        return level, traces
    w = level.width
    s = LevelScenario(
        level.name,
        level.width,
        level.height,
        {(r, w - 1 - c) for r, c in level.solid_tiles},
        (level.start[0], w - 1 - level.start[1]),
        (level.goal[0], w - 1 - level.goal[1]),
        [
            PatrolEnemy([(r, w - 1 - c)
                        for r, c in e.path], e.index, e.direction)
            for e in level.enemies
        ],
    )
    return s, {k: [(r, w - 1 - c) for r, c in v] for k, v in traces.items()}


def choose_patrol_label_anchor(
    level: LevelScenario,
    traces: TraceMap,
) -> tuple[tuple[float, float], int] | None:
    """Chooses a vertically offset patrol-label anchor over one patrol path."""
    if not level.enemies:
        return None
    trace_points = {
        (int(r), int(c))
        for trace in traces.values()
        for r, c in compress_path_trace(trace)[0]
    }
    anchor_points = {
        (int(level.start[0]), int(level.start[1])),
        (int(level.goal[0]), int(level.goal[1])),
    }

    def min_distance(
        target_row: float,
        target_col: float,
        points: Sequence[tuple[int, int]],
    ) -> float:
        return (
            min(abs(target_row - point_row) + abs(target_col - point_col)
                for point_row, point_col in points)
            if points
            else float(max(level.width, level.height))
        )

    best_choice = None
    for enemy in level.enemies:
        unique_path = list(dict.fromkeys((int(r), int(c)) for r, c in enemy.path))
        if not unique_path:
            continue
        row = float(sum(path_row for path_row, _ in unique_path)) / len(unique_path)
        col = float(sum(path_col for _, path_col in unique_path)) / len(unique_path)
        trace_clearance = min_distance(row, col, tuple(trace_points))
        anchor_clearance = min_distance(row, col, tuple(anchor_points))
        edge_margin = min(
            row,
            level.height - 1 - row,
            col,
            level.width - 1 - col,
        )
        span_bonus = max(
            abs(unique_path[-1][1] - unique_path[0][1]),
            abs(unique_path[-1][0] - unique_path[0][0]),
        )
        score = (
            2.4 * trace_clearance
            + 1.4 * anchor_clearance
            + 0.8 * edge_margin
            + 0.4 * span_bonus
        )
        choice = (
            score,
            trace_clearance,
            anchor_clearance,
            edge_margin,
            row,
            col,
        )
        best_choice = (
            choice
            if best_choice is None or choice > best_choice
            else best_choice
        )
    if best_choice is None:
        return None

    _, _, _, _, row, col = best_choice
    avoid_points = tuple(trace_points | anchor_points)

    def offset_score(label_row: float) -> float:
        if not 0.0 <= label_row <= level.height - 1:
            return -1e9
        edge_room = min(label_row, level.height - 1 - label_row)
        return 1.5 * edge_room + min_distance(label_row, col, avoid_points)

    above_score = offset_score(row - 1.0)
    below_score = offset_score(row + 1.0)
    y_offset = 9 if above_score >= below_score else -11
    return (float(col), float(row)), y_offset


def render_trajectory_showcase_plot(
    level: LevelScenario,
    traces: TraceMap,
    output_path: Path,
) -> None:
    """Builds the qualitative multi-panel trajectory showcase figure."""
    level, traces = normalize_showcase_orientation(level, traces)
    ns = sort_controller_names(str(k) for k in traces)
    patrol_label = choose_patrol_label_anchor(level, traces)
    terrain_zorder = 0.2
    patrol_path_zorder = 1.6
    trace_zorder = 3.2
    marker_zorder = 5.0
    dwell_label_zorder = 6.0
    fig, ax = plt.subplots(2, 2, figsize=(PLOT_WIDTH, TALL_PLOT_HEIGHT))
    style_plot_title(fig, resolve_plot_title("trajectory_showcase"))
    ax = ax.reshape(2, 2)
    for i, (a, n) in enumerate(zip(ax.flat, ns)):
        a.set_facecolor(COLOR_PALETTE["sky"])
        for r, c in level.solid_tiles:
            a.add_patch(
                Rectangle(
                    (c - 0.5, r - 0.5),
                    1,
                    1,
                    facecolor=COLOR_PALETTE["ground"],
                    edgecolor="none",
                    linewidth=0,
                    zorder=terrain_zorder,
                )
            )
        for e in level.enemies:
            identity_line_handle(
                a.plot(
                    [c for _, c in e.path],
                    [r for r, _ in e.path],
                    color=COLOR_PALETTE["fire"],
                    linewidth=1.9,
                    linestyle=(0, (2, 2)),
                    drawstyle="steps-post",
                    alpha=0.72,
                    zorder=patrol_path_zorder,
                )[0]
            )
        t, rep = compress_path_trace(traces[n])
        identity_line_handle(
            a.plot(
                [c for _, c in t],
                [r for r, _ in t],
                color=CONTROLLER_LINE_COLORS.get(
                    n, CONTROLLER_FILL_COLORS.get(n, COLOR_PALETTE["brick"])
                ),
                linewidth=2.8,
                drawstyle="steps-post",
                zorder=trace_zorder,
            )[0]
        )
        if len(t) > 2:
            identity_line_handle(
                a.plot(
                    [c for _, c in t[1:-1]],
                    [r for r, _ in t[1:-1]],
                    linestyle="none",
                    marker=CONTROLLER_MARKERS.get(n, "s"),
                    markersize=4.8,
                    color=CONTROLLER_LINE_COLORS.get(
                        n, CONTROLLER_FILL_COLORS.get(n, COLOR_PALETTE["brick"])
                    ),
                    markeredgecolor=COLOR_PALETTE["ink"],
                    markeredgewidth=0.9,
                    zorder=trace_zorder + 0.1,
                )[0]
            )
        for r, c, k in rep:
            if k < 3:
                continue
            a.text(
                c + 0.22,
                r - 0.22,
                f"x{k}",
                ha="left",
                va="center",
                fontsize=PAPER_CALLOUT_LABEL_SIZE,
                fontfamily=PLOT_FONT_FAMILY,
                fontweight="bold",
                color=CONTROLLER_LINE_COLORS.get(
                    n, CONTROLLER_FILL_COLORS.get(n, COLOR_PALETTE["brick"])
                ),
                bbox=dict(
                    facecolor=COLOR_PALETTE["cloud"],
                    edgecolor=COLOR_PALETTE["ink"],
                    linewidth=0.8,
                    boxstyle="square,pad=.12",
                ),
                clip_on=True,
                zorder=dwell_label_zorder,
            )
        a.scatter(
            [level.start[1]],
            [level.start[0]],
            c=COLOR_PALETTE["coin"],
            s=120,
            marker="s",
            edgecolors=COLOR_PALETTE["ink"],
            linewidths=1.4,
            zorder=marker_zorder,
        )
        a.scatter(
            [level.goal[1]],
            [level.goal[0]],
            c=COLOR_PALETTE["cloud"],
            s=130,
            marker="*",
            edgecolors=COLOR_PALETTE["ink"],
            linewidths=1.4,
            zorder=marker_zorder,
        )
        if i == 0:
            annotate_object_tag(
                a, "Start", (level.start[1], level.start[0]
                             ), COLOR_PALETTE["coin_dark"]
            )
            annotate_object_tag(
                a, "Goal", (level.goal[1], level.goal[0]), COLOR_PALETTE["ink"]
            )
            if patrol_label is not None:
                patrol_xy, patrol_offset = patrol_label
                annotate_object_tag(
                    a,
                    "Patrol",
                    patrol_xy,
                    COLOR_PALETTE["fire"],
                    y_offset=patrol_offset,
                )
        style_axes(
            a,
            CONTROLLER_SHORT_NAMES.get(n, n),
            None,
            None,
            COLOR_PALETTE["sky"],
            "",
        )
        a.set_xticks(range(0, level.width, 4))
        a.set_yticks(range(0, level.height, 2))
        a.set_xlim(-0.5, level.width - 0.5)
        a.set_ylim(level.height - 0.5, -0.5)
        a.set_xticks(np.arange(-0.5, level.width, 1), minor=True)
        a.set_yticks(np.arange(-0.5, level.height, 1), minor=True)
        a.grid(
            which="minor",
            color=COLOR_PALETTE["cloud_line"],
            linewidth=0.62,
            alpha=0.24)
        a.tick_params(axis="both", labelsize=PAPER_CALLOUT_LABEL_SIZE, pad=1.8)
        a.tick_params(which="minor", bottom=False, left=False)
        hide_axis_labels(a, hide_x=i < 2, hide_y=bool(i % 2))
    apply_named_layout("trajectory_showcase", fig, ax)
    save_plot(fig, output_path)
