"""Shared configuration values for experiments, outputs, and plots."""

import os
from collections.abc import Iterable
from pathlib import Path


# =============================================================================
# Project paths and experiment constants.
# =============================================================================
def _looks_like_project_root(path: Path) -> bool:
    """Returns whether a directory looks like the source checkout root."""
    return (
        (path / "pyproject.toml").exists()
        and (path / "src" / "platformer_lab").is_dir()
    )


def _search_project_root(start: Path) -> Path | None:
    """Walks up from a starting path until a project root is found."""
    resolved_start = start.resolve()
    for candidate in (resolved_start, *resolved_start.parents):
        if _looks_like_project_root(candidate):
            return candidate
    return None


def _resolve_project_root() -> Path:
    """Finds the runtime project root or falls back to the current directory."""
    project_root_override = os.environ.get("PLATFORMER_LAB_PROJECT_ROOT")
    if project_root_override:
        return Path(project_root_override).expanduser().resolve()
    cwd_project_root = _search_project_root(Path.cwd())
    if cwd_project_root is not None:
        return cwd_project_root
    source_project_root = _search_project_root(Path(__file__).resolve().parents[2])
    if source_project_root is not None:
        return source_project_root
    return Path.cwd().resolve()


def _resolve_output_dir(project_root: Path) -> Path:
    """Resolves the output directory, allowing an explicit override."""
    output_dir_override = os.environ.get("PLATFORMER_LAB_OUTPUT_DIR")
    if output_dir_override:
        return Path(output_dir_override).expanduser().resolve()
    return (project_root / "outputs").resolve()


PROJECT_ROOT = _resolve_project_root()
OUTPUT_DIR = _resolve_output_dir(PROJECT_ROOT)
METRICS_OUTPUT_DIR = OUTPUT_DIR / "metrics"
CHECKPOINT_OUTPUT_DIR = OUTPUT_DIR / "checkpoints"
CONFIG_OUTPUT_DIR = OUTPUT_DIR / "configs"
CACHE_OUTPUT_DIR = OUTPUT_DIR / "cache"
PLOT_OUTPUT_DIR = OUTPUT_DIR / "plots"
ARTIFACT_OUTPUT_DIRS = (
    OUTPUT_DIR,
    METRICS_OUTPUT_DIR,
    CHECKPOINT_OUTPUT_DIR,
    CONFIG_OUTPUT_DIR,
    CACHE_OUTPUT_DIR,
    PLOT_OUTPUT_DIR,
)
GRID_WIDTH = 24
GRID_HEIGHT = 14

# Primary benchmark experiment.
PRIMARY_TRAIN_EPISODE_COUNT = 480
PRIMARY_EVALUATION_EPISODE_COUNT = 24
PRIMARY_MAX_STEP_COUNT = 80
PRIMARY_TRAINING_SEED = 20260324
PRIMARY_BENCHMARK_EVALUATION_SEED = 20260426

# Follow-up study sampling.
ANALYSIS_MAX_STEP_COUNT = 60
SENSITIVITY_STUDY_LEVEL_COUNT = 3
ABLATION_STUDY_LEVEL_COUNT = 3

# Seed sweep study.
SEED_SWEEP_SEED_PAIRS = [
    (20260324, 20260426),
    (20260331, 20260503),
    (20260407, 20260510),
    (20260414, 20260517),
    (20260421, 20260524),
]
SEED_SWEEP_TRAIN_EPISODE_COUNT = 480
SEED_SWEEP_EVALUATION_EPISODE_COUNT = 12

# Noise robustness study.
NOISE_ROBUSTNESS_LEVELS = [0.0, 0.08, 0.16]
NOISE_ROBUSTNESS_EVALUATION_SEEDS = (
    20270000,
    20270001,
    20270002,
    20270003,
)

# Holdout generalization study.
HOLDOUT_GENERALIZATION_TEMPLATE_GROUPS = ("tower", "overhang")
HOLDOUT_GENERALIZATION_EVALUATION_EPISODE_COUNT = 16
HOLDOUT_GENERALIZATION_EVALUATION_SEED = 20260601

RISK_MODEL_PENALTY = 0.3
RISK_CLONE_STATE_PENALTY = 0.2
RISK_CLONE_ACTION_PENALTY_WEIGHT = 0.12
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_SEED = 20260402
PLOT_WIDTH = 12.8
SHORT_PLOT_HEIGHT = 5.2
TALL_PLOT_HEIGHT = 8.6
PLOT_DPI = 180
PLOT_FONT_NAME = "Departure Mono"
PLOT_FONT_FILENAME = "DepartureMono-Regular.otf"
PAPER_FIGURE_TITLE_SIZE = 12.0
PAPER_AXES_TITLE_SIZE = 10.0
PAPER_AXES_LABEL_SIZE = 10.0
PAPER_TICK_LABEL_SIZE = 8.0
PAPER_DENSE_TICK_LABEL_SIZE = 8.0
PAPER_INLINE_LABEL_SIZE = 8.0
PAPER_CALLOUT_LABEL_SIZE = 8.0
COLOR_PALETTE = {
    "sky": "#5C94FC",
    "cloud": "#FFF7D6",
    "panel": "#FDF4D5",
    "ink": "#1A1A1A",
    "grid": "#D8C08C",
    "brick": "#C84C0C",
    "pipe": "#2E9E48",
    "coin": "#F2C230",
    "coin_dark": "#AE7F00",
    "risk_line": "#D89C00",
    "violet": "#6A5ACD",
    "fire": "#FF7A21",
    "ground": "#7C4A21",
    "err": "#4E473A",
    "cloud_line": "#F4E7AE",
}
CONTROLLER_DISPLAY_ORDER = [
    "Risk-Aware Value MPC",
    "Value MPC",
    "Dynamic A*",
    "Static A*",
]
CONTROLLER_FILL_COLORS = {
    "Dynamic A*": COLOR_PALETTE["pipe"],
    "Value MPC": COLOR_PALETTE["brick"],
    "Risk-Aware Value MPC": COLOR_PALETTE["coin"],
    "Static A*": COLOR_PALETTE["ground"],
}
CONTROLLER_LINE_COLORS = {
    "Dynamic A*": COLOR_PALETTE["pipe"],
    "Value MPC": COLOR_PALETTE["brick"],
    "Risk-Aware Value MPC": COLOR_PALETTE["risk_line"],
    "Static A*": COLOR_PALETTE["ground"],
}
CONTROLLER_MARKERS = {
    "Risk-Aware Value MPC": "s",
    "Value MPC": "D",
    "Dynamic A*": "^",
    "Static A*": "o",
}
CONTROLLER_SHORT_NAMES = {
    "Dynamic A*": "Dynamic A*",
    "Value MPC": "Value MPC",
    "Risk-Aware Value MPC": "Risk-Aware MPC",
    "Static A*": "Static A*",
}
CONTROLLER_COMPACT_NAMES = {
    "Dynamic A*": "Dynamic\nA*",
    "Value MPC": "Value\nMPC",
    "Risk-Aware Value MPC": "Risk-Aware\nMPC",
    "Static A*": "Static\nA*",
}
ABLATION_COLORS = {
    "Full Risk-Aware MPC": COLOR_PALETTE["coin"],
    "No Learned Terminal": COLOR_PALETTE["brick"],
    "No Heuristic Terminal": COLOR_PALETTE["pipe"],
    "No Risk Predictor": COLOR_PALETTE["violet"],
    "No Predictive Hazard": COLOR_PALETTE["ground"],
}
PRIMARY_VALUE_MPC_CHECKPOINT_PATH = (
    CHECKPOINT_OUTPUT_DIR / "primary_value_mpc_checkpoint.npz"
)
PRIMARY_VALUE_TRAINING_HISTORY_PATH = (
    METRICS_OUTPUT_DIR / "primary_value_training_history.csv"
)
PRIMARY_RISK_TRAINING_HISTORY_PATH = (
    METRICS_OUTPUT_DIR / "primary_risk_training_history.csv"
)
PRIMARY_BENCHMARK_SUMMARY_METRICS_PATH = (
    METRICS_OUTPUT_DIR / "primary_benchmark_summary_metrics.csv"
)
SENSITIVITY_STUDY_METRICS_PATH = (
    METRICS_OUTPUT_DIR / "sensitivity_study_metrics.csv"
)
ABLATION_STUDY_METRICS_PATH = (
    METRICS_OUTPUT_DIR / "ablation_study_metrics.csv"
)
SEED_SWEEP_RAW_METRICS_PATH = (
    METRICS_OUTPUT_DIR / "seed_sweep_raw_metrics.csv"
)
SEED_SWEEP_SUMMARY_METRICS_PATH = (
    METRICS_OUTPUT_DIR / "seed_sweep_summary_metrics.csv"
)
SEED_SWEEP_TRAINING_HISTORY_PATH = (
    METRICS_OUTPUT_DIR / "seed_sweep_training_history.csv"
)
SEED_SWEEP_PAIRWISE_ADVANTAGE_METRICS_PATH = (
    METRICS_OUTPUT_DIR / "seed_sweep_pairwise_advantage_metrics.csv"
)
NOISE_ROBUSTNESS_METRICS_PATH = (
    METRICS_OUTPUT_DIR / "noise_robustness_metrics.csv"
)
HOLDOUT_GENERALIZATION_VALUE_MPC_CHECKPOINT_PATH = (
    CHECKPOINT_OUTPUT_DIR / "holdout_generalization_value_mpc_checkpoint.npz"
)
HOLDOUT_GENERALIZATION_CONFIG_PATH = (
    CONFIG_OUTPUT_DIR / "holdout_generalization.json"
)
HOLDOUT_GENERALIZATION_METRICS_PATH = (
    METRICS_OUTPUT_DIR / "holdout_generalization_metrics.csv"
)
TRAJECTORY_SHOWCASE_CACHE_PATH = (
    CACHE_OUTPUT_DIR / "trajectory_showcase_v2.json"
)


# =============================================================================
# Plot metadata.
# =============================================================================
PLOT_SPECS = {
    "primary_value_training": {
        "title": "Value Training",
        "axis_count": 2,
        "width_in": PLOT_WIDTH,
        "height_in": SHORT_PLOT_HEIGHT,
        "legend_required": True,
        "output_file": "primary_value_training.svg",
        "layout": {"left": 0.07, "right": 0.98, "bottom": 0.066},
    },
    "primary_risk_training": {
        "title": "Risk Training",
        "axis_count": 2,
        "width_in": PLOT_WIDTH,
        "height_in": SHORT_PLOT_HEIGHT,
        "legend_required": True,
        "output_file": "primary_risk_training.svg",
        "layout": {"left": 0.07, "right": 0.98, "bottom": 0.066},
    },
    "primary_benchmark_overview": {
        "title": "Benchmark Overview",
        "axis_count": 6,
        "width_in": PLOT_WIDTH,
        "height_in": TALL_PLOT_HEIGHT,
        "legend_required": False,
        "output_file": "primary_benchmark_overview.svg",
        "layout": {"left": 0.058, "right": 0.985, "bottom": 0.07},
    },
    "seed_sweep_training": {
        "title": "Seed Sweep Training",
        "axis_count": 2,
        "width_in": PLOT_WIDTH,
        "height_in": SHORT_PLOT_HEIGHT,
        "legend_required": True,
        "output_file": "seed_sweep_training.svg",
        "layout": {"left": 0.07, "right": 0.98, "bottom": 0.066},
    },
    "seed_sweep_benchmark_overview": {
        "title": "Seed Sweep Benchmarks",
        "axis_count": 6,
        "width_in": PLOT_WIDTH,
        "height_in": TALL_PLOT_HEIGHT,
        "legend_required": False,
        "output_file": "seed_sweep_benchmark_overview.svg",
        "layout": {"left": 0.058, "right": 0.985, "bottom": 0.07},
    },
    "sensitivity_study": {
        "title": "Sensitivity Study",
        "axis_count": 4,
        "width_in": PLOT_WIDTH,
        "height_in": TALL_PLOT_HEIGHT,
        "legend_required": False,
        "output_file": "sensitivity_study.svg",
        "layout": {"left": 0.07, "right": 0.98, "bottom": 0.066},
    },
    "ablation_study": {
        "title": "Ablation Study",
        "axis_count": 4,
        "width_in": PLOT_WIDTH,
        "height_in": TALL_PLOT_HEIGHT,
        "legend_required": False,
        "output_file": "ablation_study.svg",
        "layout": {"left": 0.07, "right": 0.98, "bottom": 0.066},
    },
    "seed_sweep_summary": {
        "title": "Seed Sweep Summary",
        "axis_count": 2,
        "width_in": PLOT_WIDTH,
        "height_in": SHORT_PLOT_HEIGHT,
        "legend_required": False,
        "output_file": "seed_sweep_summary.svg",
        "layout": {"left": 0.07, "right": 0.98, "bottom": 0.074},
    },
    "noise_robustness": {
        "title": "Noise Robustness",
        "axis_count": 2,
        "width_in": PLOT_WIDTH,
        "height_in": SHORT_PLOT_HEIGHT,
        "legend_required": False,
        "output_file": "noise_robustness.svg",
        "layout": {"left": 0.07, "right": 0.98, "bottom": 0.07},
    },
    "trajectory_showcase": {
        "title": "Trajectory Showcase",
        "axis_count": 4,
        "width_in": PLOT_WIDTH,
        "height_in": TALL_PLOT_HEIGHT,
        "legend_required": True,
        "output_file": "trajectory_showcase.svg",
        "layout": {"left": 0.065, "right": 0.98, "bottom": 0.066},
    },
}
PLOT_NAMES = sorted(PLOT_SPECS)
PLOT_CANONICAL_NAMES = tuple(PLOT_NAMES)
AUTO_TICK_MARKER = object()


def sort_controller_names(controller_names: Iterable[str]) -> list[str]:
    """Sorts controller labels with the report-preferred display order."""
    return sorted(
        dict.fromkeys(controller_names),
        key=lambda x: (
            CONTROLLER_DISPLAY_ORDER.index(x)
            if x in CONTROLLER_DISPLAY_ORDER
            else 99,
            x,
        ),
    )
