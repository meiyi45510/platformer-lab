"""Shared configuration values for experiments and outputs."""

import os
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
CHECKPOINT_OUTPUT_DIR = OUTPUT_DIR / "checkpoint"
CONFIG_OUTPUT_DIR = OUTPUT_DIR / "config"
ARTIFACT_OUTPUT_DIRS = (
    OUTPUT_DIR,
    CHECKPOINT_OUTPUT_DIR,
    CONFIG_OUTPUT_DIR,
)
GRID_WIDTH = 24
GRID_HEIGHT = 14

# Primary benchmark experiment.
PRIMARY_TRAIN_EPISODE_COUNT = 480
PRIMARY_EVALUATION_EPISODE_COUNT = 24
PRIMARY_MAX_STEP_COUNT = 80
PRIMARY_TRAINING_SEED = 20260324
PRIMARY_BENCHMARK_EVALUATION_SEED = 20260426

RISK_CLONE_STATE_PENALTY = 0.2
RISK_CLONE_ACTION_PENALTY_WEIGHT = 0.12

PRIMARY_VALUE_MPC_CHECKPOINT_PATH = (
    CHECKPOINT_OUTPUT_DIR / "primary.npz"
)

# Holdout generalization study.
HOLDOUT_GENERALIZATION_TEMPLATE_GROUPS = ("tower", "overhang")
HOLDOUT_GENERALIZATION_EVALUATION_EPISODE_COUNT = 16
HOLDOUT_GENERALIZATION_EVALUATION_SEED = 20260601
HOLDOUT_GENERALIZATION_VALUE_MPC_CHECKPOINT_PATH = (
    CHECKPOINT_OUTPUT_DIR / "holdout.npz"
)
HOLDOUT_GENERALIZATION_CONFIG_PATH = (
    CONFIG_OUTPUT_DIR / "holdout.json"
)
