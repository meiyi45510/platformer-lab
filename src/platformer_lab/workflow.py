"""Command-line pipeline for training and evaluation."""

import argparse
import re
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np

from .artifacts import (
    build_controller_runtime_metadata,
    create_risk_clone,
    load_controller_snapshot,
    save_controller_snapshot,
    write_json_data,
)
from .settings import (
    ARTIFACT_OUTPUT_DIRS,
    GRID_HEIGHT,
    GRID_WIDTH,
    HOLDOUT_GENERALIZATION_CONFIG_PATH,
    HOLDOUT_GENERALIZATION_EVALUATION_EPISODE_COUNT,
    HOLDOUT_GENERALIZATION_EVALUATION_SEED,
    HOLDOUT_GENERALIZATION_TEMPLATE_GROUPS,
    HOLDOUT_GENERALIZATION_VALUE_MPC_CHECKPOINT_PATH,
    PRIMARY_BENCHMARK_EVALUATION_SEED,
    PRIMARY_EVALUATION_EPISODE_COUNT,
    PRIMARY_MAX_STEP_COUNT,
    PRIMARY_TRAIN_EPISODE_COUNT,
    PRIMARY_TRAINING_SEED,
    PRIMARY_VALUE_MPC_CHECKPOINT_PATH,
    OUTPUT_DIR,
    PROJECT_ROOT,
)
from .controllers.value_mpc import ValueMpcController
from .environment import (
    DynamicAStarController,
    LEVEL_FAMILY_NAMES,
    LevelScenario,
    StaticAStarController,
    level_family_name,
    sample_level_scenario,
)
from .evaluation import (
    SupportsController,
    evaluate_controller_set,
    summarize_evaluation_records,
)

# =============================================================================
# LevelScenario sampling.
# =============================================================================
def sample_level_batch(seed_value: int, count: int) -> list[LevelScenario]:
    """Samples a deterministic batch of levels from all template families."""
    rng = np.random.default_rng(seed_value)
    return [sample_level_scenario(rng) for _ in range(count)]


# =============================================================================
# Logging helpers.
# =============================================================================
def display_path(path: Path) -> str:
    """Formats project-internal paths relative to the project root."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


CONTROLLER_CLI_KEYS = {
    "Static A*": "static_astar",
    "Dynamic A*": "dynamic_astar",
    "Value MPC": "value_mpc",
    "Risk-Aware Value MPC": "risk_mpc",
}


def cli_key(text: str) -> str:
    """Converts free-form labels into stable, short CLI tokens."""
    if text in CONTROLLER_CLI_KEYS:
        return CONTROLLER_CLI_KEYS[text]
    normalized = text.lower().replace("*", " star ")
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


def format_log_value(value: object) -> str:
    """Formats one structured log value for CLI output."""
    if isinstance(value, Path):
        text = display_path(value)
    elif isinstance(value, str):
        text = value
    elif isinstance(value, (int, float, bool)):
        text = str(value)
    elif value is None:
        text = "None"
    else:
        text = type(value).__name__
    return text if text and all(
        character.isalnum() or character in "._,/=-"
        for character in text
    ) else repr(text)


LOG_FIELD_ORDER = (
    "status",
    "action",
    "stage",
    "task",
    "report",
    "controller",
    "controller_count",
    "mode",
    "checkpoint",
    "episodes",
    "outputs",
    "success",
    "steps",
    "blocked",
    "hazard_fail",
    "efficiency",
    "decision_ms",
)


def ordered_log_fields(fields: Mapping[str, object]) -> list[tuple[str, object]]:
    """Orders structured log fields for stable, scan-friendly CLI output."""
    ordered_items = [
        (key, fields[key]) for key in LOG_FIELD_ORDER if key in fields
    ]
    ordered_items += [
        (key, value)
        for key, value in sorted(fields.items())
        if key not in LOG_FIELD_ORDER
    ]
    return ordered_items


def log_event(channel: str, **fields: object) -> None:
    """Prints one structured CLI log line."""
    payload = " ".join(
        f"{key}={format_log_value(value)}"
        for key, value in ordered_log_fields(fields)
    )
    print(f"[{channel}] {payload}" if payload else f"[{channel}]", flush=True)


# =============================================================================
# Holdout study.
# =============================================================================
def run_holdout_study() -> None:
    """Trains on a held-out template split and saves checkpoint + config."""
    holdout_families = tuple(
        family
        for family in (
            level_family_name(group)
            for group in HOLDOUT_GENERALIZATION_TEMPLATE_GROUPS
        )
        if family in LEVEL_FAMILY_NAMES
    )
    training_families = tuple(
        family
        for family in LEVEL_FAMILY_NAMES
        if family not in holdout_families
    )
    controller = ValueMpcController(
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        holdout_template_families=holdout_families,
    )
    controller.fit_controller(
        PRIMARY_TRAIN_EPISODE_COUNT,
        PRIMARY_TRAINING_SEED,
        PRIMARY_MAX_STEP_COUNT,
    )
    risk_controller = create_risk_clone(controller)
    save_controller_snapshot(
        HOLDOUT_GENERALIZATION_VALUE_MPC_CHECKPOINT_PATH,
        controller,
        risk_controller,
    )
    write_json_data(
        HOLDOUT_GENERALIZATION_CONFIG_PATH,
        {
            "checkpoint_file": HOLDOUT_GENERALIZATION_VALUE_MPC_CHECKPOINT_PATH.name,
            "holdout_families": holdout_families,
            "train_families": training_families,
            "train_seed": PRIMARY_TRAINING_SEED,
            "train_eval_seed": HOLDOUT_GENERALIZATION_EVALUATION_SEED,
            "holdout_eval_seed": HOLDOUT_GENERALIZATION_EVALUATION_SEED + 1,
            "train_episodes": PRIMARY_TRAIN_EPISODE_COUNT,
            "eval_episodes": HOLDOUT_GENERALIZATION_EVALUATION_EPISODE_COUNT,
            "max_steps": PRIMARY_MAX_STEP_COUNT,
            "controller": {
                "label": str(controller.name),
                "config": controller.configuration_parameters(),
                "runtime": build_controller_runtime_metadata(controller),
            },
            "risk_clone": {
                "risk_penalty": float(
                    risk_controller.risk_penalty),
                "action_risk_penalty": float(
                    risk_controller.action_risk_penalty),
                "beam_width": int(
                    risk_controller.beam_width),
            },
        },
    )


# =============================================================================
# Pipeline entry point.
# =============================================================================
def run_workflow(resume_checkpoint: str | None = None) -> None:
    """Runs the full training and evaluation pipeline."""
    for output_dir in ARTIFACT_OUTPUT_DIRS:
        output_dir.mkdir(parents=True, exist_ok=True)
    resume_path = Path(resume_checkpoint) if resume_checkpoint else None

    # 1) Train the primary controller, optionally warm-starting from a saved model.
    if resume_path is not None:
        controller = load_controller_snapshot(
            resume_path, ValueMpcController.name
        )
        log_event(
            "run",
            status="start",
            stage="train",
            task="primary_train",
            mode="resume",
            checkpoint=resume_path,
            episodes=PRIMARY_TRAIN_EPISODE_COUNT,
        )
        controller.fit_controller(
            PRIMARY_TRAIN_EPISODE_COUNT,
            PRIMARY_TRAINING_SEED,
            PRIMARY_MAX_STEP_COUNT,
            True,
        )
    else:
        controller = ValueMpcController(width=GRID_WIDTH, height=GRID_HEIGHT)
        log_event(
            "run",
            status="start",
            stage="train",
            task="primary_train",
            mode="fresh",
            episodes=PRIMARY_TRAIN_EPISODE_COUNT,
        )
        controller.fit_controller(
            PRIMARY_TRAIN_EPISODE_COUNT,
            PRIMARY_TRAINING_SEED,
            PRIMARY_MAX_STEP_COUNT,
        )
    risk_controller = create_risk_clone(controller)

    # 2) Save the trained controller.
    save_controller_snapshot(
        PRIMARY_VALUE_MPC_CHECKPOINT_PATH,
        controller,
        risk_controller,
    )

    # 3) Evaluate all controllers on the shared benchmark levels.
    log_event(
        "run",
        status="start",
        stage="benchmark",
        task="levels",
        episodes=PRIMARY_EVALUATION_EPISODE_COUNT,
    )
    levels = sample_level_batch(
        PRIMARY_BENCHMARK_EVALUATION_SEED,
        PRIMARY_EVALUATION_EPISODE_COUNT,
    )
    controllers = {
        StaticAStarController.name: StaticAStarController(),
        DynamicAStarController.name: DynamicAStarController(),
        ValueMpcController.name: controller,
        "Risk-Aware Value MPC": risk_controller,
    }
    rows = []
    for controller_name, controller_instance in controllers.items():
        log_event(
            "run",
            status="start",
            stage="benchmark",
            task="eval",
            controller=cli_key(controller_name),
        )
        rows += evaluate_controller_set(
            levels,
            cast(SupportsController, controller_instance),
            PRIMARY_MAX_STEP_COUNT,
        )

    # 4) Run holdout study.
    log_event(
        "run",
        status="start",
        stage="study",
        task="holdout",
    )
    run_holdout_study()

    # 5) Summarize and report.
    summary = summarize_evaluation_records(rows)
    log_event(
        "summary",
        action="emit",
        report="metrics",
        controller_count=len(summary),
    )
    for row in summary:
        log_event(
            "metric",
            report="metrics",
            controller=cli_key(str(row["controller_name"])),
            success=f"{row['success_rate']:.3f}",
            steps=f"{row['avg_steps']:.2f}",
            blocked=f"{row['avg_blocked_moves']:.2f}",
            hazard_fail=f"{row.get('avg_hazard_failures', 0.0):.3f}",
            efficiency=f"{row['avg_path_efficiency']:.3f}",
            decision_ms=f"{row['avg_decision_time_ms']:.3f}",
        )
    log_event("done", status="ok", outputs=OUTPUT_DIR)


def parse_args() -> argparse.Namespace:
    """Parses command-line flags for the experiment pipeline entry point."""
    parser = argparse.ArgumentParser()
    parser._optionals.title = "options"
    parser.add_argument(
        "-r", "--resume",
        metavar="CHECKPOINT",
        help="resume from checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    """Runs the full training and evaluation pipeline."""
    args = parse_args()
    run_workflow(args.resume)


if __name__ == "__main__":
    main()
