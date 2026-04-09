"""Command-line pipeline for training, evaluation, and report generation."""

import argparse
import hashlib
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import lru_cache, partial
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from .artifacts import (
    controller_configuration,
    create_risk_clone,
    load_available_rows,
    load_controller_snapshot,
    load_showcase_cache,
    requires_refresh,
    save_controller_snapshot,
    save_showcase_cache,
    ShowcaseTraceMap,
    write_csv_rows,
    write_json_data,
)
from .settings import (
    ABLATION_STUDY_METRICS_PATH,
    ABLATION_STUDY_LEVEL_COUNT,
    NOISE_ROBUSTNESS_METRICS_PATH,
    NOISE_ROBUSTNESS_EVALUATION_SEEDS,
    NOISE_ROBUSTNESS_LEVELS,
    ANALYSIS_MAX_STEP_COUNT,
    ARTIFACT_OUTPUT_DIRS,
    BOOTSTRAP_CONFIDENCE_LEVEL,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    PLOT_CANONICAL_NAMES,
    PLOT_NAMES,
    GRID_HEIGHT,
    GRID_WIDTH,
    HOLDOUT_GENERALIZATION_CONFIG_PATH,
    HOLDOUT_GENERALIZATION_METRICS_PATH,
    HOLDOUT_GENERALIZATION_EVALUATION_EPISODE_COUNT,
    HOLDOUT_GENERALIZATION_EVALUATION_SEED,
    HOLDOUT_GENERALIZATION_TEMPLATE_GROUPS,
    HOLDOUT_GENERALIZATION_VALUE_MPC_CHECKPOINT_PATH,
    PRIMARY_BENCHMARK_SUMMARY_METRICS_PATH,
    PRIMARY_BENCHMARK_EVALUATION_SEED,
    PRIMARY_EVALUATION_EPISODE_COUNT,
    PRIMARY_MAX_STEP_COUNT,
    PRIMARY_TRAIN_EPISODE_COUNT,
    PRIMARY_TRAINING_SEED,
    PRIMARY_VALUE_TRAINING_HISTORY_PATH,
    PRIMARY_VALUE_MPC_CHECKPOINT_PATH,
    PRIMARY_RISK_TRAINING_HISTORY_PATH,
    CONTROLLER_DISPLAY_ORDER,
    OUTPUT_DIR,
    PROJECT_ROOT,
    SEED_SWEEP_PAIRWISE_ADVANTAGE_METRICS_PATH,
    SEED_SWEEP_RAW_METRICS_PATH,
    SEED_SWEEP_SUMMARY_METRICS_PATH,
    SEED_SWEEP_TRAINING_HISTORY_PATH,
    SEED_SWEEP_EVALUATION_EPISODE_COUNT,
    SEED_SWEEP_SEED_PAIRS,
    SEED_SWEEP_TRAIN_EPISODE_COUNT,
    SENSITIVITY_STUDY_LEVEL_COUNT,
    SENSITIVITY_STUDY_METRICS_PATH,
    TRAJECTORY_SHOWCASE_CACHE_PATH,
    sort_controller_names,
)
from .controllers.value_mpc import ValueMpcController
from .environment import (
    DynamicAStarController,
    LEVEL_FAMILY_NAMES,
    LEVEL_TEMPLATES,
    LevelScenario,
    StaticAStarController,
    level_family_name,
    sample_level_scenario,
)
from .evaluation import (
    SupportsController,
    evaluate_controller_set,
    run_rollout,
    summarize_evaluation_records,
)

AnalysisRow = dict[str, Any]


@lru_cache(maxsize=1)
def _plots_module() -> ModuleType:
    """Imports plotting helpers lazily so non-plot CLI paths stay lightweight."""
    from . import plots

    return plots


# =============================================================================
# Statistical aggregation helpers.
# =============================================================================
def interquartile_mean(values: Sequence[float] | np.ndarray) -> float:
    """Computes the mean after trimming the lowest and highest quartiles."""
    values = np.sort(np.array(values, float))
    count = len(values)
    lower_index = count // 4
    upper_index = count - lower_index
    return float(
        values[lower_index:upper_index].mean()
        if upper_index > lower_index
        else values.mean()
    )


def bootstrap_interval(
    values: Sequence[float] | np.ndarray,
    summary_fn: Callable[[np.ndarray], float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL,
    seed: int = 0,
) -> tuple[float, float]:
    """Estimates a bootstrap confidence interval for a summary statistic."""
    values = np.array(values, float)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    bootstrap_values = np.array(
        [
            summary_fn(values[rng.integers(0, len(values), len(values))])
            for _ in range(resamples)
        ],
        float,
    )
    tail_probability = (1 - confidence_level) / 2
    return float(np.quantile(bootstrap_values, tail_probability)), float(
        np.quantile(bootstrap_values, 1 - tail_probability)
    )


def deterministic_seed(*parts: object) -> int:
    """Derives a stable integer seed from arbitrary hashable inputs."""
    return int(hashlib.sha256(repr(parts).encode()).hexdigest()[:8], 16)


def aggregate_seed_sweep_statistics(
    rows: Sequence[Mapping[str, Any]],
    group_key: str = "controller_name",
    metric_keys: Sequence[str] = (),
) -> list[AnalysisRow]:
    """Aggregates repeated experiments with bootstrap CIs and IQM stats."""
    summary_rows: list[AnalysisRow] = []
    for group_name in sort_controller_names(
        str(row[group_key]) for row in rows
    ):
        group_rows = [
            row for row in rows if str(row[group_key]) == group_name
        ]
        summary_row = {
            group_key: group_name,
            "repeat_count": float(len(group_rows)),
        }
        for metric_key in metric_keys:
            metric_values = np.array(
                [float(row[metric_key]) for row in group_rows], float
            )
            ci_low, ci_high = bootstrap_interval(
                metric_values,
                np.mean,
                seed=deterministic_seed(
                    "m",
                    group_name,
                    metric_key,
                    BOOTSTRAP_SEED,
                ),
            )
            iqm_ci_low, iqm_ci_high = bootstrap_interval(
                metric_values,
                interquartile_mean,
                seed=deterministic_seed(
                    "i",
                    group_name,
                    metric_key,
                    BOOTSTRAP_SEED,
                ),
            )
            summary_row |= {
                f"{metric_key}_mean": float(metric_values.mean()),
                f"{metric_key}_std": float(metric_values.std()),
                f"{metric_key}_ci_low": ci_low,
                f"{metric_key}_ci_high": ci_high,
                f"{metric_key}_iqm": interquartile_mean(metric_values),
                f"{metric_key}_iqm_ci_low": iqm_ci_low,
                f"{metric_key}_iqm_ci_high": iqm_ci_high,
            }
        summary_rows.append(summary_row)
    return summary_rows


def aggregate_seed_sweep_training(
    rows: Sequence[Mapping[str, Any]],
) -> list[AnalysisRow]:
    """Aggregates epoch-wise training curves across multiple random seeds."""
    summary_rows: list[AnalysisRow] = []
    metric_keys = ["train_loss", "val_loss", "train_mae", "val_mae"]
    for epoch in sorted({float(row["epoch"]) for row in rows}):
        epoch_rows = [row for row in rows if float(row["epoch"]) == epoch]
        summary_row = {"epoch": epoch, "seed_count": float(len(epoch_rows))}
        for metric_key in metric_keys:
            metric_values = np.array(
                [float(row[metric_key]) for row in epoch_rows], float
            )
            ci_low, ci_high = bootstrap_interval(
                metric_values,
                np.median,
                seed=deterministic_seed(
                    "curve", metric_key, epoch, len(
                        metric_values), BOOTSTRAP_SEED
                ),
            )
            summary_row |= {
                f"{metric_key}_median": float(np.median(metric_values)),
                f"{metric_key}_mean": float(metric_values.mean()),
                f"{metric_key}_ci_low": ci_low,
                f"{metric_key}_ci_high": ci_high,
            }
        summary_rows.append(summary_row)
    return summary_rows


def compute_pairwise_advantage(
    rows: Sequence[Mapping[str, Any]],
    metric_keys: Sequence[str],
    group_key: str = "controller_name",
    repeat_key: str = "repeat",
) -> list[AnalysisRow]:
    """Computes pairwise improvement probabilities between repeated runs."""
    rows_by_controller: dict[str, dict[int, Mapping[str, Any]]] = {}
    for row in rows:
        rows_by_controller.setdefault(str(row[group_key]), {})[
            int(float(row[repeat_key]))
        ] = row
    summary_rows = []
    for metric_key in metric_keys:
        controller_names = sort_controller_names(rows_by_controller)
        for index, controller_a in enumerate(controller_names):
            for controller_b in controller_names[index + 1:]:
                shared_repeats = sorted(
                    set(rows_by_controller.get(controller_a, {}))
                    & set(rows_by_controller.get(controller_b, {}))
                )
                if not shared_repeats:
                    continue
                differences = np.array(
                    [
                        float(
                            rows_by_controller[controller_a][repeat][
                                metric_key
                            ]
                        )
                        - float(
                            rows_by_controller[controller_b][repeat][
                                metric_key
                            ]
                        )
                        for repeat in shared_repeats
                    ],
                    float,
                )
                ci_low, ci_high = bootstrap_interval(
                    differences,
                    pairwise_improvement_probability,
                    seed=deterministic_seed(
                        "poi",
                        metric_key,
                        controller_a,
                        controller_b,
                        len(differences),
                        BOOTSTRAP_SEED,
                    ),
                )
                summary_rows.append(
                    {
                        "metric": metric_key,
                        "controller_a": controller_a,
                        "controller_b": controller_b,
                        "improvement_probability": pairwise_improvement_probability(
                            differences
                        ),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "sample_count": float(len(differences)),
                    }
                )
    return sorted(
        summary_rows,
        key=lambda summary_row: (
            summary_row["metric"],
            (
                CONTROLLER_DISPLAY_ORDER.index(summary_row["controller_a"])
                if summary_row["controller_a"] in CONTROLLER_DISPLAY_ORDER
                else 99
            ),
            (
                CONTROLLER_DISPLAY_ORDER.index(summary_row["controller_b"])
                if summary_row["controller_b"] in CONTROLLER_DISPLAY_ORDER
                else 99
            ),
        ),
    )


def pairwise_improvement_probability(values: np.ndarray) -> float:
    """Returns the win probability with ties split evenly."""
    return float(np.mean((values > 0) + 0.5 * (values == 0)))


# =============================================================================
# LevelScenario sampling helpers.
# =============================================================================
def sample_level_batch(seed_value: int, count: int) -> list[LevelScenario]:
    """Samples a deterministic batch of levels from all template families."""
    rng = np.random.default_rng(seed_value)
    return [sample_level_scenario(rng) for _ in range(count)]


def template_indices_for_families(families: Iterable[str]) -> list[int]:
    """Resolves template indices for the requested template families."""
    selected_families = {level_family_name(family) for family in families}
    return [
        template_index
        for template_index, template in enumerate(LEVEL_TEMPLATES)
        if level_family_name(template.name) in selected_families
    ]


def sample_level_batch_by_family(
    seed_value: int,
    count: int,
    families: Iterable[str],
) -> list[LevelScenario]:
    """Samples levels restricted to the requested template families."""
    template_ids = template_indices_for_families(families)
    rng = np.random.default_rng(seed_value)
    return [sample_level_scenario(rng, template_ids) for _ in range(count)]


# =============================================================================
# Analysis helpers.
# =============================================================================
def collect_level_traces(
    level: LevelScenario,
    controllers: Mapping[str, SupportsController],
    require_success: bool = False,
) -> ShowcaseTraceMap | None:
    """Runs each controller on one level and collects its trajectory trace."""
    traces: ShowcaseTraceMap = {}
    for name, controller in controllers.items():
        result = run_rollout(level, controller, PRIMARY_MAX_STEP_COUNT, True)
        traces[name] = result["trace"]
        if require_success and float(result["success"]) < 0.5:
            return None
    return traces


def prefer_showcase_candidate(
    best: tuple[Any, ...] | None,
    key: Any,
    *payload: Any,
) -> tuple[Any, ...]:
    """Keeps the better showcase candidate according to the ranking key."""
    return (key, *payload) if best is None or key < best[0] else best


def display_path(path: Path) -> str:
    """Formats project-internal paths relative to the project root."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def cli_key(text: str) -> str:
    """Converts free-form labels into stable snake_case CLI tokens."""
    normalized = text.lower().replace("*", " star ")
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


def format_log_value(value: object) -> str:
    """Formats one structured log value for CLI output."""
    text = display_path(value) if isinstance(value, Path) else str(value)
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
    "plot_keys",
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


def rebuild_showcase_trace_from_outputs(
) -> tuple[LevelScenario, ShowcaseTraceMap]:
    """Recreates the showcase level by replaying saved controller artifacts."""
    plots = _plots_module()
    controller = load_controller_snapshot(
        PRIMARY_VALUE_MPC_CHECKPOINT_PATH,
        ValueMpcController.name,
    )
    risk_controller = create_risk_clone(controller)
    controllers = {
        StaticAStarController.name: StaticAStarController(),
        DynamicAStarController.name: DynamicAStarController(),
        ValueMpcController.name: controller,
        "Risk-Aware Value MPC": risk_controller,
    }
    levels = sample_level_batch(
        PRIMARY_BENCHMARK_EVALUATION_SEED, PRIMARY_EVALUATION_EPISODE_COUNT)
    best = None
    for level_index, level in enumerate(levels):
        traces = collect_level_traces(level, controllers, True)
        if traces:
            best = prefer_showcase_candidate(
                best,
                plots.showcase_sort_key(level, traces),
                level_index,
                level,
                traces,
            )
    if best:
        return best[2], best[3]
    level = levels[0]
    return level, {
        controller_name: run_rollout(
            level, controller_instance, PRIMARY_MAX_STEP_COUNT, True
        )["trace"]
        for controller_name, controller_instance in controllers.items()
    }


def build_showcase_trace(
    levels: Sequence[LevelScenario],
    controllers: Mapping[str, SupportsController],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Selects one representative benchmark level and its controller traces."""
    plots = _plots_module()
    best = None
    for level_index in range(len(levels)):
        successful_controllers = {
            row["controller_name"]
            for row in rows
            if int(row["level_index"]) == level_index
            and float(row["success"]) == 1.0
            and row["controller_name"] in controllers
        }
        if successful_controllers != set(controllers):
            continue
        traces = collect_level_traces(levels[level_index], controllers)
        best = prefer_showcase_candidate(
            best,
            plots.showcase_sort_key(levels[level_index], traces),
            level_index,
            traces,
        )
    if best:
        return {"level_index": [best[1]], "traces": best[2]}
    return {
        "level_index": [0],
        "traces": {
            controller_name: run_rollout(
                levels[0], controller_instance, PRIMARY_MAX_STEP_COUNT, True
            )["trace"]
            for controller_name, controller_instance in controllers.items()
        },
    }


def select_analysis_levels(
    levels: Sequence[LevelScenario],
    rows: Sequence[Mapping[str, Any]],
    controller_name: str,
    count: int,
) -> list[LevelScenario]:
    """Chooses representative levels for follow-up success/failure analysis."""
    controller_rows = [
        row for row in rows if row["controller_name"] == controller_name]
    if not controller_rows:
        return list(levels[:count])
    failed_rows = sorted(
        [row for row in controller_rows if not row["success"]],
        key=lambda row: (row["total_reward"], -row["steps"]),
        reverse=True,
    )
    successful_rows = sorted(
        [row for row in controller_rows if row["success"]],
        key=lambda row: (row["steps"], -row["total_reward"]),
    )
    level_ids = []
    if failed_rows:
        level_ids.append(int(failed_rows[0]["level_index"]))
    if successful_rows:
        for quantile_index in np.linspace(
            0,
            len(successful_rows) - 1,
            num=min(max(1, count - len(level_ids)), len(successful_rows)),
        ):
            level_ids.append(
                int(successful_rows[int(round(quantile_index))]["level_index"])
            )
        for row in sorted(
            controller_rows,
            key=lambda row: (row["success"], row["total_reward"]),
            reverse=True,
        ):
            level_ids.append(int(row["level_index"]))
    selected_ids = []
    for level_id in level_ids:
        if level_id not in selected_ids:
            selected_ids.append(level_id)
        if len(selected_ids) == count:
            break
    return [levels[level_id] for level_id in selected_ids]


def run_sensitivity(
    controller: ValueMpcController,
    levels: Sequence[LevelScenario],
    rows: Sequence[Mapping[str, Any]],
) -> list[AnalysisRow]:
    """Runs the planning-horizon and enemy-count sensitivity experiments."""
    selected_levels = select_analysis_levels(
        levels, rows, controller.name, SENSITIVITY_STUDY_LEVEL_COUNT
    )
    summary_rows: list[AnalysisRow] = []
    for planning_horizon in [2, 3, 4]:
        summary_row = summarize_evaluation_records(
            evaluate_controller_set(
                selected_levels,
                controller.clone_controller_variant(
                    planning_horizon=planning_horizon,
                    beam_width=min(3, controller.beam_width),
                    label=f"Risk-Aware MPC H{planning_horizon}",
                ),
                ANALYSIS_MAX_STEP_COUNT,
            )
        )[0]
        summary_rows.append(
            {"study": "planning_horizon", "setting": str(planning_horizon)}
            | summary_row
        )
    for enemy_limit in [1, 2, 3]:
        level_variants = [
            level.with_enemy_limit(enemy_limit) for level in selected_levels
        ]
        for candidate_controller in [
            DynamicAStarController(),
            controller.clone_controller_variant(
                beam_width=min(3, controller.beam_width), label=controller.name
            ),
        ]:
            summary_rows.append(
                {"study": "enemy_count", "setting": str(enemy_limit)}
                | summarize_evaluation_records(
                    evaluate_controller_set(
                        level_variants,
                        candidate_controller,
                        ANALYSIS_MAX_STEP_COUNT,
                    )
                )[0]
            )
    return summary_rows


def run_ablation(
    controller: ValueMpcController,
    levels: Sequence[LevelScenario],
    rows: Sequence[Mapping[str, Any]],
) -> list[AnalysisRow]:
    """Runs the controller ablation experiments on representative levels."""
    selected_levels = select_analysis_levels(
        levels, rows, controller.name, ABLATION_STUDY_LEVEL_COUNT
    )
    summary_rows: list[AnalysisRow] = []
    for candidate_controller in [
        controller.clone_controller_variant(
            beam_width=min(3, controller.beam_width),
            label="Full Risk-Aware MPC",
        ),
        controller.clone_controller_variant(
            learned_terminal_weight=0.0,
            heuristic_terminal_weight=1.0,
            beam_width=min(3, controller.beam_width),
            label="No Learned Terminal",
        ),
        controller.clone_controller_variant(
            learned_terminal_weight=1.0,
            heuristic_terminal_weight=0.0,
            beam_width=min(3, controller.beam_width),
            label="No Heuristic Terminal",
        ),
        controller.clone_controller_variant(
            risk_penalty=0.0,
            action_risk_penalty=0.0,
            beam_width=min(3, controller.beam_width),
            label="No Risk Predictor",
        ),
        controller.clone_controller_variant(
            use_predictive_hazards=False,
            beam_width=min(3, controller.beam_width),
            label="No Predictive Hazard",
        ),
    ]:
        summary_rows.append(
            summarize_evaluation_records(
                evaluate_controller_set(
                    selected_levels, candidate_controller, ANALYSIS_MAX_STEP_COUNT
                )
            )[0]
        )
    return summary_rows


def run_seed_sweep_study() -> tuple[
    list[AnalysisRow],
    list[AnalysisRow],
    list[AnalysisRow],
    list[AnalysisRow],
]:
    """Repeats end-to-end training across seeds to estimate stability."""
    rows: list[AnalysisRow] = []
    training_curve_rows: list[AnalysisRow] = []
    metric_keys = [
        "success_rate",
        "avg_steps",
        "avg_total_reward",
        "avg_path_efficiency",
        "avg_hazard_failures",
        "avg_decision_time_ms",
    ]
    for repeat_index, (train_seed, eval_seed) in enumerate(SEED_SWEEP_SEED_PAIRS):
        controller = ValueMpcController(width=GRID_WIDTH, height=GRID_HEIGHT)
        training_history = controller.fit_controller(
            SEED_SWEEP_TRAIN_EPISODE_COUNT, train_seed, PRIMARY_MAX_STEP_COUNT
        )
        training_curve_rows += [
            {
                "repeat": float(repeat_index),
                "train_seed": float(train_seed),
                "epoch": float(history_row["epoch"]),
                "train_loss": float(history_row["train_loss"]),
                "val_loss": float(history_row["val_loss"]),
                "train_mae": float(history_row["train_mae"]),
                "val_mae": float(history_row["val_mae"]),
            }
            for history_row in training_history
        ]
        risk_controller = create_risk_clone(controller)
        controllers = {
            StaticAStarController.name: StaticAStarController(),
            DynamicAStarController.name: DynamicAStarController(),
            ValueMpcController.name: controller,
            "Risk-Aware Value MPC": risk_controller,
        }
        levels = sample_level_batch(
            eval_seed,
            SEED_SWEEP_EVALUATION_EPISODE_COUNT,
        )
        evaluation_rows = []
        for controller_instance in controllers.values():
            evaluation_rows += evaluate_controller_set(
                levels, controller_instance, PRIMARY_MAX_STEP_COUNT
            )
        rows += [
            {
                "repeat": float(repeat_index),
                "train_seed": float(train_seed),
                "eval_seed": float(eval_seed),
            }
            | summary_row
            for summary_row in summarize_evaluation_records(evaluation_rows)
        ]
    return (
        rows,
        aggregate_seed_sweep_statistics(rows, "controller_name", metric_keys),
        aggregate_seed_sweep_training(training_curve_rows),
        compute_pairwise_advantage(rows, ["success_rate", "avg_total_reward"]),
    )


def run_noise_robustness(
    levels: Sequence[LevelScenario],
    controllers: Mapping[str, SupportsController],
) -> list[AnalysisRow]:
    """Measures controller robustness under injected action noise."""
    summary_rows: list[AnalysisRow] = []
    controllers = {
        name: controller
        for name, controller in controllers.items()
        if name in {"Dynamic A*", "Value MPC", "Risk-Aware Value MPC"}
    }
    for noise_level in NOISE_ROBUSTNESS_LEVELS:
        evaluation_rows = []
        for controller in controllers.values():
            for seed in NOISE_ROBUSTNESS_EVALUATION_SEEDS:
                evaluation_rows += evaluate_controller_set(
                    levels,
                    controller,
                    PRIMARY_MAX_STEP_COUNT,
                    noise_level,
                    seed + int(noise_level * 1000),
                )
        summary_rows += [
            {"action_noise": float(noise_level)} | summary_row
            for summary_row in summarize_evaluation_records(evaluation_rows)
        ]
    return sorted(
        summary_rows,
        key=lambda row: (
            float(row["action_noise"]),
            (
                CONTROLLER_DISPLAY_ORDER.index(str(row["controller_name"]))
                if str(row["controller_name"]) in CONTROLLER_DISPLAY_ORDER
                else 99
            ),
        ),
    )


def run_holdout_generalization_study() -> list[AnalysisRow]:
    """Trains with held-out template families and evaluates both splits."""
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
            "model_file": HOLDOUT_GENERALIZATION_VALUE_MPC_CHECKPOINT_PATH.name,
            "holdout_template_families": holdout_families,
            "in_distribution_template_families": training_families,
            "train_seed": PRIMARY_TRAINING_SEED,
            "eval_seed": {
                "in_distribution": HOLDOUT_GENERALIZATION_EVALUATION_SEED,
                "holdout": HOLDOUT_GENERALIZATION_EVALUATION_SEED + 1,
            },
            "training_episodes": PRIMARY_TRAIN_EPISODE_COUNT,
            "evaluation_episodes": HOLDOUT_GENERALIZATION_EVALUATION_EPISODE_COUNT,
            "max_steps": PRIMARY_MAX_STEP_COUNT,
            "controller": controller_configuration(controller),
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
    controllers = {
        DynamicAStarController.name: DynamicAStarController(),
        ValueMpcController.name: controller,
        "Risk-Aware Value MPC": risk_controller,
    }
    summary_rows: list[AnalysisRow] = []
    for split_name, split_families, eval_seed in [
        (
            "in_distribution",
            training_families,
            HOLDOUT_GENERALIZATION_EVALUATION_SEED,
        ),
        (
            "holdout",
            holdout_families,
            HOLDOUT_GENERALIZATION_EVALUATION_SEED + 1,
        ),
    ]:
        levels = sample_level_batch_by_family(
            eval_seed,
            HOLDOUT_GENERALIZATION_EVALUATION_EPISODE_COUNT,
            split_families,
        )
        evaluation_rows = []
        for controller_instance in controllers.values():
            evaluation_rows += evaluate_controller_set(
                levels, controller_instance, PRIMARY_MAX_STEP_COUNT
            )
        summary_rows += [
            {
                "split": split_name,
                "eval_seed": float(eval_seed),
                "train_holdout": "|".join(holdout_families),
                "template_families": "|".join(split_families),
            }
            | summary_row
            for summary_row in summarize_evaluation_records(evaluation_rows)
        ]
    return summary_rows


def build_showcase_payload() -> dict[str, Any]:
    """Loads or rebuilds the cached qualitative trajectory example."""
    model_path = PRIMARY_VALUE_MPC_CHECKPOINT_PATH
    cache_path = TRAJECTORY_SHOWCASE_CACHE_PATH
    if cache_path.exists() and (
        not model_path.exists()
        or (
            cache_path.stat().st_mtime_ns
            >= model_path.stat().st_mtime_ns
        )
    ):
        level, traces = load_showcase_cache(cache_path)
        return {"level": level, "traces": traces}
    if not model_path.exists():
        return {}
    level, traces = rebuild_showcase_trace_from_outputs()
    save_showcase_cache(level, traces, TRAJECTORY_SHOWCASE_CACHE_PATH)
    return {"level": level, "traces": traces}


def should_refresh_plot(
    plot_name: str,
    output_path: Path,
    force_redraw: bool,
) -> bool:
    """Decides whether one plot should be regenerated from available artifacts."""
    dependencies = list(PLOT_DEPENDENCIES.get(plot_name, ()))
    if plot_name == "trajectory_showcase":
        cache_missing_but_rebuildable = (
            not TRAJECTORY_SHOWCASE_CACHE_PATH.exists()
            and PRIMARY_VALUE_MPC_CHECKPOINT_PATH.exists()
        )
        return requires_refresh(
            output_path,
            dependencies,
            force_redraw or cache_missing_but_rebuildable,
        )
    return requires_refresh(output_path, dependencies, force_redraw)


# =============================================================================
# Analysis orchestration and CLI entry points.
# =============================================================================
PLOT_DEPENDENCIES = {
    "primary_value_training": [PRIMARY_VALUE_TRAINING_HISTORY_PATH],
    "primary_risk_training": [PRIMARY_RISK_TRAINING_HISTORY_PATH],
    "primary_benchmark_overview": [PRIMARY_BENCHMARK_SUMMARY_METRICS_PATH],
    "seed_sweep_training": [SEED_SWEEP_TRAINING_HISTORY_PATH],
    "seed_sweep_benchmark_overview": [SEED_SWEEP_SUMMARY_METRICS_PATH],
    "sensitivity_study": [SENSITIVITY_STUDY_METRICS_PATH],
    "ablation_study": [ABLATION_STUDY_METRICS_PATH],
    "seed_sweep_summary": [SEED_SWEEP_SUMMARY_METRICS_PATH],
    "noise_robustness": [NOISE_ROBUSTNESS_METRICS_PATH],
    "trajectory_showcase": [
        TRAJECTORY_SHOWCASE_CACHE_PATH,
        PRIMARY_VALUE_MPC_CHECKPOINT_PATH,
    ],
}

PLOT_DATA_LOADERS = {
    "primary_value_training": partial(
        load_available_rows, PRIMARY_VALUE_TRAINING_HISTORY_PATH
    ),
    "primary_risk_training": partial(
        load_available_rows, PRIMARY_RISK_TRAINING_HISTORY_PATH
    ),
    "primary_benchmark_overview": partial(
        load_available_rows, PRIMARY_BENCHMARK_SUMMARY_METRICS_PATH
    ),
    "seed_sweep_training": partial(
        load_available_rows, SEED_SWEEP_TRAINING_HISTORY_PATH
    ),
    "seed_sweep_benchmark_overview": partial(
        load_available_rows, SEED_SWEEP_SUMMARY_METRICS_PATH
    ),
    "sensitivity_study": partial(load_available_rows, SENSITIVITY_STUDY_METRICS_PATH),
    "ablation_study": partial(load_available_rows, ABLATION_STUDY_METRICS_PATH),
    "seed_sweep_summary": partial(
        load_available_rows, SEED_SWEEP_SUMMARY_METRICS_PATH
    ),
    "noise_robustness": partial(load_available_rows, NOISE_ROBUSTNESS_METRICS_PATH),
    "trajectory_showcase": build_showcase_payload,
}


def regenerate_cached_plots(
    names: Iterable[str] | None = None,
    force_redraw: bool = False,
) -> None:
    """Rebuilds selected plots from cached CSV/JSON/NPZ artifacts only."""
    for output_dir in ARTIFACT_OUTPUT_DIRS:
        output_dir.mkdir(parents=True, exist_ok=True)
    plot_names = sorted(names or PLOT_NAMES)
    plots = _plots_module()
    plot_renderers = {
        "primary_value_training": plots.render_value_training_plot,
        "primary_risk_training": plots.render_risk_training_plot,
        "primary_benchmark_overview": plots.render_benchmark_overview_plot,
        "seed_sweep_training": partial(
            plots.render_value_training_plot,
            plot_name="seed_sweep_training",
        ),
        "seed_sweep_benchmark_overview": partial(
            plots.render_benchmark_overview_plot,
            plot_name="seed_sweep_benchmark_overview",
        ),
        "sensitivity_study": plots.render_sensitivity_plot,
        "ablation_study": plots.render_ablation_plot,
        "seed_sweep_summary": plots.render_seed_sweep_plot,
        "noise_robustness": plots.render_noise_robustness_plot,
        "trajectory_showcase": lambda data, render_output_path: (
            plots.render_trajectory_showcase_plot(
                data["level"],
                data["traces"],
                render_output_path,
            )
        ),
    }
    regenerated_names = []
    skipped_names = []
    for plot_name in plot_names:
        output_path = plots.resolve_plot_path(plot_name)
        if not should_refresh_plot(plot_name, output_path, force_redraw):
            continue
        plot_data = PLOT_DATA_LOADERS[plot_name]()
        if plot_data:
            plot_renderers[plot_name](plot_data, output_path)
            regenerated_names.append(plot_name)
        else:
            skipped_names.append(plot_name)
    regenerated = (
        ",".join(regenerated_names)
        if regenerated_names
        else "none"
    )
    log_event("plot", action="redraw", status="done", plot_keys=regenerated)
    if skipped_names:
        log_event(
            "plot",
            action="redraw",
            status="skipped_missing_inputs",
            plot_keys=",".join(skipped_names),
        )


def run_workflow(resume_model: str | None = None) -> None:
    """Runs the full training, evaluation, and artifact-generation pipeline."""
    for output_dir in ARTIFACT_OUTPUT_DIRS:
        output_dir.mkdir(parents=True, exist_ok=True)
    plots = _plots_module()
    resume_path = Path(resume_model) if resume_model else None

    # 1) Train the primary controller, optionally warm-starting from a saved
    # model.
    if resume_path is not None:
        controller = load_controller_snapshot(
            resume_path, ValueMpcController.name
        )
        log_event(
            "run",
            status="start",
            stage="train",
            task="primary_value_training",
            mode="resume",
            checkpoint=resume_path,
            episodes=PRIMARY_TRAIN_EPISODE_COUNT,
        )
        training_history = controller.fit_controller(
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
            task="primary_value_training",
            mode="expert_data",
            episodes=PRIMARY_TRAIN_EPISODE_COUNT,
        )
        training_history = controller.fit_controller(
            PRIMARY_TRAIN_EPISODE_COUNT,
            PRIMARY_TRAINING_SEED,
            PRIMARY_MAX_STEP_COUNT,
        )
    risk_controller = create_risk_clone(controller)

    # 2) Save the trained controller and the raw training histories.
    save_controller_snapshot(
        PRIMARY_VALUE_MPC_CHECKPOINT_PATH,
        controller,
        risk_controller,
    )
    write_csv_rows(PRIMARY_VALUE_TRAINING_HISTORY_PATH, training_history)
    plots.render_value_training_plot(
        training_history,
        plots.resolve_plot_path("primary_value_training"),
    )
    if controller.risk_training_history:
        write_csv_rows(
            PRIMARY_RISK_TRAINING_HISTORY_PATH,
            controller.risk_training_history,
        )
        plots.render_risk_training_plot(
            controller.risk_training_history, plots.resolve_plot_path(
                "primary_risk_training")
        )

    # 3) Evaluate all controllers on the shared benchmark levels.
    log_event(
        "run",
        status="start",
        stage="benchmark",
        task="level_batch",
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
            task="evaluate",
            controller=cli_key(controller_name),
        )
        rows += evaluate_controller_set(levels,
                                        controller_instance, PRIMARY_MAX_STEP_COUNT)

    # 4) Run supplementary analyses and persist both CSV outputs and plots.
    summary = summarize_evaluation_records(rows)
    write_csv_rows(PRIMARY_BENCHMARK_SUMMARY_METRICS_PATH, summary)
    plots.render_benchmark_overview_plot(
        summary, plots.resolve_plot_path("primary_benchmark_overview")
    )

    log_event("run", status="start", stage="study", task="sensitivity_study")
    sensitivity_rows = run_sensitivity(risk_controller, levels, rows)
    write_csv_rows(SENSITIVITY_STUDY_METRICS_PATH, sensitivity_rows)
    plots.render_sensitivity_plot(
        sensitivity_rows, plots.resolve_plot_path("sensitivity_study")
    )

    log_event("run", status="start", stage="study", task="ablation_study")
    ablation_rows = run_ablation(risk_controller, levels, rows)
    write_csv_rows(ABLATION_STUDY_METRICS_PATH, ablation_rows)
    plots.render_ablation_plot(
        ablation_rows,
        plots.resolve_plot_path("ablation_study"),
    )

    log_event("run", status="start", stage="study", task="seed_sweep")
    (
        seed_sweep_rows,
        seed_sweep_summary_rows,
        seed_sweep_curve_rows,
        pairwise_advantage_rows,
    ) = run_seed_sweep_study()
    write_csv_rows(SEED_SWEEP_RAW_METRICS_PATH, seed_sweep_rows)
    write_csv_rows(SEED_SWEEP_SUMMARY_METRICS_PATH, seed_sweep_summary_rows)
    write_csv_rows(SEED_SWEEP_TRAINING_HISTORY_PATH, seed_sweep_curve_rows)
    write_csv_rows(
        SEED_SWEEP_PAIRWISE_ADVANTAGE_METRICS_PATH,
        pairwise_advantage_rows,
    )
    plots.render_seed_sweep_plot(
        seed_sweep_summary_rows,
        plots.resolve_plot_path("seed_sweep_summary"),
    )
    plots.render_benchmark_overview_plot(
        seed_sweep_summary_rows,
        plots.resolve_plot_path("seed_sweep_benchmark_overview"),
        plot_name="seed_sweep_benchmark_overview",
    )
    plots.render_value_training_plot(
        seed_sweep_curve_rows,
        plots.resolve_plot_path("seed_sweep_training"),
        plot_name="seed_sweep_training",
    )

    log_event("run", status="start", stage="study", task="noise_robustness")
    action_noise_rows = run_noise_robustness(levels, controllers)
    write_csv_rows(NOISE_ROBUSTNESS_METRICS_PATH, action_noise_rows)
    plots.render_noise_robustness_plot(
        action_noise_rows, plots.resolve_plot_path("noise_robustness")
    )

    log_event(
        "run",
        status="start",
        stage="study",
        task="holdout_generalization",
    )
    write_csv_rows(
        HOLDOUT_GENERALIZATION_METRICS_PATH,
        run_holdout_generalization_study(),
    )

    # 5) Export a representative trajectory visualization for qualitative
    # review.
    showcase_data = build_showcase_trace(
        levels,
        {
            controller_name: controllers[controller_name]
            for controller_name in [
                StaticAStarController.name,
                DynamicAStarController.name,
                ValueMpcController.name,
                "Risk-Aware Value MPC",
            ]
        },
        rows,
    )
    traces = showcase_data["traces"]
    level = levels[int(showcase_data["level_index"][0])]
    log_event("run", status="start", stage="showcase", task="trajectory_showcase")
    plots.render_trajectory_showcase_plot(
        level,
        traces,
        plots.resolve_plot_path("trajectory_showcase"),
    )
    save_showcase_cache(level, traces, TRAJECTORY_SHOWCASE_CACHE_PATH)

    log_event(
        "summary",
        action="emit",
        report="controller_metrics",
        controller_count=len(summary),
    )
    for row in summary:
        log_event(
            "metric",
            report="controller_metrics",
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
    parser = argparse.ArgumentParser(
        description="Run pipeline or redraw cached plots."
    )
    parser._optionals.title = "options"
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Redraw cached plots only.",
    )
    parser.add_argument(
        "--plot",
        dest="plot",
        action="append",
        metavar="PLOT_KEY",
        help=(
            "Restrict redraw to plot keys: "
            + ", ".join(PLOT_CANONICAL_NAMES)
        ),
    )
    parser.add_argument(
        "--force-redraw",
        action="store_true",
        help="Ignore freshness checks.",
    )
    parser.add_argument(
        "--resume-model",
        metavar="CHECKPOINT",
        help="Resume from checkpoint.",
    )
    args = parser.parse_args()
    if args.plot:
        unknown_names = sorted({name for name in args.plot if name not in PLOT_NAMES})
        if unknown_names:
            invalid_label = (
                "invalid plot key" if len(unknown_names) == 1 else "invalid plot keys"
            )
            parser.error(
                invalid_label
                + ": "
                + ", ".join(unknown_names)
                + "; valid keys: "
                + ", ".join(PLOT_CANONICAL_NAMES)
            )
    return args


def main() -> None:
    """Dispatches to the full workflow or cached-plot regeneration flow."""
    args = parse_args()
    selected_plots = set(args.plot) if args.plot else None
    (
        regenerate_cached_plots(selected_plots, args.force_redraw)
        if args.plots_only or args.plot
        else run_workflow(args.resume_model)
    )


if __name__ == "__main__":
    main()
