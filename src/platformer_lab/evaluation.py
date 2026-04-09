"""Rollout and evaluation utilities for platformer controllers."""

import time
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .environment import (
    ACTION_DISTURBANCE_MAP,
    AVAILABLE_ACTIONS,
    GridPosition,
    LevelScenario,
    ObservationData,
    PlatformerEnv,
    plan_astar_actions,
)

MetricValue = float | str | list[GridPosition]
MetricRow = dict[str, MetricValue]


class SupportsController(Protocol):
    """Minimal controller protocol used by rollout and evaluation helpers."""

    name: str

    def select_action(
        self,
        observation: ObservationData,
        environment: PlatformerEnv | None = None,
    ) -> int: ...


# =============================================================================
# Rollout helpers.
# =============================================================================
def sample_noisy_action(
    action: int,
    rng: np.random.Generator,
    noise_prob: float,
) -> int:
    """Applies the configured action-noise model to a chosen action."""
    disturbed_actions = ACTION_DISTURBANCE_MAP.get(action, AVAILABLE_ACTIONS)
    return (
        action
        if noise_prob <= 1e-8 or float(rng.random()) >= noise_prob
        else int(
            disturbed_actions[int(rng.integers(0, len(disturbed_actions)))]
        )
    )


def run_rollout(
    level: LevelScenario,
    controller: SupportsController,
    max_steps: int = 120,
    return_trace: bool = False,
    rng: np.random.Generator | None = None,
    action_noise: float = 0.0,
) -> MetricRow:
    """Executes one closed-loop evaluation run and collects task metrics."""
    environment = PlatformerEnv(level, max_steps)
    trace: list[GridPosition] = [level.start]
    times: list[float] = []
    hazards = 0.0
    hazard_failure = 0.0
    for _ in range(max_steps):
        observation = environment.observation()
        start_time = time.perf_counter()
        action = controller.select_action(observation, environment)
        times.append((time.perf_counter() - start_time) * 1000.0)
        step_result = environment.step(
            sample_noisy_action(action, rng, action_noise)
            if rng is not None
            else action
        )
        trace.append(environment.agent_position)
        hazards += float(step_result.info["hazard_collision"])
        hazard_failure = max(
            hazard_failure,
            float(
                step_result.info["hazard_collision"] > 0
                and step_result.done
                and not environment.success
            ),
        )
        if step_result.done:
            break
    plan = plan_astar_actions(
        level, (level.start[0], level.start[1], 0), set()
    )
    path_efficiency = (
        len(plan) / environment.path_length
        if environment.success and environment.path_length > 0 and plan
        else 0.0
    )
    metrics = {
        "success": float(environment.success),
        "steps": float(environment.steps),
        "path_length": float(environment.path_length),
        "blocked_moves": float(environment.blocked_actions),
        "total_reward": float(environment.total_reward),
        "path_efficiency": float(path_efficiency),
        "avg_decision_time_ms": float(np.mean(times) if times else 0.0),
        "hazard_failures": hazard_failure,
        "hazard_collisions": hazards,
    }
    if return_trace:
        metrics["trace"] = trace
    return metrics


def evaluate_controller_set(
    levels: Sequence[LevelScenario],
    controller: SupportsController,
    max_steps: int = 120,
    action_noise: float = 0.0,
    deterministic_seed: int | None = None,
) -> list[MetricRow]:
    """Evaluates one controller over a level batch and emits per-level rows."""
    rng = (
        np.random.default_rng(deterministic_seed)
        if deterministic_seed is not None
        else None
    )
    name = getattr(controller, "name")
    rows = []
    for level_index, level in enumerate(levels):
        rollout = run_rollout(
            level, controller, max_steps, rng=rng, action_noise=action_noise
        )
        rows.append(
            {
                "controller_name": name,
                "level_index": float(level_index),
                "success": float(rollout["success"]),
                "steps": float(rollout["steps"]),
                "path_length": float(rollout["path_length"]),
                "blocked_moves": float(rollout["blocked_moves"]),
                "total_reward": float(rollout["total_reward"]),
                "path_efficiency": float(rollout["path_efficiency"]),
                "avg_decision_time_ms": float(rollout["avg_decision_time_ms"]),
                "hazard_failures": float(rollout["hazard_failures"]),
                "hazard_collisions": float(rollout["hazard_collisions"]),
            }
        )
    return rows


# =============================================================================
# Summary helpers.
# =============================================================================
def summarize_evaluation_records(
    records: Sequence[MetricRow],
) -> list[MetricRow]:
    """Aggregates level rows into one summary record per controller."""
    rows_by_controller: dict[str, list[MetricRow]] = {}
    for record in records:
        rows_by_controller.setdefault(
            str(record["controller_name"]),
            [],
        ).append(record)
    rows: list[MetricRow] = []
    for controller_name, controller_rows in rows_by_controller.items():
        rows.append(
            {
                "controller_name": controller_name,
                "success_rate": float(
                    np.mean([float(row["success"]) for row in controller_rows])
                ),
                "avg_steps": float(
                    np.mean([row["steps"] for row in controller_rows])
                ),
                "avg_path_length": float(
                    np.mean([row["path_length"] for row in controller_rows])
                ),
                "avg_blocked_moves": float(
                    np.mean([row["blocked_moves"] for row in controller_rows])
                ),
                "avg_total_reward": float(
                    np.mean([row["total_reward"] for row in controller_rows])
                ),
                "avg_path_efficiency": float(
                    np.mean(
                        [row["path_efficiency"] for row in controller_rows]
                    )
                ),
                "avg_hazard_failures": float(
                    np.mean(
                        [
                            row.get("hazard_failures", 0.0)
                            for row in controller_rows
                        ]
                    )
                ),
                "avg_hazard_collisions": float(
                    np.mean(
                        [
                            row.get("hazard_collisions", 0.0)
                            for row in controller_rows
                        ]
                    )
                ),
                "avg_decision_time_ms": float(
                    np.mean(
                        [row["avg_decision_time_ms"] for row in controller_rows]
                    )
                ),
            }
        )
    return sorted(rows, key=lambda x: x["success_rate"], reverse=True)
