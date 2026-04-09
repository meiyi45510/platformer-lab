"""Planning and rollout-scoring utilities for learned MPC control."""

from typing import Any

import numpy as np

from .value_mpc_base import RiskCache, ValueMpcBase
from ..environment import (
    ACTION_DISTURBANCE_MAP,
    ObservationData,
    PlatformerEnv,
    fallback_action_choice,
    nearest_patrol_distance,
    plan_astar_actions,
)


class ValueMpcPlanningMixin(ValueMpcBase):
    """Planning logic shared by the learned MPC controller."""

    def _state_risk_penalty(
        self,
        observation: ObservationData,
        environment: PlatformerEnv | None = None,
        risk_cache: RiskCache | None = None,
    ) -> float:
        """Returns the state-risk penalty for the current observation."""
        if self.risk_penalty <= 0:
            return 0.0
        distance = nearest_patrol_distance(
            observation["position"],
            set(observation["enemy_positions"]),
        )
        proximity = float(np.clip((3.5 - distance) / 3.5, 0.0, 1.0))
        air_fraction = 1.0 - float(observation["grounded"])
        scale = max(proximity, 0.45 * air_fraction)
        if scale <= 0.08:
            return 0.0
        if risk_cache is not None and environment is not None:
            key = environment.state_signature()
            prob = risk_cache.get(key)
            if prob is None:
                prob = self.predict_state_risk(observation, environment)
                risk_cache[key] = prob
        else:
            prob = self.predict_state_risk(observation, environment)
        return self.risk_penalty * scale * prob

    def _action_risk_penalty(
        self,
        observation: ObservationData,
        action: int,
        environment: PlatformerEnv | None = None,
        risk_cache: RiskCache | None = None,
    ) -> float:
        """Returns the cached action-risk penalty for a candidate action."""
        if self.action_risk_penalty <= 0:
            return 0.0
        distance = nearest_patrol_distance(
            observation["position"],
            set(observation["enemy_positions"]),
        )
        proximity = float(np.clip((4.0 - distance) / 4.0, 0.0, 1.0))
        air_fraction = 1.0 - float(observation["grounded"])
        scale = max(proximity, 0.28 * air_fraction)
        if scale <= 0.06:
            return 0.0
        if risk_cache is not None and environment is not None:
            key = ("action", environment.state_signature(), int(action))
            prob = risk_cache.get(key)
            if prob is None:
                prob = self.predict_action_hazard(
                    observation, action, environment)
                risk_cache[key] = prob
        else:
            prob = self.predict_action_hazard(observation, action, environment)
        return self.action_risk_penalty * scale * prob

    def _terminal_rollout_score(
        self,
        environment: PlatformerEnv,
        risk_cache: RiskCache | None = None,
    ) -> float:
        """Scores a rollout leaf using learned value and risk penalties."""
        observation = environment.observation()
        learned_value = self.predict_state_value(observation, environment)
        heuristic_value = self._heuristic_bootstrap_value(observation)
        weight_total = (
            self.learned_terminal_weight
            + self.heuristic_terminal_weight
            or 1.0
        )
        if self.use_value_residual:
            return (
                heuristic_value
                + (self.learned_terminal_weight / weight_total)
                * (learned_value - heuristic_value)
                - self._state_risk_penalty(
                    observation,
                    environment,
                    risk_cache,
                )
            )
        return (
            self.learned_terminal_weight * learned_value
            + self.heuristic_terminal_weight * heuristic_value
        ) / weight_total - self._state_risk_penalty(
            observation,
            environment,
            risk_cache,
        )

    def _guide_plan_actions(
        self,
        environment: PlatformerEnv,
    ) -> tuple[int, ...]:
        """Builds or reuses a guide plan from the embedded A* teacher."""
        key = (id(environment.level), environment.state_signature())
        if (
            self.reuse_guide_plan
            and self._guide_follow_state == key
            and self._guide_follow_plan
        ):
            return self._guide_follow_plan
        state = (
            environment.agent_position[0],
            environment.agent_position[1],
            environment.velocity_y,
        )
        return tuple(
            plan_astar_actions(
                environment.level,
                state,
                environment.enemy_positions(),
                environment.enemies,
                self.use_predictive_hazards,
                5000,
            )
            or ()
        )

    def _score_action_rollout(
        self,
        environment: PlatformerEnv,
        action: int,
        horizon: int,
        memo: dict[tuple[Any, ...], float] | None,
        risk_cache: RiskCache | None,
        beam: int | None,
    ) -> float:
        """Scores one deterministic action rollout branch."""
        observation = environment.observation()
        child = environment.clone()
        return (
            child.step(action).reward
            - self._action_risk_penalty(observation,
                                        action, environment, risk_cache)
            - self._state_risk_penalty(child.observation(), child, risk_cache)
            + self.gamma
            * self._rollout_score(child, horizon - 1, memo, risk_cache, beam)
        )

    def _score_noisy_action_rollout(
        self,
        environment: PlatformerEnv,
        action: int,
        horizon: int,
        memo: dict[tuple[Any, ...], float] | None,
        risk_cache: RiskCache | None,
        beam: int | None,
    ) -> float:
        """Scores one action under the disturbance-aware action-noise model."""
        disturbed_actions = ACTION_DISTURBANCE_MAP.get(action, (action,))
        disturbance_prob = self.disturbance_assumed_prob / len(
            disturbed_actions
        )
        score = 0.0
        for disturbed_action in disturbed_actions:
            score += (
                disturbance_prob
                + (
                    1.0 - self.disturbance_assumed_prob
                    if disturbed_action == action
                    else 0.0
                )
            ) * self._score_action_rollout(
                environment,
                disturbed_action,
                horizon,
                memo,
                risk_cache,
                beam,
            )
        return score

    def _resolve_planning_budget(
        self,
        observation: ObservationData,
    ) -> tuple[int, int]:
        """Adjusts horizon and beam width for easy or dangerous states."""
        if not self.use_adaptive_planning:
            return self.planning_horizon, self.beam_width
        goal_distance = float(observation["estimate_goal_distance"])
        enemy_distance = nearest_patrol_distance(
            observation["position"], set(observation["enemy_positions"])
        )
        horizon = self.planning_horizon
        beam_width = self.beam_width
        if enemy_distance >= 6:
            beam_width = max(3, beam_width - 1)
            if goal_distance >= 10:
                horizon = max(2, horizon - 1)
            if goal_distance >= 16:
                horizon = max(2, horizon - 2)
        elif goal_distance <= 4:
            beam_width = max(3, beam_width - 1)
        return horizon, beam_width

    def _candidate_action_ids(
        self,
        environment: PlatformerEnv,
        beam: int | None = None,
    ) -> list[int]:
        """Ranks candidate actions before beam-search rollout scoring."""
        observation = environment.observation()
        row, col = observation["position"]
        _, goal_col = observation["goal"]
        if observation["grounded"]:
            actions = (
                [2, 5, 3, 0, 1, 4]
                if goal_col > col
                else (
                    [1, 4, 3, 0, 2, 5]
                    if goal_col < col
                    else [3, 0, 1, 2, 4, 5]
                )
            )
        else:
            actions = (
                [2, 0, 1]
                if goal_col > col
                else [1, 0, 2] if goal_col < col else [0, 1, 2]
            )
        hazards = (
            environment.enemy_positions()
            | environment.future_enemy_positions(1)
        )
        if observation["grounded"] and goal_col > col and (
                row, col + 1) in hazards:
            actions = [5, 3, 2, 0, 1, 4]
        if observation["grounded"] and goal_col < col and (
                row, col - 1) in hazards:
            actions = [4, 3, 1, 0, 2, 5]
        return actions[: max(3, self.beam_width if beam is None else beam)]

    def _rollout_score(
        self,
        environment: PlatformerEnv,
        depth: int,
        memo: dict[tuple[Any, ...], float] | None = None,
        risk_cache: RiskCache | None = None,
        beam: int | None = None,
    ) -> float:
        """Recursively scores the best rollout from the current state."""
        if environment.done:
            return 0.0
        key = (
            (
                depth,
                self.beam_width if beam is None else beam,
                environment.state_signature(),
            )
            if memo is not None
            else None
        )
        if key is not None and key in memo:
            return memo[key]
        if depth <= 0:
            value = self._terminal_rollout_score(environment, risk_cache)
            if key is not None:
                memo[key] = value
            return value
        best_score = -1e9
        for action in self._candidate_action_ids(environment, beam):
            observation = environment.observation()
            child = environment.clone()
            reward = (
                child.step(action).reward -
                self._action_risk_penalty(
                    observation,
                    action,
                    environment,
                    risk_cache,
                ) -
                self._state_risk_penalty(
                    child.observation(),
                    child,
                    risk_cache))
            best_score = max(
                best_score,
                reward
                + self.gamma
                * self._rollout_score(child, depth - 1,
                                      memo, risk_cache, beam),
            )
        if key is not None:
            memo[key] = best_score
        return best_score

    def select_action(
        self,
        observation: ObservationData,
        environment: PlatformerEnv | None = None,
    ) -> int:
        """Selects the next action via guide plans and rollout scoring."""
        if environment is None:
            return fallback_action_choice(observation)
        horizon, beam_width = self._resolve_planning_budget(observation)
        guide_plan = (
            self._guide_plan_actions(
                environment) if self.guide_bonus > 0 else ()
        )
        preferred_action = (
            guide_plan[0] if guide_plan else fallback_action_choice(
                observation)
        )
        memo = {} if self.use_rollout_cache else None
        risk_cache = (
            {}
            if (
                self.risk_penalty > 0
                or self.action_risk_penalty > 0
            )
            else None
        )
        danger = (
            self.use_disturbance_aware_planning
            and self.risk_penalty > 0
            and (
                nearest_patrol_distance(
                    observation["position"],
                    set(observation["enemy_positions"]),
                )
                <= self.disturbance_radius
                or not observation["grounded"]
            )
        )
        best_action = (0, -1e9)
        for action in self._candidate_action_ids(environment, beam_width):
            score = (
                self._score_noisy_action_rollout(
                    environment,
                    action,
                    horizon,
                    memo,
                    risk_cache,
                    beam_width,
                )
                if danger
                else self._score_action_rollout(
                    environment,
                    action,
                    horizon,
                    memo,
                    risk_cache,
                    beam_width,
                )
            ) + (self.guide_bonus if action == preferred_action else 0)
            if score > best_action[1]:
                best_action = (action, score)
        if (
            self.reuse_guide_plan
            and guide_plan
            and best_action[0] == guide_plan[0]
        ):
            child = environment.clone()
            child.step(best_action[0])
            self._guide_follow_state = (
                (id(environment.level), child.state_signature())
                if (not child.done and len(guide_plan) > 1)
                else None
            )
            self._guide_follow_plan = (
                guide_plan[1:] if self._guide_follow_state is not None else ()
            )
        else:
            self._guide_follow_state = None
            self._guide_follow_plan = ()
        return best_action[0]
