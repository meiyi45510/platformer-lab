"""Training-set construction and fitting utilities for learned MPC."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from .value_mpc_base import GradientResult, OptimizerState, ValueMpcBase
from ..environment import (
    AVAILABLE_ACTIONS,
    DynamicAStarController,
    FloatMatrix,
    IntVector,
    LEVEL_TEMPLATES,
    LevelScenario,
    ObservationData,
    PlatformerEnv,
    TrainingDataset,
    fallback_action_choice,
    level_family_name,
    nearest_patrol_distance,
    sample_level_scenario,
)
from ..evaluation import sample_noisy_action


class ValueMpcTrainingMixin(ValueMpcBase):
    """Training and data-collection routines for the learned MPC controller."""

    def _value_training_loss(
        self,
        inputs: FloatMatrix,
        targets: FloatMatrix,
    ) -> tuple[float, float]:
        """Computes the regularized value-head training loss and MAE."""
        predictions = self._f(inputs)[2]
        residuals = predictions - targets
        delta = self.value_huber_delta
        scaled = np.sqrt(1 + (residuals / delta) ** 2)
        loss = float(np.mean(delta * delta * (scaled - 1)))
        loss += 0.5 * self.weight_decay * self._l2_weight_norm(
            self.value_hidden_weights,
            self.value_output_weights,
        )
        return loss, float(np.mean(np.abs(residuals)))

    def _binary_head_loss(
        self,
        inputs: FloatMatrix,
        targets: FloatMatrix,
        forward_fn: Any,
        weight_decay: float,
        value_hidden_weights: np.ndarray,
        value_output_weights: np.ndarray,
    ) -> tuple[float, float]:
        """Computes the regularized loss and MAE for a binary hazard head."""
        probs = np.clip(forward_fn(inputs)[3], 1e-6, 1 - 1e-6)
        weights = self._binary_class_weights(targets)
        loss = self._weighted_binary_cross_entropy(targets, probs, weights)
        loss += 0.5 * weight_decay * self._l2_weight_norm(
            value_hidden_weights,
            value_output_weights,
        )
        return loss, float(np.mean(np.abs(probs - targets)))

    def _state_risk_loss(
        self,
        inputs: FloatMatrix,
        targets: FloatMatrix,
    ) -> tuple[float, float]:
        """Computes the regularized state-risk loss and MAE."""
        return self._binary_head_loss(
            inputs,
            targets,
            self._forward_state_risk_head,
            self.risk_weight_decay,
            self.state_risk_hidden_weights,
            self.state_risk_output_weights,
        )

    def _action_risk_loss_term(
        self,
        inputs: FloatMatrix,
        targets: FloatMatrix,
    ) -> tuple[float, float]:
        """Computes the regularized action-risk loss and MAE."""
        return self._binary_head_loss(
            inputs,
            targets,
            self._forward_action_risk_head,
            self.action_risk_weight_decay,
            self.action_risk_hidden_weights,
            self.action_risk_output_weights,
        )

    def _value_gradients(
        self,
        inputs: FloatMatrix,
        targets: FloatMatrix,
        mae_weight: float,
        epsilon: float,
    ) -> GradientResult:
        """Computes clipped gradients for the terminal value head."""
        hidden_pre, hidden, predictions = self._f(inputs)
        residuals = predictions - targets
        delta = self.value_huber_delta
        scaled = np.sqrt(1 + (residuals / delta) ** 2)
        grad = residuals / scaled + mae_weight * residuals / np.sqrt(
            residuals * residuals + epsilon * epsilon
        )
        grad = grad[:, None] / len(targets)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            grad_value_output_weights = np.clip(
                hidden.T @ grad
                + self.weight_decay * self.value_output_weights,
                -0.8,
                0.8,
            )
            grad_value_output_bias = np.clip(grad.sum(0), -0.8, 0.8)
            grad_hidden = grad @ self.value_output_weights.T
            grad_hidden[hidden_pre <= 0] = 0
            grad_value_hidden_weights = np.clip(
                inputs.T @ grad_hidden
                + self.weight_decay * self.value_hidden_weights,
                -0.8,
                0.8,
            )
            grad_value_hidden_bias = np.clip(grad_hidden.sum(0), -0.8, 0.8)
        loss = float(np.mean(delta * delta * (scaled - 1)))
        loss += 0.5 * self.weight_decay * self._l2_weight_norm(
            self.value_hidden_weights,
            self.value_output_weights,
        )
        return (
            loss,
            float(np.mean(np.abs(residuals))),
            grad_value_hidden_weights,
            grad_value_hidden_bias,
            grad_value_output_weights,
            grad_value_output_bias,
        )

    def _state_risk_gradients(
        self,
        inputs: FloatMatrix,
        targets: FloatMatrix,
    ) -> GradientResult:
        """Computes clipped gradients for the state-risk head."""
        hidden_pre, hidden, _, probs = self._forward_state_risk_head(
            inputs, 1.0)
        return self._binary_head_gradients(
            inputs,
            targets,
            hidden_pre,
            hidden,
            probs,
            self.state_risk_hidden_weights,
            self.state_risk_output_weights,
            self.risk_weight_decay,
        )

    def _action_risk_gradients(
        self,
        inputs: FloatMatrix,
        targets: FloatMatrix,
    ) -> GradientResult:
        """Computes clipped gradients for the action-risk head."""
        hidden_pre, hidden, _, probs = self._forward_action_risk_head(
            inputs, 1.0)
        return self._binary_head_gradients(
            inputs,
            targets,
            hidden_pre,
            hidden,
            probs,
            self.action_risk_hidden_weights,
            self.action_risk_output_weights,
            self.action_risk_weight_decay,
        )

    @staticmethod
    def _make_batches(
        sample_count: int,
        batch_size: int,
        rng: np.random.Generator,
    ) -> list[IntVector]:
        """Builds shuffled mini-batch index slices."""
        indices = np.asarray(rng.permutation(sample_count), dtype=np.int32)
        batch_size = max(1, min(int(batch_size), sample_count))
        return [
            np.asarray(indices[start: start + batch_size], dtype=np.int32)
            for start in range(0, sample_count, batch_size)
        ]

    def _empty_training_set(self) -> TrainingDataset:
        """Returns an empty training set with the observation feature shape."""
        return (
            np.empty((0, self.input_dim), np.float32),
            np.empty(0, np.float32),
            np.empty(0, np.int32),
        )

    def _pack_training_set(
        self,
        features: list[FloatMatrix],
        targets: list[float],
        groups: list[int],
    ) -> TrainingDataset:
        """Stacks collected feature rows into a typed training tuple."""
        return (
            self._empty_training_set()
            if not targets
            else (
                np.vstack(features).astype(np.float32),
                np.array(targets, np.float32),
                np.array(groups, np.int32),
            )
        )

    def _merge_training_sets(self, *sets: TrainingDataset) -> TrainingDataset:
        """Concatenates grouped training sets with offset group ids."""
        merged_features = []
        merged_targets = []
        merged_groups = []
        offset = 0
        for features, targets, groups in sets:
            if len(targets) == 0:
                continue
            merged_features.append(np.asarray(features, np.float32))
            merged_targets.append(np.asarray(targets, np.float32))
            groups = np.asarray(groups, np.int32)
            merged_groups.append(groups + offset)
            offset += int(groups.max()) + 1 if len(groups) else 0
        return (
            self._empty_training_set()
            if not merged_targets
            else (
                np.vstack(merged_features).astype(np.float32),
                np.concatenate(merged_targets).astype(np.float32),
                np.concatenate(merged_groups).astype(np.int32),
            )
        )

    @staticmethod
    def _split_grouped_indices(
        groups: IntVector,
        rng: np.random.Generator,
        val_frac: float = 0.2,
    ) -> tuple[IntVector, IntVector]:
        """Splits grouped samples into train and validation partitions."""
        group_ids = np.asarray(groups, np.int32)
        if len(group_ids) < 2:
            singleton_ids = np.arange(len(group_ids), dtype=np.int32)
            return singleton_ids, singleton_ids
        unique_groups = np.unique(group_ids)
        if len(unique_groups) < 2:
            singleton_ids = np.arange(len(group_ids), dtype=np.int32)
            return singleton_ids, singleton_ids
        permuted_groups = unique_groups[rng.permutation(len(unique_groups))]
        val_group_count = max(
            1,
            min(
                len(unique_groups) - 1,
                int(round(val_frac * len(unique_groups))),
            ),
        )
        val_mask = np.isin(group_ids, permuted_groups[:val_group_count])
        train_ids = np.asarray(np.where(~val_mask)[0], dtype=np.int32)
        val_ids = np.asarray(np.where(val_mask)[0], dtype=np.int32)
        if len(train_ids) == 0 or len(val_ids) == 0:
            shuffled_ids = np.asarray(
                rng.permutation(len(group_ids)),
                dtype=np.int32,
            )
            val_count = max(1, min(len(group_ids) - 1,
                                   int(round(val_frac * len(group_ids)))))
            val_ids = shuffled_ids[:val_count]
            train_ids = shuffled_ids[val_count:]
        return train_ids, val_ids

    def _build_risk_targets(self, hazards: Sequence[float]) -> list[float]:
        """Builds discounted future-hazard targets for risk learning."""
        horizon = max(1, self.risk_horizon)
        targets = [0.0] * len(hazards)
        for index in range(len(hazards)):
            target = 0.0
            for future in range(index, min(len(hazards), index + horizon)):
                target = max(
                    target,
                    (self.risk_decay ** (future - index)) *
                    float(hazards[future]),
                )
            targets[index] = target
        return targets

    @staticmethod
    def _append_grouped_values(
        features: list[FloatMatrix],
        targets: list[float],
        groups: list[int],
        group_id: int,
        indices: Sequence[int],
        episode_features: Sequence[FloatMatrix],
        values: Sequence[float],
    ) -> None:
        """Appends selected per-step values while preserving the episode id."""
        for index in indices:
            features.append(episode_features[index])
            targets.append(values[index])
            groups.append(group_id)

    def _append_target_sequence(
        self,
        features: list[FloatMatrix],
        targets: list[float],
        groups: list[int],
        group_id: int,
        episode_features: Sequence[FloatMatrix],
        values: Sequence[float],
        augment: bool = False,
    ) -> None:
        """Appends targets and optionally oversamples hard examples."""
        indices = list(range(len(episode_features)))
        if augment:
            hard_indices = {
                neighbor
                for index, value in enumerate(values)
                if value > 0.08
                for neighbor in range(
                    max(0, index - 1),
                    min(len(values), index + 2),
                )
            }
            indices += sorted(hard_indices)
        self._append_grouped_values(
            features,
            targets,
            groups,
            group_id,
            indices,
            episode_features,
            values,
        )

    @staticmethod
    def _build_optimizer_state(*parameters: np.ndarray) -> OptimizerState:
        """Builds the Adam moment buffers for a parameter group."""
        return {
            "m": [np.zeros_like(parameter) for parameter in parameters],
            "v": [np.zeros_like(parameter) for parameter in parameters],
            "t": 0,
        }

    @staticmethod
    def _apply_adam_update(
        parameters: list[np.ndarray],
        gradients: list[np.ndarray],
        optimizer: OptimizerState,
        learning_rate: float,
        clip_ranges: Sequence[tuple[float, float]],
    ) -> None:
        """Applies one clipped Adam update to a parameter group."""
        optimizer["t"] += 1
        step = optimizer["t"]
        beta1 = 0.9
        beta2 = 0.999
        corr1 = 1 - beta1**step
        corr2 = 1 - beta2**step
        for index, (parameter, gradient, (low, high)) in enumerate(
            zip(parameters, gradients, clip_ranges)
        ):
            optimizer["m"][index] = (
                beta1 * optimizer["m"][index] + (1 - beta1) * gradient
            )
            optimizer["v"][index] = beta2 * optimizer["v"][index] + (
                1 - beta2
            ) * (gradient * gradient)
            mean = optimizer["m"][index] / corr1
            variance = optimizer["v"][index] / corr2
            parameter -= learning_rate * mean / (np.sqrt(variance) + 1e-8)
            parameter[...] = np.nan_to_num(
                np.clip(parameter, low, high),
                nan=0.0,
                posinf=high,
                neginf=low,
            )

    def _update_value_weights(
        self,
        grad_value_hidden_weights: np.ndarray,
        grad_value_hidden_bias: np.ndarray,
        grad_value_output_weights: np.ndarray,
        grad_value_output_bias: np.ndarray,
        optimizer: OptimizerState,
        learning_rate: float,
    ) -> None:
        """Updates the value-head parameters."""
        self._apply_adam_update(
            [
                self.value_hidden_weights,
                self.value_hidden_bias,
                self.value_output_weights,
                self.value_output_bias,
            ],
            [
                grad_value_hidden_weights,
                grad_value_hidden_bias,
                grad_value_output_weights,
                grad_value_output_bias,
            ],
            optimizer,
            learning_rate,
            [(-0.6, 0.6), (-1.5, 1.5), (-0.6, 0.6), (-1.5, 1.5)],
        )

    def _update_state_risk_weights(
        self,
        grad_value_hidden_weights: np.ndarray,
        grad_value_hidden_bias: np.ndarray,
        grad_value_output_weights: np.ndarray,
        grad_value_output_bias: np.ndarray,
        optimizer: OptimizerState,
        learning_rate: float,
    ) -> None:
        """Updates the state-risk-head parameters."""
        self._apply_adam_update(
            [
                self.state_risk_hidden_weights,
                self.state_risk_hidden_bias,
                self.state_risk_output_weights,
                self.state_risk_output_bias,
            ],
            [
                grad_value_hidden_weights,
                grad_value_hidden_bias,
                grad_value_output_weights,
                grad_value_output_bias,
            ],
            optimizer,
            learning_rate,
            [(-0.6, 0.6), (-1.5, 1.5), (-0.6, 0.6), (-1.5, 1.5)],
        )

    def _update_action_risk_weights(
        self,
        grad_value_hidden_weights: np.ndarray,
        grad_value_hidden_bias: np.ndarray,
        grad_value_output_weights: np.ndarray,
        grad_value_output_bias: np.ndarray,
        optimizer: OptimizerState,
        learning_rate: float,
    ) -> None:
        """Updates the action-risk-head parameters."""
        self._apply_adam_update(
            [
                self.action_risk_hidden_weights,
                self.action_risk_hidden_bias,
                self.action_risk_output_weights,
                self.action_risk_output_bias,
            ],
            [
                grad_value_hidden_weights,
                grad_value_hidden_bias,
                grad_value_output_weights,
                grad_value_output_bias,
            ],
            optimizer,
            learning_rate,
            [(-0.6, 0.6), (-1.5, 1.5), (-0.6, 0.6), (-1.5, 1.5)],
        )

    @staticmethod
    def _aggregation_profile_settings(
        episodes: int,
        warm_start: bool = False,
    ) -> tuple[int, float, float, bool]:
        """Returns schedule settings for policy-improvement rounds."""
        if episodes <= 0:
            return 0, 0.35 if warm_start else 0.55, 0.03, True
        if warm_start:
            return (
                min(episodes, max(0, min(24, episodes // 32 + 8))),
                0.34,
                0.025,
                True,
            )
        return min(
            episodes, max(
                0, min(
                    24, episodes // 32 + 4))), 0.5, 0.03, True

    def _aggregation_rounds(
        self,
        episodes: int,
        warm_start: bool = False,
    ) -> list[tuple[int, float, float, bool]]:
        """Builds per-round collection counts and exploration settings."""
        total, teacher_prob, random_prob, bootstrap = (
            self._aggregation_profile_settings(
                episodes,
                warm_start,
            )
        )
        round_count = min(self.aggregation_rounds, max(1, total))
        if total <= 0:
            return []
        weights = np.array(
            [1.0 / (1.18**index) for index in range(round_count)],
            float,
        )
        counts = np.floor(total * weights / weights.sum()).astype(int)
        for index in range(total - int(counts.sum())):
            counts[index % round_count] += 1
        rounds = []
        for index, count in enumerate(counts):
            if count <= 0:
                continue
            scale = index / max(1, round_count - 1)
            rounds.append(
                (
                    int(count),
                    max(0.08, teacher_prob * (1 - 0.34 * scale)),
                    max(0.008, random_prob * (1 - 0.4 * scale)),
                    bootstrap,
                )
            )
        return rounds

    @staticmethod
    def _hazard_signal(
        observation: ObservationData,
        hazard: float = 0.0,
    ) -> float:
        """Builds a dense hazard target from collisions and proximity."""
        distance = nearest_patrol_distance(
            observation["position"],
            set(observation["enemy_positions"]),
        )
        proximity = float(np.clip((3.5 - distance) / 3.5, 0.0, 1.0))
        air_fraction = 1.0 - float(observation["grounded"])
        return float(
            np.clip(
                max(float(hazard), 0.55 * proximity +
                    0.18 * proximity * air_fraction),
                0.0,
                1.0,
            )
        )

    def _discount_returns(self, rewards: Sequence[float]) -> list[float]:
        """Computes discounted returns for one rollout."""
        running = 0.0
        returns = [0.0] * len(rewards)
        for index in range(len(rewards) - 1, -1, -1):
            running = float(rewards[index]) + self.gamma * running
            returns[index] = running
        return returns

    def _bootstrap_value_prediction(
        self,
        features: FloatMatrix,
        bias: float = 0.0,
    ) -> float:
        """Predicts a bootstrap terminal value from encoded features."""
        bias = float(bias)
        if not self.fitted:
            return bias
        value = float(
            self._f(np.asarray(features, np.float32).reshape(1, -1))[2][0]
            * self.target_std
            + self.target_mean
        )
        return value + bias if self.use_value_residual else value

    def _build_bootstrapped_returns(
        self,
        episode_features: Sequence[FloatMatrix],
        rewards: Sequence[float],
        baselines: Sequence[float] | None = None,
    ) -> list[float]:
        """Builds truncated bootstrapped returns for value fitting."""
        horizon = max(1, int(self.bootstrap_steps))
        count = len(rewards)
        returns = [0.0] * count
        for start in range(count):
            total = 0.0
            discount = 1.0
            stop = start
            while stop < count and stop < start + horizon:
                total += discount * float(rewards[stop])
                discount *= self.gamma
                stop += 1
            if stop < count:
                total += discount * self._bootstrap_value_prediction(
                    episode_features[stop],
                    0.0 if baselines is None else baselines[stop],
                )
            returns[start] = total
        return returns

    def _append_return_targets(
        self,
        features: list[FloatMatrix],
        targets: list[float],
        groups: list[int],
        group_id: int,
        episode_features: Sequence[FloatMatrix],
        rewards: Sequence[float],
        baselines: Sequence[float] | None = None,
        mask: Sequence[bool] | None = None,
        bootstrap: bool = False,
        hazards: Sequence[float] | None = None,
    ) -> None:
        """Converts one rollout into value-learning supervision targets."""
        returns = (
            self._build_bootstrapped_returns(
                episode_features, rewards, baselines)
            if bootstrap and self.bootstrap_steps > 0
            else self._discount_returns(rewards)
        )
        risk_targets = (
            None
            if hazards is None or self.risk_target_weight <= 0
            else self._build_risk_targets(hazards)
        )
        indices = (
            range(len(episode_features))
            if mask is None
            else [i for i, keep in enumerate(mask) if keep]
        )
        values = []
        for index in range(len(episode_features)):
            target = returns[index] - (
                (0.0 if baselines is None else float(baselines[index]))
                if self.use_value_residual
                else 0.0
            )
            if risk_targets is not None:
                target -= self.risk_target_weight * risk_targets[index]
            values.append(target)
        self._append_grouped_values(
            features,
            targets,
            groups,
            group_id,
            indices,
            episode_features,
            values,
        )

    @staticmethod
    def _hard_replay_selection(
        observations: Sequence[ObservationData],
        rewards: Sequence[float],
        infos: Sequence[dict[str, float]],
        environment: PlatformerEnv,
    ) -> list[bool]:
        """Marks hard rollout segments for failure-replay sampling."""
        step_count = len(rewards)
        mask = [False] * step_count
        bad_episode = (
            (not environment.success)
            or environment.steps > 34
            or environment.total_reward < 20
        )
        for index, observation in enumerate(observations):
            distance = nearest_patrol_distance(
                observation["position"], set(observation["enemy_positions"])
            )
            hard = bool(
                infos[index]["hazard_collision"]
                or infos[index]["horizontal_blocked"]
                or rewards[index] < -0.55
                or distance <= 2.0
                or (
                    index + 1 < step_count
                    and observations[index + 1]["estimate_goal_distance"]
                    >= observation["estimate_goal_distance"]
                )
            )
            if hard:
                for neighbor in range(
                        max(0, index - 2), min(step_count, index + 3)):
                    mask[neighbor] = True
        if bad_episode:
            for neighbor in range(
                max(0, step_count - max(4, step_count // 3)), step_count
            ):
                mask[neighbor] = True
        return mask if any(mask) else (
            [True] * step_count if bad_episode else mask)

    def _enemy_curriculum(
        self,
        episodes: int,
        warm_start: bool = False,
    ) -> list[int | None]:
        """Builds the per-rollout enemy-count curriculum schedule."""
        if episodes <= 0 or not self.use_curriculum_learning:
            return [None] * max(0, episodes)
        limits = tuple(
            sorted(
                dict.fromkeys(
                    max(1, int(limit))
                    for limit in self.curriculum_enemy_limits
                    if int(limit) > 0
                )
            )
        ) or (3,)
        weights = np.array(
            list(self.curriculum_phase_weights[: len(limits)])
            + [1.0] * max(0, len(limits) - len(self.curriculum_phase_weights)),
            float,
        )
        if len(limits) > 1:
            if warm_start:
                weights *= np.linspace(1.0, 1.4, len(limits))
            else:
                weights = (weights**-0.25) * np.linspace(
                    1.22, 0.98, len(limits)
                )
        weights = np.clip(weights, 1e-6, None)
        counts = np.floor(episodes * weights / weights.sum()).astype(int)
        for index in range(episodes - int(counts.sum())):
            counts[len(counts) - 1 - index % len(counts)] += 1
        schedule = []
        for limit, count in zip(limits, counts):
            schedule.extend([limit] * int(count))
        return schedule[:episodes] if schedule else [None] * episodes

    def _eligible_template_indices(self) -> list[int]:
        """Returns template ids that are not part of the holdout split."""
        holdout_families = set(self.holdout_template_families)
        indices = [
            index
            for index, template in enumerate(LEVEL_TEMPLATES)
            if level_family_name(template.name) not in holdout_families
        ]
        return indices or list(range(len(LEVEL_TEMPLATES)))

    def _template_schedule(
        self,
        episodes: int,
        rng: np.random.Generator,
    ) -> list[int]:
        """Builds the per-rollout template sampling schedule."""
        template_ids = self._eligible_template_indices()
        if episodes <= 0:
            return []
        if not self.use_template_balanced_training:
            return [
                int(template_ids[int(rng.integers(0, len(template_ids)))])
                for _ in range(episodes)
            ]
        schedule = []
        while len(schedule) < episodes:
            schedule.extend(
                [
                    int(template_ids[index])
                    for index in rng.permutation(len(template_ids))
                ]
            )
        return schedule[:episodes]

    def _sample_training_level(
        self,
        rng: np.random.Generator,
        enemy_limit: int | None = None,
        template_idx: int | None = None,
    ) -> LevelScenario:
        """Samples one training level with optional template/enemy limits."""
        level = sample_level_scenario(
            rng,
            (
                self._eligible_template_indices()
                if template_idx is None
                else [template_idx]
            ),
        )
        return (
            level
            if enemy_limit is None
            else level.with_enemy_limit(
                max(1, min(int(enemy_limit), len(level.enemies)))
            )
        )

    def _fit_value_model_head(
        self,
        features: FloatMatrix,
        targets: FloatMatrix,
        group_ids: IntVector,
        rng: np.random.Generator,
        epochs: int,
        reinitialize: bool,
    ) -> list[dict[str, float]]:
        """Fits the terminal value model used by short-horizon MPC rollouts."""
        train_ids, val_ids = self._split_grouped_indices(group_ids, rng)
        train_features, val_features = features[train_ids], features[val_ids]
        train_targets_raw, val_targets_raw = (
            targets[train_ids],
            targets[val_ids],
        )
        self.target_mean = float(train_targets_raw.mean())
        self.target_std = float(train_targets_raw.std() + 1e-6)
        target_mean = np.float32(self.target_mean)
        target_std = np.float32(self.target_std)
        train_targets = np.asarray(
            (train_targets_raw - target_mean) / target_std,
            dtype=np.float32,
        )
        val_targets = np.asarray(
            (val_targets_raw - target_mean) / target_std,
            dtype=np.float32,
        )
        if reinitialize:
            self._initialize_value_weights(int(rng.integers(0, 10**9)))
        history, best_val, best_snapshot, optimizer, batch_size = (
            self._prepare_fit_state(
                len(train_targets),
                self.batch_size,
                self.value_hidden_weights,
                self.value_hidden_bias,
                self.value_output_weights,
                self.value_output_bias,
            )
        )
        for epoch in range(epochs):
            progress = epoch / max(1, epochs - 1)
            learning_rate_now = 0.28 * self.learning_rate * (
                1 - 0.35 * progress
            )
            mae_weight = self.value_mae_weight * (0.4 + 1.2 * progress)
            for batch_ids in self._make_batches(
                    len(train_targets), batch_size, rng):
                (
                    _,
                    _,
                    grad_value_hidden_weights,
                    grad_value_hidden_bias,
                    grad_value_output_weights,
                    grad_value_output_bias,
                ) = self._value_gradients(
                    train_features[batch_ids],
                    train_targets[batch_ids],
                    mae_weight,
                    8e-4 if progress < 0.8 else 6e-4,
                )
                self._update_value_weights(
                    grad_value_hidden_weights,
                    grad_value_hidden_bias,
                    grad_value_output_weights,
                    grad_value_output_bias,
                    optimizer,
                    learning_rate_now,
                )
            train_loss, train_mae = self._value_training_loss(
                train_features, train_targets
            )
            val_loss, val_mae = self._value_training_loss(
                val_features, val_targets
            )
            self._append_training_metrics(
                history,
                epoch,
                "train_loss",
                "val_loss",
                train_loss,
                val_loss,
                train_mae,
                val_mae,
                len(targets),
                len(np.unique(group_ids[train_ids])),
                len(np.unique(group_ids[val_ids])),
            )
            if val_loss < best_val:
                best_val = val_loss
                best_snapshot = self._parameter_snapshot(
                    self.value_hidden_weights,
                    self.value_hidden_bias,
                    self.value_output_weights,
                    self.value_output_bias,
                )
        (
            self.value_hidden_weights,
            self.value_hidden_bias,
            self.value_output_weights,
            self.value_output_bias,
        ) = best_snapshot
        self.fitted = True
        return history

    def _fit_state_risk_model(
        self,
        features: FloatMatrix,
        targets: FloatMatrix,
        group_ids: IntVector,
        rng: np.random.Generator,
        epochs: int,
        reinitialize: bool,
    ) -> list[dict[str, float]]:
        """Fits the state-risk prediction head."""
        return self._fit_binary_head(
            features,
            targets,
            group_ids,
            rng,
            epochs,
            reinitialize,
            self._initialize_state_risk_weights,
            self._state_risk_gradients,
            self._state_risk_loss,
            self._update_state_risk_weights,
            self._forward_state_risk_head,
            self.risk_batch_size,
            self.risk_learning_rate,
            (
                "state_risk_hidden_weights",
                "state_risk_hidden_bias",
                "state_risk_output_weights",
                "state_risk_output_bias",
            ),
            "risk_fitted",
            "risk_temperature",
            "risk_training_history",
        )

    def _fit_action_risk_model(
        self,
        features: FloatMatrix,
        targets: FloatMatrix,
        group_ids: IntVector,
        rng: np.random.Generator,
        epochs: int,
        reinitialize: bool,
    ) -> list[dict[str, float]]:
        """Fits the action-conditioned risk prediction head."""
        return self._fit_binary_head(
            features,
            targets,
            group_ids,
            rng,
            epochs,
            reinitialize,
            self._initialize_action_risk_weights,
            self._action_risk_gradients,
            self._action_risk_loss_term,
            self._update_action_risk_weights,
            self._forward_action_risk_head,
            self.action_risk_batch_size,
            self.action_risk_learning_rate,
            (
                "action_risk_hidden_weights",
                "action_risk_hidden_bias",
                "action_risk_output_weights",
                "action_risk_output_bias",
            ),
            "action_risk_fitted",
            "action_risk_temperature",
            "action_risk_training_history",
        )

    def _append_state_risk_targets(
        self,
        features: list[FloatMatrix],
        targets: list[float],
        groups: list[int],
        group_id: int,
        episode_features: Sequence[FloatMatrix],
        hazards: list[float],
        augment: bool = False,
    ) -> None:
        """Appends supervision targets for the state-risk model."""
        self._append_target_sequence(
            features,
            targets,
            groups,
            group_id,
            episode_features,
            self._build_risk_targets(hazards),
            augment,
        )

    def _append_action_risk_targets(
        self,
        features: list[FloatMatrix],
        targets: list[float],
        groups: list[int],
        group_id: int,
        episode_features: Sequence[FloatMatrix],
        hazards: list[float],
        augment: bool = False,
    ) -> None:
        """Appends supervision targets for the action-risk model."""
        self._append_target_sequence(
            features,
            targets,
            groups,
            group_id,
            episode_features,
            self._build_risk_targets(hazards),
            augment,
        )

    def _teacher_policy_bundle(
        self,
    ) -> tuple[ValueMpcBase, DynamicAStarController]:
        """Builds the teacher and cloned policy used for policy aggregation."""
        return self.clone_controller_variant(
            risk_penalty=0.0,
            action_risk_penalty=0.0,
            beam_width=min(3, self.beam_width),
            reuse_guide_plan=False,
            label=self.name,
        ), DynamicAStarController(self.use_teacher_predictive_hazards)

    def _risk_data_policies(
        self,
    ) -> tuple[DynamicAStarController, ValueMpcBase]:
        """Builds the policy pair used to collect risk-model rollouts."""
        return DynamicAStarController(True), self.clone_controller_variant(
            risk_penalty=max(0.35, self.risk_penalty),
            action_risk_penalty=0.0,
            beam_width=min(3, self.beam_width),
            reuse_guide_plan=False,
            label=self.name,
        )

    @staticmethod
    def _sample_mixed_policy_action(
        observation: ObservationData,
        environment: PlatformerEnv,
        rng: np.random.Generator,
        mix_value: float,
        teacher: DynamicAStarController,
        policy: ValueMpcBase,
    ) -> int:
        """Samples one action from the mixed teacher/policy collector."""
        action = (
            teacher.select_action(observation, environment)
            if mix_value < 0.3
            else (
                policy.select_action(observation, environment)
                if mix_value < 0.75
                else (
                    fallback_action_choice(observation)
                    if mix_value < 0.9
                    else int(rng.integers(0, len(AVAILABLE_ACTIONS)))
                )
            )
        )
        if float(rng.random()) < (0.18 if mix_value < 0.75 else 0.28):
            action = int(rng.integers(0, len(AVAILABLE_ACTIONS)))
        return action

    def _collect_risk_training_set(
        self,
        episodes: int,
        rng: np.random.Generator,
        max_steps: int,
        action_conditioned: bool,
        curriculum: list[int | None] | None = None,
        templates: list[int] | None = None,
    ) -> TrainingDataset:
        """Collects rollouts for either the state-risk or action-risk head."""
        if episodes <= 0:
            return self._empty_training_set()
        teacher, policy = self._risk_data_policies()
        features: list[FloatMatrix] = []
        targets: list[float] = []
        groups: list[int] = []
        for episode_index in range(episodes):
            environment = PlatformerEnv(
                self._sample_training_level(
                    rng,
                    None if curriculum is None else curriculum[episode_index],
                    None if templates is None else templates[episode_index],
                ),
                max_steps,
            )
            episode_features = []
            hazards = []
            mix_value = float(rng.random())
            action_noise = (
                self.risk_action_noise
                if mix_value < 0.8
                else min(0.3, self.risk_action_noise + 0.08)
            )
            for _ in range(max_steps):
                observation = environment.observation()
                if not action_conditioned:
                    episode_features.append(
                        self.encode_observation(observation, environment)
                    )
                action = self._sample_mixed_policy_action(
                    observation,
                    environment,
                    rng,
                    mix_value,
                    teacher,
                    policy,
                )
                if action_conditioned:
                    episode_features.append(
                        self.encode_action(observation, action, environment)
                    )
                step = environment.step(
                    sample_noisy_action(action, rng, action_noise))
                hazards.append(
                    self._hazard_signal(
                        observation,
                        float(step.info["hazard_collision"]),
                    )
                )
                if step.done:
                    break
            if action_conditioned:
                self._append_action_risk_targets(
                    features,
                    targets,
                    groups,
                    episode_index,
                    episode_features,
                    hazards,
                    True,
                )
            else:
                self._append_state_risk_targets(
                    features,
                    targets,
                    groups,
                    episode_index,
                    episode_features,
                    hazards,
                    True,
                )
        return self._pack_training_set(features, targets, groups)

    def _collect_teacher_training_set(
        self,
        episodes: int,
        rng: np.random.Generator,
        max_steps: int,
        bootstrap: bool = False,
        curriculum: list[int | None] | None = None,
        templates: list[int] | None = None,
    ) -> TrainingDataset:
        """Collects teacher rollouts for supervised value learning."""
        teacher = DynamicAStarController(self.use_teacher_predictive_hazards)
        features: list[FloatMatrix] = []
        targets: list[float] = []
        groups: list[int] = []
        for episode_index in range(episodes):
            environment = PlatformerEnv(
                self._sample_training_level(
                    rng,
                    None if curriculum is None else curriculum[episode_index],
                    None if templates is None else templates[episode_index],
                ),
                max_steps,
            )
            episode_features = []
            rewards = []
            baselines = []
            hazards = []
            for _ in range(max_steps):
                observation = environment.observation()
                episode_features.append(
                    self.encode_observation(observation, environment)
                )
                baselines.append(self._heuristic_bootstrap_value(observation))
                step = environment.step(
                    teacher.select_action(observation, environment))
                rewards.append(step.reward)
                hazards.append(
                    self._hazard_signal(
                        observation,
                        float(step.info["hazard_collision"]),
                    )
                )
                if step.done:
                    break
            self._append_return_targets(
                features,
                targets,
                groups,
                episode_index,
                episode_features,
                rewards,
                baselines,
                bootstrap=bootstrap,
                hazards=hazards,
            )
        return self._pack_training_set(features, targets, groups)

    def _collect_policy_training_set(
        self,
        episodes: int,
        rng: np.random.Generator,
        max_steps: int,
        bootstrap: bool = True,
        curriculum: list[int | None] | None = None,
        templates: list[int] | None = None,
        teacher_prob: float = 0.28,
        random_prob: float = 0.05,
    ) -> TrainingDataset:
        """Collects on-policy rollouts for value-model aggregation rounds."""
        if episodes <= 0:
            return self._empty_training_set()
        controller, teacher = self._teacher_policy_bundle()
        features: list[FloatMatrix] = []
        targets: list[float] = []
        groups: list[int] = []
        for episode_index in range(episodes):
            environment = PlatformerEnv(
                self._sample_training_level(
                    rng,
                    None if curriculum is None else curriculum[episode_index],
                    None if templates is None else templates[episode_index],
                ),
                max_steps,
            )
            episode_features = []
            rewards = []
            baselines = []
            hazards = []
            for _ in range(max_steps):
                observation = environment.observation()
                episode_features.append(
                    self.encode_observation(observation, environment)
                )
                baselines.append(self._heuristic_bootstrap_value(observation))
                use_teacher = (
                    not controller.fitted
                    or float(rng.random()) < teacher_prob
                )
                action = (
                    teacher.select_action(observation, environment)
                    if use_teacher
                    else controller.select_action(observation, environment)
                )
                if float(rng.random()) < random_prob:
                    action = int(rng.integers(0, len(AVAILABLE_ACTIONS)))
                step = environment.step(
                    sample_noisy_action(
                        action, rng, self.training_action_noise)
                )
                rewards.append(step.reward)
                hazards.append(
                    self._hazard_signal(
                        observation,
                        float(step.info["hazard_collision"]),
                    )
                )
                if step.done:
                    break
            self._append_return_targets(
                features,
                targets,
                groups,
                episode_index,
                episode_features,
                rewards,
                baselines,
                bootstrap=bootstrap,
                hazards=hazards,
            )
        return self._pack_training_set(features, targets, groups)

    def _collect_failure_replay_set(
        self,
        episodes: int,
        rng: np.random.Generator,
        max_steps: int,
        bootstrap: bool = True,
        curriculum: list[int | None] | None = None,
        templates: list[int] | None = None,
    ) -> TrainingDataset:
        """Collects hard replay rollouts for failure-focused value updates."""
        if episodes <= 0:
            return self._empty_training_set()
        controller, teacher = self._teacher_policy_bundle()
        features: list[FloatMatrix] = []
        targets: list[float] = []
        groups: list[int] = []
        for episode_index in range(episodes):
            environment = PlatformerEnv(
                self._sample_training_level(
                    rng,
                    None if curriculum is None else curriculum[episode_index],
                    None if templates is None else templates[episode_index],
                ),
                max_steps,
            )
            episode_features = []
            rewards = []
            baselines = []
            hazards = []
            observations = []
            infos = []
            acting_policy = teacher if float(
                rng.random()) < 0.35 else controller
            for _ in range(max_steps):
                observation = environment.observation()
                observations.append(observation)
                episode_features.append(
                    self.encode_observation(observation, environment)
                )
                baselines.append(self._heuristic_bootstrap_value(observation))
                action = acting_policy.select_action(observation, environment)
                if acting_policy is controller and float(rng.random()) < 0.12:
                    action = int(rng.integers(0, len(AVAILABLE_ACTIONS)))
                step = environment.step(
                    sample_noisy_action(
                        action,
                        rng,
                        max(
                            self.training_action_noise,
                            0.08 if acting_policy is controller else 0.05,
                        ),
                    )
                )
                rewards.append(step.reward)
                infos.append(dict(step.info))
                hazards.append(
                    self._hazard_signal(
                        observation,
                        float(step.info["hazard_collision"]),
                    )
                )
                if step.done:
                    break
            mask = self._hard_replay_selection(
                observations,
                rewards,
                infos,
                environment,
            )
            if any(mask):
                self._append_return_targets(
                    features,
                    targets,
                    groups,
                    episode_index,
                    episode_features,
                    rewards,
                    baselines,
                    mask,
                    bootstrap,
                    hazards,
                )
        return self._pack_training_set(features, targets, groups)

    def _collect_state_risk_training_set(
        self,
        episodes: int,
        rng: np.random.Generator,
        max_steps: int,
        curriculum: list[int | None] | None = None,
        templates: list[int] | None = None,
    ) -> TrainingDataset:
        """Collects supervision for the state-risk model."""
        return self._collect_risk_training_set(
            episodes,
            rng,
            max_steps,
            action_conditioned=False,
            curriculum=curriculum,
            templates=templates,
        )

    def _collect_action_risk_training_set(
        self,
        episodes: int,
        rng: np.random.Generator,
        max_steps: int,
        curriculum: list[int | None] | None = None,
        templates: list[int] | None = None,
    ) -> TrainingDataset:
        """Collects supervision for the action-risk model."""
        return self._collect_risk_training_set(
            episodes,
            rng,
            max_steps,
            action_conditioned=True,
            curriculum=curriculum,
            templates=templates,
        )

    def fit_controller(
        self,
        episodes: int,
        deterministic_seed: int,
        max_steps: int = 120,
        warm_start: bool = False,
    ) -> list[dict[str, float]]:
        """Trains the value model first, then the state and action risk models.

        Args:
            episodes: Number of rollout episodes used for value learning.
            deterministic_seed: Seed used for sampling, data collection,
                and fitting.
            max_steps: Maximum rollout length per level.
            warm_start: Whether to continue from already fitted parameters.

        Returns:
            Epoch-level history for the value-model training phase.
        """
        rng = np.random.default_rng(deterministic_seed)
        cold_start = not (warm_start and self.fitted)

        # 1) Build curriculum schedules for value learning and risk learning.
        teacher_curriculum = self._enemy_curriculum(episodes, warm_start)
        base_templates = self._template_schedule(episodes, rng)
        risk_curriculum = self._enemy_curriculum(
            self.risk_dataset_episodes,
            warm_start,
        )
        risk_templates = self._template_schedule(
            self.risk_dataset_episodes, rng)
        aggregation_rounds = self._aggregation_rounds(
            episodes, warm_start and self.fitted
        )

        # 2) Cold-start with expert trajectories before on-policy aggregation
        # begins.
        if cold_start:
            teacher_features, teacher_targets, teacher_group_ids = (
                self._collect_teacher_training_set(
                    episodes,
                    rng,
                    max_steps,
                    False,
                    teacher_curriculum,
                    self._template_schedule(episodes, rng),
                )
            )
            self._fit_value_model_head(
                teacher_features,
                teacher_targets,
                teacher_group_ids,
                rng,
                max(10, self.epochs // 3),
                True,
            )

        # 3) Fit the value model, then enrich the replay pool with aggregation
        # rounds.
        (
            pooled_features,
            pooled_targets,
            pooled_group_ids,
        ) = self._collect_teacher_training_set(
            episodes,
            rng,
            max_steps,
            True,
            teacher_curriculum,
            base_templates,
        )
        history = self._fit_value_model_head(
            pooled_features,
            pooled_targets,
            pooled_group_ids,
            rng,
            max(6, self.epochs // 2) if aggregation_rounds else self.epochs,
            False,
        )
        if aggregation_rounds:
            round_weights = np.array(
                [count for count, _, _, _ in aggregation_rounds], float
            )
            replay_counts = (
                np.floor(
                    self.failure_replay_episodes
                    * round_weights
                    / round_weights.sum()
                ).astype(int)
                if round_weights.sum() > 0
                else np.zeros(len(aggregation_rounds), int)
            )
            for index in range(self.failure_replay_episodes -
                               int(replay_counts.sum())):
                replay_counts[index % len(replay_counts)] += 1
            for round_index, (
                (count, teacher_prob, random_prob, bootstrap),
                replay_count,
            ) in enumerate(zip(aggregation_rounds, replay_counts)):
                curriculum = self._enemy_curriculum(count, warm_start)
                replay_curriculum = self._enemy_curriculum(
                    int(replay_count), warm_start
                )
                (
                    policy_features,
                    policy_targets,
                    policy_group_ids,
                ) = self._collect_policy_training_set(
                    count,
                    rng,
                    max_steps,
                    bootstrap,
                    curriculum,
                    self._template_schedule(count, rng),
                    teacher_prob,
                    random_prob,
                )
                (
                    replay_features,
                    replay_targets,
                    replay_group_ids,
                ) = self._collect_failure_replay_set(
                    int(replay_count),
                    rng,
                    max_steps,
                    True,
                    replay_curriculum,
                    self._template_schedule(int(replay_count), rng),
                )
                (
                    pooled_features,
                    pooled_targets,
                    pooled_group_ids,
                ) = self._merge_training_sets(
                    (pooled_features, pooled_targets, pooled_group_ids),
                    (policy_features, policy_targets, policy_group_ids),
                    (replay_features, replay_targets, replay_group_ids),
                )
                history = self._fit_value_model_head(
                    pooled_features,
                    pooled_targets,
                    pooled_group_ids,
                    rng,
                    (
                        self.epochs
                        if round_index == len(aggregation_rounds) - 1
                        else max(5, self.epochs // 2)
                    ),
                    False,
                )

        # 4) Fit the two auxiliary risk models from dedicated risky rollouts.
        risk_features, risk_targets, risk_group_ids = (
            self._collect_state_risk_training_set(
                self.risk_dataset_episodes,
                rng,
                max_steps,
                risk_curriculum,
                risk_templates,
            )
        )
        if len(risk_targets) > 1:
            self._fit_state_risk_model(
                risk_features,
                risk_targets,
                risk_group_ids,
                rng,
                self.risk_epochs,
                not (warm_start and self.risk_fitted),
            )
        else:
            self.risk_fitted = False
            self.risk_training_history = []

        action_features, action_targets, action_group_ids = (
            self._collect_action_risk_training_set(
                self.risk_dataset_episodes,
                rng,
                max_steps,
                risk_curriculum,
                risk_templates,
            )
        )
        if len(action_targets) > 1:
            self._fit_action_risk_model(
                action_features,
                action_targets,
                action_group_ids,
                rng,
                self.action_risk_epochs,
                not (warm_start and self.action_risk_fitted),
            )
        else:
            self.action_risk_fitted = False
            self.action_risk_training_history = []
        return history
