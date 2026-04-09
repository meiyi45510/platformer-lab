"""Core configuration, encoding, and model-head utilities for learned MPC."""

import math
from typing import Any

import numpy as np

from .value_mpc_base import (
    ArrayTuple,
    ForwardOutput,
    GradientResult,
    OptimizerState,
    SigmoidHeadOutput,
    ValueMpcBase,
    ValueMpcBaseT,
)
from ..settings import (
    RISK_CLONE_ACTION_PENALTY_WEIGHT,
    RISK_CLONE_STATE_PENALTY,
)
from ..environment import (
    ACTION_SPACE_SIZE,
    FloatMatrix,
    IntVector,
    LEVEL_FAMILY_INDEX,
    MAX_FALL_VELOCITY,
    ObservationData,
    PlatformerEnv,
    estimate_goal_distance,
    level_family_name,
    nearest_patrol_distance,
)

MODEL_PARAMETER_FIELDS = (
    "value_hidden_weights",
    "value_hidden_bias",
    "value_output_weights",
    "value_output_bias",
    "state_risk_hidden_weights",
    "state_risk_hidden_bias",
    "state_risk_output_weights",
    "state_risk_output_bias",
    "action_risk_hidden_weights",
    "action_risk_hidden_bias",
    "action_risk_output_weights",
    "action_risk_output_bias",
)
CONFIGURATION_FIELDS = (
    "width",
    "height",
    "hidden_dim",
    "learning_rate",
    "weight_decay",
    "value_huber_delta",
    "value_mae_weight",
    "epochs",
    "batch_size",
    "planning_horizon",
    "gamma",
    "beam_width",
    "use_rollout_cache",
    "use_predictive_hazards",
    "use_adaptive_planning",
    "learned_terminal_weight",
    "heuristic_terminal_weight",
    "risk_hidden_dim",
    "risk_learning_rate",
    "risk_weight_decay",
    "risk_epochs",
    "risk_batch_size",
    "risk_horizon",
    "risk_penalty",
    "risk_decay",
    "risk_dataset_episodes",
    "failure_replay_episodes",
    "bootstrap_steps",
    "use_curriculum_learning",
    "curriculum_enemy_limits",
    "curriculum_phase_weights",
    "use_value_residual",
    "use_teacher_predictive_hazards",
    "guide_bonus",
    "reuse_guide_plan",
    "use_disturbance_aware_planning",
    "disturbance_assumed_prob",
    "disturbance_radius",
    "aggregation_rounds",
    "risk_target_weight",
    "training_action_noise",
    "risk_action_noise",
    "action_risk_hidden_dim",
    "action_risk_learning_rate",
    "action_risk_weight_decay",
    "action_risk_epochs",
    "action_risk_batch_size",
    "action_risk_penalty",
    "use_template_balanced_training",
    "holdout_template_families",
)


class ValueMpcCoreMixin(ValueMpcBase):
    """Shared controller state, feature encoding, and neural-head helpers."""

    def __init__(
        self,
        width: int = 24,
        height: int = 14,
        hidden_dim: int = 128,
        learning_rate: float = 0.0036,
        weight_decay: float = 1e-4,
        value_huber_delta: float = 0.8,
        value_mae_weight: float = 0.12,
        epochs: int = 16,
        batch_size: int = 256,
        planning_horizon: int = 4,
        gamma: float = 0.96,
        beam_width: int = 4,
        use_rollout_cache: bool = True,
        use_predictive_hazards: bool = True,
        use_adaptive_planning: bool = False,
        learned_terminal_weight: float = 0.5,
        heuristic_terminal_weight: float = 0.5,
        risk_hidden_dim: int = 96,
        risk_learning_rate: float = 0.003,
        risk_weight_decay: float = 1e-4,
        risk_epochs: int = 8,
        risk_batch_size: int = 256,
        risk_horizon: int = 4,
        risk_penalty: float = 0.0,
        risk_decay: float = 0.85,
        risk_dataset_episodes: int = 28,
        failure_replay_episodes: int = 16,
        bootstrap_steps: int = 6,
        use_curriculum_learning: bool = True,
        curriculum_enemy_limits: tuple[int, ...] = (1, 2, 3),
        curriculum_phase_weights: tuple[float, ...] = (1, 1, 12),
        use_value_residual: bool = True,
        use_teacher_predictive_hazards: bool = True,
        guide_bonus: float = 1.4,
        reuse_guide_plan: bool = True,
        use_disturbance_aware_planning: bool = False,
        disturbance_assumed_prob: float = 0.12,
        disturbance_radius: float = 4.0,
        aggregation_rounds: int = 3,
        risk_target_weight: float = 0.22,
        training_action_noise: float = 0.04,
        risk_action_noise: float = 0.16,
        action_risk_hidden_dim: int = 80,
        action_risk_learning_rate: float = 0.0026,
        action_risk_weight_decay: float = 1e-4,
        action_risk_epochs: int = 8,
        action_risk_batch_size: int = 256,
        action_risk_penalty: float = 0.0,
        use_template_balanced_training: bool = True,
        holdout_template_families: tuple[str, ...] = (),
        label: str | None = None,
    ) -> None:
        """Initializes controller hyperparameters, caches, and model heads."""
        self.width = width
        self.height = height
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.value_huber_delta = value_huber_delta
        self.value_mae_weight = value_mae_weight
        self.epochs = epochs
        self.batch_size = batch_size
        self.planning_horizon = planning_horizon
        self.gamma = gamma
        self.beam_width = beam_width
        self.use_rollout_cache = use_rollout_cache
        self.use_predictive_hazards = use_predictive_hazards
        self.use_adaptive_planning = use_adaptive_planning
        self.learned_terminal_weight = learned_terminal_weight
        self.heuristic_terminal_weight = heuristic_terminal_weight
        self.risk_hidden_dim = risk_hidden_dim
        self.risk_learning_rate = risk_learning_rate
        self.risk_weight_decay = risk_weight_decay
        self.risk_epochs = risk_epochs
        self.risk_batch_size = risk_batch_size
        self.risk_horizon = risk_horizon
        self.risk_penalty = risk_penalty
        self.risk_decay = risk_decay
        self.risk_dataset_episodes = risk_dataset_episodes
        self.failure_replay_episodes = failure_replay_episodes
        self.bootstrap_steps = bootstrap_steps
        self.use_curriculum_learning = bool(use_curriculum_learning)
        self.curriculum_enemy_limits = tuple(
            int(x) for x in curriculum_enemy_limits
        )
        self.curriculum_phase_weights = tuple(
            float(x) for x in curriculum_phase_weights
        )
        self.use_value_residual = use_value_residual
        self.use_teacher_predictive_hazards = use_teacher_predictive_hazards
        self.guide_bonus = guide_bonus
        self.reuse_guide_plan = bool(reuse_guide_plan)
        self.use_disturbance_aware_planning = bool(
            use_disturbance_aware_planning
        )
        self.disturbance_assumed_prob = float(
            np.clip(disturbance_assumed_prob, 0.0, 0.49)
        )
        self.disturbance_radius = float(max(0.0, disturbance_radius))
        self.aggregation_rounds = max(1, int(aggregation_rounds))
        self.risk_target_weight = float(max(0.0, risk_target_weight))
        self.training_action_noise = float(
            np.clip(training_action_noise, 0.0, 0.35)
        )
        self.risk_action_noise = float(np.clip(risk_action_noise, 0.0, 0.45))
        self.action_risk_hidden_dim = int(action_risk_hidden_dim)
        self.action_risk_learning_rate = float(action_risk_learning_rate)
        self.action_risk_weight_decay = float(action_risk_weight_decay)
        self.action_risk_epochs = int(action_risk_epochs)
        self.action_risk_batch_size = int(action_risk_batch_size)
        self.action_risk_penalty = float(max(0.0, action_risk_penalty))
        self.use_template_balanced_training = bool(
            use_template_balanced_training)
        self.holdout_template_families = tuple(
            family
            for family in (
                level_family_name(name) for name in holdout_template_families
            )
            if family in LEVEL_FAMILY_INDEX
        )
        self.input_dim = width * height * 6 + 5
        self.action_input_dim = self.input_dim + ACTION_SPACE_SIZE
        self.fitted = False
        self.risk_fitted = False
        self.action_risk_fitted = False
        self.target_mean = 0.0
        self.target_std = 1.0
        self.risk_temperature = 1.0
        self.action_risk_temperature = 1.0
        self.risk_training_history = []
        self.action_risk_training_history = []
        self._feature_base_cache = {}
        self._guide_follow_state = None
        self._guide_follow_plan = ()
        self.risk_clone_state_penalty = float(RISK_CLONE_STATE_PENALTY)
        self.risk_clone_action_penalty = float(
            RISK_CLONE_ACTION_PENALTY_WEIGHT)
        self.name = label or self.name
        self._initialize_value_weights(0)
        self._initialize_state_risk_weights(1)
        self._initialize_action_risk_weights(2)

    @property
    def risk_clone_state_penalty(self) -> float:
        """Returns the state-risk penalty used by the exported risk clone."""
        return self._risk_clone_penalty

    @risk_clone_state_penalty.setter
    def risk_clone_state_penalty(self, value: float) -> None:
        """Stores the state-risk penalty used by the exported risk clone."""
        self._risk_clone_penalty = float(value)

    @property
    def risk_clone_action_penalty(self) -> float:
        """Returns the action-risk penalty used by the exported risk clone."""
        return self._risk_clone_action_risk_penalty

    @risk_clone_action_penalty.setter
    def risk_clone_action_penalty(self, value: float) -> None:
        """Stores the action-risk penalty used by the exported risk clone."""
        self._risk_clone_action_risk_penalty = float(value)

    @staticmethod
    def _initialize_block(
        input_dim: int,
        hidden_dim: int,
        deterministic_seed: int,
    ) -> ArrayTuple:
        """Initializes a two-layer MLP block with He-style random weights."""
        rng = np.random.default_rng(deterministic_seed)
        return (
            rng.normal(0, math.sqrt(2 / input_dim), (input_dim, hidden_dim)),
            np.zeros(hidden_dim),
            rng.normal(0, math.sqrt(2 / hidden_dim), (hidden_dim, 1)),
            np.zeros(1),
        )

    @staticmethod
    def _sanitize_block_weights(
        value_hidden_weights: np.ndarray,
        value_hidden_bias: np.ndarray,
        value_output_weights: np.ndarray,
        value_output_bias: np.ndarray,
    ) -> ArrayTuple:
        """Clips model weights into stable numeric ranges after loading."""
        return (
            np.nan_to_num(
                np.clip(value_hidden_weights, -0.6, 0.6),
                nan=0.0,
                posinf=0.6,
                neginf=-0.6,
            ),
            np.nan_to_num(
                np.clip(value_hidden_bias, -1.5, 1.5),
                nan=0.0,
                posinf=1.5,
                neginf=-1.5,
            ),
            np.nan_to_num(
                np.clip(value_output_weights, -0.6, 0.6),
                nan=0.0,
                posinf=0.6,
                neginf=-0.6,
            ),
            np.nan_to_num(
                np.clip(value_output_bias, -1.5, 1.5),
                nan=0.0,
                posinf=1.5,
                neginf=-1.5,
            ),
        )

    def _initialize_value_weights(self, deterministic_seed: int) -> None:
        """Initializes the terminal value head."""
        (
            self.value_hidden_weights,
            self.value_hidden_bias,
            self.value_output_weights,
            self.value_output_bias,
        ) = self._initialize_block(
            self.input_dim,
            self.hidden_dim,
            deterministic_seed,
        )

    def _initialize_state_risk_weights(self, deterministic_seed: int) -> None:
        """Initializes the state-risk prediction head."""
        (
            self.state_risk_hidden_weights,
            self.state_risk_hidden_bias,
            self.state_risk_output_weights,
            self.state_risk_output_bias,
        ) = self._initialize_block(
            self.input_dim,
            self.risk_hidden_dim,
            deterministic_seed,
        )

    def _initialize_action_risk_weights(self, deterministic_seed: int) -> None:
        """Initializes the action-conditioned risk prediction head."""
        (
            self.action_risk_hidden_weights,
            self.action_risk_hidden_bias,
            self.action_risk_output_weights,
            self.action_risk_output_bias,
        ) = self._initialize_block(
            self.action_input_dim,
            self.action_risk_hidden_dim,
            deterministic_seed,
        )

    def _sanitize_loaded_weights(self) -> None:
        """Sanitizes all loaded model heads after restoring a checkpoint."""
        (
            self.value_hidden_weights,
            self.value_hidden_bias,
            self.value_output_weights,
            self.value_output_bias,
        ) = self._sanitize_block_weights(
            self.value_hidden_weights,
            self.value_hidden_bias,
            self.value_output_weights,
            self.value_output_bias,
        )
        (
            self.state_risk_hidden_weights,
            self.state_risk_hidden_bias,
            self.state_risk_output_weights,
            self.state_risk_output_bias,
        ) = (
            self._sanitize_block_weights(
                self.state_risk_hidden_weights,
                self.state_risk_hidden_bias,
                self.state_risk_output_weights,
                self.state_risk_output_bias,
            )
        )
        (
            self.action_risk_hidden_weights,
            self.action_risk_hidden_bias,
            self.action_risk_output_weights,
            self.action_risk_output_bias,
        ) = self._sanitize_block_weights(
            self.action_risk_hidden_weights,
            self.action_risk_hidden_bias,
            self.action_risk_output_weights,
            self.action_risk_output_bias,
        )
        self.risk_temperature = float(
            np.clip(
                np.nan_to_num(self.risk_temperature, nan=1.0,
                              posinf=3.0, neginf=0.5),
                0.5,
                3.0,
            )
        )
        self.action_risk_temperature = float(
            np.clip(
                np.nan_to_num(
                    self.action_risk_temperature,
                    nan=1.0,
                    posinf=3.0,
                    neginf=0.5,
                ),
                0.5,
                3.0,
            )
        )

    def _parameter_values(self) -> dict[str, Any]:
        """Returns the config values needed to rebuild the controller."""
        return {name: getattr(self, name) for name in CONFIGURATION_FIELDS}

    def sanitize_loaded_parameters(self) -> None:
        """Sanitizes parameters restored from checkpoint artifacts."""
        self._sanitize_loaded_weights()

    def configuration_parameters(self) -> dict[str, Any]:
        """Exposes the controller settings that define its behavior."""
        return self._parameter_values()

    def clone_controller_variant(
        self: ValueMpcBaseT,
        **overrides: Any,
    ) -> ValueMpcBaseT:
        """Clones the controller and applies configuration overrides."""
        params = self._parameter_values()
        params["label"] = self.name
        params.update(overrides)
        controller = type(self)(**params)
        for name in MODEL_PARAMETER_FIELDS:
            setattr(controller, name, np.array(getattr(self, name), copy=True))
        controller.target_mean = self.target_mean
        controller.target_std = self.target_std
        controller.risk_temperature = self.risk_temperature
        controller.action_risk_temperature = self.action_risk_temperature
        controller.fitted = self.fitted
        controller.risk_fitted = self.risk_fitted
        controller.action_risk_fitted = self.action_risk_fitted
        controller.risk_training_history = [
            dict(row) for row in self.risk_training_history
        ]
        controller.action_risk_training_history = [
            dict(row) for row in self.action_risk_training_history
        ]
        controller._feature_base_cache = self._feature_base_cache
        controller._guide_follow_state = None
        controller._guide_follow_plan = ()
        controller.risk_clone_state_penalty = self.risk_clone_state_penalty
        controller.risk_clone_action_penalty = self.risk_clone_action_penalty
        return controller

    def encode_observation(
        self,
        observation: ObservationData,
        environment: PlatformerEnv | None = None,
    ) -> FloatMatrix:
        """Encodes one observation into the dense MLP feature vector.

        The representation concatenates static geometry, enemy occupancy at
        multiple horizons, the current agent location, and a small set of
        low-dimensional motion and proximity features.
        """
        width = int(observation["width"])
        height = int(observation["height"])
        grid_area = width * height
        cache_key = (
            None
            if environment is None
            else (
                id(environment.level),
                width,
                height,
                int(environment.level.goal[0]),
                int(environment.level.goal[1]),
            )
        )
        features = self._feature_base_cache.get(cache_key)
        if features is None:
            features = np.zeros(grid_area * 6 + 5, np.float32)
            for row, col in observation["solid_tiles"]:
                features[row * width + col] = 1
            goal_row, goal_col = observation["goal"]
            features[5 * grid_area + goal_row * width + goal_col] = 1
            if cache_key is not None:
                self._feature_base_cache[cache_key] = features
        features = np.array(features, copy=True)
        enemies = set(observation["enemy_positions"])
        row, col = observation["position"]
        goal_row, goal_col = observation["goal"]
        future_1 = environment.future_enemy_positions(
            1) if environment else enemies
        future_2 = environment.future_enemy_positions(
            2) if environment else enemies
        for enemy_row, enemy_col in enemies:
            features[grid_area + enemy_row * width + enemy_col] = 1
        for enemy_row, enemy_col in future_1:
            features[2 * grid_area + enemy_row * width + enemy_col] = 1
        for enemy_row, enemy_col in future_2:
            features[3 * grid_area + enemy_row * width + enemy_col] = 1
        features[4 * grid_area + row * width + col] = 1
        patrol_distance = nearest_patrol_distance((row, col), enemies)
        features[6 * grid_area: 6 * grid_area + 5] = [
            float(observation["velocity_y"]) / MAX_FALL_VELOCITY,
            float(observation["grounded"]),
            (goal_row - row) / max(1.0, height),
            (goal_col - col) / max(1.0, width),
            min(patrol_distance / max(width, height), 1.5),
        ]
        return features

    def encode_action(
        self,
        observation: ObservationData,
        action: int,
        environment: PlatformerEnv | None = None,
    ) -> FloatMatrix:
        """Extends the observation encoding with a one-hot action vector."""
        features = np.empty(self.action_input_dim, np.float32)
        features[: self.input_dim] = self.encode_observation(
            observation, environment)
        features[self.input_dim:] = 0.0
        features[self.input_dim + int(action)] = 1.0
        return features

    @staticmethod
    def _relu_activation(values: np.ndarray) -> np.ndarray:
        """Applies the ReLU nonlinearity."""
        return np.maximum(values, 0.0)

    def _f(self, inputs: FloatMatrix) -> ForwardOutput:
        """Runs the terminal value head forward."""
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            hidden_pre = (
                inputs @ self.value_hidden_weights + self.value_hidden_bias
            )
            hidden = self._relu_activation(hidden_pre)
            output = (hidden @ self.value_output_weights +
                      self.value_output_bias)[:, 0]
        return hidden_pre, hidden, output

    def _forward_sigmoid_head(
        self,
        inputs: FloatMatrix,
        value_hidden_weights: np.ndarray,
        value_hidden_bias: np.ndarray,
        value_output_weights: np.ndarray,
        value_output_bias: np.ndarray,
        temperature: float,
    ) -> SigmoidHeadOutput:
        """Runs a sigmoid head forward and returns logits and probabilities."""
        temperature = max(0.5, float(temperature))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            hidden_pre = inputs @ value_hidden_weights + value_hidden_bias
            hidden = self._relu_activation(hidden_pre)
            logits = (hidden @ value_output_weights + value_output_bias)[:, 0]
            probs = 1 / (1 + np.exp(-np.clip(logits / temperature, -20, 20)))
        return hidden_pre, hidden, logits, probs

    def _forward_state_risk_head(
        self,
        inputs: FloatMatrix,
        temperature: float | None = None,
    ) -> SigmoidHeadOutput:
        """Runs the state-risk head forward."""
        return self._forward_sigmoid_head(
            inputs,
            self.state_risk_hidden_weights,
            self.state_risk_hidden_bias,
            self.state_risk_output_weights,
            self.state_risk_output_bias,
            self.risk_temperature if temperature is None else temperature,
        )

    def _forward_action_risk_head(
        self,
        inputs: FloatMatrix,
        temperature: float | None = None,
    ) -> SigmoidHeadOutput:
        """Runs the action-risk head forward."""
        return self._forward_sigmoid_head(
            inputs,
            self.action_risk_hidden_weights,
            self.action_risk_hidden_bias,
            self.action_risk_output_weights,
            self.action_risk_output_bias,
            (
                self.action_risk_temperature
                if temperature is None
                else temperature
            ),
        )

    @staticmethod
    def _heuristic_bootstrap_value(observation: ObservationData) -> float:
        """Returns the heuristic terminal value used before learned fitting."""
        return -0.45 * float(
            observation.get(
                "estimate_goal_distance",
                estimate_goal_distance(
                    observation["position"], observation["goal"]),
            )
        )

    @staticmethod
    def _l2_weight_norm(*weights: np.ndarray) -> float:
        """Computes the combined squared L2 norm of parameter tensors."""
        return float(sum(float(np.square(weight).sum()) for weight in weights))

    @staticmethod
    def _parameter_snapshot(
        *configuration_parameters: np.ndarray,
    ) -> ArrayTuple:
        """Copies parameter arrays for later best-checkpoint restoration."""
        return tuple(np.array(parameter, copy=True)
                     for parameter in configuration_parameters)

    def _prepare_fit_state(
        self,
        sample_count: int,
        batch_size: int,
        *configuration_parameters: np.ndarray,
    ) -> tuple[list[dict[str, float]], float, ArrayTuple, OptimizerState, int]:
        """Builds optimizer state and checkpoint trackers for fitting."""
        return (
            [],
            1e9,
            self._parameter_snapshot(*configuration_parameters),
            self._build_optimizer_state(*configuration_parameters),
            max(1, min(sample_count, int(batch_size))),
        )

    @staticmethod
    def _append_training_metrics(
        history: list[dict[str, float]],
        epoch: int,
        train_key: str,
        val_key: str,
        train_loss: float,
        val_loss: float,
        train_mae: float,
        val_mae: float,
        sample_count: int,
        train_group_count: int,
        val_group_count: int,
    ) -> None:
        """Appends one epoch summary to a training-history list."""
        history.append(
            {
                "epoch": float(epoch),
                train_key: train_loss,
                val_key: val_loss,
                "train_mae": train_mae,
                "val_mae": val_mae,
                "sample_count": float(sample_count),
                "train_group_count": float(train_group_count),
                "val_group_count": float(val_group_count),
            }
        )

    @staticmethod
    def _binary_class_weights(
        targets: FloatMatrix,
        threshold: float = 0.05,
    ) -> FloatMatrix:
        """Builds inverse-frequency weights for imbalanced binary targets."""
        positive_mask = targets > threshold
        positive_rate = float(positive_mask.mean()) if len(targets) else 0.0
        scale = min(6.0, max(1.0, (1 - positive_rate) /
                    max(positive_rate, 1e-4)))
        return 1 + (scale - 1) * positive_mask.astype(np.float32)

    @staticmethod
    def _weighted_binary_cross_entropy(
        targets: FloatMatrix,
        probs: FloatMatrix,
        weights: FloatMatrix,
    ) -> float:
        """Computes weighted binary cross-entropy for one hazard head."""
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        return -float(
            np.mean(
                weights * (targets * np.log(probs) +
                           (1 - targets) * np.log(1 - probs))
            )
        )

    @staticmethod
    def _calibrate_head_temperature(
        logits: np.ndarray,
        targets: FloatMatrix,
    ) -> float:
        """Calibrates a sigmoid head temperature on held-out logits."""
        if len(targets) < 8:
            return 1.0
        best = (1.0, float("inf"))
        for temperature in np.linspace(0.65, 2.2, 32):
            probs = np.clip(
                1 / (1 + np.exp(-np.clip(logits / temperature, -20, 20))),
                1e-6,
                1 - 1e-6,
            )
            loss = -float(
                np.mean(targets * np.log(probs) +
                        (1 - targets) * np.log(1 - probs))
            )
            if loss < best[1]:
                best = (float(temperature), loss)
        return best[0]

    def _binary_head_gradients(
        self,
        inputs: FloatMatrix,
        targets: FloatMatrix,
        hidden_pre: np.ndarray,
        hidden: np.ndarray,
        probs: np.ndarray,
        value_hidden_weights: np.ndarray,
        value_output_weights: np.ndarray,
        weight_decay: float,
    ) -> GradientResult:
        """Computes clipped gradients for either binary hazard head."""
        weights = self._binary_class_weights(targets)
        grad = ((probs - targets) * weights)[:, None] / max(1, len(targets))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            grad_value_output_weights = np.clip(
                hidden.T @ grad + weight_decay * value_output_weights,
                -0.8,
                0.8,
            )
            grad_value_output_bias = np.clip(grad.sum(0), -0.8, 0.8)
            grad_hidden = grad @ value_output_weights.T
            grad_hidden[hidden_pre <= 0] = 0
            grad_value_hidden_weights = np.clip(
                inputs.T @ grad_hidden + weight_decay * value_hidden_weights,
                -0.8,
                0.8,
            )
            grad_value_hidden_bias = np.clip(grad_hidden.sum(0), -0.8, 0.8)
        loss = self._weighted_binary_cross_entropy(targets, probs, weights)
        loss += 0.5 * weight_decay * self._l2_weight_norm(
            value_hidden_weights,
            value_output_weights,
        )
        mae = float(np.mean(np.abs(probs - targets)))
        return (
            loss,
            mae,
            grad_value_hidden_weights,
            grad_value_hidden_bias,
            grad_value_output_weights,
            grad_value_output_bias,
        )

    def _fit_binary_head(
        self,
        features: FloatMatrix,
        targets: FloatMatrix,
        group_ids: IntVector,
        rng: np.random.Generator,
        epochs: int,
        reinitialize: bool,
        init_fn: Any,
        grad_fn: Any,
        loss_fn: Any,
        update_fn: Any,
        forward_fn: Any,
        batch_size: int,
        learning_rate: float,
        param_names: tuple[str, str, str, str],
        fitted_attr: str,
        temperature_attr: str,
        history_attr: str,
    ) -> list[dict[str, float]]:
        """Fits one binary hazard head with grouped train/validation splits."""
        train_ids, val_ids = self._split_grouped_indices(group_ids, rng)
        train_features, val_features = features[train_ids], features[val_ids]
        train_targets, val_targets = targets[train_ids], targets[val_ids]
        setattr(self, temperature_attr, 1.0)
        if reinitialize:
            init_fn(int(rng.integers(0, 10**9)))
        parameter_values = tuple(getattr(self, name) for name in param_names)
        history, best_val, best_snapshot, optimizer, batch_size = (
            self._prepare_fit_state(
                len(train_targets),
                batch_size,
                *parameter_values,
            )
        )
        for epoch in range(epochs):
            progress = epoch / max(1, epochs - 1)
            learning_rate_now = 0.35 * learning_rate * (1 - 0.25 * progress)
            for batch_ids in self._make_batches(
                    len(train_targets), batch_size, rng):
                (
                    _,
                    _,
                    grad_value_hidden_weights,
                    grad_value_hidden_bias,
                    grad_value_output_weights,
                    grad_value_output_bias,
                ) = grad_fn(
                    train_features[batch_ids],
                    train_targets[batch_ids],
                )
                update_fn(
                    grad_value_hidden_weights,
                    grad_value_hidden_bias,
                    grad_value_output_weights,
                    grad_value_output_bias,
                    optimizer,
                    learning_rate_now,
                )
            train_loss, train_mae = loss_fn(train_features, train_targets)
            val_loss, val_mae = loss_fn(val_features, val_targets)
            self._append_training_metrics(
                history,
                epoch,
                "train_bce",
                "val_bce",
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
                    *(getattr(self, name) for name in param_names)
                )
        for name, value in zip(param_names, best_snapshot):
            setattr(self, name, value)
        setattr(self, fitted_attr, True)
        setattr(
            self,
            temperature_attr,
            self._calibrate_head_temperature(
                forward_fn(val_features, 1.0)[2], val_targets
            ),
        )
        setattr(self, history_attr, history)
        return history

    def predict_state_value(
        self,
        observation: ObservationData,
        environment: PlatformerEnv | None = None,
    ) -> float:
        """Predicts the terminal value estimate for a state."""
        heuristic_value = self._heuristic_bootstrap_value(observation)
        if not self.fitted:
            return heuristic_value
        value = float(
            self._f(
                self.encode_observation(
                    observation, environment).reshape(1, -1)
            )[2][0]
            * self.target_std
            + self.target_mean
        )
        return value + heuristic_value if self.use_value_residual else value

    def predict_state_risk(
        self,
        observation: ObservationData,
        environment: PlatformerEnv | None = None,
    ) -> float:
        """Predicts the hazard probability for the current state."""
        if not self.risk_fitted:
            distance = nearest_patrol_distance(
                observation["position"], set(observation["enemy_positions"])
            )
            return float(
                np.clip(
                    (3 - distance) / 3 + (
                        0.12
                        if (
                            not observation["grounded"] and distance <= 4
                        )
                        else 0
                    ),
                    0,
                    1,
                ))
        return float(
            self._forward_state_risk_head(
                self.encode_observation(
                    observation, environment).reshape(1, -1)
            )[3][0]
        )

    def predict_action_hazard(
        self,
        observation: ObservationData,
        action: int,
        environment: PlatformerEnv | None = None,
    ) -> float:
        """Predicts the hazard probability after taking a candidate action."""
        if not self.action_risk_fitted:
            return self.predict_state_risk(observation, environment)
        return float(
            self._forward_action_risk_head(
                self.encode_action(observation, action,
                                   environment).reshape(1, -1)
            )[3][0]
        )
