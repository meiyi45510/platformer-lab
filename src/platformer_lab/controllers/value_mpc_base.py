"""Shared attribute and interface declarations for learned-MPC mixins."""

from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

from ..environment import (
    FloatMatrix,
    IntVector,
    ObservationData,
    PlatformerEnv,
)

ArrayTuple = tuple[np.ndarray, ...]
ForwardOutput = tuple[np.ndarray, np.ndarray, np.ndarray]
SigmoidHeadOutput = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
GradientResult = tuple[
    float,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]
OptimizerState = dict[str, Any]
RiskCache = dict[Any, float]
ValueMpcBaseT = TypeVar("ValueMpcBaseT", bound="ValueMpcBase")


class ValueMpcBase:
    """Typed base class that exposes cross-mixin state to static analyzers."""

    name: str
    width: int
    height: int
    hidden_dim: int
    learning_rate: float
    weight_decay: float
    value_huber_delta: float
    value_mae_weight: float
    epochs: int
    batch_size: int
    planning_horizon: int
    gamma: float
    beam_width: int
    use_rollout_cache: bool
    use_predictive_hazards: bool
    use_adaptive_planning: bool
    learned_terminal_weight: float
    heuristic_terminal_weight: float
    risk_hidden_dim: int
    risk_learning_rate: float
    risk_weight_decay: float
    risk_epochs: int
    risk_batch_size: int
    risk_horizon: int
    risk_penalty: float
    risk_decay: float
    risk_dataset_episodes: int
    failure_replay_episodes: int
    bootstrap_steps: int
    use_curriculum_learning: bool
    curriculum_enemy_limits: tuple[int, ...]
    curriculum_phase_weights: tuple[float, ...]
    use_value_residual: bool
    use_teacher_predictive_hazards: bool
    guide_bonus: float
    reuse_guide_plan: bool
    use_disturbance_aware_planning: bool
    disturbance_assumed_prob: float
    disturbance_radius: float
    aggregation_rounds: int
    risk_target_weight: float
    training_action_noise: float
    risk_action_noise: float
    action_risk_hidden_dim: int
    action_risk_learning_rate: float
    action_risk_weight_decay: float
    action_risk_epochs: int
    action_risk_batch_size: int
    action_risk_penalty: float
    use_template_balanced_training: bool
    holdout_template_families: tuple[str, ...]
    input_dim: int
    action_input_dim: int
    fitted: bool
    risk_fitted: bool
    action_risk_fitted: bool
    target_mean: float
    target_std: float
    risk_temperature: float
    action_risk_temperature: float
    risk_training_history: list[dict[str, float]]
    action_risk_training_history: list[dict[str, float]]
    _feature_base_cache: dict[Any, np.ndarray]
    _guide_follow_state: Any | None
    _guide_follow_plan: tuple[int, ...]
    _risk_clone_penalty: float
    _risk_clone_action_risk_penalty: float
    risk_clone_state_penalty: float
    risk_clone_action_penalty: float
    value_hidden_weights: np.ndarray
    value_hidden_bias: np.ndarray
    value_output_weights: np.ndarray
    value_output_bias: np.ndarray
    state_risk_hidden_weights: np.ndarray
    state_risk_hidden_bias: np.ndarray
    state_risk_output_weights: np.ndarray
    state_risk_output_bias: np.ndarray
    action_risk_hidden_weights: np.ndarray
    action_risk_hidden_bias: np.ndarray
    action_risk_output_weights: np.ndarray
    action_risk_output_bias: np.ndarray

    @staticmethod
    def _build_optimizer_state(*parameters: np.ndarray) -> OptimizerState:
        """Builds optimizer state for parameter updates."""
        raise NotImplementedError

    def _parameter_values(self) -> dict[str, Any]:
        """Returns the config values needed to rebuild the controller."""
        raise NotImplementedError

    @staticmethod
    def _split_grouped_indices(
        groups: IntVector,
        rng: np.random.Generator,
        val_frac: float = 0.2,
    ) -> tuple[IntVector, IntVector]:
        """Splits grouped samples into train and validation ids."""
        raise NotImplementedError

    @staticmethod
    def _make_batches(
        sample_count: int,
        batch_size: int,
        rng: np.random.Generator,
    ) -> list[IntVector]:
        """Builds shuffled mini-batch index slices."""
        raise NotImplementedError

    def select_action(
        self,
        observation: ObservationData,
        environment: PlatformerEnv | None = None,
    ) -> int:
        """Selects the next action for the current environment state."""
        raise NotImplementedError

    def predict_state_value(
        self,
        observation: ObservationData,
        environment: PlatformerEnv | None = None,
    ) -> float:
        """Predicts the terminal value estimate for a state."""
        raise NotImplementedError

    def predict_state_risk(
        self,
        observation: ObservationData,
        environment: PlatformerEnv | None = None,
    ) -> float:
        """Predicts the hazard probability for a state."""
        raise NotImplementedError

    def predict_action_hazard(
        self,
        observation: ObservationData,
        action: int,
        environment: PlatformerEnv | None = None,
    ) -> float:
        """Predicts the hazard probability for a candidate action."""
        raise NotImplementedError

    @staticmethod
    def _heuristic_bootstrap_value(observation: ObservationData) -> float:
        """Returns the heuristic terminal value used by the controller."""
        raise NotImplementedError

    def _f(self, inputs: FloatMatrix) -> ForwardOutput:
        """Runs the terminal value head forward."""
        raise NotImplementedError

    def _forward_state_risk_head(
        self,
        inputs: FloatMatrix,
        temperature: float | None = None,
    ) -> SigmoidHeadOutput:
        """Runs the state-risk head forward."""
        raise NotImplementedError

    def _forward_action_risk_head(
        self,
        inputs: FloatMatrix,
        temperature: float | None = None,
    ) -> SigmoidHeadOutput:
        """Runs the action-risk head forward."""
        raise NotImplementedError

    @staticmethod
    def _binary_class_weights(
        targets: FloatMatrix,
        threshold: float = 0.05,
    ) -> FloatMatrix:
        """Builds inverse-frequency weights for binary targets."""
        raise NotImplementedError

    @staticmethod
    def _weighted_binary_cross_entropy(
        targets: FloatMatrix,
        probs: FloatMatrix,
        weights: FloatMatrix,
    ) -> float:
        """Computes weighted binary cross-entropy."""
        raise NotImplementedError

    @staticmethod
    def _l2_weight_norm(*weights: np.ndarray) -> float:
        """Computes the combined squared L2 norm."""
        raise NotImplementedError

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
        """Computes gradients for a binary risk head."""
        raise NotImplementedError

    def _prepare_fit_state(
        self,
        sample_count: int,
        batch_size: int,
        *configuration_parameters: np.ndarray,
    ) -> tuple[list[dict[str, float]], float, ArrayTuple, OptimizerState, int]:
        """Builds optimizer state and checkpoint trackers."""
        raise NotImplementedError

    def _initialize_value_weights(self, deterministic_seed: int) -> None:
        """Initializes the terminal value head."""
        raise NotImplementedError

    def _initialize_state_risk_weights(self, deterministic_seed: int) -> None:
        """Initializes the state-risk head."""
        raise NotImplementedError

    def _initialize_action_risk_weights(self, deterministic_seed: int) -> None:
        """Initializes the action-risk head."""
        raise NotImplementedError

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
        raise NotImplementedError

    @staticmethod
    def _parameter_snapshot(
        *configuration_parameters: np.ndarray,
    ) -> ArrayTuple:
        """Copies parameter arrays for later restoration."""
        raise NotImplementedError

    @staticmethod
    def _calibrate_head_temperature(
        logits: np.ndarray,
        targets: FloatMatrix,
    ) -> float:
        """Calibrates a sigmoid head temperature."""
        raise NotImplementedError

    def _fit_binary_head(
        self,
        features: FloatMatrix,
        targets: FloatMatrix,
        group_ids: IntVector,
        rng: np.random.Generator,
        epochs: int,
        reinitialize: bool,
        init_fn: Callable[[int], None],
        grad_fn: Callable[[FloatMatrix, FloatMatrix], GradientResult],
        loss_fn: Callable[[FloatMatrix, FloatMatrix], tuple[float, float]],
        update_fn: Callable[..., None],
        forward_fn: Callable[[FloatMatrix, float | None], SigmoidHeadOutput],
        batch_size: int,
        learning_rate: float,
        param_names: tuple[str, str, str, str],
        fitted_attr: str,
        temperature_attr: str,
        history_attr: str,
    ) -> list[dict[str, float]]:
        """Fits one binary hazard head."""
        raise NotImplementedError

    def clone_controller_variant(
        self: ValueMpcBaseT,
        **overrides: Any,
    ) -> ValueMpcBaseT:
        """Clones the controller and applies configuration overrides."""
        raise NotImplementedError

    def encode_observation(
        self,
        observation: ObservationData,
        environment: PlatformerEnv | None = None,
    ) -> FloatMatrix:
        """Encodes one observation into the controller feature vector."""
        raise NotImplementedError

    def encode_action(
        self,
        observation: ObservationData,
        action: int,
        environment: PlatformerEnv | None = None,
    ) -> FloatMatrix:
        """Encodes an observation and action into one feature vector."""
        raise NotImplementedError
