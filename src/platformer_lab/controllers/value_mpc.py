"""Learned value and risk MPC controller used in the main experiments."""

from .value_mpc_core import ValueMpcCoreMixin
from .value_mpc_planning import ValueMpcPlanningMixin
from .value_mpc_training import ValueMpcTrainingMixin


class ValueMpcController(
    ValueMpcTrainingMixin,
    ValueMpcPlanningMixin,
    ValueMpcCoreMixin,
):
    """Lightweight value-based MPC controller with learned risk penalties.

    The controller learns:
    1. A terminal value model for short-horizon MPC rollouts.
    2. A state-risk model for hazardous regions.
    3. An action-conditioned risk model for disturbance-aware planning.
    """

    name = "Value MPC"
