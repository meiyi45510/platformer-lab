"""Artifact serialization helpers for experiment checkpoints."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .controllers.value_mpc import ValueMpcController
from .controllers.value_mpc_core import MODEL_PARAMETER_FIELDS

SNAPSHOT_SCHEMA_VERSION = 2


# =============================================================================
# JSON serialization helpers.
# =============================================================================
def to_json_compatible(value: Any) -> Any:
    """Recursively converts NumPy-heavy data into JSON-serializable objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [to_json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): to_json_compatible(item) for key, item in value.items()
        }
    return value


def write_json_data(path: Path, obj: Any) -> None:
    """Serializes structured data to JSON with readable indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            to_json_compatible(obj),
            handle,
            ensure_ascii=False,
            indent=2,
        )


VALUE_PARAMETER_FIELDS = MODEL_PARAMETER_FIELDS[0:4]
STATE_RISK_PARAMETER_FIELDS = MODEL_PARAMETER_FIELDS[4:8]
ACTION_RISK_PARAMETER_FIELDS = MODEL_PARAMETER_FIELDS[8:12]


def capture_array_parameters(
    controller: ValueMpcController,
    names: Sequence[str],
) -> dict[str, np.ndarray]:
    """Copies selected array-like controller attributes into a dictionary."""
    return {name: getattr(controller, name) for name in names}


def restore_array_parameters(
    target: ValueMpcController,
    data: Mapping[str, Any],
    names: Sequence[str],
) -> None:
    """Restores selected array parameters onto a controller instance."""
    for field_name in names:
        setattr(target, field_name, np.array(data[field_name], copy=True))


def archive_field(archive: Mapping[str, Any], name: str) -> Any:
    """Reads one field from a controller archive with scalar unboxing."""
    value = archive[name]
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def create_risk_clone(
    controller: ValueMpcController,
    label: str = "Risk-Aware Value MPC",
) -> ValueMpcController:
    """Builds the lightweight risk-aware controller variant used in reports."""
    return controller.clone_controller_variant(
        risk_penalty=float(controller.risk_clone_state_penalty),
        action_risk_penalty=float(controller.risk_clone_action_penalty),
        beam_width=min(3, controller.beam_width),
        use_adaptive_planning=False,
        use_disturbance_aware_planning=False,
        disturbance_assumed_prob=0.0,
        disturbance_radius=0.0,
        label=label,
    )


def build_controller_runtime_metadata(
    controller: ValueMpcController,
) -> dict[str, Any]:
    """Builds the runtime metadata persisted beside controller weights."""
    return {
        "target_mean": float(controller.target_mean),
        "target_std": float(controller.target_std),
        "risk_temperature": float(controller.risk_temperature),
        "action_risk_temperature": float(
            controller.action_risk_temperature
        ),
        "fitted": bool(controller.fitted),
        "risk_fitted": bool(controller.risk_fitted),
        "action_risk_fitted": bool(controller.action_risk_fitted),
    }


def build_controller_snapshot_metadata(
    controller: ValueMpcController,
    risk_controller: ValueMpcController | None = None,
) -> dict[str, Any]:
    """Builds the serialized metadata block stored beside controller weights."""
    risk_controller = risk_controller or create_risk_clone(controller)
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "label": str(controller.name),
        "config": controller.configuration_parameters(),
        "runtime": build_controller_runtime_metadata(controller),
        "risk_clone": {
            "risk_penalty": float(risk_controller.risk_penalty),
            "action_risk_penalty": float(
                risk_controller.action_risk_penalty
            ),
        },
    }


def serialize_snapshot_metadata_json(
    controller: ValueMpcController,
    risk_controller: ValueMpcController | None = None,
) -> np.ndarray:
    """Serializes checkpoint metadata into a compact JSON string."""
    return np.array(
        json.dumps(
            to_json_compatible(
                build_controller_snapshot_metadata(
                    controller, risk_controller
                )
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def capture_controller_state(
    controller: ValueMpcController,
    risk_controller: ValueMpcController | None = None,
) -> dict[str, Any]:
    """Collects controller weights and metadata for snapshot persistence."""
    return (
        capture_array_parameters(controller, VALUE_PARAMETER_FIELDS)
        | capture_array_parameters(controller, STATE_RISK_PARAMETER_FIELDS)
        | capture_array_parameters(controller, ACTION_RISK_PARAMETER_FIELDS)
        | {
            "metadata_json": serialize_snapshot_metadata_json(
                controller, risk_controller
            )
        }
    )


def save_controller_snapshot(
    path: Path,
    controller: ValueMpcController,
    risk_controller: ValueMpcController | None = None,
) -> None:
    """Persists controller weights and metadata for reproducible reuse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **capture_controller_state(controller, risk_controller))


def controller_configuration(controller: ValueMpcController) -> dict[str, Any]:
    """Returns the controller settings that should appear in saved metadata."""
    return controller.configuration_parameters() | build_controller_runtime_metadata(
        controller
    ) | {
        "label": str(controller.name)
    }


def load_controller_snapshot(
    path: Path,
    label: str | None = None,
) -> ValueMpcController:
    """Restores a controller archive written by the current codebase."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive_field(archive, "metadata_json")))
        if int(metadata["schema_version"]) != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(f"unsupported checkpoint schema: {path}")
        controller = ValueMpcController(
            **metadata["config"],
            label=(
                str(metadata["label"])
                if label is None
                else label
            ),
        )
        restore_array_parameters(controller, archive, VALUE_PARAMETER_FIELDS)
        restore_array_parameters(
            controller, archive, STATE_RISK_PARAMETER_FIELDS
        )
        restore_array_parameters(
            controller, archive, ACTION_RISK_PARAMETER_FIELDS
        )
        runtime = metadata["runtime"]
        controller.risk_temperature = float(runtime["risk_temperature"])
        controller.action_risk_temperature = float(
            runtime["action_risk_temperature"]
        )
        controller.sanitize_loaded_parameters()
        controller.target_mean = float(runtime["target_mean"])
        controller.target_std = float(runtime["target_std"])
        controller.fitted = bool(runtime["fitted"])
        controller.risk_fitted = bool(runtime["risk_fitted"])
        controller.action_risk_fitted = bool(
            runtime["action_risk_fitted"]
        )
        risk_clone = metadata["risk_clone"]
        controller.risk_clone_state_penalty = float(
            risk_clone["risk_penalty"]
        )
        controller.risk_clone_action_penalty = float(
            risk_clone["action_risk_penalty"]
        )
        return controller
