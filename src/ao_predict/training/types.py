"""Public contracts for AO Predict model training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from astropy import units as u


@dataclass(frozen=True)
class ModelTrainingDataConfig:
    """Ordered named feature and target values supplied to model training.

    Public data configurations are passive. ``train_model`` performs coupled
    shape, numerical, unit, schema, and partition validation before fitting.
    Mapping insertion order defines model input and output order. Physical and
    scientifically dimensionless values carry Astropy units directly; plain
    NumPy arrays represent genuinely nonphysical values.

    Attributes:
        features: Unit-free feature names mapped to scalar value arrays.
        targets: Unit-free target names mapped to strictly positive value arrays.
    """

    features: Mapping[str, np.ndarray | u.Quantity]
    targets: Mapping[str, np.ndarray | u.Quantity]


@dataclass(frozen=True)
class TrainModelRequest:
    """Complete request for one restart-safe dense-regression training run.

    Exactly one of ``validation_data``, ``validation_count``, or
    ``validation_fraction`` is required. Automatic partitioning splits whole
    simulations and uses ``split_seed`` when supplied. ``cpu_threads`` changes
    PyTorch's process-wide CPU thread count.

    Attributes:
        model_path: Caller-selected path stem for the model and companions.
        training_data: Explicit training data or the complete automatic-split pool.
        hidden_widths: Dense hidden-layer widths; an empty tuple is linear.
        batch_size: Maximum number of aligned examples in a training batch.
        validation_data: Optional explicit validation partition.
        validation_count: Optional number of simulations withheld automatically.
        validation_fraction: Optional fraction of simulations withheld automatically.
        split_seed: Optional automatic-partition seed.
        overwrite: Replace an existing derived output set and start fresh.
        training_seed: Optional model-initialization and batch-order seed.
        device: Explicit CPU, CUDA, or MPS PyTorch device name.
        cpu_threads: Optional positive process-wide CPU thread count.
        validation_batch_size: Maximum validation examples per execution batch.
        base_learning_rate: Adam learning rate reached after warmup.
        weight_decay: Coupled Adam weight decay.
        warmup_epochs: Number of epochs in linear learning-rate warmup.
        warmup_start_fraction: Initial fraction of ``base_learning_rate``.
        minimum_training_epochs: Early-stopping eligibility gate in epochs.
        validation_check_epochs: Training-set exposures between validation checks.
        learning_rate_reduction_factor: Plateau learning-rate multiplier.
        minimum_learning_rate: Plateau learning-rate floor.
        scheduler_patience_checks: Number of consecutive unsuccessful eligible
            checks that triggers a learning-rate reduction.
        scheduler_minimum_improvement_fraction: Relative validation-objective
            decrease required from the best eligible scheduler objective.
        early_stopping_patience_checks: Number of consecutive unsuccessful
            eligible checks that triggers early-stopping termination.
        early_stopping_minimum_improvement_percent: Absolute validation-Error
            decrease, in percentage points, required from the current
            early-stopping reference.
        maximum_validation_checks: Final scheduled validation-check bound.
    """

    model_path: str | Path
    training_data: ModelTrainingDataConfig
    hidden_widths: tuple[int, ...]
    batch_size: int
    validation_data: ModelTrainingDataConfig | None = None
    validation_count: int | None = None
    validation_fraction: float | None = None
    split_seed: int | None = None
    overwrite: bool = False
    training_seed: int | None = None
    device: str = "cpu"
    cpu_threads: int | None = None
    validation_batch_size: int | None = None
    base_learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    warmup_start_fraction: float = 0.1
    minimum_training_epochs: int = 20
    validation_check_epochs: float = 1.0
    learning_rate_reduction_factor: float = 0.5
    minimum_learning_rate: float = 1.0e-6
    scheduler_patience_checks: int = 4
    scheduler_minimum_improvement_fraction: float = 0.001
    early_stopping_patience_checks: int = 9
    early_stopping_minimum_improvement_percent: float = 0.001
    maximum_validation_checks: int = 1_000


class TrainingTerminationReason(StrEnum):
    """Reason a successful model-training run terminated."""

    EARLY_STOPPING = "early_stopping"
    MAXIMUM_VALIDATION_CHECKS = "maximum_validation_checks"


@dataclass(frozen=True)
class TrainingValidationRecord:
    """Measurements from one complete validation check.

    The training objective covers examples fitted since the preceding check,
    or since the run began for the first record. Validation measurements cover
    the complete validation partition. Learning rates bracket scheduler action.

    Attributes:
        validation_check: One-based completed validation-check number.
        training_epochs: Cumulative training examples seen divided by the
            resolved number of training examples.
        optimizer_updates: Cumulative number of parameter updates.
        training_examples_seen: Cumulative number of examples consumed by
            training batches.
        training_objective: Mean physical relative MSE for training examples
            consumed since the preceding validation check.
        validation_objective: Physical relative MSE over the complete
            validation partition.
        validation_error_percent: Square root of ``validation_objective``
            expressed as a percentage.
        learning_rate_before: Learning rate used before scheduler processing at
            this validation boundary.
        learning_rate_after: Learning rate retained after scheduler processing.
    """

    validation_check: int
    training_epochs: float
    optimizer_updates: int
    training_examples_seen: int
    training_objective: float
    validation_objective: float
    validation_error_percent: float
    learning_rate_before: float
    learning_rate_after: float


@dataclass(frozen=True)
class TrainModelResult:
    """Published result from one complete model-training run.

    The result is returned only after the best model package is validated and
    published, the training log is finalized, and recovery is removed.
    ``validation_mask`` is a read-only AO Predict-owned copy for automatic
    splits and is ``None`` for caller-supplied explicit partitions.

    Attributes:
        model_path: Normalized caller-selected model path stem.
        termination_reason: Successful lifecycle termination condition.
        training_seed: Effective model-initialization and batch-order seed.
        split_seed: Effective automatic-partition seed, or ``None`` for an
            explicit validation partition.
        validation_mask: Read-only automatic validation membership over the
            original simulation pool, or ``None`` for explicit validation.
        optimizer_updates: Total number of completed parameter updates.
        training_examples_seen: Total number of examples consumed by training
            batches.
        validation_checks: Total number of completed validation boundaries.
        best_validation_check: One-based validation check whose model state was
            published.
        best_validation_objective: Complete-partition physical relative MSE for
            the published model state.
        best_model_validation_error_percent: Square root of the best validation
            objective expressed as a percentage.
        validation_history: Ordered measurements for every validation boundary.
    """

    model_path: Path
    termination_reason: TrainingTerminationReason
    training_seed: int
    split_seed: int | None
    validation_mask: np.ndarray | None
    optimizer_updates: int
    training_examples_seen: int
    validation_checks: int
    best_validation_check: int
    best_validation_objective: float
    best_model_validation_error_percent: float
    validation_history: tuple[TrainingValidationRecord, ...]


class ModelTrainingValidationError(ValueError):
    """Raised when a model-training request has invalid coupled inputs."""

    def __init__(self, issues: list[str]):
        self.issues = list(issues)
        message = "Model training validation failed:\n- " + "\n- ".join(self.issues)
        super().__init__(message)


class TrainingRecoveryMismatchError(ValueError):
    """Raised when retained recovery cannot continue the requested run."""

    def __init__(self, mismatches: list[str]):
        self.mismatches = list(mismatches)
        message = "Training recovery does not match the request:\n- " + "\n- ".join(
            self.mismatches
        )
        super().__init__(message)
