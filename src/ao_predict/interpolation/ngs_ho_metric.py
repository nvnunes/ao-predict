"""NGS high-order metric interpolation artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial import Delaunay, QhullError

from ao_predict.simulation.stats import (
    EE_GEOMETRY_ENCIRCLED,
    PsfMetadata,
    compute_psf_ee,
    compute_psf_fwhm,
    compute_psf_sr,
)

from ._core import (
    RbfInterpolationConfig,
    evaluate_scaled_rbf_model,
    field_coordinates,
    load_payload,
    make_scaled_rbf_model,
    require_finite_vector,
    require_positive_scalar,
    require_positive_vector,
    require_pupil,
    save_payload,
    validate_payload_kind,
    validate_psf_array,
    validate_rbf_config,
    zenith_angle_to_airmass,
)


NGS_HO_METRIC_ARTIFACT_KIND = "ao_predict_ngs_ho_metric_interpolator"
NGS_HO_METRIC_ARTIFACT_VERSION = 1
REQUIRED_NGS_HO_METRICS = ("ee", "fwhm_mas", "sr")
OPTIONAL_NGS_HO_METRICS: tuple[str, ...] = ()
_FIELD_ATOL = 1.0e-10


@dataclass(frozen=True)
class NgsHoMetricSamples:
    """Measured NGS high-order metric samples for interpolation.

    Attributes:
        zenith_angle_deg: NGS HO metric zenith-angle planes in degrees.
        x_arcsec: NGS field x-coordinates in arcseconds.
        y_arcsec: NGS field y-coordinates in arcseconds.
        ee: Encircled energy metric with shape ``(planes, points)``.
        fwhm_mas: FWHM metric in milliarcseconds with shape
            ``(planes, points)``.
        sr: Strehl metric with shape ``(planes, points)``.
        provenance: Optional source provenance strings.
    """

    zenith_angle_deg: np.ndarray
    x_arcsec: np.ndarray
    y_arcsec: np.ndarray
    ee: np.ndarray
    fwhm_mas: np.ndarray
    sr: np.ndarray
    provenance: tuple[str, ...] = ()


@dataclass(frozen=True)
class NgsHoPsfSamples:
    """NGS high-order PSF samples used to measure NGS HO metrics.

    Attributes:
        zenith_angle_deg: NGS HO PSF zenith-angle planes in degrees.
        x_arcsec: NGS field x-coordinates in arcseconds.
        y_arcsec: NGS field y-coordinates in arcseconds.
        psfs: PSF array with shape ``(planes, points, y, x)``.
        wavelength_um: Scalar, per-plane, or per-PSF wavelength metadata.
        pixel_scale_mas: Scalar, per-plane, or per-PSF pixel-scale metadata.
        tel_diameter_m: Telescope diameter in meters.
        tel_pupil: Shared two-dimensional telescope pupil.
        provenance: Optional source provenance strings.
    """

    zenith_angle_deg: np.ndarray
    x_arcsec: np.ndarray
    y_arcsec: np.ndarray
    psfs: np.ndarray
    wavelength_um: float | np.ndarray
    pixel_scale_mas: float | np.ndarray
    tel_diameter_m: float
    tel_pupil: np.ndarray
    provenance: tuple[str, ...] = ()


@dataclass(frozen=True)
class NgsHoMetricPrediction:
    """NGS high-order metric prediction returned by artifact evaluation.

    Attributes:
        ee: Predicted encircled-energy values with one value per query point.
        fwhm_mas: Predicted FWHM values in milliarcseconds with one value per
            query point.
        sr: Predicted Strehl-ratio values with one value per query point. SR is
            required for v1 NGS-HO-metric artifacts.
    """

    ee: np.ndarray
    fwhm_mas: np.ndarray
    sr: np.ndarray


@dataclass(frozen=True)
class NgsHoMetricReplaySummary:
    """Source-node replay residual summary for an NGS-HO-metric artifact.

    Attributes:
        metric_rms: RMS residual by metric name for source-node replay.
        metric_max_abs: Maximum absolute residual by metric name for
            source-node replay.
        num_planes: Number of zenith-angle planes replayed.
        num_points: Number of NGS field points per plane.
    """

    metric_rms: dict[str, float]
    metric_max_abs: dict[str, float]
    num_planes: int
    num_points: int


@dataclass(frozen=True)
class NgsHoMetricInterpolator:
    """Versioned NGS-HO-metric interpolation artifact.

    The v1 artifact stores required ``ee``, ``fwhm_mas``, and ``sr`` metric
    models over derived airmass and NGS field coordinates. Queries are
    validated against the stored zenith-angle/airmass range and source field
    support before RBF evaluation.

    Attributes:
        zenith_angle_deg_axis: Supported zenith-angle planes in degrees.
        airmass_axis: Airmass values derived from ``zenith_angle_deg_axis``.
        x_arcsec: Source NGS field x-coordinates in arcseconds.
        y_arcsec: Source NGS field y-coordinates in arcseconds.
        metric_names: Persisted metric names. V1 requires ``ee``,
            ``fwhm_mas``, and ``sr``.
        interpolation_config: RBF configuration used to fit metric models.
        model: Scaled RBF model payload keyed by metric name.
        provenance: Optional source provenance strings.
        builder: Builder metadata persisted with the artifact.
    """

    zenith_angle_deg_axis: np.ndarray
    airmass_axis: np.ndarray
    x_arcsec: np.ndarray
    y_arcsec: np.ndarray
    metric_names: tuple[str, ...]
    interpolation_config: RbfInterpolationConfig
    model: Mapping[str, Any]
    provenance: tuple[str, ...] = ()
    builder: Mapping[str, Any] = field(default_factory=dict)


def build_ngs_ho_metric_samples_from_psfs(samples: NgsHoPsfSamples) -> NgsHoMetricSamples:
    """Measure NGS-HO metrics from NGS-HO PSF samples.

    The measurement path computes AO Predict default SR and FWHM with default
    PSF preprocessing. EE is measured as encircled energy using an aperture
    diameter of ``2 * fwhm_mas`` for each PSF.

    Args:
        samples: NGS-HO PSF sample cube and full ``PsfMetadata`` inputs.

    Returns:
        Metric samples carrying required ``ee``, ``fwhm_mas``, and ``sr``.

    Raises:
        TypeError: If ``samples`` has the wrong contract type.
        ValueError: If sample shapes or PSF metadata are invalid, or if any
            measured metric cannot satisfy the metric-sample contract.
    """

    prepared = _prepare_psf_samples(samples)
    psfs = prepared.psfs.reshape((-1, *prepared.psfs.shape[2:]))
    metadata = PsfMetadata(
        wavelength_um=_expand_psf_metadata(prepared.wavelength_um, prepared.psfs.shape[:2], label="wavelength_um"),
        pixel_scale_mas=_expand_psf_metadata(prepared.pixel_scale_mas, prepared.psfs.shape[:2], label="pixel_scale_mas"),
        tel_diameter_m=float(prepared.tel_diameter_m),
        tel_pupil=np.asarray(prepared.tel_pupil, dtype=np.float32),
    )
    sr = np.asarray(compute_psf_sr(psfs, metadata, preprocess="default"), dtype=float)
    fwhm_mas = np.asarray(compute_psf_fwhm(psfs, metadata, preprocess="default"), dtype=float)
    ee_aperture_diameters_mas = np.asarray(2.0 * fwhm_mas, dtype=np.float32).reshape(-1, 1)
    ee = np.asarray(
        compute_psf_ee(
            psfs,
            metadata,
            ee_apertures_mas=ee_aperture_diameters_mas,
            ee_geometry=EE_GEOMETRY_ENCIRCLED,
            preprocess="default",
        ),
        dtype=float,
    ).reshape(-1)
    n_planes, n_points = prepared.psfs.shape[:2]
    return NgsHoMetricSamples(
        zenith_angle_deg=np.asarray(prepared.zenith_angle_deg, dtype=float),
        x_arcsec=np.asarray(prepared.x_arcsec, dtype=float),
        y_arcsec=np.asarray(prepared.y_arcsec, dtype=float),
        ee=ee.reshape(n_planes, n_points),
        fwhm_mas=fwhm_mas.reshape(n_planes, n_points),
        sr=sr.reshape(n_planes, n_points),
        provenance=tuple(prepared.provenance),
    )


def build_ngs_ho_metric_interpolator_from_psfs(
    samples: NgsHoPsfSamples,
    *,
    interpolation_config: RbfInterpolationConfig | None = None,
) -> NgsHoMetricInterpolator:
    """Build an NGS-HO-metric artifact from NGS-HO PSF samples.

    This is the high-level PSF path: PSFs are first measured into
    ``NgsHoMetricSamples``, then the same metric interpolator builder used for
    direct metric input is applied.

    Args:
        samples: NGS-HO PSF samples.
        interpolation_config: Optional scaled RBF configuration for metric
            interpolation.

    Returns:
        Validated NGS-HO-metric interpolation artifact.
    """

    return build_ngs_ho_metric_interpolator(
        build_ngs_ho_metric_samples_from_psfs(samples),
        interpolation_config=interpolation_config,
    )


def build_ngs_ho_metric_interpolator(
    samples: NgsHoMetricSamples,
    *,
    interpolation_config: RbfInterpolationConfig | None = None,
) -> NgsHoMetricInterpolator:
    """Build an NGS-HO-metric interpolation artifact from metric samples.

    Direct metric input must already provide ``ee``, ``fwhm_mas``, and ``sr``
    over a rectangular ``zenith_angle_deg x field-point`` sample layout.

    Args:
        samples: Measured NGS-HO metric samples.
        interpolation_config: Optional scaled RBF configuration. When omitted,
            the shared RBF default is used.

    Returns:
        Validated in-memory NGS-HO-metric artifact.

    Raises:
        TypeError: If ``samples`` or ``interpolation_config`` has the wrong
            contract type.
        ValueError: If metric arrays, field coordinates, or RBF inputs are
            invalid.
    """

    config = validate_rbf_config(interpolation_config or RbfInterpolationConfig())
    prepared = _prepare_metric_samples(samples)
    coordinates = _training_coordinates(prepared)
    values_by_name = {
        "ee": np.asarray(prepared.ee, dtype=float).reshape(-1),
        "fwhm_mas": np.asarray(prepared.fwhm_mas, dtype=float).reshape(-1),
        "sr": np.asarray(prepared.sr, dtype=float).reshape(-1),
    }
    model = make_scaled_rbf_model(coordinates, values_by_name, config)
    return NgsHoMetricInterpolator(
        zenith_angle_deg_axis=np.asarray(prepared.zenith_angle_deg, dtype=float),
        airmass_axis=np.asarray(zenith_angle_to_airmass(prepared.zenith_angle_deg), dtype=float),
        x_arcsec=np.asarray(prepared.x_arcsec, dtype=float),
        y_arcsec=np.asarray(prepared.y_arcsec, dtype=float),
        metric_names=tuple(values_by_name),
        interpolation_config=config,
        model=model,
        provenance=tuple(prepared.provenance),
        builder={
            "name": "ao_predict.interpolation.ngs_ho_metric",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def evaluate_ngs_ho_metric_interpolator(
    interpolator: NgsHoMetricInterpolator,
    *,
    zenith_angle_deg: float | np.ndarray,
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
) -> NgsHoMetricPrediction:
    """Evaluate an NGS-HO-metric artifact at supported query points.

    Args:
        interpolator: Loaded or newly built NGS-HO-metric artifact.
        zenith_angle_deg: Scalar or per-point zenith angles in degrees.
        x_arcsec: Query NGS field x-coordinates in arcseconds.
        y_arcsec: Query NGS field y-coordinates in arcseconds, matching
            ``x_arcsec`` length.

    Returns:
        Predicted ``ee``, ``fwhm_mas``, and ``sr`` arrays.

    Raises:
        ValueError: If the query is outside the artifact airmass range or field
            support, or if interpolated metric values violate metric bounds.
    """

    validate_ngs_ho_metric_query(
        interpolator,
        zenith_angle_deg=zenith_angle_deg,
        x_arcsec=x_arcsec,
        y_arcsec=y_arcsec,
    )
    coordinates = _query_coordinates(zenith_angle_deg, x_arcsec, y_arcsec)
    output = evaluate_scaled_rbf_model(interpolator.model, coordinates)
    _validate_predicted_metrics(output)
    return NgsHoMetricPrediction(
        ee=np.asarray(output["ee"], dtype=float),
        fwhm_mas=np.asarray(output["fwhm_mas"], dtype=float),
        sr=np.asarray(output["sr"], dtype=float),
    )


def validate_ngs_ho_metric_query(
    interpolator: NgsHoMetricInterpolator,
    *,
    zenith_angle_deg: float | np.ndarray,
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
) -> None:
    """Validate NGS-HO-metric query support without returning predictions.

    The check broadcasts scalar zenith angle over field coordinates, validates
    the derived airmass against the artifact support, and validates NGS field
    coordinates against the source field support.

    Raises:
        TypeError: If ``interpolator`` has the wrong contract type.
        ValueError: If query shapes are invalid or the requested query lies
            outside artifact support.
    """

    if not isinstance(interpolator, NgsHoMetricInterpolator):
        raise TypeError("interpolator must be an NgsHoMetricInterpolator instance.")
    _validate_interpolator(interpolator)
    zenith, x, y = _query_values(zenith_angle_deg, x_arcsec, y_arcsec)
    _validate_zenith_support(interpolator, zenith)
    _validate_field_support(interpolator, x, y)


def validate_ngs_ho_metric_interpolator(interpolator: NgsHoMetricInterpolator) -> None:
    """Validate an NGS-HO-metric artifact contract.

    This validator checks the in-memory metadata/model consistency used by
    save, load, replay, and evaluation paths. In v1, ``metric_names`` and model
    outputs must match exactly and include ``ee``, ``fwhm_mas``, and ``sr``.

    Raises:
        TypeError: If ``interpolator`` has the wrong contract type.
        ValueError: If required metadata or model fields are malformed.
    """

    if not isinstance(interpolator, NgsHoMetricInterpolator):
        raise TypeError("interpolator must be an NgsHoMetricInterpolator instance.")
    _validate_interpolator(interpolator)


def replay_ngs_ho_metric_interpolator(
    interpolator: NgsHoMetricInterpolator,
    samples: NgsHoMetricSamples,
) -> NgsHoMetricReplaySummary:
    """Replay an NGS-HO-metric artifact at source sample nodes.

    Args:
        interpolator: NGS-HO-metric artifact to validate.
        samples: Source metric samples expected to match the artifact support.

    Returns:
        Per-metric RMS and maximum absolute source-node residuals.

    Raises:
        ValueError: If source replay queries are unsupported or predicted
            metrics violate metric bounds.
    """

    prepared = _prepare_metric_samples(samples)
    prediction = evaluate_ngs_ho_metric_interpolator(
        interpolator,
        zenith_angle_deg=np.repeat(prepared.zenith_angle_deg, prepared.x_arcsec.size),
        x_arcsec=np.tile(prepared.x_arcsec, prepared.zenith_angle_deg.size),
        y_arcsec=np.tile(prepared.y_arcsec, prepared.zenith_angle_deg.size),
    )
    reference = {
        "ee": np.asarray(prepared.ee, dtype=float).reshape(-1),
        "fwhm_mas": np.asarray(prepared.fwhm_mas, dtype=float).reshape(-1),
    }
    measured = {
        "ee": prediction.ee,
        "fwhm_mas": prediction.fwhm_mas,
    }
    reference["sr"] = np.asarray(prepared.sr, dtype=float).reshape(-1)
    measured["sr"] = prediction.sr
    return NgsHoMetricReplaySummary(
        metric_rms={
            name: float(np.sqrt(np.mean((measured[name] - reference[name]) ** 2)))
            for name in reference
        },
        metric_max_abs={
            name: float(np.max(np.abs(measured[name] - reference[name])))
            for name in reference
        },
        num_planes=int(prepared.zenith_angle_deg.size),
        num_points=int(prepared.x_arcsec.size),
    )


def save_ngs_ho_metric_interpolator(
    interpolator: NgsHoMetricInterpolator,
    path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Save an NGS-HO-metric interpolation artifact.

    Args:
        interpolator: Artifact to validate and persist.
        path: Destination pickle payload path.
        overwrite: When ``False``, existing files are rejected.

    Raises:
        FileExistsError: If ``path`` exists and ``overwrite`` is ``False``.
        TypeError: If ``interpolator`` has the wrong contract type.
        ValueError: If the artifact fails validation before persistence.
    """

    save_payload(_ngs_to_payload(interpolator), Path(path), overwrite=overwrite)


def load_ngs_ho_metric_interpolator(path: Path) -> NgsHoMetricInterpolator:
    """Load and validate an NGS-HO-metric interpolation artifact.

    Args:
        path: Pickle payload path produced by
            ``save_ngs_ho_metric_interpolator``.

    Returns:
        Validated in-memory NGS-HO-metric artifact.

    Raises:
        ValueError: If the payload kind, version, metadata, or model contract
            does not match the v1 artifact schema.
    """

    return _ngs_from_payload(load_payload(Path(path)))


def _prepare_metric_samples(samples: NgsHoMetricSamples) -> NgsHoMetricSamples:
    if not isinstance(samples, NgsHoMetricSamples):
        raise TypeError("samples must be an NgsHoMetricSamples instance.")
    zenith_angle_deg = require_finite_vector(samples.zenith_angle_deg, label="zenith_angle_deg")
    coordinates = field_coordinates(samples.x_arcsec, samples.y_arcsec)
    if zenith_angle_deg.size == 0:
        raise ValueError("At least one zenith-angle plane is required.")
    expected_shape = (zenith_angle_deg.size, coordinates.shape[0])
    ee = _prepare_metric_array(samples.ee, label="ee", expected_shape=expected_shape)
    fwhm_mas = _prepare_metric_array(samples.fwhm_mas, label="fwhm_mas", expected_shape=expected_shape)
    sr = _prepare_metric_array(samples.sr, label="sr", expected_shape=expected_shape)
    return NgsHoMetricSamples(
        zenith_angle_deg=zenith_angle_deg,
        x_arcsec=np.asarray(samples.x_arcsec, dtype=float).reshape(-1),
        y_arcsec=np.asarray(samples.y_arcsec, dtype=float).reshape(-1),
        ee=ee,
        fwhm_mas=fwhm_mas,
        sr=sr,
        provenance=tuple(str(value) for value in samples.provenance),
    )


def _prepare_psf_samples(samples: NgsHoPsfSamples) -> NgsHoPsfSamples:
    if not isinstance(samples, NgsHoPsfSamples):
        raise TypeError("samples must be an NgsHoPsfSamples instance.")
    zenith_angle_deg = require_finite_vector(samples.zenith_angle_deg, label="zenith_angle_deg")
    coordinates = field_coordinates(samples.x_arcsec, samples.y_arcsec)
    psfs = validate_psf_array(samples.psfs, label="psfs", ndim=4)
    if psfs.shape[:2] != (zenith_angle_deg.size, coordinates.shape[0]):
        raise ValueError(
            "psfs shape must be (planes, points, y, x); "
            f"got {psfs.shape} for {zenith_angle_deg.size} planes and {coordinates.shape[0]} points."
        )
    # Validate metadata now; expansion is repeated during metric measurement.
    _expand_psf_metadata(samples.wavelength_um, psfs.shape[:2], label="wavelength_um")
    _expand_psf_metadata(samples.pixel_scale_mas, psfs.shape[:2], label="pixel_scale_mas")
    return NgsHoPsfSamples(
        zenith_angle_deg=zenith_angle_deg,
        x_arcsec=np.asarray(samples.x_arcsec, dtype=float).reshape(-1),
        y_arcsec=np.asarray(samples.y_arcsec, dtype=float).reshape(-1),
        psfs=psfs,
        wavelength_um=samples.wavelength_um,
        pixel_scale_mas=samples.pixel_scale_mas,
        tel_diameter_m=require_positive_scalar(samples.tel_diameter_m, label="tel_diameter_m"),
        tel_pupil=require_pupil(samples.tel_pupil),
        provenance=tuple(str(value) for value in samples.provenance),
    )


def _prepare_metric_array(value: Any, *, label: str, expected_shape: tuple[int, int]) -> np.ndarray:
    if value is None:
        raise ValueError(f"{label} is required.")
    metric = np.asarray(value, dtype=float)
    if metric.shape != expected_shape:
        raise ValueError(f"{label} must have shape {expected_shape}, got {metric.shape}.")
    if not np.all(np.isfinite(metric)):
        raise ValueError(f"{label} must contain only finite values.")
    if np.any(metric <= 0.0):
        raise ValueError(f"{label} must contain only values > 0.")
    if label == "ee" and np.any(metric > 1.0):
        raise ValueError("ee must not contain values > 1.")
    return metric


def _expand_psf_metadata(value: Any, shape: tuple[int, int], *, label: str) -> float | np.ndarray:
    n_planes, n_points = shape
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        scalar = require_positive_scalar(array, label=label)
        return scalar
    if array.shape == (n_planes,):
        vector = require_positive_vector(array, label=label, length=n_planes)
        return np.repeat(vector, n_points)
    if array.shape == shape:
        flat = require_positive_vector(array.reshape(-1), label=label, length=n_planes * n_points)
        return flat
    flat = np.asarray(array, dtype=float).reshape(-1)
    if flat.size == n_planes * n_points:
        return require_positive_vector(flat, label=label, length=n_planes * n_points)
    raise ValueError(f"{label} must be scalar, per-plane, or per-PSF; got shape {array.shape}.")


def _training_coordinates(samples: NgsHoMetricSamples) -> np.ndarray:
    airmasses = np.asarray(zenith_angle_to_airmass(samples.zenith_angle_deg), dtype=float)
    n_planes = samples.zenith_angle_deg.size
    n_points = samples.x_arcsec.size
    return np.column_stack(
        [
            np.repeat(airmasses, n_points),
            np.tile(samples.x_arcsec, n_planes),
            np.tile(samples.y_arcsec, n_planes),
        ]
    )


def _query_values(zenith_angle_deg: Any, x_arcsec: Any, y_arcsec: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = require_finite_vector(x_arcsec, label="x_arcsec")
    if x.size == 0:
        raise ValueError("At least one query field coordinate is required.")
    y = require_finite_vector(y_arcsec, label="y_arcsec", length=x.size)
    zenith = require_finite_vector(zenith_angle_deg, label="zenith_angle_deg")
    if zenith.size == 1 and x.size != 1:
        zenith = np.full(x.shape, float(zenith[0]), dtype=float)
    if zenith.size != x.size:
        raise ValueError(
            "zenith_angle_deg, x_arcsec, and y_arcsec must have matching lengths; "
            f"got {zenith.size}, {x.size}, {y.size}."
        )
    return zenith, x, y


def _query_coordinates(zenith_angle_deg: Any, x_arcsec: Any, y_arcsec: Any) -> np.ndarray:
    zenith, x, y = _query_values(zenith_angle_deg, x_arcsec, y_arcsec)
    return np.column_stack([zenith_angle_to_airmass(zenith), x, y])


def _validate_zenith_support(interpolator: NgsHoMetricInterpolator, zenith_angle_deg: np.ndarray) -> None:
    airmass = np.asarray(zenith_angle_to_airmass(zenith_angle_deg), dtype=float)
    axis = np.asarray(interpolator.airmass_axis, dtype=float).reshape(-1)
    minimum = float(np.min(axis))
    maximum = float(np.max(axis))
    if np.any((airmass < minimum) & ~np.isclose(airmass, minimum, rtol=0.0, atol=_FIELD_ATOL)):
        raise ValueError(f"airmass query is below the supported range minimum {minimum}.")
    if np.any((airmass > maximum) & ~np.isclose(airmass, maximum, rtol=0.0, atol=_FIELD_ATOL)):
        raise ValueError(f"airmass query is above the supported range maximum {maximum}.")


def _validate_field_support(interpolator: NgsHoMetricInterpolator, x_arcsec: np.ndarray, y_arcsec: np.ndarray) -> None:
    source = np.unique(field_coordinates(interpolator.x_arcsec, interpolator.y_arcsec), axis=0)
    query = np.column_stack([x_arcsec, y_arcsec])
    if source.shape[0] < 3 or np.linalg.matrix_rank(source - np.mean(source, axis=0), tol=_FIELD_ATOL) < 2:
        _validate_degenerate_field_support(source, query)
        return
    try:
        simplex = Delaunay(source).find_simplex(query, tol=_FIELD_ATOL)
    except QhullError:
        _validate_degenerate_field_support(source, query)
        return
    if np.any(simplex < 0):
        raise ValueError("NGS field query coordinates are outside the source field support.")


def _validate_degenerate_field_support(source: np.ndarray, query: np.ndarray) -> None:
    if source.shape[0] == 1:
        if not np.all(np.isclose(query, source[0], rtol=0.0, atol=_FIELD_ATOL)):
            raise ValueError("NGS field query coordinates are outside the single-point source field support.")
        return

    center = np.mean(source, axis=0)
    _, singular_values, vh = np.linalg.svd(source - center, full_matrices=False)
    if singular_values[0] <= _FIELD_ATOL:
        raise ValueError("NGS source field support is degenerate.")
    direction = vh[0]
    source_projection = (source - center) @ direction
    query_centered = query - center
    query_projection = query_centered @ direction
    perpendicular = query_centered - np.outer(query_projection, direction)
    if np.any(np.linalg.norm(perpendicular, axis=1) > _FIELD_ATOL):
        raise ValueError("NGS field query coordinates are outside the line source field support.")
    if np.any(query_projection < np.min(source_projection) - _FIELD_ATOL) or np.any(
        query_projection > np.max(source_projection) + _FIELD_ATOL
    ):
        raise ValueError("NGS field query coordinates are outside the line source field support.")


def _validate_predicted_metrics(metrics: Mapping[str, np.ndarray]) -> None:
    for name, values in metrics.items():
        values = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Interpolator produced non-finite {name} values.")
        if np.any(values <= 0.0):
            raise ValueError(f"Interpolator produced non-positive {name} values.")
        if name == "ee" and np.any(values > 1.0):
            raise ValueError("Interpolator produced ee values > 1.")


def _ngs_to_payload(interpolator: NgsHoMetricInterpolator) -> dict[str, Any]:
    if not isinstance(interpolator, NgsHoMetricInterpolator):
        raise TypeError("interpolator must be an NgsHoMetricInterpolator instance.")
    _validate_interpolator(interpolator)
    return {
        "kind": NGS_HO_METRIC_ARTIFACT_KIND,
        "version": NGS_HO_METRIC_ARTIFACT_VERSION,
        "builder": dict(interpolator.builder),
        "interpolation_config": interpolator.interpolation_config,
        "metadata": {
            "zenith_angle_deg_axis": np.asarray(interpolator.zenith_angle_deg_axis, dtype=float),
            "airmass_axis": np.asarray(interpolator.airmass_axis, dtype=float),
            "x_arcsec": np.asarray(interpolator.x_arcsec, dtype=float),
            "y_arcsec": np.asarray(interpolator.y_arcsec, dtype=float),
            "metric_names": tuple(interpolator.metric_names),
            "provenance": tuple(interpolator.provenance),
        },
        "model": dict(interpolator.model),
    }


def _ngs_from_payload(payload: Mapping[str, Any]) -> NgsHoMetricInterpolator:
    validate_payload_kind(payload, kind=NGS_HO_METRIC_ARTIFACT_KIND, version=NGS_HO_METRIC_ARTIFACT_VERSION)
    metadata = dict(payload.get("metadata", {}))
    config = validate_rbf_config(payload.get("interpolation_config", RbfInterpolationConfig()))
    metric_names = tuple(str(value) for value in metadata.get("metric_names", ()))
    _validate_metric_names(metric_names, label="NGS HO metric artifact")
    interpolator = NgsHoMetricInterpolator(
        zenith_angle_deg_axis=require_finite_vector(
            metadata.get("zenith_angle_deg_axis"),
            label="metadata.zenith_angle_deg_axis",
        ),
        airmass_axis=require_finite_vector(metadata.get("airmass_axis"), label="metadata.airmass_axis"),
        x_arcsec=require_finite_vector(metadata.get("x_arcsec"), label="metadata.x_arcsec"),
        y_arcsec=require_finite_vector(metadata.get("y_arcsec"), label="metadata.y_arcsec"),
        metric_names=metric_names,
        interpolation_config=config,
        model=dict(payload.get("model", {})),
        provenance=tuple(str(value) for value in metadata.get("provenance", ())),
        builder=dict(payload.get("builder", {})),
    )
    _validate_interpolator(interpolator)
    return interpolator


def _validate_interpolator(interpolator: NgsHoMetricInterpolator) -> None:
    if interpolator.zenith_angle_deg_axis.size == 0:
        raise ValueError("zenith_angle_deg_axis must not be empty.")
    if interpolator.airmass_axis.shape != interpolator.zenith_angle_deg_axis.shape:
        raise ValueError("airmass_axis shape must match zenith_angle_deg_axis shape.")
    if interpolator.x_arcsec.shape != interpolator.y_arcsec.shape:
        raise ValueError("x_arcsec and y_arcsec shapes must match.")
    _validate_metric_names(interpolator.metric_names, label="NGS HO metric artifact")
    model_names = set(dict(interpolator.model.get("models", {})))
    metric_names = set(interpolator.metric_names)
    if model_names != metric_names:
        missing = metric_names - model_names
        extra = model_names - metric_names
        details = []
        if missing:
            details.append(f"missing model metrics: {', '.join(sorted(missing))}")
        if extra:
            details.append(f"extra model metrics: {', '.join(sorted(extra))}")
        raise ValueError(f"NGS HO metric artifact model does not match metric_names ({'; '.join(details)}).")


def _validate_metric_names(metric_names: tuple[str, ...], *, label: str) -> None:
    if len(set(metric_names)) != len(metric_names):
        raise ValueError(f"{label} contains duplicate metrics.")
    missing = set(REQUIRED_NGS_HO_METRICS) - set(metric_names)
    if missing:
        raise ValueError(f"{label} is missing required metrics: {', '.join(sorted(missing))}.")
    unsupported = set(metric_names) - set(REQUIRED_NGS_HO_METRICS) - set(OPTIONAL_NGS_HO_METRICS)
    if unsupported:
        raise ValueError(f"{label} contains unsupported metrics: {', '.join(sorted(unsupported))}.")
