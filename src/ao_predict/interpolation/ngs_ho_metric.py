"""NGS high-order metric interpolation artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from astropy import units as u
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import Delaunay, QhullError

from ao_predict.simulation.stats import (
    EE_GEOMETRY_ENCIRCLED,
    PsfMetadata,
    compute_psf_ee,
    compute_psf_fwhm,
    compute_psf_sr,
)
from ao_predict._units import quantity_value, require_quantity

from ._core import (
    RegularGridInterpolationConfig,
    RbfInterpolationConfig,
    axis_index,
    evaluate_scaled_rbf_model,
    field_coordinates,
    grid_field_values,
    load_payload,
    make_scaled_rbf_model,
    rectangular_field_axes,
    require_finite_vector,
    require_positive_scalar,
    require_positive_vector,
    require_pupil,
    save_payload,
    snap_rectangular_field_query,
    unique_sorted,
    validate_rectangular_field_query,
    validate_psf_array,
    validate_regular_grid_config,
    validate_rbf_config,
    zenith_angle_to_airmass,
)


NGS_HO_METRIC_ARTIFACT_KIND = "ao_predict_ngs_ho_metric_interpolator"
NGS_HO_METRIC_ARTIFACT_VERSION = 1
NGS_HO_METRIC_STRATEGY_REGULAR_GRID = "regular_grid"
NGS_HO_METRIC_STRATEGY_RBF = "rbf"
NGS_COORD_AIRMASS = "airmass"
NGS_COORD_X = "x"
NGS_COORD_Y = "y"
NGS_REGULAR_GRID_FIELD_ORDER = (NGS_COORD_Y, NGS_COORD_X)
NGS_RBF_FIELD_ORDER = (NGS_COORD_X, NGS_COORD_Y)
NGS_REGULAR_GRID_COORDINATE_ORDERS = frozenset(
    {
        NGS_REGULAR_GRID_FIELD_ORDER,
        (NGS_COORD_AIRMASS, *NGS_REGULAR_GRID_FIELD_ORDER),
    }
)
NGS_RBF_COORDINATE_ORDERS = frozenset(
    {
        NGS_RBF_FIELD_ORDER,
        (NGS_COORD_AIRMASS, *NGS_RBF_FIELD_ORDER),
    }
)
REQUIRED_NGS_HO_METRICS = ("ee", "fwhm", "sr")
OPTIONAL_NGS_HO_METRICS: tuple[str, ...] = ()
_FIELD_ATOL = 1.0e-10


@dataclass(frozen=True, kw_only=True)
class NgsHoMetricSamples:
    """Measured NGS high-order metric samples for interpolation.

    ``zenith_angle`` is optional. Multiple unique zenith values activate an
    airmass interpolation coordinate. A scalar or single unique zenith value is
    stored as fixed support metadata.

    Attributes:
        x: NGS field x-coordinates compatible with arcseconds.
        y: NGS field y-coordinates compatible with arcseconds.
        ee: Dimensionless enclosed-energy quantities.
        fwhm: FWHM quantities compatible with milliarcseconds.
        sr: Dimensionless Strehl-ratio quantities.
        zenith_angle: Optional scalar or per-plane angle compatible with
            degrees.
        provenance: Optional source-provenance strings.
    """

    x: u.Quantity
    y: u.Quantity
    ee: u.Quantity
    fwhm: u.Quantity
    sr: u.Quantity
    zenith_angle: u.Quantity | None = None
    provenance: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class NgsHoPsfSamples:
    """NGS high-order PSF samples used to measure NGS HO metrics.

    Wavelength remains PSF-stat metadata only; it is never an interpolation
    coordinate for NGS-HO metric artifacts.

    Attributes:
        x: NGS field x-coordinates compatible with arcseconds.
        y: NGS field y-coordinates compatible with arcseconds.
        psfs: PSF array with shape ``(points, y, x)`` for fixed-plane samples
            or ``(planes, points, y, x)`` for multi-plane samples.
        wavelength: PSF wavelength compatible with microns.
        pixel_scale: PSF pixel scale compatible with milliarcseconds.
        tel_diameter: Telescope diameter compatible with metres.
        tel_pupil: Dimensionless two-dimensional telescope pupil.
        zenith_angle: Optional scalar or per-plane angle compatible with
            degrees.
        provenance: Optional source-provenance strings.
    """

    x: u.Quantity
    y: u.Quantity
    psfs: np.ndarray
    wavelength: u.Quantity
    pixel_scale: u.Quantity
    tel_diameter: u.Quantity
    tel_pupil: u.Quantity
    zenith_angle: u.Quantity | None = None
    provenance: tuple[str, ...] = ()


@dataclass(frozen=True)
class NgsHoMetricPrediction:
    """NGS high-order metric prediction returned by artifact evaluation.

    Attributes:
        ee: Dimensionless enclosed-energy quantities at the query points.
        fwhm: FWHM quantities in milliarcseconds at the query points.
        sr: Dimensionless Strehl-ratio quantities at the query points.
    """

    ee: u.Quantity
    fwhm: u.Quantity
    sr: u.Quantity


@dataclass(frozen=True)
class NgsHoMetricReplaySummary:
    """Source-node replay residual summary for an NGS-HO-metric artifact.

    Metric residual mappings retain each metric's units: enclosed energy and
    Strehl ratio are dimensionless, and FWHM is in milliarcseconds.

    Attributes:
        metric_rms: Root-mean-square source-node residual by metric name.
        metric_max_abs: Maximum absolute source-node residual by metric name.
        num_planes: Number of replayed physical-support planes.
        num_points: Number of field points per plane.
    """

    metric_rms: dict[str, u.Quantity]
    metric_max_abs: dict[str, u.Quantity]
    num_planes: int
    num_points: int


@dataclass(frozen=True)
class NgsHoMetricInterpolator:
    """Versioned NGS-HO-metric interpolation artifact.

    Attributes:
        coordinate_order: Active interpolation-coordinate names.
        zenith_angle_axis: Supported zenith angles in degrees.
        airmass_axis: Dimensionless airmass values corresponding to the zenith
            axis.
        x: Supported field x-coordinates in arcseconds.
        y: Supported field y-coordinates in arcseconds.
        metric_names: Ordered names of the interpolated metrics.
        interpolation_config: Interpolation-strategy configuration.
        model: Validated strategy-specific interpolation state.
        provenance: Optional source-provenance strings.
        builder: Artifact-builder provenance.
    """

    coordinate_order: tuple[str, ...]
    zenith_angle_axis: u.Quantity
    airmass_axis: u.Quantity
    x: u.Quantity
    y: u.Quantity
    metric_names: tuple[str, ...]
    interpolation_config: RegularGridInterpolationConfig | RbfInterpolationConfig
    model: Mapping[str, Any]
    provenance: tuple[str, ...] = ()
    builder: Mapping[str, Any] = field(default_factory=dict)


def build_ngs_ho_metric_samples_from_psfs(samples: NgsHoPsfSamples) -> NgsHoMetricSamples:
    """Measure NGS-HO metrics from NGS-HO PSF samples."""

    prepared = _prepare_psf_samples(samples)
    psfs = prepared.psfs.reshape((-1, *prepared.psfs.shape[2:]))
    metadata = PsfMetadata(
        wavelength=_expand_psf_metadata(prepared.wavelength, prepared.psfs.shape[:2], unit=u.um, label="wavelength") * u.um,
        pixel_scale=_expand_psf_metadata(prepared.pixel_scale, prepared.psfs.shape[:2], unit=u.mas, label="pixel_scale") * u.mas,
        tel_diameter=prepared.tel_diameter,
        tel_pupil=prepared.tel_pupil,
    )
    sr = np.asarray(compute_psf_sr(psfs, metadata, preprocess="default"), dtype=float)
    fwhm = np.asarray(compute_psf_fwhm(psfs, metadata, preprocess="default"), dtype=float)
    ee_aperture_diameters = np.asarray(2.0 * fwhm, dtype=np.float32).reshape(-1, 1) * u.mas
    ee = np.asarray(
        compute_psf_ee(
            psfs,
            metadata,
            ee_apertures=ee_aperture_diameters,
            ee_geometry=EE_GEOMETRY_ENCIRCLED,
            preprocess="default",
        ),
        dtype=float,
    ).reshape(-1)
    n_planes, n_points = prepared.psfs.shape[:2]
    return NgsHoMetricSamples(
        zenith_angle=prepared.zenith_angle,
        x=prepared.x,
        y=prepared.y,
        ee=ee.reshape(n_planes, n_points) * u.dimensionless_unscaled,
        fwhm=fwhm.reshape(n_planes, n_points) * u.mas,
        sr=sr.reshape(n_planes, n_points) * u.dimensionless_unscaled,
        provenance=tuple(prepared.provenance),
    )


def build_ngs_ho_metric_interpolator_from_psfs(
    samples: NgsHoPsfSamples,
    *,
    interpolation_config: RegularGridInterpolationConfig | RbfInterpolationConfig | None = None,
) -> NgsHoMetricInterpolator:
    """Build an NGS-HO-metric artifact from NGS-HO PSF samples."""

    return build_ngs_ho_metric_interpolator(
        build_ngs_ho_metric_samples_from_psfs(samples),
        interpolation_config=interpolation_config,
    )


def build_ngs_ho_metric_interpolator(
    samples: NgsHoMetricSamples,
    *,
    interpolation_config: RegularGridInterpolationConfig | RbfInterpolationConfig | None = None,
) -> NgsHoMetricInterpolator:
    """Build an NGS-HO-metric interpolation artifact from metric samples."""

    config = _validate_interpolation_config(interpolation_config or RegularGridInterpolationConfig())
    prepared = _prepare_metric_samples(samples)
    values_by_name = {
        "ee": prepared.ee.to_value(u.dimensionless_unscaled),
        "fwhm": prepared.fwhm.to_value(u.mas),
        "sr": prepared.sr.to_value(u.dimensionless_unscaled),
    }
    zenith_axis = _zenith_axis(prepared)
    coordinate_order = _coordinate_order_for_config(config, zenith_axis)
    if isinstance(config, RbfInterpolationConfig):
        coordinates = _training_coordinates(prepared, coordinate_order)
        model = make_scaled_rbf_model(
            coordinates,
            {name: values.reshape(-1) for name, values in values_by_name.items()},
            config,
        )
        x = prepared.x
        y = prepared.y
    else:
        x, y, model = _make_regular_grid_metric_model(
            prepared,
            values_by_name,
            config,
            coordinate_order,
            zenith_axis,
        )
    model = dict(model)
    model["metric_units"] = {"ee": u.dimensionless_unscaled, "fwhm": u.mas, "sr": u.dimensionless_unscaled}
    if "metric_grids" in model:
        model["metric_grids"] = {
            name: np.asarray(grid) * model["metric_units"][name]
            for name, grid in dict(model["metric_grids"]).items()
        }
    return NgsHoMetricInterpolator(
        coordinate_order=coordinate_order,
        zenith_angle_axis=zenith_axis * u.deg,
        airmass_axis=zenith_angle_to_airmass(zenith_axis * u.deg),
        x=x if isinstance(x, u.Quantity) else x * u.arcsec,
        y=y if isinstance(y, u.Quantity) else y * u.arcsec,
        metric_names=tuple(values_by_name),
        interpolation_config=config,
        model=model,
        provenance=tuple(prepared.provenance),
        builder={
            "name": "ao_predict.interpolation.ngs_ho_metric",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "interpolation_strategy": _interpolation_strategy(config),
        },
    )


def evaluate_ngs_ho_metric_interpolator(
    interpolator: NgsHoMetricInterpolator,
    *,
    zenith_angle: u.Quantity | None = None,
    x: u.Quantity,
    y: u.Quantity,
) -> NgsHoMetricPrediction:
    """Evaluate an NGS-HO-metric artifact at supported query points.

    Args:
        interpolator: Validated NGS-HO-metric artifact.
        zenith_angle: Optional query angle compatible with degrees. It is
            required when the artifact has an active airmass coordinate.
        x: Query field x-coordinates compatible with arcseconds.
        y: Query field y-coordinates compatible with arcseconds.

    Returns:
        Dimensionless enclosed-energy and Strehl-ratio quantities plus FWHM
        quantities in milliarcseconds.

    Raises:
        TypeError: If the artifact or query values use unsupported types.
        ValueError: If units, shapes, values, or query support are invalid.
    """

    query = _prepare_validated_query(
        interpolator,
        zenith_angle=zenith_angle,
        x=x,
        y=y,
    )
    config = _validate_interpolation_config(interpolator.interpolation_config)
    if isinstance(config, RbfInterpolationConfig):
        coordinates = _query_coordinates(interpolator, query)
        output = evaluate_scaled_rbf_model(interpolator.model, coordinates)
    else:
        output = _evaluate_regular_grid_metric_model(interpolator, config, query=query)
    _validate_predicted_metrics(output)
    return NgsHoMetricPrediction(
        ee=np.asarray(output["ee"], dtype=float) * u.dimensionless_unscaled,
        fwhm=np.asarray(output["fwhm"], dtype=float) * u.mas,
        sr=np.asarray(output["sr"], dtype=float) * u.dimensionless_unscaled,
    )


def validate_ngs_ho_metric_query(
    interpolator: NgsHoMetricInterpolator,
    *,
    zenith_angle: u.Quantity | None = None,
    x: u.Quantity,
    y: u.Quantity,
) -> None:
    """Validate NGS-HO-metric query support without returning predictions.

    Args:
        interpolator: Validated NGS-HO-metric artifact.
        zenith_angle: Optional query angle compatible with degrees. It is
            required when the artifact has an active airmass coordinate.
        x: Query field x-coordinates compatible with arcseconds.
        y: Query field y-coordinates compatible with arcseconds.

    Raises:
        TypeError: If the artifact or query values use unsupported types.
        ValueError: If units, shapes, values, or query support are invalid.
    """

    _prepare_validated_query(
        interpolator,
        zenith_angle=zenith_angle,
        x=x,
        y=y,
    )


def _prepare_validated_query(
    interpolator: NgsHoMetricInterpolator,
    *,
    zenith_angle: u.Quantity | None = None,
    x: u.Quantity,
    y: u.Quantity,
) -> dict[str, np.ndarray]:
    if not isinstance(interpolator, NgsHoMetricInterpolator):
        raise TypeError("interpolator must be an NgsHoMetricInterpolator instance.")
    _validate_interpolator(interpolator)
    query = _query_values(interpolator, zenith_angle, x, y)
    if isinstance(_validate_interpolation_config(interpolator.interpolation_config), RegularGridInterpolationConfig):
        _validate_regular_grid_field_support(interpolator, query[NGS_COORD_X], query[NGS_COORD_Y])
    else:
        _validate_rbf_field_support(interpolator, query[NGS_COORD_X], query[NGS_COORD_Y])
    return query


def validate_ngs_ho_metric_interpolator(interpolator: NgsHoMetricInterpolator) -> None:
    """Validate an NGS-HO-metric artifact contract."""

    if not isinstance(interpolator, NgsHoMetricInterpolator):
        raise TypeError("interpolator must be an NgsHoMetricInterpolator instance.")
    _validate_interpolator(interpolator)


def replay_ngs_ho_metric_interpolator(
    interpolator: NgsHoMetricInterpolator,
    samples: NgsHoMetricSamples,
) -> NgsHoMetricReplaySummary:
    """Replay an NGS-HO-metric artifact at source sample nodes."""

    prepared = _prepare_metric_samples(samples)
    query_kwargs: dict[str, Any] = {
        "x": np.tile(prepared.x.to_value(u.arcsec), prepared.ee.shape[0]) * u.arcsec,
        "y": np.tile(prepared.y.to_value(u.arcsec), prepared.ee.shape[0]) * u.arcsec,
    }
    if prepared.zenith_angle is not None:
        query_kwargs["zenith_angle"] = np.repeat(prepared.zenith_angle.to_value(u.deg), prepared.x.size) * u.deg
    prediction = evaluate_ngs_ho_metric_interpolator(interpolator, **query_kwargs)
    reference = {
        "ee": prepared.ee.to_value(u.dimensionless_unscaled).reshape(-1),
        "fwhm": prepared.fwhm.to_value(u.mas).reshape(-1),
        "sr": prepared.sr.to_value(u.dimensionless_unscaled).reshape(-1),
    }
    measured = {
        "ee": prediction.ee.to_value(u.dimensionless_unscaled),
        "fwhm": prediction.fwhm.to_value(u.mas),
        "sr": prediction.sr.to_value(u.dimensionless_unscaled),
    }
    units = {"ee": u.dimensionless_unscaled, "fwhm": u.mas, "sr": u.dimensionless_unscaled}
    return NgsHoMetricReplaySummary(
        metric_rms={
            name: float(np.sqrt(np.mean((measured[name] - reference[name]) ** 2)))
            * units[name]
            for name in reference
        },
        metric_max_abs={
            name: float(np.max(np.abs(measured[name] - reference[name])))
            * units[name]
            for name in reference
        },
        num_planes=int(prepared.ee.shape[0]),
        num_points=int(prepared.x.size),
    )


def save_ngs_ho_metric_interpolator(
    interpolator: NgsHoMetricInterpolator,
    path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Save an NGS-HO-metric interpolation artifact."""

    save_payload(_ngs_to_payload(interpolator), Path(path), overwrite=overwrite)


def load_ngs_ho_metric_interpolator(path: Path) -> NgsHoMetricInterpolator:
    """Load and validate an NGS-HO-metric interpolation artifact."""

    return _ngs_from_payload(load_payload(Path(path)))


# Sample preparation


def _prepare_metric_samples(samples: NgsHoMetricSamples) -> NgsHoMetricSamples:
    if not isinstance(samples, NgsHoMetricSamples):
        raise TypeError("samples must be an NgsHoMetricSamples instance.")
    coordinates = field_coordinates(samples.x, samples.y)
    n_planes = _infer_metric_plane_count(samples, coordinates.shape[0])
    zenith_angle = _prepare_optional_plane_values(samples.zenith_angle, label="zenith_angle", length=n_planes)
    if zenith_angle is None and n_planes != 1:
        raise ValueError("Multiple NGS HO metric planes require zenith_angle.")
    expected_shape = (n_planes, coordinates.shape[0])
    ee = _prepare_metric_array(samples.ee, unit=u.dimensionless_unscaled, label="ee", expected_shape=expected_shape)
    fwhm = _prepare_metric_array(samples.fwhm, unit=u.mas, label="fwhm", expected_shape=expected_shape)
    sr = _prepare_metric_array(samples.sr, unit=u.dimensionless_unscaled, label="sr", expected_shape=expected_shape)
    return NgsHoMetricSamples(
        zenith_angle=zenith_angle,
        x=quantity_value(samples.x, u.arcsec, label="x", dtype=float).reshape(-1) * u.arcsec,
        y=quantity_value(samples.y, u.arcsec, label="y", dtype=float).reshape(-1) * u.arcsec,
        ee=ee,
        fwhm=fwhm,
        sr=sr,
        provenance=tuple(str(value) for value in samples.provenance),
    )


def _prepare_psf_samples(samples: NgsHoPsfSamples) -> NgsHoPsfSamples:
    if not isinstance(samples, NgsHoPsfSamples):
        raise TypeError("samples must be an NgsHoPsfSamples instance.")
    coordinates = field_coordinates(samples.x, samples.y)
    psfs = np.asarray(samples.psfs, dtype=np.float32)
    if psfs.ndim == 3:
        psfs = psfs[np.newaxis, ...]
    psfs = validate_psf_array(psfs, label="psfs", ndim=4)
    if psfs.shape[1] != coordinates.shape[0]:
        raise ValueError(
            "psfs point dimension must match field coordinates; "
            f"got {psfs.shape} for {coordinates.shape[0]} points."
        )
    zenith_angle = _prepare_optional_plane_values(samples.zenith_angle, label="zenith_angle", length=psfs.shape[0])
    _expand_psf_metadata(samples.wavelength, psfs.shape[:2], unit=u.um, label="wavelength")
    _expand_psf_metadata(samples.pixel_scale, psfs.shape[:2], unit=u.mas, label="pixel_scale")
    return NgsHoPsfSamples(
        zenith_angle=zenith_angle,
        x=quantity_value(samples.x, u.arcsec, label="x", dtype=float).reshape(-1) * u.arcsec,
        y=quantity_value(samples.y, u.arcsec, label="y", dtype=float).reshape(-1) * u.arcsec,
        psfs=psfs,
        wavelength=require_quantity(samples.wavelength, u.um, label="wavelength"),
        pixel_scale=require_quantity(samples.pixel_scale, u.mas, label="pixel_scale"),
        tel_diameter=require_positive_scalar(quantity_value(samples.tel_diameter, u.m, label="tel_diameter"), label="tel_diameter") * u.m,
        tel_pupil=require_pupil(quantity_value(samples.tel_pupil, u.dimensionless_unscaled, label="tel_pupil")) * u.dimensionless_unscaled,
        provenance=tuple(str(value) for value in samples.provenance),
    )


def _infer_metric_plane_count(samples: NgsHoMetricSamples, n_points: int) -> int:
    counts: list[int] = []
    metric_units = {"ee": u.one, "fwhm": u.mas, "sr": u.one}
    for name in REQUIRED_NGS_HO_METRICS:
        metric = quantity_value(
            getattr(samples, name), metric_units[name], label=name, dtype=float
        )
        if metric.ndim == 1:
            if metric.shape[0] != n_points:
                raise ValueError(f"{name} must have length {n_points}; got shape {metric.shape}.")
            counts.append(1)
        elif metric.ndim == 2:
            if metric.shape[1] != n_points:
                raise ValueError(f"{name} second dimension must be {n_points}; got shape {metric.shape}.")
            counts.append(int(metric.shape[0]))
        else:
            raise ValueError(f"{name} must have shape (points,) or (planes, points); got {metric.shape}.")
    if samples.zenith_angle is not None:
        zenith = quantity_value(
            samples.zenith_angle, u.deg, label="zenith_angle", dtype=float
        )
        if zenith.ndim != 0:
            counts.append(int(zenith.reshape(-1).size))
    n_planes = max(counts) if counts else 1
    if any(count not in {1, n_planes} for count in counts):
        raise ValueError("NGS HO metric plane counts must match across metrics and zenith_angle.")
    return int(n_planes)


def _prepare_optional_plane_values(value: Any, *, label: str, length: int) -> u.Quantity | None:
    if value is None:
        return None
    array = quantity_value(value, u.deg, label=label, dtype=float)
    if array.ndim == 0:
        vector = np.full(int(length), float(array), dtype=float)
    else:
        vector = array.reshape(-1)
        if vector.size != int(length):
            raise ValueError(f"{label} must be scalar or have one value per plane; got length {vector.size}.")
    return require_finite_vector(vector, label=label, length=length) * u.deg


def _prepare_metric_array(value: Any, *, unit: u.UnitBase, label: str, expected_shape: tuple[int, int]) -> u.Quantity:
    if value is None:
        raise ValueError(f"{label} is required.")
    metric = quantity_value(value, unit, label=label, dtype=float)
    if metric.ndim == 1 and expected_shape[0] == 1:
        metric = metric[np.newaxis, :]
    if metric.shape != expected_shape:
        raise ValueError(f"{label} must have shape {expected_shape}, got {metric.shape}.")
    if not np.all(np.isfinite(metric)):
        raise ValueError(f"{label} must contain only finite values.")
    if np.any(metric <= 0.0):
        raise ValueError(f"{label} must contain only values > 0.")
    if label == "ee" and np.any(metric > 1.0):
        raise ValueError("ee must not contain values > 1.")
    return metric * unit


def _expand_psf_metadata(value: Any, shape: tuple[int, int], *, unit: u.UnitBase, label: str) -> float | np.ndarray:
    n_planes, n_points = shape
    array = quantity_value(value, unit, label=label, dtype=float)
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


# Build helpers


def _zenith_axis(samples: NgsHoMetricSamples) -> np.ndarray:
    if samples.zenith_angle is None:
        return np.asarray([], dtype=float)
    return unique_sorted(samples.zenith_angle.to_value(u.deg), label="zenith_angle")


def _coordinate_order_for_config(
    config: RegularGridInterpolationConfig | RbfInterpolationConfig,
    zenith_axis: np.ndarray,
) -> tuple[str, ...]:
    has_airmass = zenith_axis.size > 1
    if isinstance(config, RegularGridInterpolationConfig):
        return (NGS_COORD_AIRMASS, *NGS_REGULAR_GRID_FIELD_ORDER) if has_airmass else NGS_REGULAR_GRID_FIELD_ORDER
    return (NGS_COORD_AIRMASS, *NGS_RBF_FIELD_ORDER) if has_airmass else NGS_RBF_FIELD_ORDER


def _validate_interpolation_config(
    config: RegularGridInterpolationConfig | RbfInterpolationConfig,
) -> RegularGridInterpolationConfig | RbfInterpolationConfig:
    if isinstance(config, RegularGridInterpolationConfig):
        return validate_regular_grid_config(config)
    if isinstance(config, RbfInterpolationConfig):
        return validate_rbf_config(config)
    raise TypeError("interpolation_config must be a RegularGridInterpolationConfig or RbfInterpolationConfig instance.")


def _interpolation_strategy(config: RegularGridInterpolationConfig | RbfInterpolationConfig) -> str:
    config = _validate_interpolation_config(config)
    if isinstance(config, RegularGridInterpolationConfig):
        return NGS_HO_METRIC_STRATEGY_REGULAR_GRID
    return NGS_HO_METRIC_STRATEGY_RBF


def _make_regular_grid_metric_model(
    samples: NgsHoMetricSamples,
    values_by_name: Mapping[str, np.ndarray],
    config: RegularGridInterpolationConfig,
    coordinate_order: tuple[str, ...],
    zenith_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    config = validate_regular_grid_config(config)
    _validate_coordinate_order_for_strategy(coordinate_order, NGS_HO_METRIC_STRATEGY_REGULAR_GRID)
    if NGS_COORD_AIRMASS in coordinate_order:
        if zenith_axis.size != samples.ee.shape[0]:
            raise ValueError("NGS HO metric samples contain duplicate zenith-angle planes.")
    elif samples.ee.shape[0] != 1:
        raise ValueError("Field-only NGS HO metric artifacts require exactly one source plane.")
    x_axis, y_axis = rectangular_field_axes(samples.x, samples.y, label="NGS HO metric samples")
    metric_grids: dict[str, np.ndarray] = {}
    active_shape = (zenith_axis.size,) if NGS_COORD_AIRMASS in coordinate_order else ()
    for name, values in values_by_name.items():
        values = np.asarray(values, dtype=float)
        metric_grid = np.full((*active_shape, y_axis.size, x_axis.size), np.nan, dtype=float)
        for plane_index in range(values.shape[0]):
            grid_index = (
                (
                    axis_index(
                        zenith_axis,
                        float(samples.zenith_angle[plane_index].to_value(u.deg)),
                        label="zenith_angle",
                    ),
                )
                if NGS_COORD_AIRMASS in coordinate_order
                else ()
            )
            metric_grid[grid_index] = grid_field_values(
                samples.x,
                samples.y,
                values[plane_index],
                x_axis,
                y_axis,
                label=f"NGS HO metric {name}",
                dtype=float,
            )
        if not np.all(np.isfinite(metric_grid)):
            raise ValueError(f"NGS HO metric {name} grid is incomplete.")
        metric_grids[str(name)] = metric_grid
    return (
        np.asarray(x_axis, dtype=float),
        np.asarray(y_axis, dtype=float),
        {"metric_grids": metric_grids, "method": config.method},
    )


def _training_coordinates(samples: NgsHoMetricSamples, coordinate_order: tuple[str, ...]) -> np.ndarray:
    columns: list[np.ndarray] = []
    n_planes = samples.ee.shape[0]
    n_points = samples.x.size
    for name in coordinate_order:
        if name == NGS_COORD_AIRMASS:
            if samples.zenith_angle is None:
                raise ValueError("zenith_angle is required for active airmass interpolation.")
            airmasses = zenith_angle_to_airmass(samples.zenith_angle).to_value(u.dimensionless_unscaled)
            columns.append(np.repeat(airmasses, n_points))
        elif name == NGS_COORD_X:
            columns.append(np.tile(samples.x.to_value(u.arcsec), n_planes))
        elif name == NGS_COORD_Y:
            columns.append(np.tile(samples.y.to_value(u.arcsec), n_planes))
        else:
            raise ValueError(f"Unsupported NGS HO metric coordinate {name!r}.")
    return np.column_stack(columns)


# Query helpers


def _query_values(interpolator: NgsHoMetricInterpolator, zenith_angle: Any, x: Any, y: Any) -> dict[str, np.ndarray]:
    x = require_finite_vector(quantity_value(x, u.arcsec, label="x"), label="x")
    if x.size == 0:
        raise ValueError("At least one query field coordinate is required.")
    y = require_finite_vector(quantity_value(y, u.arcsec, label="y"), label="y", length=x.size)
    query: dict[str, np.ndarray] = {NGS_COORD_X: x, NGS_COORD_Y: y}
    if NGS_COORD_AIRMASS in interpolator.coordinate_order:
        if zenith_angle is None:
            raise ValueError("zenith_angle is required by this NGS HO metric artifact.")
        zenith = require_finite_vector(quantity_value(zenith_angle, u.deg, label="zenith_angle"), label="zenith_angle")
        if zenith.size == 1 and x.size != 1:
            zenith = np.full(x.shape, float(zenith[0]), dtype=float)
        if zenith.size != x.size:
            raise ValueError(
                "zenith_angle, x, and y must have matching lengths; "
                f"got {zenith.size}, {x.size}, {y.size}."
            )
        airmass = zenith_angle_to_airmass(zenith * u.deg).to_value(u.dimensionless_unscaled)
        _validate_airmass_support(interpolator, airmass)
        query[NGS_COORD_AIRMASS] = airmass
    elif zenith_angle is not None and interpolator.zenith_angle_axis.size == 1:
        zenith = require_finite_vector(quantity_value(zenith_angle, u.deg, label="zenith_angle"), label="zenith_angle")
        fixed_zenith = float(interpolator.zenith_angle_axis[0].to_value(u.deg))
        if not np.all(np.isclose(zenith, fixed_zenith, rtol=0.0, atol=_FIELD_ATOL)):
            raise ValueError(
                f"zenith_angle query does not match fixed artifact value {fixed_zenith}."
            )
    return query


def _query_coordinates(interpolator: NgsHoMetricInterpolator, query: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([np.asarray(query[name], dtype=float) for name in interpolator.coordinate_order])


def _evaluate_regular_grid_metric_model(
    interpolator: NgsHoMetricInterpolator,
    config: RegularGridInterpolationConfig,
    *,
    query: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    config = validate_regular_grid_config(config)
    coordinates = field_coordinates(query[NGS_COORD_X] * u.arcsec, query[NGS_COORD_Y] * u.arcsec)
    coordinates = snap_rectangular_field_query(interpolator.x.to_value(u.arcsec), interpolator.y.to_value(u.arcsec), coordinates)
    query = dict(query)
    query[NGS_COORD_X] = coordinates[:, 0]
    query[NGS_COORD_Y] = coordinates[:, 1]
    points = _query_coordinates(interpolator, query)
    axes = _regular_grid_axes(interpolator)
    return {
        name: np.asarray(
            RegularGridInterpolator(
                axes,
                quantity_value(grid, dict(interpolator.model["metric_units"])[name], label=f"metric grid {name}"),
                method=config.method,
                bounds_error=True,
            )(points),
            dtype=float,
        )
        for name, grid in dict(interpolator.model.get("metric_grids", {})).items()
    }


def _regular_grid_axes(interpolator: NgsHoMetricInterpolator) -> tuple[np.ndarray, ...]:
    axes: list[np.ndarray] = []
    for name in interpolator.coordinate_order:
        if name == NGS_COORD_AIRMASS:
            axes.append(interpolator.airmass_axis.to_value(u.dimensionless_unscaled))
        elif name == NGS_COORD_Y:
            axes.append(interpolator.y.to_value(u.arcsec))
        elif name == NGS_COORD_X:
            axes.append(interpolator.x.to_value(u.arcsec))
        else:
            raise ValueError(f"Unsupported NGS HO metric coordinate {name!r}.")
    return tuple(axes)


def _validate_airmass_support(interpolator: NgsHoMetricInterpolator, airmass: np.ndarray) -> None:
    axis = interpolator.airmass_axis.to_value(u.dimensionless_unscaled).reshape(-1)
    minimum = float(np.min(axis))
    maximum = float(np.max(axis))
    if np.any((airmass < minimum) & ~np.isclose(airmass, minimum, rtol=0.0, atol=_FIELD_ATOL)):
        raise ValueError(f"airmass query is below the supported range minimum {minimum}.")
    if np.any((airmass > maximum) & ~np.isclose(airmass, maximum, rtol=0.0, atol=_FIELD_ATOL)):
        raise ValueError(f"airmass query is above the supported range maximum {maximum}.")


def _validate_regular_grid_field_support(interpolator: NgsHoMetricInterpolator, x: np.ndarray, y: np.ndarray) -> None:
    validate_rectangular_field_query(
        interpolator.x.to_value(u.arcsec),
        interpolator.y.to_value(u.arcsec),
        np.column_stack([x, y]),
        label="NGS HO metric",
        atol=_FIELD_ATOL,
    )


def _validate_rbf_field_support(interpolator: NgsHoMetricInterpolator, x: np.ndarray, y: np.ndarray) -> None:
    source = np.unique(field_coordinates(interpolator.x, interpolator.y), axis=0)
    query = np.column_stack([x, y])
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


# Persistence and validation


def _ngs_to_payload(interpolator: NgsHoMetricInterpolator) -> dict[str, Any]:
    if not isinstance(interpolator, NgsHoMetricInterpolator):
        raise TypeError("interpolator must be an NgsHoMetricInterpolator instance.")
    _validate_interpolator(interpolator)
    return {
        "kind": NGS_HO_METRIC_ARTIFACT_KIND,
        "version": NGS_HO_METRIC_ARTIFACT_VERSION,
        "builder": dict(interpolator.builder),
        "strategy": _interpolation_strategy(interpolator.interpolation_config),
        "interpolation_config": interpolator.interpolation_config,
        "interpolation": {"coordinate_order": tuple(interpolator.coordinate_order)},
        "metadata": {
            "zenith_angle_axis": interpolator.zenith_angle_axis,
            "airmass_axis": interpolator.airmass_axis,
            "x": interpolator.x,
            "y": interpolator.y,
            "metric_names": tuple(interpolator.metric_names),
            "provenance": tuple(interpolator.provenance),
        },
        "model": dict(interpolator.model),
    }


def _ngs_from_payload(payload: Mapping[str, Any]) -> NgsHoMetricInterpolator:
    if payload.get("kind") != NGS_HO_METRIC_ARTIFACT_KIND:
        raise ValueError(f"Unsupported artifact kind: {payload.get('kind')!r}.")
    version = payload.get("version")
    if version != NGS_HO_METRIC_ARTIFACT_VERSION:
        raise ValueError(f"Unsupported artifact version: {version!r}.")
    metadata = dict(payload.get("metadata", {}))
    config = _payload_interpolation_config(payload)
    strategy = _interpolation_strategy(config)
    interpolation = dict(payload.get("interpolation", {}))
    coordinate_order = tuple(str(value) for value in interpolation.get("coordinate_order", ()))
    _validate_coordinate_order_for_strategy(coordinate_order, strategy)
    metric_names = tuple(str(value) for value in metadata.get("metric_names", ()))
    _validate_metric_names(metric_names, label="NGS HO metric artifact")
    interpolator = NgsHoMetricInterpolator(
        coordinate_order=coordinate_order,
        zenith_angle_axis=require_quantity(metadata.get("zenith_angle_axis"), u.deg, label="metadata.zenith_angle_axis"),
        airmass_axis=require_quantity(metadata.get("airmass_axis"), u.dimensionless_unscaled, label="metadata.airmass_axis"),
        x=require_quantity(metadata.get("x"), u.arcsec, label="metadata.x"),
        y=require_quantity(metadata.get("y"), u.arcsec, label="metadata.y"),
        metric_names=metric_names,
        interpolation_config=config,
        model=dict(payload.get("model", {})),
        provenance=tuple(str(value) for value in metadata.get("provenance", ())),
        builder=dict(payload.get("builder", {})),
    )
    _validate_interpolator(interpolator)
    return interpolator


def _payload_interpolation_config(payload: Mapping[str, Any]) -> RegularGridInterpolationConfig | RbfInterpolationConfig:
    raw_config = payload.get("interpolation_config")
    strategy = payload.get("strategy")
    if strategy == NGS_HO_METRIC_STRATEGY_REGULAR_GRID:
        return validate_regular_grid_config(raw_config)
    if strategy == NGS_HO_METRIC_STRATEGY_RBF:
        return validate_rbf_config(raw_config)
    raise ValueError(f"Unsupported NGS HO metric interpolation strategy: {strategy!r}.")


def _validate_coordinate_order_for_strategy(coordinate_order: tuple[str, ...], strategy: str) -> None:
    valid = NGS_REGULAR_GRID_COORDINATE_ORDERS if strategy == NGS_HO_METRIC_STRATEGY_REGULAR_GRID else NGS_RBF_COORDINATE_ORDERS
    if tuple(coordinate_order) not in valid:
        raise ValueError(f"Unsupported NGS HO metric coordinate_order for {strategy}: {tuple(coordinate_order)!r}.")
    if "wavelength" in coordinate_order:
        raise ValueError("NGS HO metric artifacts must not use wavelength as an interpolation coordinate.")


def _validate_interpolator(interpolator: NgsHoMetricInterpolator) -> None:
    require_quantity(interpolator.zenith_angle_axis, u.deg, label="zenith_angle_axis")
    require_quantity(interpolator.airmass_axis, u.dimensionless_unscaled, label="airmass_axis")
    require_quantity(interpolator.x, u.arcsec, label="x")
    require_quantity(interpolator.y, u.arcsec, label="y")
    if interpolator.airmass_axis.shape != interpolator.zenith_angle_axis.shape:
        raise ValueError("airmass_axis shape must match zenith_angle_axis shape.")
    _validate_metric_names(interpolator.metric_names, label="NGS HO metric artifact")
    config = _validate_interpolation_config(interpolator.interpolation_config)
    strategy = _interpolation_strategy(config)
    _validate_coordinate_order_for_strategy(tuple(interpolator.coordinate_order), strategy)
    if NGS_COORD_AIRMASS in interpolator.coordinate_order:
        if interpolator.zenith_angle_axis.size <= 1:
            raise ValueError("active airmass coordinate requires multiple zenith_angle_axis values.")
    elif interpolator.zenith_angle_axis.size > 1:
        raise ValueError("multiple zenith_angle_axis values require active airmass coordinate.")
    if isinstance(config, RegularGridInterpolationConfig):
        if interpolator.x.ndim != 1 or interpolator.y.ndim != 1:
            raise ValueError("x and y must be 1-D rectangular field axes for regular-grid artifacts.")
        if interpolator.x.size == 0 or interpolator.y.size == 0:
            raise ValueError("x and y field axes must not be empty.")
        if np.any(np.diff(interpolator.x) <= 0.0) or np.any(np.diff(interpolator.y) <= 0.0):
            raise ValueError("x and y field axes must be strictly increasing.")
        _validate_regular_grid_metric_model(interpolator)
        return
    if interpolator.x.shape != interpolator.y.shape:
        raise ValueError("x and y shapes must match.")
    if interpolator.x.size == 0:
        raise ValueError("x and y must not be empty.")
    _validate_rbf_metric_model(interpolator)


def _validate_rbf_metric_model(interpolator: NgsHoMetricInterpolator) -> None:
    model = dict(interpolator.model)
    _validate_model_metric_units(model)
    model_names = set(dict(model.get("models", {})))
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
    coord_mean = np.asarray(model.get("coord_mean"), dtype=float)
    if coord_mean.shape != (len(interpolator.coordinate_order),):
        raise ValueError("NGS HO metric RBF coord_mean shape must match coordinate_order.")


def _validate_regular_grid_metric_model(interpolator: NgsHoMetricInterpolator) -> None:
    metric_grids = dict(interpolator.model.get("metric_grids", {}))
    metric_units = _validate_model_metric_units(interpolator.model)
    model_names = set(metric_grids)
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
    expected_shape = _expected_regular_grid_shape(interpolator)
    for name, grid in metric_grids.items():
        grid = quantity_value(grid, metric_units[name], label=f"metric grid {name}", dtype=float)
        if grid.shape != expected_shape:
            raise ValueError(f"NGS HO metric artifact grid {name!r} must have shape {expected_shape}; got {grid.shape}.")
        if not np.all(np.isfinite(grid)):
            raise ValueError(f"NGS HO metric artifact grid {name!r} must contain only finite values.")


def _validate_model_metric_units(model: Mapping[str, Any]) -> dict[str, u.UnitBase]:
    """Validate the fixed units carried by an NGS metric model."""
    expected = {"ee": u.dimensionless_unscaled, "fwhm": u.mas, "sr": u.dimensionless_unscaled}
    raw = model.get("metric_units")
    if not isinstance(raw, Mapping) or set(raw) != set(expected):
        raise ValueError("NGS HO metric artifact model must declare units for ee, fwhm, and sr.")
    for name, unit in expected.items():
        if u.Unit(raw[name]) != unit:
            raise ValueError(f"NGS HO metric artifact {name!r} unit must be {unit}.")
    return expected


def _expected_regular_grid_shape(interpolator: NgsHoMetricInterpolator) -> tuple[int, ...]:
    sizes = {
        NGS_COORD_AIRMASS: interpolator.airmass_axis.size,
        NGS_COORD_Y: interpolator.y.size,
        NGS_COORD_X: interpolator.x.size,
    }
    return tuple(int(sizes[name]) for name in interpolator.coordinate_order)


def _validate_metric_names(metric_names: tuple[str, ...], *, label: str) -> None:
    if len(set(metric_names)) != len(metric_names):
        raise ValueError(f"{label} contains duplicate metrics.")
    missing = set(REQUIRED_NGS_HO_METRICS) - set(metric_names)
    if missing:
        raise ValueError(f"{label} is missing required metrics: {', '.join(sorted(missing))}.")
    unsupported = set(metric_names) - set(REQUIRED_NGS_HO_METRICS) - set(OPTIONAL_NGS_HO_METRICS)
    if unsupported:
        raise ValueError(f"{label} contains unsupported metrics: {', '.join(sorted(unsupported))}.")
