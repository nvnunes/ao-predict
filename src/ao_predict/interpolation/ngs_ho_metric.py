"""NGS high-order metric interpolation artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import Delaunay, QhullError

from ao_predict.simulation.stats import (
    EE_GEOMETRY_ENCIRCLED,
    PsfMetadata,
    compute_psf_ee,
    compute_psf_fwhm,
    compute_psf_sr,
)

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
NGS_COORD_X = "x_arcsec"
NGS_COORD_Y = "y_arcsec"
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
REQUIRED_NGS_HO_METRICS = ("ee", "fwhm_mas", "sr")
OPTIONAL_NGS_HO_METRICS: tuple[str, ...] = ()
_FIELD_ATOL = 1.0e-10


@dataclass(frozen=True, kw_only=True)
class NgsHoMetricSamples:
    """Measured NGS high-order metric samples for interpolation.

    ``zenith_angle_deg`` is optional. Multiple unique zenith values activate an
    airmass interpolation coordinate. A scalar or single unique zenith value is
    stored as fixed support metadata.
    """

    x_arcsec: np.ndarray
    y_arcsec: np.ndarray
    ee: np.ndarray
    fwhm_mas: np.ndarray
    sr: np.ndarray
    zenith_angle_deg: float | np.ndarray | None = None
    provenance: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class NgsHoPsfSamples:
    """NGS high-order PSF samples used to measure NGS HO metrics.

    Wavelength remains PSF-stat metadata only; it is never an interpolation
    coordinate for NGS-HO metric artifacts.
    """

    x_arcsec: np.ndarray
    y_arcsec: np.ndarray
    psfs: np.ndarray
    wavelength_um: float | np.ndarray
    pixel_scale_mas: float | np.ndarray
    tel_diameter_m: float
    tel_pupil: np.ndarray
    zenith_angle_deg: float | np.ndarray | None = None
    provenance: tuple[str, ...] = ()


@dataclass(frozen=True)
class NgsHoMetricPrediction:
    """NGS high-order metric prediction returned by artifact evaluation."""

    ee: np.ndarray
    fwhm_mas: np.ndarray
    sr: np.ndarray


@dataclass(frozen=True)
class NgsHoMetricReplaySummary:
    """Source-node replay residual summary for an NGS-HO-metric artifact."""

    metric_rms: dict[str, float]
    metric_max_abs: dict[str, float]
    num_planes: int
    num_points: int


@dataclass(frozen=True)
class NgsHoMetricInterpolator:
    """Versioned NGS-HO-metric interpolation artifact."""

    coordinate_order: tuple[str, ...]
    zenith_angle_deg_axis: np.ndarray
    airmass_axis: np.ndarray
    x_arcsec: np.ndarray
    y_arcsec: np.ndarray
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
        zenith_angle_deg=None if prepared.zenith_angle_deg is None else np.asarray(prepared.zenith_angle_deg, dtype=float),
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
        "ee": np.asarray(prepared.ee, dtype=float),
        "fwhm_mas": np.asarray(prepared.fwhm_mas, dtype=float),
        "sr": np.asarray(prepared.sr, dtype=float),
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
        x_arcsec = np.asarray(prepared.x_arcsec, dtype=float)
        y_arcsec = np.asarray(prepared.y_arcsec, dtype=float)
    else:
        x_arcsec, y_arcsec, model = _make_regular_grid_metric_model(
            prepared,
            values_by_name,
            config,
            coordinate_order,
            zenith_axis,
        )
    return NgsHoMetricInterpolator(
        coordinate_order=coordinate_order,
        zenith_angle_deg_axis=zenith_axis,
        airmass_axis=np.asarray(zenith_angle_to_airmass(zenith_axis), dtype=float),
        x_arcsec=x_arcsec,
        y_arcsec=y_arcsec,
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
    zenith_angle_deg: float | np.ndarray | None = None,
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
) -> NgsHoMetricPrediction:
    """Evaluate an NGS-HO-metric artifact at supported query points."""

    query = _prepare_validated_query(
        interpolator,
        zenith_angle_deg=zenith_angle_deg,
        x_arcsec=x_arcsec,
        y_arcsec=y_arcsec,
    )
    config = _validate_interpolation_config(interpolator.interpolation_config)
    if isinstance(config, RbfInterpolationConfig):
        coordinates = _query_coordinates(interpolator, query)
        output = evaluate_scaled_rbf_model(interpolator.model, coordinates)
    else:
        output = _evaluate_regular_grid_metric_model(interpolator, config, query=query)
    _validate_predicted_metrics(output)
    return NgsHoMetricPrediction(
        ee=np.asarray(output["ee"], dtype=float),
        fwhm_mas=np.asarray(output["fwhm_mas"], dtype=float),
        sr=np.asarray(output["sr"], dtype=float),
    )


def validate_ngs_ho_metric_query(
    interpolator: NgsHoMetricInterpolator,
    *,
    zenith_angle_deg: float | np.ndarray | None = None,
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
) -> None:
    """Validate NGS-HO-metric query support without returning predictions."""

    _prepare_validated_query(
        interpolator,
        zenith_angle_deg=zenith_angle_deg,
        x_arcsec=x_arcsec,
        y_arcsec=y_arcsec,
    )


def _prepare_validated_query(
    interpolator: NgsHoMetricInterpolator,
    *,
    zenith_angle_deg: float | np.ndarray | None = None,
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
) -> dict[str, np.ndarray]:
    if not isinstance(interpolator, NgsHoMetricInterpolator):
        raise TypeError("interpolator must be an NgsHoMetricInterpolator instance.")
    _validate_interpolator(interpolator)
    query = _query_values(interpolator, zenith_angle_deg, x_arcsec, y_arcsec)
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
        "x_arcsec": np.tile(prepared.x_arcsec, prepared.ee.shape[0]),
        "y_arcsec": np.tile(prepared.y_arcsec, prepared.ee.shape[0]),
    }
    if prepared.zenith_angle_deg is not None:
        query_kwargs["zenith_angle_deg"] = np.repeat(prepared.zenith_angle_deg, prepared.x_arcsec.size)
    prediction = evaluate_ngs_ho_metric_interpolator(interpolator, **query_kwargs)
    reference = {
        "ee": np.asarray(prepared.ee, dtype=float).reshape(-1),
        "fwhm_mas": np.asarray(prepared.fwhm_mas, dtype=float).reshape(-1),
        "sr": np.asarray(prepared.sr, dtype=float).reshape(-1),
    }
    measured = {"ee": prediction.ee, "fwhm_mas": prediction.fwhm_mas, "sr": prediction.sr}
    return NgsHoMetricReplaySummary(
        metric_rms={name: float(np.sqrt(np.mean((measured[name] - reference[name]) ** 2))) for name in reference},
        metric_max_abs={name: float(np.max(np.abs(measured[name] - reference[name]))) for name in reference},
        num_planes=int(prepared.ee.shape[0]),
        num_points=int(prepared.x_arcsec.size),
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
    coordinates = field_coordinates(samples.x_arcsec, samples.y_arcsec)
    n_planes = _infer_metric_plane_count(samples, coordinates.shape[0])
    zenith_angle_deg = _prepare_optional_plane_values(samples.zenith_angle_deg, label="zenith_angle_deg", length=n_planes)
    if zenith_angle_deg is None and n_planes != 1:
        raise ValueError("Multiple NGS HO metric planes require zenith_angle_deg.")
    expected_shape = (n_planes, coordinates.shape[0])
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
    coordinates = field_coordinates(samples.x_arcsec, samples.y_arcsec)
    psfs = np.asarray(samples.psfs, dtype=np.float32)
    if psfs.ndim == 3:
        psfs = psfs[np.newaxis, ...]
    psfs = validate_psf_array(psfs, label="psfs", ndim=4)
    if psfs.shape[1] != coordinates.shape[0]:
        raise ValueError(
            "psfs point dimension must match field coordinates; "
            f"got {psfs.shape} for {coordinates.shape[0]} points."
        )
    zenith_angle_deg = _prepare_optional_plane_values(samples.zenith_angle_deg, label="zenith_angle_deg", length=psfs.shape[0])
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


def _infer_metric_plane_count(samples: NgsHoMetricSamples, n_points: int) -> int:
    counts: list[int] = []
    for name in REQUIRED_NGS_HO_METRICS:
        metric = np.asarray(getattr(samples, name), dtype=float)
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
    if samples.zenith_angle_deg is not None:
        zenith = np.asarray(samples.zenith_angle_deg, dtype=float)
        if zenith.ndim != 0:
            counts.append(int(zenith.reshape(-1).size))
    n_planes = max(counts) if counts else 1
    if any(count not in {1, n_planes} for count in counts):
        raise ValueError("NGS HO metric plane counts must match across metrics and zenith_angle_deg.")
    return int(n_planes)


def _prepare_optional_plane_values(value: Any, *, label: str, length: int) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        vector = np.full(int(length), float(array), dtype=float)
    else:
        vector = array.reshape(-1)
        if vector.size != int(length):
            raise ValueError(f"{label} must be scalar or have one value per plane; got length {vector.size}.")
    return require_finite_vector(vector, label=label, length=length)


def _prepare_metric_array(value: Any, *, label: str, expected_shape: tuple[int, int]) -> np.ndarray:
    if value is None:
        raise ValueError(f"{label} is required.")
    metric = np.asarray(value, dtype=float)
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


# Build helpers


def _zenith_axis(samples: NgsHoMetricSamples) -> np.ndarray:
    if samples.zenith_angle_deg is None:
        return np.asarray([], dtype=float)
    return unique_sorted(samples.zenith_angle_deg, label="zenith_angle_deg")


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
    x_axis, y_axis = rectangular_field_axes(samples.x_arcsec, samples.y_arcsec, label="NGS HO metric samples")
    metric_grids: dict[str, np.ndarray] = {}
    active_shape = (zenith_axis.size,) if NGS_COORD_AIRMASS in coordinate_order else ()
    for name, values in values_by_name.items():
        values = np.asarray(values, dtype=float)
        metric_grid = np.full((*active_shape, y_axis.size, x_axis.size), np.nan, dtype=float)
        for plane_index in range(values.shape[0]):
            grid_index = (
                (axis_index(zenith_axis, float(samples.zenith_angle_deg[plane_index]), label="zenith_angle_deg"),)
                if NGS_COORD_AIRMASS in coordinate_order
                else ()
            )
            metric_grid[grid_index] = grid_field_values(
                samples.x_arcsec,
                samples.y_arcsec,
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
    n_points = samples.x_arcsec.size
    for name in coordinate_order:
        if name == NGS_COORD_AIRMASS:
            if samples.zenith_angle_deg is None:
                raise ValueError("zenith_angle_deg is required for active airmass interpolation.")
            airmasses = np.asarray(zenith_angle_to_airmass(samples.zenith_angle_deg), dtype=float)
            columns.append(np.repeat(airmasses, n_points))
        elif name == NGS_COORD_X:
            columns.append(np.tile(samples.x_arcsec, n_planes))
        elif name == NGS_COORD_Y:
            columns.append(np.tile(samples.y_arcsec, n_planes))
        else:
            raise ValueError(f"Unsupported NGS HO metric coordinate {name!r}.")
    return np.column_stack(columns)


# Query helpers


def _query_values(interpolator: NgsHoMetricInterpolator, zenith_angle_deg: Any, x_arcsec: Any, y_arcsec: Any) -> dict[str, np.ndarray]:
    x = require_finite_vector(x_arcsec, label="x_arcsec")
    if x.size == 0:
        raise ValueError("At least one query field coordinate is required.")
    y = require_finite_vector(y_arcsec, label="y_arcsec", length=x.size)
    query: dict[str, np.ndarray] = {NGS_COORD_X: x, NGS_COORD_Y: y}
    if NGS_COORD_AIRMASS in interpolator.coordinate_order:
        if zenith_angle_deg is None:
            raise ValueError("zenith_angle_deg is required by this NGS HO metric artifact.")
        zenith = require_finite_vector(zenith_angle_deg, label="zenith_angle_deg")
        if zenith.size == 1 and x.size != 1:
            zenith = np.full(x.shape, float(zenith[0]), dtype=float)
        if zenith.size != x.size:
            raise ValueError(
                "zenith_angle_deg, x_arcsec, and y_arcsec must have matching lengths; "
                f"got {zenith.size}, {x.size}, {y.size}."
            )
        airmass = np.asarray(zenith_angle_to_airmass(zenith), dtype=float)
        _validate_airmass_support(interpolator, airmass)
        query[NGS_COORD_AIRMASS] = airmass
    elif zenith_angle_deg is not None and interpolator.zenith_angle_deg_axis.size == 1:
        zenith = require_finite_vector(zenith_angle_deg, label="zenith_angle_deg")
        if not np.all(np.isclose(zenith, float(interpolator.zenith_angle_deg_axis[0]), rtol=0.0, atol=_FIELD_ATOL)):
            raise ValueError(
                f"zenith_angle_deg query does not match fixed artifact value {float(interpolator.zenith_angle_deg_axis[0])}."
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
    coordinates = field_coordinates(query[NGS_COORD_X], query[NGS_COORD_Y])
    coordinates = snap_rectangular_field_query(interpolator.x_arcsec, interpolator.y_arcsec, coordinates)
    query = dict(query)
    query[NGS_COORD_X] = coordinates[:, 0]
    query[NGS_COORD_Y] = coordinates[:, 1]
    points = _query_coordinates(interpolator, query)
    axes = _regular_grid_axes(interpolator)
    return {
        name: np.asarray(
            RegularGridInterpolator(
                axes,
                np.asarray(grid, dtype=float),
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
            axes.append(np.asarray(interpolator.airmass_axis, dtype=float))
        elif name == NGS_COORD_Y:
            axes.append(np.asarray(interpolator.y_arcsec, dtype=float))
        elif name == NGS_COORD_X:
            axes.append(np.asarray(interpolator.x_arcsec, dtype=float))
        else:
            raise ValueError(f"Unsupported NGS HO metric coordinate {name!r}.")
    return tuple(axes)


def _validate_airmass_support(interpolator: NgsHoMetricInterpolator, airmass: np.ndarray) -> None:
    axis = np.asarray(interpolator.airmass_axis, dtype=float).reshape(-1)
    minimum = float(np.min(axis))
    maximum = float(np.max(axis))
    if np.any((airmass < minimum) & ~np.isclose(airmass, minimum, rtol=0.0, atol=_FIELD_ATOL)):
        raise ValueError(f"airmass query is below the supported range minimum {minimum}.")
    if np.any((airmass > maximum) & ~np.isclose(airmass, maximum, rtol=0.0, atol=_FIELD_ATOL)):
        raise ValueError(f"airmass query is above the supported range maximum {maximum}.")


def _validate_regular_grid_field_support(interpolator: NgsHoMetricInterpolator, x_arcsec: np.ndarray, y_arcsec: np.ndarray) -> None:
    validate_rectangular_field_query(
        interpolator.x_arcsec,
        interpolator.y_arcsec,
        np.column_stack([x_arcsec, y_arcsec]),
        label="NGS HO metric",
        atol=_FIELD_ATOL,
    )


def _validate_rbf_field_support(interpolator: NgsHoMetricInterpolator, x_arcsec: np.ndarray, y_arcsec: np.ndarray) -> None:
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
    if "wavelength_um" in coordinate_order:
        raise ValueError("NGS HO metric artifacts must not use wavelength_um as an interpolation coordinate.")


def _validate_interpolator(interpolator: NgsHoMetricInterpolator) -> None:
    if interpolator.airmass_axis.shape != interpolator.zenith_angle_deg_axis.shape:
        raise ValueError("airmass_axis shape must match zenith_angle_deg_axis shape.")
    _validate_metric_names(interpolator.metric_names, label="NGS HO metric artifact")
    config = _validate_interpolation_config(interpolator.interpolation_config)
    strategy = _interpolation_strategy(config)
    _validate_coordinate_order_for_strategy(tuple(interpolator.coordinate_order), strategy)
    if NGS_COORD_AIRMASS in interpolator.coordinate_order:
        if interpolator.zenith_angle_deg_axis.size <= 1:
            raise ValueError("active airmass coordinate requires multiple zenith_angle_deg_axis values.")
    elif interpolator.zenith_angle_deg_axis.size > 1:
        raise ValueError("multiple zenith_angle_deg_axis values require active airmass coordinate.")
    if isinstance(config, RegularGridInterpolationConfig):
        if interpolator.x_arcsec.ndim != 1 or interpolator.y_arcsec.ndim != 1:
            raise ValueError("x_arcsec and y_arcsec must be 1-D rectangular field axes for regular-grid artifacts.")
        if interpolator.x_arcsec.size == 0 or interpolator.y_arcsec.size == 0:
            raise ValueError("x_arcsec and y_arcsec field axes must not be empty.")
        if np.any(np.diff(interpolator.x_arcsec) <= 0.0) or np.any(np.diff(interpolator.y_arcsec) <= 0.0):
            raise ValueError("x_arcsec and y_arcsec field axes must be strictly increasing.")
        _validate_regular_grid_metric_model(interpolator)
        return
    if interpolator.x_arcsec.shape != interpolator.y_arcsec.shape:
        raise ValueError("x_arcsec and y_arcsec shapes must match.")
    if interpolator.x_arcsec.size == 0:
        raise ValueError("x_arcsec and y_arcsec must not be empty.")
    _validate_rbf_metric_model(interpolator)


def _validate_rbf_metric_model(interpolator: NgsHoMetricInterpolator) -> None:
    model = dict(interpolator.model)
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
        grid = np.asarray(grid, dtype=float)
        if grid.shape != expected_shape:
            raise ValueError(f"NGS HO metric artifact grid {name!r} must have shape {expected_shape}; got {grid.shape}.")
        if not np.all(np.isfinite(grid)):
            raise ValueError(f"NGS HO metric artifact grid {name!r} must contain only finite values.")


def _expected_regular_grid_shape(interpolator: NgsHoMetricInterpolator) -> tuple[int, ...]:
    sizes = {
        NGS_COORD_AIRMASS: interpolator.airmass_axis.size,
        NGS_COORD_Y: interpolator.y_arcsec.size,
        NGS_COORD_X: interpolator.x_arcsec.size,
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
