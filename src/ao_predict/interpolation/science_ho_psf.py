"""Science high-order PSF interpolation artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ao_predict.simulation.stats import (
    EE_GEOMETRY_ENCIRCLED,
    PsfMetadata,
    compute_psf_ee,
    compute_psf_fwhm,
)
from ao_predict.simulation.validation import validate_meta_field_name

from ._core import (
    axis_index,
    field_coordinates,
    grid_field_values,
    interpolation_axis_weights,
    load_payload,
    rectangular_field_axes,
    require_finite_vector,
    require_positive_scalar,
    require_positive_vector,
    require_pupil,
    save_payload,
    snap_rectangular_field_query,
    unique_sorted,
    validate_rectangular_field_query,
    validate_payload_kind,
    zenith_angle_to_airmass,
)


SCIENCE_HO_PSF_ARTIFACT_KIND = "ao_predict_science_ho_psf_interpolator"
SCIENCE_HO_PSF_ARTIFACT_VERSION = 1
SCIENCE_HO_PSF_FIELD_QUERY_ATOL_ARCSEC = 1.0e-6
SCIENCE_COORD_AIRMASS = "airmass"
SCIENCE_COORD_WAVELENGTH = "wavelength_um"
SCIENCE_COORD_Y = "y_arcsec"
SCIENCE_COORD_X = "x_arcsec"
SCIENCE_FIELD_COORDINATE_ORDER = (SCIENCE_COORD_Y, SCIENCE_COORD_X)
SCIENCE_VALID_COORDINATE_ORDERS = frozenset(
    {
        SCIENCE_FIELD_COORDINATE_ORDER,
        (SCIENCE_COORD_WAVELENGTH, *SCIENCE_FIELD_COORDINATE_ORDER),
        (SCIENCE_COORD_AIRMASS, *SCIENCE_FIELD_COORDINATE_ORDER),
        (SCIENCE_COORD_AIRMASS, SCIENCE_COORD_WAVELENGTH, *SCIENCE_FIELD_COORDINATE_ORDER),
    }
)
_AXIS_ATOL = 1.0e-10


@dataclass(frozen=True, kw_only=True)
class ScienceHoPsfSamples:
    """Science high-order PSF samples used to build an interpolation artifact.

    ``wavelength_um`` is required physical PSF metadata. It becomes an active
    interpolation coordinate only when more than one wavelength is supplied.
    ``zenith_angle_deg`` is optional; when omitted, the artifact has no zenith
    or airmass support check. When supplied with one unique value, it is stored
    as fixed metadata rather than as an active interpolation coordinate.

    Attributes:
        x_arcsec: Science field x-coordinates in arcseconds.
        y_arcsec: Science field y-coordinates in arcseconds.
        psfs: PSF array with shape ``(points, y, x)`` for fixed-plane samples
            or ``(planes, points, y, x)`` for physical-axis samples.
        wavelength_um: Scalar fixed wavelength or one wavelength per plane.
        pixel_scale_mas: Scalar fixed pixel scale or one value per plane.
        tel_diameter_m: Telescope diameter in meters.
        tel_pupil: Shared two-dimensional telescope pupil.
        zenith_angle_deg: Optional scalar fixed zenith angle or one value per
            plane. Multiple unique values activate airmass interpolation.
        meta: Optional source metadata fields. Values may be artifact-global
            finite scalars or finite vectors with one value per source plane.
        provenance: Optional source provenance strings.
    """

    x_arcsec: np.ndarray
    y_arcsec: np.ndarray
    psfs: np.ndarray
    wavelength_um: float | np.ndarray
    pixel_scale_mas: float | np.ndarray
    tel_diameter_m: float
    tel_pupil: np.ndarray
    zenith_angle_deg: float | np.ndarray | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)
    provenance: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScienceHoPsfPrediction:
    """Science high-order PSF prediction returned by an artifact evaluation."""

    psfs: np.ndarray
    pixel_scale_mas: float
    metadata: PsfMetadata
    meta: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ScienceHoPsfReplaySummary:
    """Source-node replay residual summary for a science-HO-PSF artifact."""

    psf_nrms_mean: float
    psf_nrms_max: float
    pixel_scale_abs_max_mas: float
    metric_rms: dict[str, float]
    metric_max_abs: dict[str, float]
    num_planes: int
    num_points: int


@dataclass(frozen=True)
class ScienceHoPsfInterpolator:
    """Versioned science-HO-PSF interpolation artifact.

    ``coordinate_order`` records the active interpolation coordinates. Fixed
    wavelength and fixed zenith values remain in their metadata axes but are not
    included in ``coordinate_order``.
    """

    coordinate_order: tuple[str, ...]
    zenith_angle_deg_axis: np.ndarray
    airmass_axis: np.ndarray
    wavelength_um_axis: np.ndarray
    x_arcsec: np.ndarray
    y_arcsec: np.ndarray
    psf_shape: tuple[int, int]
    pixel_scale_mas_grid: np.ndarray
    tel_diameter_m: float
    tel_pupil: np.ndarray
    psf_grid: np.ndarray
    meta: Mapping[str, Any] = field(default_factory=dict)
    provenance: tuple[str, ...] = ()
    builder: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _ScienceHoPsfRuntimeInterpolator:
    """Process-local prepared interpolator for a science-HO-PSF artifact."""

    artifact: ScienceHoPsfInterpolator
    psf_interpolator: RegularGridInterpolator


def build_science_ho_psf_interpolator(samples: ScienceHoPsfSamples) -> ScienceHoPsfInterpolator:
    """Build a science high-order PSF interpolation artifact.

    Active physical axes are inferred from supplied sample diversity. Field
    coordinates must form a complete rectangular ``y_arcsec x x_arcsec`` grid.
    """

    prepared = _prepare_samples(samples)
    field_x_axis, field_y_axis = rectangular_field_axes(
        prepared.x_arcsec,
        prepared.y_arcsec,
        label="Science HO PSF samples",
    )
    wavelength_axis = unique_sorted(prepared.wavelength_um, label="wavelength_um")
    zenith_axis = (
        unique_sorted(prepared.zenith_angle_deg, label="zenith_angle_deg")
        if prepared.zenith_angle_deg is not None
        else np.asarray([], dtype=float)
    )
    coordinate_order = _coordinate_order_for_axes(zenith_axis, wavelength_axis)
    active_shape = _active_physical_shape(coordinate_order, zenith_axis, wavelength_axis)
    plane_indices = np.full(active_shape, -1, dtype=int)
    psf_grid = np.full(
        (*active_shape, field_y_axis.size, field_x_axis.size, *prepared.psfs.shape[2:]),
        np.nan,
        dtype=np.float32,
    )

    for plane_index in range(prepared.psfs.shape[0]):
        grid_index = _active_grid_index(
            coordinate_order,
            zenith_axis,
            wavelength_axis,
            None if prepared.zenith_angle_deg is None else float(prepared.zenith_angle_deg[plane_index]),
            float(prepared.wavelength_um[plane_index]),
        )
        if plane_indices[grid_index] != -1:
            raise ValueError("Duplicate science HO PSF plane for active physical coordinates.")
        plane_indices[grid_index] = plane_index
        psf_grid[grid_index] = grid_field_values(
            prepared.x_arcsec,
            prepared.y_arcsec,
            prepared.psfs[plane_index],
            field_x_axis,
            field_y_axis,
            label="Science HO PSF sample",
            dtype=np.float32,
        )

    if np.any(plane_indices < 0):
        raise ValueError("Science HO PSF samples must form a complete active physical-coordinate grid.")

    pixel_scale_grid = np.full(active_shape, np.nan, dtype=float)
    for plane_index, pixel_scale_mas in enumerate(prepared.pixel_scale_mas):
        grid_index = _active_grid_index(
            coordinate_order,
            zenith_axis,
            wavelength_axis,
            None if prepared.zenith_angle_deg is None else float(prepared.zenith_angle_deg[plane_index]),
            float(prepared.wavelength_um[plane_index]),
        )
        pixel_scale_grid[grid_index] = float(pixel_scale_mas)
    meta = _build_artifact_meta(prepared.meta, prepared, coordinate_order, zenith_axis, wavelength_axis, active_shape)

    return ScienceHoPsfInterpolator(
        coordinate_order=coordinate_order,
        zenith_angle_deg_axis=zenith_axis,
        airmass_axis=np.asarray(zenith_angle_to_airmass(zenith_axis), dtype=float),
        wavelength_um_axis=wavelength_axis,
        x_arcsec=np.asarray(field_x_axis, dtype=float),
        y_arcsec=np.asarray(field_y_axis, dtype=float),
        psf_shape=tuple(int(v) for v in prepared.psfs.shape[2:]),
        pixel_scale_mas_grid=np.asarray(pixel_scale_grid, dtype=float),
        tel_diameter_m=float(prepared.tel_diameter_m),
        tel_pupil=np.asarray(prepared.tel_pupil, dtype=np.float32),
        meta=meta,
        psf_grid=psf_grid,
        provenance=tuple(prepared.provenance),
        builder={
            "name": "ao_predict.interpolation.science_ho_psf",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "interpolation_method": _interpolation_method(coordinate_order),
        },
    )


def _prepare_science_ho_psf_runtime_interpolator(
    interpolator: ScienceHoPsfInterpolator,
) -> _ScienceHoPsfRuntimeInterpolator:
    """Prepare a runtime interpolator for repeated science-HO-PSF queries."""

    validate_science_ho_psf_interpolator(interpolator)
    return _ScienceHoPsfRuntimeInterpolator(
        artifact=interpolator,
        psf_interpolator=RegularGridInterpolator(
            _interpolation_axes(interpolator),
            np.asarray(interpolator.psf_grid, dtype=np.float32),
            method="linear",
            bounds_error=True,
        ),
    )


def _evaluate_science_ho_psf_runtime_interpolator(
    runtime_interpolator: _ScienceHoPsfRuntimeInterpolator,
    *,
    zenith_angle_deg: float | None = None,
    wavelength_um: float | None = None,
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
) -> ScienceHoPsfPrediction:
    """Evaluate a prepared runtime science-HO-PSF interpolator."""

    if not isinstance(runtime_interpolator, _ScienceHoPsfRuntimeInterpolator):
        raise TypeError("runtime_interpolator must be a _ScienceHoPsfRuntimeInterpolator instance.")
    interpolator = runtime_interpolator.artifact
    coordinates = field_coordinates(x_arcsec, y_arcsec)
    _validate_field_query(interpolator, coordinates)
    coordinates = _snap_field_query(interpolator, coordinates)
    physical_values = _validate_and_prepare_query_physical_values(
        interpolator,
        zenith_angle_deg=zenith_angle_deg,
        wavelength_um=wavelength_um,
    )
    points = _query_points(interpolator, physical_values, coordinates)
    psfs = np.asarray(runtime_interpolator.psf_interpolator(points), dtype=np.float32).reshape(
        coordinates.shape[0],
        *interpolator.psf_shape,
    )
    weights = _plane_weights(interpolator, physical_values)
    pixel_scale_mas = 0.0
    meta = _initial_evaluated_meta(interpolator.meta, grid_shape=_active_grid_shape(interpolator))
    for weight, grid_index in weights:
        pixel_scale_mas += float(weight) * float(np.asarray(interpolator.pixel_scale_mas_grid)[grid_index])
        _accumulate_grid_meta(meta, interpolator.meta, weight=float(weight), grid_index=grid_index)
    _validate_psf_flux(psfs, label="predicted psfs")
    wavelength_value = _prediction_wavelength_um(interpolator, physical_values)
    pixel_scale_mas = float(pixel_scale_mas)
    return ScienceHoPsfPrediction(
        psfs=psfs,
        pixel_scale_mas=pixel_scale_mas,
        metadata=PsfMetadata(
            wavelength_um=wavelength_value,
            pixel_scale_mas=pixel_scale_mas,
            tel_diameter_m=float(interpolator.tel_diameter_m),
            tel_pupil=np.asarray(interpolator.tel_pupil, dtype=np.float32),
        ),
        meta=meta,
    )


def evaluate_science_ho_psf_interpolator(
    interpolator: ScienceHoPsfInterpolator,
    *,
    zenith_angle_deg: float | None = None,
    wavelength_um: float | None = None,
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
) -> ScienceHoPsfPrediction:
    """Evaluate a science-HO-PSF artifact at supported coordinates.

    ``zenith_angle_deg`` is required only when the artifact has an active
    ``airmass`` coordinate. ``wavelength_um`` is required only when wavelength
    is active; otherwise a supplied value is checked against the fixed artifact
    wavelength.
    """

    return _evaluate_science_ho_psf_runtime_interpolator(
        _prepare_science_ho_psf_runtime_interpolator(interpolator),
        zenith_angle_deg=zenith_angle_deg,
        wavelength_um=wavelength_um,
        x_arcsec=x_arcsec,
        y_arcsec=y_arcsec,
    )


def validate_science_ho_psf_query(
    interpolator: ScienceHoPsfInterpolator,
    *,
    zenith_angle_deg: float | None = None,
    wavelength_um: float | None = None,
) -> None:
    """Validate science-HO-PSF physical-coordinate support without evaluating PSFs."""

    if not isinstance(interpolator, ScienceHoPsfInterpolator):
        raise TypeError("interpolator must be a ScienceHoPsfInterpolator instance.")
    _validate_interpolator(interpolator)
    physical_values = _validate_and_prepare_query_physical_values(
        interpolator,
        zenith_angle_deg=zenith_angle_deg,
        wavelength_um=wavelength_um,
    )
    _plane_weights(interpolator, physical_values)


def validate_science_ho_psf_interpolator(interpolator: ScienceHoPsfInterpolator) -> None:
    """Validate a science-HO-PSF artifact contract."""

    if not isinstance(interpolator, ScienceHoPsfInterpolator):
        raise TypeError("interpolator must be a ScienceHoPsfInterpolator instance.")
    _validate_interpolator(interpolator)


def replay_science_ho_psf_interpolator(
    interpolator: ScienceHoPsfInterpolator,
    samples: ScienceHoPsfSamples,
) -> ScienceHoPsfReplaySummary:
    """Replay a science-HO-PSF artifact at source sample nodes."""

    prepared = _prepare_samples(samples)
    residuals: list[np.ndarray] = []
    pixel_scale_errors: list[float] = []
    metric_errors: dict[str, list[np.ndarray]] = {"fwhm_mas": [], "ee": []}
    for index in range(prepared.psfs.shape[0]):
        wavelength_um = float(prepared.wavelength_um[index])
        prediction = evaluate_science_ho_psf_interpolator(
            interpolator,
            zenith_angle_deg=None if prepared.zenith_angle_deg is None else float(prepared.zenith_angle_deg[index]),
            wavelength_um=wavelength_um,
            x_arcsec=prepared.x_arcsec,
            y_arcsec=prepared.y_arcsec,
        )
        reference = np.asarray(prepared.psfs[index], dtype=np.float32)
        residuals.append(_psf_nrms(reference, prediction.psfs))
        pixel_scale_errors.append(abs(float(prepared.pixel_scale_mas[index]) - prediction.pixel_scale_mas))
        plane_metric_errors = _science_metric_errors(
            reference,
            prediction.psfs,
            wavelength_um=wavelength_um,
            reference_pixel_scale_mas=float(prepared.pixel_scale_mas[index]),
            prediction_metadata=prediction.metadata,
            tel_diameter_m=float(prepared.tel_diameter_m),
            tel_pupil=prepared.tel_pupil,
        )
        for name, values in plane_metric_errors.items():
            metric_errors[name].append(values)
    residual = np.concatenate(residuals)
    metric_error_arrays = {name: np.concatenate(values) for name, values in metric_errors.items()}
    return ScienceHoPsfReplaySummary(
        psf_nrms_mean=float(np.mean(residual)),
        psf_nrms_max=float(np.max(residual)),
        pixel_scale_abs_max_mas=float(np.max(pixel_scale_errors)),
        metric_rms={name: float(np.sqrt(np.mean(values**2))) for name, values in metric_error_arrays.items()},
        metric_max_abs={name: float(np.max(np.abs(values))) for name, values in metric_error_arrays.items()},
        num_planes=int(prepared.psfs.shape[0]),
        num_points=int(prepared.psfs.shape[1]),
    )


def save_science_ho_psf_interpolator(
    interpolator: ScienceHoPsfInterpolator,
    path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Save a science-HO-PSF interpolation artifact."""

    save_payload(_science_to_payload(interpolator), Path(path), overwrite=overwrite)


def load_science_ho_psf_interpolator(path: Path) -> ScienceHoPsfInterpolator:
    """Load and validate a science-HO-PSF interpolation artifact."""

    return _science_from_payload(load_payload(Path(path)))


# Sample preparation and grid construction


def _prepare_samples(samples: ScienceHoPsfSamples) -> ScienceHoPsfSamples:
    if not isinstance(samples, ScienceHoPsfSamples):
        raise TypeError("samples must be a ScienceHoPsfSamples instance.")
    coordinates = field_coordinates(samples.x_arcsec, samples.y_arcsec)
    psfs = _validate_source_psfs(samples.psfs, label="psfs", num_points=coordinates.shape[0])
    num_planes = psfs.shape[0]
    wavelength_um = _prepare_required_plane_values(
        samples.wavelength_um,
        label="wavelength_um",
        length=num_planes,
        positive=True,
    )
    pixel_scale_mas = _prepare_required_plane_values(
        samples.pixel_scale_mas,
        label="pixel_scale_mas",
        length=num_planes,
        positive=True,
    )
    zenith_angle_deg = _prepare_optional_plane_values(
        samples.zenith_angle_deg,
        label="zenith_angle_deg",
        length=num_planes,
    )
    return ScienceHoPsfSamples(
        x_arcsec=np.asarray(samples.x_arcsec, dtype=float).reshape(-1),
        y_arcsec=np.asarray(samples.y_arcsec, dtype=float).reshape(-1),
        psfs=psfs,
        wavelength_um=wavelength_um,
        pixel_scale_mas=pixel_scale_mas,
        tel_diameter_m=require_positive_scalar(samples.tel_diameter_m, label="tel_diameter_m"),
        tel_pupil=require_pupil(samples.tel_pupil),
        zenith_angle_deg=zenith_angle_deg,
        meta=_prepare_source_meta(samples.meta, num_planes=num_planes),
        provenance=tuple(str(value) for value in samples.provenance),
    )


def _validate_source_psfs(psfs: Any, *, label: str, num_points: int) -> np.ndarray:
    array = np.asarray(psfs, dtype=np.float32)
    if array.ndim == 3:
        array = array[np.newaxis, ...]
    if array.ndim != 4:
        raise ValueError(f"{label} must have ndim=3 or ndim=4, got shape {array.shape}.")
    if array.shape[1] != int(num_points):
        raise ValueError(
            f"{label} point dimension must match {int(num_points)} field coordinates; got shape {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values.")
    _validate_psf_flux(array, label=label)
    return array


def _prepare_required_plane_values(value: Any, *, label: str, length: int, positive: bool) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        vector = np.full(int(length), float(array), dtype=float)
    else:
        vector = array.reshape(-1)
        if vector.size != int(length):
            raise ValueError(f"{label} must be scalar or have one value per plane; got length {vector.size}.")
    if positive:
        return require_positive_vector(vector, label=label, length=length)
    return require_finite_vector(vector, label=label, length=length)


def _prepare_optional_plane_values(value: Any, *, label: str, length: int) -> np.ndarray | None:
    if value is None:
        return None
    return _prepare_required_plane_values(value, label=label, length=length, positive=False)


def _coordinate_order_for_axes(zenith_axis: np.ndarray, wavelength_axis: np.ndarray) -> tuple[str, ...]:
    order: list[str] = []
    if zenith_axis.size > 1:
        order.append(SCIENCE_COORD_AIRMASS)
    if wavelength_axis.size > 1:
        order.append(SCIENCE_COORD_WAVELENGTH)
    order.extend(SCIENCE_FIELD_COORDINATE_ORDER)
    coordinate_order = tuple(order)
    _validate_coordinate_order(coordinate_order)
    return coordinate_order


def _validate_coordinate_order(coordinate_order: tuple[str, ...]) -> None:
    if tuple(coordinate_order) not in SCIENCE_VALID_COORDINATE_ORDERS:
        raise ValueError(f"Unsupported science HO PSF coordinate_order: {tuple(coordinate_order)!r}.")


def _active_physical_shape(
    coordinate_order: tuple[str, ...],
    zenith_axis: np.ndarray,
    wavelength_axis: np.ndarray,
) -> tuple[int, ...]:
    shape: list[int] = []
    if SCIENCE_COORD_AIRMASS in coordinate_order:
        shape.append(int(zenith_axis.size))
    if SCIENCE_COORD_WAVELENGTH in coordinate_order:
        shape.append(int(wavelength_axis.size))
    return tuple(shape)


def _active_grid_index(
    coordinate_order: tuple[str, ...],
    zenith_axis: np.ndarray,
    wavelength_axis: np.ndarray,
    zenith_angle_deg: float | None,
    wavelength_um: float,
) -> tuple[int, ...]:
    index: list[int] = []
    if SCIENCE_COORD_AIRMASS in coordinate_order:
        if zenith_angle_deg is None:
            raise ValueError("zenith_angle_deg is required for active airmass interpolation.")
        index.append(axis_index(zenith_axis, float(zenith_angle_deg), label="zenith_angle_deg"))
    if SCIENCE_COORD_WAVELENGTH in coordinate_order:
        index.append(axis_index(wavelength_axis, float(wavelength_um), label="wavelength_um"))
    return tuple(index)


# Query helpers


def _validate_and_prepare_query_physical_values(
    interpolator: ScienceHoPsfInterpolator,
    *,
    zenith_angle_deg: float | None,
    wavelength_um: float | None,
) -> dict[str, float]:
    values: dict[str, float] = {}
    if SCIENCE_COORD_AIRMASS in interpolator.coordinate_order:
        if zenith_angle_deg is None:
            raise ValueError("zenith_angle_deg is required by this science HO PSF artifact.")
        values[SCIENCE_COORD_AIRMASS] = float(zenith_angle_to_airmass(float(zenith_angle_deg)))
    elif zenith_angle_deg is not None and interpolator.zenith_angle_deg_axis.size == 1:
        _validate_fixed_value(
            float(zenith_angle_deg),
            float(interpolator.zenith_angle_deg_axis[0]),
            label="zenith_angle_deg",
        )

    if SCIENCE_COORD_WAVELENGTH in interpolator.coordinate_order:
        if wavelength_um is None:
            raise ValueError("wavelength_um is required by this science HO PSF artifact.")
        values[SCIENCE_COORD_WAVELENGTH] = require_positive_scalar(wavelength_um, label="wavelength_um")
    else:
        fixed_wavelength = float(interpolator.wavelength_um_axis[0])
        if wavelength_um is not None:
            _validate_fixed_value(
                require_positive_scalar(wavelength_um, label="wavelength_um"),
                fixed_wavelength,
                label="wavelength_um",
            )
        values[SCIENCE_COORD_WAVELENGTH] = fixed_wavelength
    return values


def _validate_fixed_value(value: float, fixed: float, *, label: str) -> None:
    if not np.isclose(float(value), float(fixed), rtol=0.0, atol=_AXIS_ATOL):
        raise ValueError(f"{label}={value} does not match fixed artifact value {fixed}.")


def _query_points(
    interpolator: ScienceHoPsfInterpolator,
    physical_values: Mapping[str, float],
    coordinates: np.ndarray,
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for name in interpolator.coordinate_order:
        if name == SCIENCE_COORD_AIRMASS:
            columns.append(np.full(coordinates.shape[0], float(physical_values[SCIENCE_COORD_AIRMASS]), dtype=float))
        elif name == SCIENCE_COORD_WAVELENGTH:
            columns.append(np.full(coordinates.shape[0], float(physical_values[SCIENCE_COORD_WAVELENGTH]), dtype=float))
        elif name == SCIENCE_COORD_Y:
            columns.append(coordinates[:, 1])
        elif name == SCIENCE_COORD_X:
            columns.append(coordinates[:, 0])
        else:
            raise ValueError(f"Unsupported science HO PSF coordinate {name!r}.")
    return np.column_stack(columns)


def _plane_weights(
    interpolator: ScienceHoPsfInterpolator,
    physical_values: Mapping[str, float],
) -> list[tuple[float, tuple[int, ...]]]:
    weights: list[tuple[float, tuple[int, ...]]] = [(1.0, ())]
    if SCIENCE_COORD_AIRMASS in interpolator.coordinate_order:
        weights = _extend_weights(
            weights,
            interpolation_axis_weights(interpolator.airmass_axis, physical_values[SCIENCE_COORD_AIRMASS], label="airmass"),
        )
    if SCIENCE_COORD_WAVELENGTH in interpolator.coordinate_order:
        weights = _extend_weights(
            weights,
            interpolation_axis_weights(
                interpolator.wavelength_um_axis,
                physical_values[SCIENCE_COORD_WAVELENGTH],
                label="wavelength_um",
            ),
        )
    return weights


def _extend_weights(
    current: list[tuple[float, tuple[int, ...]]],
    axis_weights: list[tuple[int, float]],
) -> list[tuple[float, tuple[int, ...]]]:
    return [
        (float(weight * axis_weight), (*index, int(axis_index_value)))
        for weight, index in current
        for axis_index_value, axis_weight in axis_weights
    ]


def _prediction_wavelength_um(interpolator: ScienceHoPsfInterpolator, physical_values: Mapping[str, float]) -> float:
    if SCIENCE_COORD_WAVELENGTH in interpolator.coordinate_order:
        return float(physical_values[SCIENCE_COORD_WAVELENGTH])
    return float(interpolator.wavelength_um_axis[0])


def _interpolation_axes(interpolator: ScienceHoPsfInterpolator) -> tuple[np.ndarray, ...]:
    axes: list[np.ndarray] = []
    for name in interpolator.coordinate_order:
        if name == SCIENCE_COORD_AIRMASS:
            axes.append(np.asarray(interpolator.airmass_axis, dtype=float))
        elif name == SCIENCE_COORD_WAVELENGTH:
            axes.append(np.asarray(interpolator.wavelength_um_axis, dtype=float))
        elif name == SCIENCE_COORD_Y:
            axes.append(np.asarray(interpolator.y_arcsec, dtype=float))
        elif name == SCIENCE_COORD_X:
            axes.append(np.asarray(interpolator.x_arcsec, dtype=float))
        else:
            raise ValueError(f"Unsupported science HO PSF coordinate {name!r}.")
    return tuple(axes)


def _active_grid_shape(interpolator: ScienceHoPsfInterpolator) -> tuple[int, ...]:
    return _active_physical_shape(
        interpolator.coordinate_order,
        np.asarray(interpolator.zenith_angle_deg_axis, dtype=float),
        np.asarray(interpolator.wavelength_um_axis, dtype=float),
    )


# Validation and replay helpers


def _validate_field_query(interpolator: ScienceHoPsfInterpolator, coordinates: np.ndarray) -> None:
    validate_rectangular_field_query(
        interpolator.x_arcsec,
        interpolator.y_arcsec,
        coordinates,
        label="science HO PSF",
        atol=SCIENCE_HO_PSF_FIELD_QUERY_ATOL_ARCSEC,
    )


def _snap_field_query(interpolator: ScienceHoPsfInterpolator, coordinates: np.ndarray) -> np.ndarray:
    return snap_rectangular_field_query(interpolator.x_arcsec, interpolator.y_arcsec, coordinates)


def _psf_nrms(reference: np.ndarray, measured: np.ndarray) -> np.ndarray:
    numerator = np.sqrt(np.mean((np.asarray(measured) - np.asarray(reference)) ** 2, axis=(-2, -1)))
    denominator = np.sqrt(np.mean(np.asarray(reference) ** 2, axis=(-2, -1)))
    return numerator / denominator


def _validate_psf_flux(psfs: np.ndarray, *, label: str) -> None:
    flux = np.sum(np.asarray(psfs, dtype=np.float64), axis=(-2, -1))
    if not np.all(np.isfinite(flux)):
        raise ValueError(f"{label} must have finite per-PSF total flux.")
    if np.any(flux <= 0.0):
        raise ValueError(f"{label} must have strictly positive per-PSF total flux.")


def _prepare_source_meta(meta: Mapping[str, Any], *, num_planes: int) -> dict[str, float | np.ndarray]:
    if not isinstance(meta, Mapping):
        raise TypeError("meta must be a mapping from field name to scalar or per-plane numeric values.")
    prepared: dict[str, float | np.ndarray] = {}
    for raw_name, raw_value in meta.items():
        name = validate_meta_field_name(raw_name, label="ScienceHoPsfSamples.meta")
        if name in prepared:
            raise ValueError(f"ScienceHoPsfSamples.meta contains duplicate field name {name!r}.")
        value = np.asarray(raw_value, dtype=float)
        if value.ndim == 0:
            scalar = float(value)
            if not np.isfinite(scalar):
                raise ValueError(f"ScienceHoPsfSamples.meta[{name!r}] must be finite.")
            prepared[name] = scalar
            continue
        vector = value.reshape(-1)
        if vector.size != int(num_planes):
            raise ValueError(
                f"ScienceHoPsfSamples.meta[{name!r}] must be scalar or have one value per plane; "
                f"got length {vector.size}, expected {int(num_planes)}."
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"ScienceHoPsfSamples.meta[{name!r}] must contain only finite values.")
        prepared[name] = vector.astype(float, copy=False)
    return prepared


def _build_artifact_meta(
    source_meta: Mapping[str, float | np.ndarray],
    samples: ScienceHoPsfSamples,
    coordinate_order: tuple[str, ...],
    zenith_axis: np.ndarray,
    wavelength_axis: np.ndarray,
    active_shape: tuple[int, ...],
) -> dict[str, float | np.ndarray]:
    artifact_meta: dict[str, float | np.ndarray] = {}
    for name, value in source_meta.items():
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            artifact_meta[name] = float(array)
            continue
        grid = np.full(active_shape, np.nan, dtype=float)
        for plane_index, meta_value in enumerate(array.reshape(-1)):
            grid_index = _active_grid_index(
                coordinate_order,
                zenith_axis,
                wavelength_axis,
                None if samples.zenith_angle_deg is None else float(samples.zenith_angle_deg[plane_index]),
                float(samples.wavelength_um[plane_index]),
            )
            grid[grid_index] = float(meta_value)
        artifact_meta[name] = grid
    return artifact_meta


def _initial_evaluated_meta(meta: Mapping[str, Any], *, grid_shape: tuple[int, ...]) -> dict[str, float]:
    evaluated: dict[str, float] = {}
    for name, value in _validate_artifact_meta(meta, grid_shape=grid_shape).items():
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            evaluated[name] = float(array)
        else:
            evaluated[name] = 0.0
    return evaluated


def _accumulate_grid_meta(
    evaluated: dict[str, float],
    meta: Mapping[str, Any],
    *,
    weight: float,
    grid_index: tuple[int, ...],
) -> None:
    for name, value in meta.items():
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            continue
        evaluated[name] += float(weight) * float(array[grid_index])


def _validate_artifact_meta(meta: Mapping[str, Any], *, grid_shape: tuple[int, ...] | None) -> dict[str, float | np.ndarray]:
    if not isinstance(meta, Mapping):
        raise TypeError("metadata.meta must be a mapping.")
    validated: dict[str, float | np.ndarray] = {}
    for raw_name, raw_value in meta.items():
        name = validate_meta_field_name(raw_name, label="metadata.meta")
        if name in validated:
            raise ValueError(f"metadata.meta contains duplicate field name {name!r}.")
        value = np.asarray(raw_value, dtype=float)
        if value.ndim == 0:
            scalar = float(value)
            if not np.isfinite(scalar):
                raise ValueError(f"metadata.meta[{name!r}] must be finite.")
            validated[name] = scalar
            continue
        if grid_shape is not None and value.shape != grid_shape:
            raise ValueError(
                f"metadata.meta[{name!r}] must be scalar or have shape {grid_shape}; got {value.shape}."
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(f"metadata.meta[{name!r}] must contain only finite values.")
        validated[name] = np.asarray(value, dtype=float)
    return validated


def _science_metric_errors(
    reference_psfs: np.ndarray,
    measured_psfs: np.ndarray,
    *,
    wavelength_um: float,
    reference_pixel_scale_mas: float,
    prediction_metadata: PsfMetadata,
    tel_diameter_m: float,
    tel_pupil: np.ndarray,
) -> dict[str, np.ndarray]:
    reference_metadata = PsfMetadata(
        wavelength_um=float(wavelength_um),
        pixel_scale_mas=float(reference_pixel_scale_mas),
        tel_diameter_m=float(tel_diameter_m),
        tel_pupil=np.asarray(tel_pupil, dtype=np.float32),
    )
    reference_fwhm = np.asarray(compute_psf_fwhm(reference_psfs, reference_metadata, preprocess="default"), dtype=float).reshape(-1)
    measured_fwhm = np.asarray(compute_psf_fwhm(measured_psfs, prediction_metadata, preprocess="default"), dtype=float).reshape(-1)
    if not np.all(np.isfinite(reference_fwhm)) or not np.all(np.isfinite(measured_fwhm)):
        raise ValueError("Science HO PSF replay could not compute finite fwhm_mas metrics.")

    ee_aperture_diameters_mas = np.asarray(2.0 * reference_fwhm, dtype=np.float32).reshape(-1, 1)
    reference_ee = np.asarray(
        compute_psf_ee(
            reference_psfs,
            reference_metadata,
            ee_apertures_mas=ee_aperture_diameters_mas,
            ee_geometry=EE_GEOMETRY_ENCIRCLED,
            preprocess="default",
        ),
        dtype=float,
    ).reshape(-1)
    measured_ee = np.asarray(
        compute_psf_ee(
            measured_psfs,
            prediction_metadata,
            ee_apertures_mas=ee_aperture_diameters_mas,
            ee_geometry=EE_GEOMETRY_ENCIRCLED,
            preprocess="default",
        ),
        dtype=float,
    ).reshape(-1)
    if not np.all(np.isfinite(reference_ee)) or not np.all(np.isfinite(measured_ee)):
        raise ValueError("Science HO PSF replay could not compute finite ee metrics.")
    return {
        "fwhm_mas": measured_fwhm - reference_fwhm,
        "ee": measured_ee - reference_ee,
    }


# Persistence and validation


def _interpolation_method(coordinate_order: tuple[str, ...]) -> str:
    label_by_coordinate = {
        SCIENCE_COORD_AIRMASS: "airmass",
        SCIENCE_COORD_WAVELENGTH: "wavelength",
        SCIENCE_COORD_Y: "y",
        SCIENCE_COORD_X: "x",
    }
    return "regular_grid_linear_" + "_".join(label_by_coordinate[name] for name in coordinate_order)


def _science_to_payload(interpolator: ScienceHoPsfInterpolator) -> dict[str, Any]:
    if not isinstance(interpolator, ScienceHoPsfInterpolator):
        raise TypeError("interpolator must be a ScienceHoPsfInterpolator instance.")
    _validate_interpolator(interpolator)
    return {
        "kind": SCIENCE_HO_PSF_ARTIFACT_KIND,
        "version": SCIENCE_HO_PSF_ARTIFACT_VERSION,
        "builder": dict(interpolator.builder),
        "interpolation": {
            "method": _interpolation_method(interpolator.coordinate_order),
            "coordinate_order": tuple(interpolator.coordinate_order),
        },
        "metadata": {
            "zenith_angle_deg_axis": np.asarray(interpolator.zenith_angle_deg_axis, dtype=float),
            "airmass_axis": np.asarray(interpolator.airmass_axis, dtype=float),
            "wavelength_um_axis": np.asarray(interpolator.wavelength_um_axis, dtype=float),
            "x_arcsec": np.asarray(interpolator.x_arcsec, dtype=float),
            "y_arcsec": np.asarray(interpolator.y_arcsec, dtype=float),
            "psf_shape": tuple(interpolator.psf_shape),
            "pixel_scale_mas_grid": np.asarray(interpolator.pixel_scale_mas_grid, dtype=float),
            "tel_diameter_m": float(interpolator.tel_diameter_m),
            "tel_pupil": np.asarray(interpolator.tel_pupil, dtype=np.float32),
            "meta": {name: np.asarray(value, dtype=float) for name, value in interpolator.meta.items()},
            "provenance": tuple(interpolator.provenance),
        },
        "model": {
            "psf_grid": np.asarray(interpolator.psf_grid, dtype=np.float32),
        },
    }


def _science_from_payload(payload: Mapping[str, Any]) -> ScienceHoPsfInterpolator:
    validate_payload_kind(payload, kind=SCIENCE_HO_PSF_ARTIFACT_KIND, version=SCIENCE_HO_PSF_ARTIFACT_VERSION)
    interpolation = dict(payload.get("interpolation", {}))
    coordinate_order = tuple(str(value) for value in interpolation.get("coordinate_order", ()))
    _validate_coordinate_order(coordinate_order)
    metadata = dict(payload.get("metadata", {}))
    model = dict(payload.get("model", {}))
    interpolator = ScienceHoPsfInterpolator(
        coordinate_order=coordinate_order,
        zenith_angle_deg_axis=require_finite_vector(
            metadata.get("zenith_angle_deg_axis"),
            label="metadata.zenith_angle_deg_axis",
        ),
        airmass_axis=require_finite_vector(metadata.get("airmass_axis"), label="metadata.airmass_axis"),
        wavelength_um_axis=require_positive_vector(
            metadata.get("wavelength_um_axis"),
            label="metadata.wavelength_um_axis",
        ),
        x_arcsec=require_finite_vector(metadata.get("x_arcsec"), label="metadata.x_arcsec"),
        y_arcsec=require_finite_vector(metadata.get("y_arcsec"), label="metadata.y_arcsec"),
        psf_shape=tuple(int(value) for value in metadata.get("psf_shape", ())),
        pixel_scale_mas_grid=np.asarray(metadata.get("pixel_scale_mas_grid"), dtype=float),
        tel_diameter_m=require_positive_scalar(metadata.get("tel_diameter_m"), label="metadata.tel_diameter_m"),
        tel_pupil=require_pupil(metadata.get("tel_pupil"), label="metadata.tel_pupil"),
        meta=dict(metadata.get("meta", {})),
        psf_grid=np.asarray(model.get("psf_grid"), dtype=np.float32),
        provenance=tuple(str(value) for value in metadata.get("provenance", ())),
        builder=dict(payload.get("builder", {})),
    )
    _validate_interpolator(interpolator)
    return interpolator


def _validate_interpolator(interpolator: ScienceHoPsfInterpolator) -> None:
    _validate_coordinate_order(tuple(interpolator.coordinate_order))
    if len(interpolator.psf_shape) != 2 or any(int(value) <= 0 for value in interpolator.psf_shape):
        raise ValueError("metadata.psf_shape must contain two positive dimensions.")
    if interpolator.airmass_axis.shape != interpolator.zenith_angle_deg_axis.shape:
        raise ValueError("airmass_axis shape must match zenith_angle_deg_axis shape.")
    if SCIENCE_COORD_AIRMASS in interpolator.coordinate_order:
        if interpolator.zenith_angle_deg_axis.size <= 1:
            raise ValueError("active airmass coordinate requires multiple zenith_angle_deg_axis values.")
    elif interpolator.zenith_angle_deg_axis.size > 1:
        raise ValueError("multiple zenith_angle_deg_axis values require active airmass coordinate.")
    if SCIENCE_COORD_WAVELENGTH in interpolator.coordinate_order:
        if interpolator.wavelength_um_axis.size <= 1:
            raise ValueError("active wavelength_um coordinate requires multiple wavelength_um_axis values.")
    elif interpolator.wavelength_um_axis.size != 1:
        raise ValueError("fixed wavelength artifacts must store exactly one wavelength_um_axis value.")
    if interpolator.x_arcsec.ndim != 1 or interpolator.y_arcsec.ndim != 1:
        raise ValueError("x_arcsec and y_arcsec must be 1-D rectangular field axes.")
    if interpolator.x_arcsec.size == 0 or interpolator.y_arcsec.size == 0:
        raise ValueError("x_arcsec and y_arcsec field axes must not be empty.")
    if np.any(np.diff(interpolator.x_arcsec) <= 0.0) or np.any(np.diff(interpolator.y_arcsec) <= 0.0):
        raise ValueError("x_arcsec and y_arcsec field axes must be strictly increasing.")
    active_shape = _active_grid_shape(interpolator)
    if np.asarray(interpolator.pixel_scale_mas_grid).shape != active_shape:
        raise ValueError(f"pixel_scale_mas_grid shape must be {active_shape}.")
    if not np.all(np.isfinite(interpolator.pixel_scale_mas_grid)):
        raise ValueError("pixel_scale_mas_grid must contain only finite values.")
    _validate_artifact_meta(interpolator.meta, grid_shape=active_shape)
    expected_psf_grid_shape = _expected_model_grid_shape(interpolator)
    psf_grid = np.asarray(interpolator.psf_grid, dtype=np.float32)
    if psf_grid.shape != expected_psf_grid_shape:
        raise ValueError(f"psf_grid shape must be {expected_psf_grid_shape}; got {psf_grid.shape}.")
    if not np.all(np.isfinite(psf_grid)):
        raise ValueError("psf_grid must contain only finite values.")
    _validate_psf_flux(psf_grid, label="psf_grid")


def _expected_model_grid_shape(interpolator: ScienceHoPsfInterpolator) -> tuple[int, ...]:
    sizes = {
        SCIENCE_COORD_AIRMASS: interpolator.airmass_axis.size,
        SCIENCE_COORD_WAVELENGTH: interpolator.wavelength_um_axis.size,
        SCIENCE_COORD_Y: interpolator.y_arcsec.size,
        SCIENCE_COORD_X: interpolator.x_arcsec.size,
    }
    return tuple(int(sizes[name]) for name in interpolator.coordinate_order) + tuple(interpolator.psf_shape)
