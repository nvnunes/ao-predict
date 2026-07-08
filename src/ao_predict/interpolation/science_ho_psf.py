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
    interpolation_axis_weights,
    load_payload,
    require_finite_vector,
    require_positive_scalar,
    require_positive_vector,
    require_pupil,
    save_payload,
    unique_sorted,
    validate_payload_kind,
    zenith_angle_to_airmass,
)


SCIENCE_HO_PSF_ARTIFACT_KIND = "ao_predict_science_ho_psf_interpolator"
SCIENCE_HO_PSF_ARTIFACT_VERSION = 1
SCIENCE_HO_PSF_INTERPOLATION_METHOD = "regular_grid_linear_airmass_wavelength_y_x"
SCIENCE_HO_PSF_FIELD_QUERY_ATOL_ARCSEC = 1.0e-6


@dataclass(frozen=True)
class ScienceHoPsfSamples:
    """Science high-order PSF samples used to build an interpolation artifact.

    The public plane axes are zenith angle and wavelength. Each plane contains
    one PSF at every field coordinate. The science field coordinates must form
    a complete rectangular ``y_arcsec x x_arcsec`` grid before building an
    artifact. Pixel scale is a plane-level value and full ``PsfMetadata``
    fields are retained for future PSF-statistic use. PSFs are interpolated in
    the source artifact flux convention; this artifact does not clip or
    normalize them.

    Attributes:
        zenith_angle_deg: Plane zenith angles in degrees, one value per plane.
        wavelength_um: Plane wavelengths in microns, one value per plane.
        x_arcsec: Science field x-coordinates in arcseconds.
        y_arcsec: Science field y-coordinates in arcseconds.
        psfs: PSF array with shape ``(planes, points, y, x)``. Each PSF must
            have finite pixels and strictly positive total flux.
        pixel_scale_mas: Plane pixel scales in milliarcseconds per pixel.
        tel_diameter_m: Telescope diameter in meters.
        tel_pupil: Shared two-dimensional telescope pupil.
        meta: Optional source metadata fields. Each key becomes an additional
            scalar ``/meta`` field for simulations that consume the artifact.
            Values may be artifact-global finite scalars or finite vectors with
            one value per zenith-angle/wavelength source plane.
        provenance: Optional source provenance strings.
    """

    zenith_angle_deg: np.ndarray
    wavelength_um: np.ndarray
    x_arcsec: np.ndarray
    y_arcsec: np.ndarray
    psfs: np.ndarray
    pixel_scale_mas: np.ndarray
    tel_diameter_m: float
    tel_pupil: np.ndarray
    meta: Mapping[str, Any] = field(default_factory=dict)
    provenance: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScienceHoPsfPrediction:
    """Science high-order PSF prediction returned by an artifact evaluation.

    Attributes:
        psfs: Predicted PSF cube with shape ``(points, y, x)`` in the source
            artifact flux convention.
        pixel_scale_mas: Artifact-backed pixel scale in milliarcseconds per
            pixel for the evaluated zenith-angle and wavelength plane.
        metadata: ``PsfMetadata`` carrying the requested wavelength, evaluated
            pixel scale, telescope diameter, and telescope pupil.
        meta: Evaluated scalar source metadata fields for the requested
            zenith-angle/wavelength plane.
    """

    psfs: np.ndarray
    pixel_scale_mas: float
    metadata: PsfMetadata
    meta: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ScienceHoPsfReplaySummary:
    """Source-node replay residual summary for a science-HO-PSF artifact.

    Attributes:
        psf_nrms_mean: Mean normalized RMS residual between flux-preserving
            replayed PSFs and source PSFs.
        psf_nrms_max: Maximum normalized RMS residual between flux-preserving
            replayed PSFs and source PSFs.
        pixel_scale_abs_max_mas: Maximum absolute pixel-scale residual in
            milliarcseconds per pixel.
        metric_rms: RMS residuals for replayed AO Predict metrics. Keys are
            metric names such as ``"fwhm_mas"`` and ``"ee"``.
        metric_max_abs: Maximum absolute residuals for replayed AO Predict
            metrics.
        num_planes: Number of zenith-angle/wavelength planes replayed.
        num_points: Number of science field points per plane.
    """

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

    The artifact stores a complete rectangular zenith-angle by wavelength by
    science-field grid. Interpolation is linear in derived airmass,
    wavelength, and rectangular field ``y_arcsec``/``x_arcsec`` axes. The
    stored PSFs preserve the source PSF flux convention instead of normalizing
    PSFs to unit flux.

    Attributes:
        zenith_angle_deg_axis: Supported zenith-angle axis in degrees.
        airmass_axis: Airmass values derived from ``zenith_angle_deg_axis``.
        wavelength_um_axis: Supported wavelength axis in microns.
        x_arcsec: Rectangular science field x-axis in arcseconds.
        y_arcsec: Rectangular science field y-axis in arcseconds.
        psf_shape: Native two-dimensional PSF image shape ``(y, x)``.
        pixel_scale_mas_grid: Plane pixel scales in milliarcseconds per pixel
            with shape ``(zenith, wavelength)``.
        tel_diameter_m: Telescope diameter in meters shared by predictions.
        tel_pupil: Telescope pupil array shared by predictions.
        meta: Source metadata fields. Each value is either an artifact-global
            scalar or a grid with the same ``(zenith, wavelength)`` shape as
            ``pixel_scale_mas_grid``.
        psf_grid: Filled rectangular source PSF grid with shape
            ``(zenith, wavelength, y, x, psf_y, psf_x)``.
        provenance: Optional source provenance strings.
        builder: Builder metadata persisted with the artifact.
    """

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
    """Process-local prepared interpolator for a science-HO-PSF artifact.

    The persisted artifact stores plain arrays and metadata. This interpolator
    owns the SciPy interpolation object built from those arrays so repeated
    runtime evaluations can reuse it without changing the persisted artifact
    format.

    Attributes:
        artifact: Science-HO-PSF artifact backing this runtime interpolator.
        psf_interpolator: Prepared SciPy interpolator for PSF grid evaluation.
    """

    artifact: ScienceHoPsfInterpolator
    psf_interpolator: RegularGridInterpolator


def build_science_ho_psf_interpolator(
    samples: ScienceHoPsfSamples,
) -> ScienceHoPsfInterpolator:
    """Build a science high-order PSF interpolation artifact.

    The artifact requires a complete rectangular
    ``zenith_angle_deg x wavelength_um x y_arcsec x x_arcsec`` grid. PSF
    interpolation is linear in derived airmass, wavelength, and science field
    axes.

    Args:
        samples: Source science-HO-PSF samples. All arrays are validated before
            grid construction. PSFs enter the rectangular grid in their source
            flux convention.

    Returns:
        A validated in-memory artifact that can be evaluated, replayed, or
        persisted with ``save_science_ho_psf_interpolator``.

    Raises:
        TypeError: If ``samples`` has the wrong contract type.
        ValueError: If sample shapes, units, metadata, or the plane grid are
            invalid.
    """

    prepared = _prepare_samples(samples)
    zenith_axis = unique_sorted(prepared.zenith_angle_deg, label="zenith_angle_deg")
    airmass_axis = zenith_angle_to_airmass(zenith_axis)
    wavelength_axis = unique_sorted(prepared.wavelength_um, label="wavelength_um")
    field_x_axis, field_y_axis = _rectangular_field_axes(prepared.x_arcsec, prepared.y_arcsec)
    plane_indices = np.full((zenith_axis.size, wavelength_axis.size), -1, dtype=int)

    psf_grid = np.full(
        (
            zenith_axis.size,
            wavelength_axis.size,
            field_y_axis.size,
            field_x_axis.size,
            *prepared.psfs.shape[2:],
        ),
        np.nan,
        dtype=np.float32,
    )
    for plane_index, (zenith_angle_deg, wavelength_um) in enumerate(
        zip(prepared.zenith_angle_deg, prepared.wavelength_um, strict=True)
    ):
        iz = axis_index(zenith_axis, float(zenith_angle_deg), label="zenith_angle_deg")
        iw = axis_index(wavelength_axis, float(wavelength_um), label="wavelength_um")
        if plane_indices[iz, iw] != -1:
            raise ValueError(
                "Duplicate science HO PSF plane at "
                f"zenith_angle_deg={zenith_angle_deg}, wavelength_um={wavelength_um}."
            )
        plane_indices[iz, iw] = plane_index
        psf_grid[iz, iw] = _plane_psf_grid(prepared, plane_index, field_x_axis, field_y_axis)

    if np.any(plane_indices < 0):
        raise ValueError("Science HO PSF samples must form a complete zenith_angle_deg x wavelength_um grid.")

    pixel_scale_grid = np.full(plane_indices.shape, np.nan, dtype=float)
    for zenith_angle_deg, wavelength_um, pixel_scale_mas in zip(
        prepared.zenith_angle_deg,
        prepared.wavelength_um,
        prepared.pixel_scale_mas,
        strict=True,
    ):
        iz = axis_index(zenith_axis, float(zenith_angle_deg), label="zenith_angle_deg")
        iw = axis_index(wavelength_axis, float(wavelength_um), label="wavelength_um")
        pixel_scale_grid[iz, iw] = float(pixel_scale_mas)
    meta = _build_artifact_meta(prepared.meta, prepared, plane_indices, zenith_axis, wavelength_axis)

    return ScienceHoPsfInterpolator(
        zenith_angle_deg_axis=zenith_axis,
        airmass_axis=np.asarray(airmass_axis, dtype=float),
        wavelength_um_axis=wavelength_axis,
        x_arcsec=np.asarray(field_x_axis, dtype=float),
        y_arcsec=np.asarray(field_y_axis, dtype=float),
        psf_shape=tuple(int(v) for v in prepared.psfs.shape[2:]),
        pixel_scale_mas_grid=pixel_scale_grid,
        tel_diameter_m=float(prepared.tel_diameter_m),
        tel_pupil=np.asarray(prepared.tel_pupil, dtype=np.float32),
        meta=meta,
        psf_grid=psf_grid,
        provenance=tuple(prepared.provenance),
        builder={
            "name": "ao_predict.interpolation.science_ho_psf",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "interpolation_method": SCIENCE_HO_PSF_INTERPOLATION_METHOD,
        },
    )


def _prepare_science_ho_psf_runtime_interpolator(
    interpolator: ScienceHoPsfInterpolator,
) -> _ScienceHoPsfRuntimeInterpolator:
    """Prepare a runtime interpolator for repeated science-HO-PSF queries.

    Args:
        interpolator: Loaded or newly built science-HO-PSF artifact.

    Returns:
        Runtime-only interpolator that caches the SciPy interpolation object.

    Raises:
        TypeError: If ``interpolator`` is not a ``ScienceHoPsfInterpolator``.
        ValueError: If the artifact contract is invalid.
    """

    validate_science_ho_psf_interpolator(interpolator)
    return _ScienceHoPsfRuntimeInterpolator(
        artifact=interpolator,
        psf_interpolator=RegularGridInterpolator(
            (
                np.asarray(interpolator.airmass_axis, dtype=float),
                np.asarray(interpolator.wavelength_um_axis, dtype=float),
                np.asarray(interpolator.y_arcsec, dtype=float),
                np.asarray(interpolator.x_arcsec, dtype=float),
            ),
            np.asarray(interpolator.psf_grid, dtype=np.float32),
            method="linear",
            bounds_error=True,
        ),
    )


def _evaluate_science_ho_psf_runtime_interpolator(
    runtime_interpolator: _ScienceHoPsfRuntimeInterpolator,
    *,
    zenith_angle_deg: float,
    wavelength_um: float,
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
) -> ScienceHoPsfPrediction:
    """Evaluate a prepared runtime science-HO-PSF interpolator.

    Args:
        runtime_interpolator: Runtime interpolator returned by
            ``_prepare_science_ho_psf_runtime_interpolator``.
        zenith_angle_deg: Query zenith angle in degrees. The derived airmass
            must fall within the artifact airmass axis.
        wavelength_um: Query wavelength in microns. The value must fall within
            the artifact wavelength axis.
        x_arcsec: Science field x-coordinates in arcseconds.
        y_arcsec: Science field y-coordinates in arcseconds, matching
            ``x_arcsec`` length.

    Returns:
        Predicted flux-preserving PSFs and artifact-backed ``PsfMetadata``.

    Raises:
        TypeError: If ``runtime_interpolator`` is not a prepared science-HO-PSF
            runtime interpolator.
        ValueError: If the query is outside supported artifact coordinates or
            predicted PSFs are invalid.
    """

    if not isinstance(runtime_interpolator, _ScienceHoPsfRuntimeInterpolator):
        raise TypeError("runtime_interpolator must be a _ScienceHoPsfRuntimeInterpolator instance.")
    interpolator = runtime_interpolator.artifact
    coordinates = field_coordinates(x_arcsec, y_arcsec)
    _validate_field_query(interpolator, coordinates)
    coordinates = _snap_field_query(interpolator, coordinates)
    airmass = float(zenith_angle_to_airmass(zenith_angle_deg))
    points = np.column_stack(
        [
            np.full(coordinates.shape[0], airmass, dtype=float),
            np.full(coordinates.shape[0], float(wavelength_um), dtype=float),
            coordinates[:, 1],
            coordinates[:, 0],
        ]
    )
    psfs = np.asarray(runtime_interpolator.psf_interpolator(points), dtype=np.float32).reshape(
        coordinates.shape[0],
        *interpolator.psf_shape,
    )
    weights = _plane_weights(interpolator, zenith_angle_deg=zenith_angle_deg, wavelength_um=wavelength_um)
    pixel_scale_mas = 0.0
    meta = _initial_evaluated_meta(interpolator.meta, grid_shape=interpolator.pixel_scale_mas_grid.shape)
    for weight, grid_index in weights:
        pixel_scale_mas += float(weight) * float(interpolator.pixel_scale_mas_grid[grid_index])
        _accumulate_grid_meta(meta, interpolator.meta, weight=float(weight), grid_index=grid_index)
    _validate_psf_flux(psfs, label="predicted psfs")
    pixel_scale_mas = float(pixel_scale_mas)
    return ScienceHoPsfPrediction(
        psfs=psfs,
        pixel_scale_mas=pixel_scale_mas,
        metadata=PsfMetadata(
            wavelength_um=float(wavelength_um),
            pixel_scale_mas=pixel_scale_mas,
            tel_diameter_m=float(interpolator.tel_diameter_m),
            tel_pupil=np.asarray(interpolator.tel_pupil, dtype=np.float32),
        ),
        meta=meta,
    )


def evaluate_science_ho_psf_interpolator(
    interpolator: ScienceHoPsfInterpolator,
    *,
    zenith_angle_deg: float,
    wavelength_um: float,
    x_arcsec: np.ndarray,
    y_arcsec: np.ndarray,
) -> ScienceHoPsfPrediction:
    """Evaluate a science-HO-PSF artifact at supported coordinates.

    Args:
        interpolator: Loaded or newly built science-HO-PSF artifact.
        zenith_angle_deg: Query zenith angle in degrees. The derived airmass
            must fall within the artifact airmass axis.
        wavelength_um: Query wavelength in microns. The value must fall within
            the artifact wavelength axis.
        x_arcsec: Science field x-coordinates in arcseconds.
        y_arcsec: Science field y-coordinates in arcseconds, matching
            ``x_arcsec`` length.

    Returns:
        Predicted flux-preserving PSFs and artifact-backed ``PsfMetadata``.

    Raises:
        ValueError: If the query is outside the supported zenith-angle or
            wavelength axes, field coordinates are malformed, or predicted PSFs
            have non-finite values or non-positive total flux.
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
    zenith_angle_deg: float,
    wavelength_um: float,
) -> None:
    """Validate science-HO-PSF plane support without evaluating PSFs.

    The check enforces the artifact's zenith-angle/airmass and wavelength
    support. Field-coordinate validation is performed by
    ``evaluate_science_ho_psf_interpolator`` because field-coordinate
    validation requires the concrete query points.

    Raises:
        ValueError: If the requested zenith angle or wavelength is outside the
            artifact support.
    """

    _plane_weights(interpolator, zenith_angle_deg=zenith_angle_deg, wavelength_um=wavelength_um)


def validate_science_ho_psf_interpolator(interpolator: ScienceHoPsfInterpolator) -> None:
    """Validate a science-HO-PSF artifact contract.

    This validator checks the in-memory object shape and grid consistency used
    by save, load, and evaluation paths. It does not replay source nodes.

    Raises:
        TypeError: If ``interpolator`` is not a ``ScienceHoPsfInterpolator``.
        ValueError: If persisted metadata or model-grid fields are malformed.
    """

    if not isinstance(interpolator, ScienceHoPsfInterpolator):
        raise TypeError("interpolator must be a ScienceHoPsfInterpolator instance.")
    _validate_interpolator(interpolator)


def replay_science_ho_psf_interpolator(
    interpolator: ScienceHoPsfInterpolator,
    samples: ScienceHoPsfSamples,
) -> ScienceHoPsfReplaySummary:
    """Replay a science-HO-PSF artifact at source sample nodes.

    Replay evaluates the artifact at every source zenith-angle/wavelength plane
    and every source science field point. The summary reports PSF NRMS,
    pixel-scale residuals, and AO Predict FWHM/EE residuals. PSF residuals are
    computed on flux-preserving PSFs; metric residuals use the normal AO Predict
    stats preprocessing path.

    Args:
        interpolator: Science-HO-PSF artifact to validate.
        samples: Source samples expected to match the artifact support.

    Returns:
        Residual summary for PSFs, pixel scale, and replayed metrics.

    Raises:
        ValueError: If replay queries are unsupported or AO Predict metrics
            cannot be computed as finite values.
    """

    prepared = _prepare_samples(samples)
    residuals: list[np.ndarray] = []
    pixel_scale_errors: list[float] = []
    metric_errors: dict[str, list[np.ndarray]] = {"fwhm_mas": [], "ee": []}
    for index, (zenith_angle_deg, wavelength_um, pixel_scale_mas) in enumerate(
        zip(prepared.zenith_angle_deg, prepared.wavelength_um, prepared.pixel_scale_mas, strict=True)
    ):
        prediction = evaluate_science_ho_psf_interpolator(
            interpolator,
            zenith_angle_deg=float(zenith_angle_deg),
            wavelength_um=float(wavelength_um),
            x_arcsec=prepared.x_arcsec,
            y_arcsec=prepared.y_arcsec,
        )
        reference = np.asarray(prepared.psfs[index], dtype=np.float32)
        residuals.append(_psf_nrms(reference, prediction.psfs))
        pixel_scale_errors.append(abs(float(pixel_scale_mas) - prediction.pixel_scale_mas))
        plane_metric_errors = _science_metric_errors(
            reference,
            prediction.psfs,
            wavelength_um=float(wavelength_um),
            reference_pixel_scale_mas=float(pixel_scale_mas),
            prediction_metadata=prediction.metadata,
            tel_diameter_m=float(prepared.tel_diameter_m),
            tel_pupil=prepared.tel_pupil,
        )
        for name, values in plane_metric_errors.items():
            metric_errors[name].append(values)
    residual = np.concatenate(residuals)
    metric_error_arrays = {
        name: np.concatenate(values)
        for name, values in metric_errors.items()
    }
    return ScienceHoPsfReplaySummary(
        psf_nrms_mean=float(np.mean(residual)),
        psf_nrms_max=float(np.max(residual)),
        pixel_scale_abs_max_mas=float(np.max(pixel_scale_errors)),
        metric_rms={
            name: float(np.sqrt(np.mean(values**2)))
            for name, values in metric_error_arrays.items()
        },
        metric_max_abs={
            name: float(np.max(np.abs(values)))
            for name, values in metric_error_arrays.items()
        },
        num_planes=int(prepared.psfs.shape[0]),
        num_points=int(prepared.psfs.shape[1]),
    )


def save_science_ho_psf_interpolator(
    interpolator: ScienceHoPsfInterpolator,
    path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Save a science-HO-PSF interpolation artifact.

    Args:
        interpolator: Artifact to validate and persist.
        path: Destination pickle payload path.
        overwrite: When ``False``, existing files are rejected.

    Raises:
        FileExistsError: If ``path`` exists and ``overwrite`` is ``False``.
        TypeError: If ``interpolator`` has the wrong contract type.
    """

    save_payload(_science_to_payload(interpolator), Path(path), overwrite=overwrite)


def load_science_ho_psf_interpolator(path: Path) -> ScienceHoPsfInterpolator:
    """Load and validate a science-HO-PSF interpolation artifact.

    Args:
        path: Pickle payload path produced by
            ``save_science_ho_psf_interpolator``.

    Returns:
        Validated in-memory science-HO-PSF artifact.

    Raises:
        ValueError: If the payload kind, version, metadata, or model grid does
            not match the v1 artifact contract.
    """

    return _science_from_payload(load_payload(Path(path)))


def _prepare_samples(samples: ScienceHoPsfSamples) -> ScienceHoPsfSamples:
    if not isinstance(samples, ScienceHoPsfSamples):
        raise TypeError("samples must be a ScienceHoPsfSamples instance.")
    zenith_angle_deg = require_finite_vector(samples.zenith_angle_deg, label="zenith_angle_deg")
    wavelength_um = require_positive_vector(samples.wavelength_um, label="wavelength_um", length=zenith_angle_deg.size)
    pixel_scale_mas = require_positive_vector(samples.pixel_scale_mas, label="pixel_scale_mas", length=zenith_angle_deg.size)
    coordinates = field_coordinates(samples.x_arcsec, samples.y_arcsec)
    psfs = _validate_source_psfs(samples.psfs, label="psfs")
    if psfs.shape[:2] != (zenith_angle_deg.size, coordinates.shape[0]):
        raise ValueError(
            "psfs shape must be (planes, points, y, x); "
            f"got {psfs.shape} for {zenith_angle_deg.size} planes and {coordinates.shape[0]} points."
        )
    return ScienceHoPsfSamples(
        zenith_angle_deg=zenith_angle_deg,
        wavelength_um=wavelength_um,
        x_arcsec=np.asarray(samples.x_arcsec, dtype=float).reshape(-1),
        y_arcsec=np.asarray(samples.y_arcsec, dtype=float).reshape(-1),
        psfs=psfs,
        pixel_scale_mas=pixel_scale_mas,
        tel_diameter_m=require_positive_scalar(samples.tel_diameter_m, label="tel_diameter_m"),
        tel_pupil=require_pupil(samples.tel_pupil),
        meta=_prepare_source_meta(samples.meta, num_planes=zenith_angle_deg.size),
        provenance=tuple(str(value) for value in samples.provenance),
    )


def _rectangular_field_axes(x_arcsec: np.ndarray, y_arcsec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coordinates = field_coordinates(x_arcsec, y_arcsec)
    x_axis = unique_sorted(coordinates[:, 0], label="x_arcsec")
    y_axis = unique_sorted(coordinates[:, 1], label="y_arcsec")
    if coordinates.shape[0] != x_axis.size * y_axis.size:
        raise ValueError(
            "Science HO PSF samples must form a complete rectangular x_arcsec x y_arcsec field grid; "
            f"got {coordinates.shape[0]} points for {x_axis.size} x {y_axis.size} axes."
        )
    seen: set[tuple[int, int]] = set()
    for x_value, y_value in coordinates:
        ix = axis_index(x_axis, float(x_value), label="x_arcsec")
        iy = axis_index(y_axis, float(y_value), label="y_arcsec")
        key = (iy, ix)
        if key in seen:
            raise ValueError(f"Duplicate science HO PSF field point at x={x_value}, y={y_value}.")
        seen.add(key)
    return x_axis, y_axis


def _plane_psf_grid(
    samples: ScienceHoPsfSamples,
    plane_index: int,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
) -> np.ndarray:
    grid = np.full((y_axis.size, x_axis.size, *samples.psfs.shape[2:]), np.nan, dtype=np.float32)
    for point_index, (x_value, y_value) in enumerate(
        zip(samples.x_arcsec, samples.y_arcsec, strict=True)
    ):
        ix = axis_index(x_axis, float(x_value), label="x_arcsec")
        iy = axis_index(y_axis, float(y_value), label="y_arcsec")
        grid[iy, ix] = np.asarray(samples.psfs[plane_index, point_index], dtype=np.float32)
    if not np.all(np.isfinite(grid)):
        raise ValueError("Science HO PSF sample field grid is incomplete.")
    return grid


def _plane_weights(
    interpolator: ScienceHoPsfInterpolator,
    *,
    zenith_angle_deg: float,
    wavelength_um: float,
) -> list[tuple[float, tuple[int, int]]]:
    airmass = float(zenith_angle_to_airmass(zenith_angle_deg))
    weights: list[tuple[float, tuple[int, int]]] = []
    for iz, zenith_weight in interpolation_axis_weights(interpolator.airmass_axis, airmass, label="airmass"):
        for iw, wavelength_weight in interpolation_axis_weights(
            interpolator.wavelength_um_axis,
            wavelength_um,
            label="wavelength_um",
        ):
            weights.append((float(zenith_weight * wavelength_weight), (iz, iw)))
    return weights


def _validate_field_query(interpolator: ScienceHoPsfInterpolator, coordinates: np.ndarray) -> None:
    x_axis = np.asarray(interpolator.x_arcsec, dtype=float)
    y_axis = np.asarray(interpolator.y_arcsec, dtype=float)
    x = np.asarray(coordinates[:, 0], dtype=float)
    y = np.asarray(coordinates[:, 1], dtype=float)
    if np.any(x < x_axis[0] - SCIENCE_HO_PSF_FIELD_QUERY_ATOL_ARCSEC) or np.any(
        x > x_axis[-1] + SCIENCE_HO_PSF_FIELD_QUERY_ATOL_ARCSEC
    ):
        raise ValueError(
            f"x_arcsec query is outside science HO PSF field support [{x_axis[0]}, {x_axis[-1]}]."
        )
    if np.any(y < y_axis[0] - SCIENCE_HO_PSF_FIELD_QUERY_ATOL_ARCSEC) or np.any(
        y > y_axis[-1] + SCIENCE_HO_PSF_FIELD_QUERY_ATOL_ARCSEC
    ):
        raise ValueError(
            f"y_arcsec query is outside science HO PSF field support [{y_axis[0]}, {y_axis[-1]}]."
        )


def _snap_field_query(interpolator: ScienceHoPsfInterpolator, coordinates: np.ndarray) -> np.ndarray:
    snapped = np.asarray(coordinates, dtype=float).copy()
    x_axis = np.asarray(interpolator.x_arcsec, dtype=float)
    y_axis = np.asarray(interpolator.y_arcsec, dtype=float)
    snapped[:, 0] = np.clip(snapped[:, 0], x_axis[0], x_axis[-1])
    snapped[:, 1] = np.clip(snapped[:, 1], y_axis[0], y_axis[-1])
    return snapped


def _psf_nrms(reference: np.ndarray, measured: np.ndarray) -> np.ndarray:
    numerator = np.sqrt(np.mean((np.asarray(measured) - np.asarray(reference)) ** 2, axis=(-2, -1)))
    denominator = np.sqrt(np.mean(np.asarray(reference) ** 2, axis=(-2, -1)))
    return numerator / denominator


def _validate_source_psfs(psfs: Any, *, label: str) -> np.ndarray:
    array = np.asarray(psfs, dtype=np.float32)
    if array.ndim != 4:
        raise ValueError(f"{label} must have ndim=4, got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values.")
    _validate_psf_flux(array, label=label)
    return array


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
    plane_indices: np.ndarray,
    zenith_axis: np.ndarray,
    wavelength_axis: np.ndarray,
) -> dict[str, float | np.ndarray]:
    artifact_meta: dict[str, float | np.ndarray] = {}
    for name, value in source_meta.items():
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            artifact_meta[name] = float(array)
            continue
        grid = np.full(plane_indices.shape, np.nan, dtype=float)
        for zenith_angle_deg, wavelength_um, meta_value in zip(
            samples.zenith_angle_deg,
            samples.wavelength_um,
            array.reshape(-1),
            strict=True,
        ):
            iz = axis_index(zenith_axis, float(zenith_angle_deg), label="zenith_angle_deg")
            iw = axis_index(wavelength_axis, float(wavelength_um), label="wavelength_um")
            grid[iz, iw] = float(meta_value)
        artifact_meta[name] = grid
    return artifact_meta


def _initial_evaluated_meta(meta: Mapping[str, Any], *, grid_shape: tuple[int, int]) -> dict[str, float]:
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
    grid_index: tuple[int, int],
) -> None:
    for name, value in meta.items():
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            continue
        evaluated[name] += float(weight) * float(array[grid_index])


def _validate_artifact_meta(meta: Mapping[str, Any], *, grid_shape: tuple[int, int] | None) -> dict[str, float | np.ndarray]:
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
    reference_fwhm = np.asarray(
        compute_psf_fwhm(reference_psfs, reference_metadata, preprocess="default"),
        dtype=float,
    ).reshape(-1)
    measured_fwhm = np.asarray(
        compute_psf_fwhm(measured_psfs, prediction_metadata, preprocess="default"),
        dtype=float,
    ).reshape(-1)
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


def _science_to_payload(interpolator: ScienceHoPsfInterpolator) -> dict[str, Any]:
    if not isinstance(interpolator, ScienceHoPsfInterpolator):
        raise TypeError("interpolator must be a ScienceHoPsfInterpolator instance.")
    _validate_interpolator(interpolator)
    return {
        "kind": SCIENCE_HO_PSF_ARTIFACT_KIND,
        "version": SCIENCE_HO_PSF_ARTIFACT_VERSION,
        "builder": dict(interpolator.builder),
        "interpolation": {
            "method": SCIENCE_HO_PSF_INTERPOLATION_METHOD,
            "coordinate_order": ("airmass", "wavelength_um", "y_arcsec", "x_arcsec"),
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
    metadata = dict(payload.get("metadata", {}))
    model = dict(payload.get("model", {}))
    interpolator = ScienceHoPsfInterpolator(
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
    if len(interpolator.psf_shape) != 2 or any(int(value) <= 0 for value in interpolator.psf_shape):
        raise ValueError("metadata.psf_shape must contain two positive dimensions.")
    if interpolator.pixel_scale_mas_grid.shape != (
        interpolator.zenith_angle_deg_axis.size,
        interpolator.wavelength_um_axis.size,
    ):
        raise ValueError("pixel_scale_mas_grid shape must match zenith and wavelength axes.")
    _validate_artifact_meta(interpolator.meta, grid_shape=interpolator.pixel_scale_mas_grid.shape)
    if interpolator.airmass_axis.shape != interpolator.zenith_angle_deg_axis.shape:
        raise ValueError("airmass_axis shape must match zenith_angle_deg_axis shape.")
    if interpolator.x_arcsec.ndim != 1 or interpolator.y_arcsec.ndim != 1:
        raise ValueError("x_arcsec and y_arcsec must be 1-D rectangular field axes.")
    if interpolator.x_arcsec.size == 0 or interpolator.y_arcsec.size == 0:
        raise ValueError("x_arcsec and y_arcsec field axes must not be empty.")
    if np.any(np.diff(interpolator.x_arcsec) <= 0.0) or np.any(np.diff(interpolator.y_arcsec) <= 0.0):
        raise ValueError("x_arcsec and y_arcsec field axes must be strictly increasing.")
    expected_psf_grid_shape = (
        interpolator.zenith_angle_deg_axis.size,
        interpolator.wavelength_um_axis.size,
        interpolator.y_arcsec.size,
        interpolator.x_arcsec.size,
        *interpolator.psf_shape,
    )
    psf_grid = np.asarray(interpolator.psf_grid, dtype=np.float32)
    if psf_grid.shape != expected_psf_grid_shape:
        raise ValueError(f"psf_grid shape must be {expected_psf_grid_shape}; got {psf_grid.shape}.")
    if not np.all(np.isfinite(psf_grid)):
        raise ValueError("psf_grid must contain only finite values.")
    _validate_psf_flux(psf_grid, label="psf_grid")
