"""Internal build-input packages for interpolation CLI workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ._core import load_payload, save_payload, validate_payload_kind
from .ngs_ho_metric import (
    NgsHoMetricSamples,
    NgsHoPsfSamples,
    _prepare_metric_samples,
    _prepare_psf_samples,
)
from .science_ho_psf import ScienceHoPsfSamples, _prepare_samples


SCIENCE_HO_PSF_INPUT_KIND = "ao_predict_science_ho_psf_inputs"
NGS_HO_PSF_INPUT_KIND = "ao_predict_ngs_ho_psf_inputs"
NGS_HO_METRIC_INPUT_KIND = "ao_predict_ngs_ho_metric_inputs"
INTERPOLATION_INPUT_VERSION = 1


def save_science_ho_psf_inputs(
    samples: ScienceHoPsfSamples,
    path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Save science-HO-PSF samples for upstream CLI artifact building.

    The saved file is an internal builder handoff package. Callers construct
    the public ``ScienceHoPsfSamples`` object, while the serialized package is
    intended to be consumed only by ``ao-predict interpolation`` commands.

    Args:
        samples: Science-HO-PSF samples to validate and package.
        path: Destination path.
        overwrite: When ``False``, existing files are rejected.

    Raises:
        FileExistsError: If ``path`` exists and ``overwrite`` is ``False``.
        TypeError: If ``samples`` is not a ``ScienceHoPsfSamples`` instance.
        ValueError: If sample shapes, units, or metadata are invalid.
    """

    prepared = _prepare_samples(samples)
    save_payload(
        {
            "kind": SCIENCE_HO_PSF_INPUT_KIND,
            "version": INTERPOLATION_INPUT_VERSION,
            "builder": _builder_metadata("science_ho_psf_inputs"),
            "samples": {
                "zenith_angle_deg": np.asarray(prepared.zenith_angle_deg, dtype=float),
                "wavelength_um": np.asarray(prepared.wavelength_um, dtype=float),
                "x_arcsec": np.asarray(prepared.x_arcsec, dtype=float),
                "y_arcsec": np.asarray(prepared.y_arcsec, dtype=float),
                "psfs": np.asarray(prepared.psfs, dtype=np.float32),
                "pixel_scale_mas": np.asarray(prepared.pixel_scale_mas, dtype=float),
                "tel_diameter_m": float(prepared.tel_diameter_m),
                "tel_pupil": np.asarray(prepared.tel_pupil, dtype=np.float32),
                "provenance": tuple(prepared.provenance),
            },
        },
        Path(path),
        overwrite=overwrite,
    )


def save_ngs_ho_psf_inputs(
    samples: NgsHoPsfSamples,
    path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Save NGS-HO-PSF samples for upstream CLI metric-artifact building.

    The saved file is an internal builder handoff package. The CLI consumes it
    to measure required NGS-HO metrics and build the durable metric
    interpolator artifact.

    Args:
        samples: NGS-HO-PSF samples to validate and package.
        path: Destination path.
        overwrite: When ``False``, existing files are rejected.

    Raises:
        FileExistsError: If ``path`` exists and ``overwrite`` is ``False``.
        TypeError: If ``samples`` is not a ``NgsHoPsfSamples`` instance.
        ValueError: If sample shapes, PSFs, or metadata are invalid.
    """

    prepared = _prepare_psf_samples(samples)
    save_payload(
        {
            "kind": NGS_HO_PSF_INPUT_KIND,
            "version": INTERPOLATION_INPUT_VERSION,
            "builder": _builder_metadata("ngs_ho_psf_inputs"),
            "samples": {
                "zenith_angle_deg": np.asarray(prepared.zenith_angle_deg, dtype=float),
                "x_arcsec": np.asarray(prepared.x_arcsec, dtype=float),
                "y_arcsec": np.asarray(prepared.y_arcsec, dtype=float),
                "psfs": np.asarray(prepared.psfs, dtype=np.float32),
                "wavelength_um": np.asarray(prepared.wavelength_um, dtype=float),
                "pixel_scale_mas": np.asarray(prepared.pixel_scale_mas, dtype=float),
                "tel_diameter_m": float(prepared.tel_diameter_m),
                "tel_pupil": np.asarray(prepared.tel_pupil, dtype=np.float32),
                "provenance": tuple(prepared.provenance),
            },
        },
        Path(path),
        overwrite=overwrite,
    )


def save_ngs_ho_metric_inputs(
    samples: NgsHoMetricSamples,
    path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Save measured NGS-HO metrics for upstream CLI artifact building.

    The saved file is an internal builder handoff package. It is useful when a
    downstream project already measured ``ee``, ``fwhm_mas``, and ``sr`` and
    wants the upstream CLI to build the durable interpolator artifact.

    Args:
        samples: NGS-HO metric samples to validate and package.
        path: Destination path.
        overwrite: When ``False``, existing files are rejected.

    Raises:
        FileExistsError: If ``path`` exists and ``overwrite`` is ``False``.
        TypeError: If ``samples`` is not a ``NgsHoMetricSamples`` instance.
        ValueError: If metric shapes, values, or coordinates are invalid.
    """

    prepared = _prepare_metric_samples(samples)
    save_payload(
        {
            "kind": NGS_HO_METRIC_INPUT_KIND,
            "version": INTERPOLATION_INPUT_VERSION,
            "builder": _builder_metadata("ngs_ho_metric_inputs"),
            "samples": {
                "zenith_angle_deg": np.asarray(prepared.zenith_angle_deg, dtype=float),
                "x_arcsec": np.asarray(prepared.x_arcsec, dtype=float),
                "y_arcsec": np.asarray(prepared.y_arcsec, dtype=float),
                "ee": np.asarray(prepared.ee, dtype=float),
                "fwhm_mas": np.asarray(prepared.fwhm_mas, dtype=float),
                "sr": np.asarray(prepared.sr, dtype=float),
                "provenance": tuple(prepared.provenance),
            },
        },
        Path(path),
        overwrite=overwrite,
    )


def _load_science_ho_psf_inputs(path: Path) -> ScienceHoPsfSamples:
    payload = load_payload(Path(path))
    validate_payload_kind(payload, kind=SCIENCE_HO_PSF_INPUT_KIND, version=INTERPOLATION_INPUT_VERSION)
    samples = dict(payload.get("samples", {}))
    return _prepare_samples(
        ScienceHoPsfSamples(
            zenith_angle_deg=np.asarray(samples.get("zenith_angle_deg"), dtype=float),
            wavelength_um=np.asarray(samples.get("wavelength_um"), dtype=float),
            x_arcsec=np.asarray(samples.get("x_arcsec"), dtype=float),
            y_arcsec=np.asarray(samples.get("y_arcsec"), dtype=float),
            psfs=np.asarray(samples.get("psfs"), dtype=np.float32),
            pixel_scale_mas=np.asarray(samples.get("pixel_scale_mas"), dtype=float),
            tel_diameter_m=float(samples.get("tel_diameter_m")),
            tel_pupil=np.asarray(samples.get("tel_pupil"), dtype=np.float32),
            provenance=tuple(str(value) for value in samples.get("provenance", ())),
        )
    )


def _load_ngs_ho_psf_inputs(path: Path) -> NgsHoPsfSamples:
    payload = load_payload(Path(path))
    validate_payload_kind(payload, kind=NGS_HO_PSF_INPUT_KIND, version=INTERPOLATION_INPUT_VERSION)
    samples = dict(payload.get("samples", {}))
    return _prepare_psf_samples(
        NgsHoPsfSamples(
            zenith_angle_deg=np.asarray(samples.get("zenith_angle_deg"), dtype=float),
            x_arcsec=np.asarray(samples.get("x_arcsec"), dtype=float),
            y_arcsec=np.asarray(samples.get("y_arcsec"), dtype=float),
            psfs=np.asarray(samples.get("psfs"), dtype=np.float32),
            wavelength_um=np.asarray(samples.get("wavelength_um"), dtype=float),
            pixel_scale_mas=np.asarray(samples.get("pixel_scale_mas"), dtype=float),
            tel_diameter_m=float(samples.get("tel_diameter_m")),
            tel_pupil=np.asarray(samples.get("tel_pupil"), dtype=np.float32),
            provenance=tuple(str(value) for value in samples.get("provenance", ())),
        )
    )


def _load_ngs_ho_metric_inputs(path: Path) -> NgsHoMetricSamples:
    payload = load_payload(Path(path))
    validate_payload_kind(payload, kind=NGS_HO_METRIC_INPUT_KIND, version=INTERPOLATION_INPUT_VERSION)
    samples = dict(payload.get("samples", {}))
    return _prepare_metric_samples(
        NgsHoMetricSamples(
            zenith_angle_deg=np.asarray(samples.get("zenith_angle_deg"), dtype=float),
            x_arcsec=np.asarray(samples.get("x_arcsec"), dtype=float),
            y_arcsec=np.asarray(samples.get("y_arcsec"), dtype=float),
            ee=np.asarray(samples.get("ee"), dtype=float),
            fwhm_mas=np.asarray(samples.get("fwhm_mas"), dtype=float),
            sr=np.asarray(samples.get("sr"), dtype=float),
            provenance=tuple(str(value) for value in samples.get("provenance", ())),
        )
    )


def _builder_metadata(name: str) -> Mapping[str, Any]:
    return {
        "name": f"ao_predict.interpolation.{name}",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
