"""Versioned interpolation artifacts for Hybrid workflows.

This public submodule owns the generic AO Predict artifact contract for
science high-order PSF interpolation and NGS high-order metric interpolation.
Objects are built, validated, evaluated, replay-tested, saved, and loaded
through the functions exported here. The package root intentionally does not
re-export these names while the Hybrid API is still being introduced.
"""

from ._core import RegularGridInterpolationConfig, RbfInterpolationConfig, zenith_angle_to_airmass
from .ngs_ho_metric import (
    NgsHoMetricInterpolator,
    NgsHoMetricPrediction,
    NgsHoMetricReplaySummary,
    NgsHoMetricSamples,
    NgsHoPsfSamples,
    build_ngs_ho_metric_interpolator,
    build_ngs_ho_metric_interpolator_from_psfs,
    build_ngs_ho_metric_samples_from_psfs,
    evaluate_ngs_ho_metric_interpolator,
    load_ngs_ho_metric_interpolator,
    replay_ngs_ho_metric_interpolator,
    save_ngs_ho_metric_interpolator,
    validate_ngs_ho_metric_interpolator,
    validate_ngs_ho_metric_query,
)
from .science_ho_psf import (
    ScienceHoPsfInterpolator,
    ScienceHoPsfPrediction,
    ScienceHoPsfReplaySummary,
    ScienceHoPsfSamples,
    build_science_ho_psf_interpolator,
    evaluate_science_ho_psf_interpolator,
    load_science_ho_psf_interpolator,
    replay_science_ho_psf_interpolator,
    save_science_ho_psf_interpolator,
    validate_science_ho_psf_interpolator,
    validate_science_ho_psf_query,
)

__all__ = [
    "build_ngs_ho_metric_interpolator",
    "build_ngs_ho_metric_interpolator_from_psfs",
    "build_ngs_ho_metric_samples_from_psfs",
    "build_science_ho_psf_interpolator",
    "evaluate_ngs_ho_metric_interpolator",
    "evaluate_science_ho_psf_interpolator",
    "load_ngs_ho_metric_interpolator",
    "load_science_ho_psf_interpolator",
    "NgsHoMetricInterpolator",
    "NgsHoMetricPrediction",
    "NgsHoMetricReplaySummary",
    "NgsHoMetricSamples",
    "NgsHoPsfSamples",
    "RegularGridInterpolationConfig",
    "RbfInterpolationConfig",
    "replay_ngs_ho_metric_interpolator",
    "replay_science_ho_psf_interpolator",
    "save_ngs_ho_metric_interpolator",
    "save_science_ho_psf_interpolator",
    "ScienceHoPsfInterpolator",
    "ScienceHoPsfPrediction",
    "ScienceHoPsfReplaySummary",
    "ScienceHoPsfSamples",
    "validate_ngs_ho_metric_interpolator",
    "validate_ngs_ho_metric_query",
    "validate_science_ho_psf_interpolator",
    "validate_science_ho_psf_query",
    "zenith_angle_to_airmass",
]
