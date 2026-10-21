"""Bounded propagation and statistics checks using stock MASTSEL primitives."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u
from ao_stats import PsfMetadata, compute_psf_stats

from ao_predict import TiptopSimulation, tiptop_full_flux_denominators
from ao_predict.simulation.validation import validate_simulation_payload_core


def _simulation():
    simulation = TiptopSimulation()
    simulation._dtype = np.dtype(np.float64)
    return simulation


def _propagation():
    # Follow stock TIPTOP's import order. Importing MASTSEL first can leave
    # TIPTOP's circular dependency initialized without MavisLO in older releases.
    importlib.import_module("tiptop")
    return importlib.import_module("mastsel.mavisPsf")


def _runtime(*, oversampling=1, opd=False, wavelengths=(1.65e-6,)):
    propagation = _propagation()
    n, pupil_size, grid = 96, 32, 12.0
    y, x = np.mgrid[:24, :24] - 11.5
    radius = np.hypot(x, y)
    # Resampling an apodized source pupil exercises fractional amplitudes and
    # prevents a binary-pupil count from accidentally satisfying the test.
    pupil = ((radius < 10.5) & (radius > 3.0)) * (0.8 + 0.1 * x / 12.0)
    mask = propagation.Field(wavelengths[0], n, grid)
    mask.sampling = propagation.zeroPad(
        propagation.congrid(pupil, [pupil_size, pupil_size]), (n - pupil_size) // 2,
    )
    frequency = np.fft.fftshift(np.fft.fftfreq(n))
    fx, fy = np.meshgrid(frequency, frequency)
    return SimpleNamespace(
        N=n, sx=pupil_size, grid_diameter=grid, freq_range=n / grid, dk=1.0,
        mask=mask, pupil=pupil, wvl=np.asarray(wavelengths), nWvl=len(wavelengths),
        wvlMax=max(wavelengths), overSamp=oversampling,
        opdMap=60.0 * np.sin(x / 4.0) * np.cos(y / 4.0) if opd else None,
        psd=2e-15 * np.exp(-(fx**2 + fy**2) / 0.03),
        LOisOn=False, doConvolve=True, doConvolveAsterism=True, jitter_FWHM=None,
    )


def _propagate(runtime, *, turbulence=True, crop=16, skip_reshape=False):
    propagation = _propagation()
    return propagation.psdSetToPsfSet(
        [runtime.psd if turbulence else np.zeros_like(runtime.psd)], runtime.pupil,
        runtime.wvl, runtime.N, runtime.sx, runtime.grid_diameter,
        runtime.freq_range, runtime.dk, crop, runtime.wvlMax, runtime.overSamp,
        opdMap=runtime.opdMap, padPSD=runtime.nWvl > 1, skip_reshape=skip_reshape,
    )


@pytest.mark.parametrize("turbulence", [False, True])
@pytest.mark.parametrize("opd", [False, True])
@pytest.mark.parametrize("oversampling", [1, 2, 3])
def test_conserved_denominator_matches_complete_propagation(turbulence, opd, oversampling):
    runtime = _runtime(oversampling=oversampling, opd=opd)
    denominator = tiptop_full_flux_denominators(runtime)[0]
    full = _propagate(runtime, turbulence=turbulence, crop=256)[0].sampling
    crop = _propagate(runtime, turbulence=turbulence)[0].sampling

    assert 0.0 < crop.sum() / denominator < 1.0
    if oversampling % 2:
        np.testing.assert_allclose(full.sum() / denominator, 1.0, rtol=2e-6)
    else:
        # Cubic half-pixel shifting uses constant boundaries and loses power.
        assert 0.998 < full.sum() / denominator < 1.0
    if opd:
        assert denominator == pytest.approx(runtime.N**2 / oversampling**2)


@pytest.mark.parametrize("opd", [False, True])
@pytest.mark.parametrize("skip_reshape", [False, True])
def test_multi_wavelength_padding_and_averaging(opd, skip_reshape):
    runtime = _runtime(oversampling=3, opd=opd, wavelengths=(1.3e-6, 1.65e-6))
    denominators = tiptop_full_flux_denominators(runtime, skip_reshape=skip_reshape)
    full = _propagate(runtime, crop=512, skip_reshape=skip_reshape)
    assert denominators.shape == (2,)
    for images, denominator in zip(full, denominators, strict=True):
        # The shorter wavelength uses an even effective factor (6), with a
        # small shift loss; the longer wavelength uses the odd factor (3).
        assert 0.998 < images[0].sampling.sum() / denominator < 1.0 + 2e-6


@pytest.mark.parametrize("lo_on,jitter", [(False, False), (False, True), (True, False), (True, True)])
def test_stock_output_scaling_is_restored_before_normalization(lo_on, jitter):
    propagation = _propagation()
    runtime = _runtime()
    ho = _propagate(runtime)
    runtime.psfLongExpPointingsArr = ho
    runtime.LOisOn = lo_on
    runtime.jitter_FWHM = 10.0 if jitter else None
    kernel = propagation.residualToSpectrum(
        [0.0, 10.0, 12.0], runtime.wvl[0], ho[0].N, 1.0 / (ho[0].N * 10.0),
    )
    raw = ho[0].sampling.copy()
    final = ho[0]
    for active in (lo_on, jitter):
        if active:
            final = propagation.convolve(final, kernel)
            raw = np.real(propagation.KernelConvolve(raw, kernel.sampling))
    runtime.results = [final]
    expected = np.maximum(raw / tiptop_full_flux_denominators(runtime)[0], 0.0)
    actual = _simulation()._extract_tiptop_psfs(runtime)
    np.testing.assert_allclose(actual[0], expected, rtol=2e-5, atol=2e-8)
    assert actual.sum() < 1.0


def test_unconvolved_low_order_results_are_not_rescaled():
    runtime = _runtime()
    runtime.LOisOn = True
    runtime.doConvolve = False
    runtime.results = _propagate(runtime)
    expected = runtime.results[0].sampling / tiptop_full_flux_denominators(runtime)[0]
    actual = TiptopSimulation()._extract_tiptop_psfs(runtime)
    np.testing.assert_allclose(actual[0], np.maximum(expected, 0.0), rtol=1e-6)


@pytest.mark.parametrize("oversampling", [1, 2, 3])
@pytest.mark.parametrize("peak_method", ["pixel_max", "gaussian_fit"])
def test_stats_preserve_full_flux_and_scalar_fwhm(oversampling, peak_method):
    runtime = _runtime(oversampling=oversampling)
    runtime.results = _propagate(runtime)
    simulation = _simulation()
    psfs = simulation._extract_tiptop_psfs(runtime)
    assert simulation.prepare_psfs_for_stats(psfs, {}, {}) is psfs
    flux = psfs.sum()
    scale = (runtime.wvl[0] / runtime.grid_diameter * oversampling * u.rad).to(u.mas)
    metadata = PsfMetadata(
        wavelength=runtime.wvl[0] * u.m, pixel_scale=scale, tel_diameter=4.0 * u.m,
        tel_pupil=runtime.pupil * u.one,
    )
    kwargs = dict(
        metadata=metadata, ee_apertures=np.array([4.0, 8.0]) * scale,
        peak_method=peak_method, fwhm_method="fourier_contour", ee_geometry="ensquared",
        preprocess=None, dtype=np.float64,
    )
    sr, ee, fwhm = compute_psf_stats(psfs, **kwargs)
    crop_sr, crop_ee, crop_fwhm = compute_psf_stats(psfs / flux, **kwargs)
    np.testing.assert_allclose(sr.value, crop_sr.value * flux, rtol=5e-5)
    np.testing.assert_allclose(ee.value, crop_ee.value * flux, rtol=1e-4)
    np.testing.assert_allclose(fwhm.value, crop_fwhm.value, rtol=1e-10)


def test_normalization_bounds_negative_interpolation_power_without_crop_rescaling():
    runtime = _runtime()
    denominator = tiptop_full_flux_denominators(runtime)[0]
    raw = np.zeros((1, 8, 8))
    raw[0, 4, 4] = 0.5 * denominator
    raw[0, 0, 0] = -1e-8 * denominator
    original = raw.copy()
    simulation = TiptopSimulation()
    normalized = simulation._normalize_tiptop_psfs(raw, runtime)
    assert normalized.sum() == pytest.approx(0.5)
    assert normalized.min() == 0.0
    np.testing.assert_array_equal(raw, original)
    raw[0, 0, 0] = -1e-3 * denominator
    with pytest.raises(ValueError, match="negative interpolation power"):
        simulation._normalize_tiptop_psfs(raw, runtime)
    raw[:] = -5e-7 * denominator
    raw[0, 4, 4] = 0.5 * denominator
    with pytest.raises(ValueError, match="negative interpolation power"):
        simulation._normalize_tiptop_psfs(raw, runtime)


def test_extraction_rejects_incomplete_or_deferred_outputs():
    runtime = _runtime()
    runtime.LOisOn = True
    runtime.doConvolveAsterism = False
    with pytest.raises(ValueError, match="deferred asterism"):
        TiptopSimulation()._extract_tiptop_psfs(runtime)
    runtime.doConvolveAsterism = True
    with pytest.raises(ValueError, match="missing final PSFs"):
        TiptopSimulation()._extract_tiptop_psfs(runtime)
    runtime = _runtime(wavelengths=(1.3e-6, 1.65e-6))
    with pytest.raises(ValueError, match="one TIPTOP science wavelength"):
        TiptopSimulation()._extract_tiptop_psfs(runtime)


@pytest.mark.parametrize("value", [np.nan, -0.01, 1.1, 0.0])
def test_stats_reject_invalid_flux_instead_of_renormalizing(value):
    psfs = np.zeros((1, 8, 8))
    psfs[0, 4, 4] = value
    with pytest.raises(ValueError, match="TIPTOP"):
        TiptopSimulation().prepare_psfs_for_stats(psfs, {}, {})


def test_denominator_rejects_missing_or_invalid_runtime_state():
    with pytest.raises(ValueError, match="retained propagation state"):
        tiptop_full_flux_denominators(SimpleNamespace())
    runtime = _runtime()
    runtime.mask.sampling[:] = 0.0
    with pytest.raises(ValueError, match="pupil power"):
        tiptop_full_flux_denominators(runtime)
    runtime = _runtime(oversampling=1.5)
    with pytest.raises(ValueError, match="positive integers"):
        tiptop_full_flux_denominators(runtime)


def test_new_execution_rejects_old_normalization_version():
    simulation = TiptopSimulation()
    payload = dict(name=simulation.name, version="0.0.1", extra_stat_fields={},
                   ngs_mag_standard=simulation.ngs_mag_standard)
    with pytest.raises(ValueError, match="version"):
        validate_simulation_payload_core(payload, expected_version=simulation.version)
