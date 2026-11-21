# Hybrid Simulation

`HybridSimulation` is AO Predict's dataset adapter over Hybrid AO PSF. Supply
the science HO PSF and NGS metric interpolator paths through
`SimulationConfig.specific_fields`. The INI selected by `config_path` sets
`[hybrid] LOModel`, which defaults to `mastsel`. MASTSEL supports
`non_psd_policy` of `error` or `clip`. With `LOModel = e2e-calibrated`,
`[inputs]` supplies the four fitted model paths and `NGSMetricsPath` must
match the simulation's NGS metric interpolator. See the
[Python API guide](../../api.md#hybrid-interpolation-inputs) for an AO Predict
configuration example and the [Hybrid AO PSF project](https://github.com/nvnunes/hybrid-ao-psf)
for the standalone engine and artifact API.

Downstream subclasses can attach a covariance correction through the existing
resolved-input seam without overriding execution:

```python
from dataclasses import replace

from ao_predict import HybridSimulation
from hybrid_ao_psf import HybridCtotCorrection


class ScaledHybridSimulation(HybridSimulation):
    def _resolve_hybrid_inputs(self, context):
        resolved = super()._resolve_hybrid_inputs(context)
        return replace(
            resolved,
            lo_model=replace(
                resolved.lo_model,
                ctot_correction=HybridCtotCorrection(
                    apply=lambda ctot, context: 2 * ctot,
                ),
            ),
        )
```

Downstream adapters can record dataset metadata in
`prepare_simulation_payload()`.
See Hybrid AO PSF for the callback's wavefront units, numerical context,
and return contract.

AO Predict passes the requested diagnostic level to Hybrid AO PSF and saves
the fields that Hybrid declares. The full angular covariance is
`/diagnostics/covariance/ctot`. MASTSEL diagnostics use
`/diagnostics/mastsel`, with one `frame_rate` per observation.
Per-star values follow the batch's NGS slot order and have `NaN` in inactive
slots. `/diagnostics/hybrid/ngs_used` records that slot mask. Debug adds
`/diagnostics/psfs_ho` and the model's detailed fields. E2E-Calibrated
stores reconstructor matrices in per-star, two-axis blocks so the same
NGS slot mask applies to the matrix fields.

::: ao_predict.simulation.hybrid.HybridSimulation

::: ao_predict.simulation.hybrid.HybridResolvedInputs
