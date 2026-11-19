# Hybrid Simulation

`HybridSimulation` is AO Predict's dataset adapter over Hybrid AO PSF. Supply
the upstream science-HO-PSF and NGS-HO-metric artifact paths through
`SimulationConfig.specific_fields`; `non_psd_policy` selects `error` (the
default) or `clip` for materially non-positive-semidefinite Ctot. See the
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
slots. `/diagnostics/hybrid/ngs_used` records that slot mask. Debug adds the
effective MASTSEL INI and `/diagnostics/psfs_ho`.

::: ao_predict.simulation.hybrid.HybridSimulation

::: ao_predict.simulation.hybrid.HybridResolvedInputs
