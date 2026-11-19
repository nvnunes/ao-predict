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
                    apply=lambda ctot, runtime: 2 * ctot,
                ),
            ),
        )
```

Hybrid AO PSF records correction `input_files` when saving a standalone result.
Downstream adapters can record dataset metadata in
`prepare_simulation_payload()`.
See Hybrid AO PSF for the callback's wavefront units, borrowed runtime lifetime
and return contract.

The diagnostics dataset `/diagnostics/hybrid/ngs/frequency` records the
scalar Hybrid LO frame rate in each active NGS slot. Inactive slots contain
`NaN`. The dataset adapter derives the trace and determinant summaries from
the angular covariance and uses the MASTSEL conversion scale for the debug
wavefront covariance.

::: ao_predict.simulation.hybrid.HybridSimulation

::: ao_predict.simulation.hybrid.HybridResolvedInputs
