# Hybrid Simulation

`HybridSimulation` is AO Predict's dataset adapter over Hybrid AO PSF. Supply
the upstream science-HO-PSF and NGS-HO-metric artifact paths through
`SimulationConfig.specific_fields`; `non_psd_policy` selects `error` (the
default) or `clip` for materially non-positive-semidefinite Ctot. See the
[Python API guide](../../api.md#hybrid-interpolation-inputs) for an AO Predict
configuration example and the [Hybrid AO PSF project](https://github.com/nvnunes/hybrid-ao-psf)
for the standalone engine and artifact API.

::: ao_predict.simulation.hybrid.HybridSimulation
