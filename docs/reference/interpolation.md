# Hybrid Interpolation

The public science-HO-PSF and NGS-HO-metric interpolation API belongs to
[Hybrid AO PSF](https://github.com/nvnunes/hybrid-ao-psf). Import builders,
providers, evaluation, replay, and artifact operations from `hybrid_ao_psf`.
AO Predict's `HybridSimulation` consumes those artifacts and owns AO
dataset lifecycle and persisted reference/provenance fields.

Supply Hybrid AO PSF-format interpolation artifacts when configuring the AO
Predict adapter. See the [Python API guide](../api.md#hybrid-interpolation-inputs)
for a configuration example.
