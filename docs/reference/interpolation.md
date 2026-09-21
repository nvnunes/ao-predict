# Hybrid Interpolation Ownership

The public science-HO-PSF and NGS-HO-metric interpolation API now belongs to
[Hybrid AO PSF](https://github.com/nvnunes/hybrid-ao-psf). Import builders,
providers, evaluation, replay, and artifact operations from `hybrid_ao_psf`.
AO Predict's `HybridSimulation` consumes those artifacts and retains only AO
dataset lifecycle and persisted reference/provenance fields.

The former `ao_predict.interpolation` implementation remains temporarily for
the coordinated downstream transition and parity comparison; new Hybrid
artifacts should use the upstream format. See the [Python API guide](../api.md#hybrid-interpolation-inputs)
for the AO adapter's configuration example.
