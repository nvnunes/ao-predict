# Analysis API

Use `load_analysis_dataset(path)` as the supported upstream read path for
persisted simulation datasets.

```python
from ao_predict.analysis import load_analysis_dataset

dataset = load_analysis_dataset("examples/sims/demo.h5")
sim = dataset.sim(0)
```

Public analysis behavior:
- `load_analysis_dataset(path, *, dataset_cls=AnalysisDataset, simulation_cls=AnalysisSimulation, extra_field_extractors=None) -> AnalysisDataset`
- `len(dataset)`
- `dataset.setup` for dataset-level setup values
- `dataset.options` for eager per-simulation option columns
- `dataset.meta` for eager loaded-analysis metadata columns and scalars
- `dataset.stats` for eager per-simulation stats columns
- `dataset.sim(i) -> AnalysisSimulation`
- `sim.config` with exactly `setup` and `options`
- `sim.meta` with per-simulation scientific metadata plus dataset-level
  telescope metadata such as `pixel_scale_mas`, `tel_diameter_m`, and
  `tel_pupil`
- `sim.stats` with core `sr`, `ee`, and `fwhm_mas` plus any declared extra
  stats
- lazy `sim.psfs`

PSFs are optional. If the dataset was created without persisted PSFs,
accessing `sim.psfs` raises a clear error explaining that PSFs were not saved.

The loader extension seam is generic and intended for downstream subclassing.
Downstream packages can pass dataset/simulation subclasses plus
`extra_field_extractors` that return `AnalysisLoadContribution` objects from an
`AnalysisLoadContext`. That allows downstream repos to add eager or lazy extra
fields without duplicating upstream object construction or exposing raw HDF5
objects on the public analysis surface. Downstream simulation subclasses can
then expose semantic properties backed by `_require_extra_field(...)`.

`AnalysisDataset` is generic over the simulation view type. Downstream
packages that define a custom simulation subclass can declare that
relationship directly:

```python
from ao_predict.analysis import AnalysisDataset, AnalysisSimulation


class CustomAnalysisSimulation(AnalysisSimulation):
    ...


class CustomAnalysisDataset(AnalysisDataset[CustomAnalysisSimulation]):
    pass
```

That lets `dataset.sim(i)` carry the custom simulation type without needing a
typed wrapper override in the dataset subclass.

Compatibility wrappers, legacy shaping, plotting, and downstream-specific
helpers remain outside `ao_predict` and should stay downstream in
`girmos-aosims`.

::: ao_predict.analysis
