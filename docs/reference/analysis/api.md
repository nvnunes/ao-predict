# Analysis API

Use `load_analysis_dataset(path)` as the supported upstream read path for
persisted simulation datasets.

```python
from ao_predict.analysis import load_analysis_dataset

dataset = load_analysis_dataset("examples/sims/demo.h5")
sim = dataset.sim(0)
```

Public analysis behavior:
- `load_analysis_dataset(path) -> AnalysisDataset`
- `len(dataset)`
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

Compatibility wrappers, legacy shaping, plotting, and downstream-specific
helpers remain outside `ao_predict` and should stay downstream in
`girmos-aosims`.

::: ao_predict.analysis
