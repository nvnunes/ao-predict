# ao-predict Docs

ao-predict provides:

- batched simulation dataset workflows with TIPTOP-backed execution
- generic interpolation artifacts for prepared simulation products
- schema-aware HDF5 persistence with resume/retry semantics
- restart-safe PyTorch training for dense AO surrogate models
- bounded physical prediction and aggregate evaluation from validated model packages
- immutable analysis views over persisted simulation data
- generic PSF and metric-field plotting from analysis views
- code-first and CLI-driven execution interfaces

## Start Here

- Guides:
  - [CLI](cli.md)
  - [Python API](api.md)
  - [Architecture](architecture.md)
- API reference:
  - [Simulation Interfaces](reference/simulation/interfaces.md)
  - [Simulation Base](reference/simulation/base.md)
  - [Simulation API](reference/simulation/api.md)
  - [Simulation Options](reference/simulation/options.md)
  - [Simulation Runner](reference/simulation/runner.md)
  - [Simulation Stats](reference/simulation/stats.md)
  - [TIPTOP Simulation](reference/simulation/tiptop.md)
  - [Validation](reference/simulation/validation.md)
  - [Interpolation](reference/interpolation.md)
  - [Persistence Store](reference/persistence/store.md)
  - [Training](reference/training.md)
  - [Prediction](reference/prediction.md)
  - [Analysis](reference/analysis/api.md)
  - [Plotting](reference/plotting.md)
- Contributor documentation:
  - [Testing and verification](testing.md)
  - [Development setup](development.md)
