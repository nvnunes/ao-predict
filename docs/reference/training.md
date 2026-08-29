# Training API

Use `ao_predict.training` to validate prepared feature and target arrays, train
the supported dense-regression family, continue an interrupted compatible run,
and publish a deployable model package.

The same public names are re-exported from the `ao_predict` package root.
Lower-level model, data-preparation, recovery, locking, and package helpers are
private implementation details.

::: ao_predict.training
