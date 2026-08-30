from __future__ import annotations

import pytest
from astropy import units as u

from ao_predict.simulation import schema
from ao_predict.simulation.api import (
    InitDatasetRequest,
    SetupConfig,
    SimulationConfig,
    init_dataset,
)
from ao_predict.simulation.options import GeneratedOptions, options_from_rows


def test_options_from_rows_preserves_stable_row_order() -> None:
    generated = options_from_rows(
        [
            {
                schema.KEY_OPTION_ZENITH_ANGLE: 15.0,
                schema.KEY_OPTION_WAVELENGTH: 1.25,
            },
            {
                schema.KEY_OPTION_ZENITH_ANGLE: 25.0,
                schema.KEY_OPTION_WAVELENGTH: 1.65,
            },
        ]
    )

    assert isinstance(generated, GeneratedOptions)
    assert generated.columns == (
        schema.KEY_OPTION_ZENITH_ANGLE,
        schema.KEY_OPTION_WAVELENGTH,
    )
    assert generated.rows == ((15.0, 1.25), (25.0, 1.65))


def test_options_from_rows_uses_explicit_columns_and_broadcast_defaults() -> None:
    generated = options_from_rows(
        [
            {
                schema.KEY_OPTION_WAVELENGTH: 1.25,
                schema.KEY_OPTION_ZENITH_ANGLE: 15.0,
            },
            {
                schema.KEY_OPTION_WAVELENGTH: 1.65,
                schema.KEY_OPTION_ZENITH_ANGLE: 25.0,
            },
        ],
        columns=[schema.KEY_OPTION_ZENITH_ANGLE],
        broadcast={schema.KEY_OPTION_WAVELENGTH: 2.2},
    )

    config = generated.to_table_options_config()

    assert config.columns == [schema.KEY_OPTION_ZENITH_ANGLE]
    assert config.rows == [[15.0], [25.0]]
    assert config.broadcast == {schema.KEY_OPTION_WAVELENGTH: 2.2}


def test_options_from_rows_supports_ragged_ngs_rows_with_null_slots() -> None:
    columns = [
        schema.KEY_OPTION_ZENITH_ANGLE,
        "ngs1_r",
        "ngs1_theta",
        "ngs1_magnitude",
        "ngs2_r",
        "ngs2_theta",
        "ngs2_magnitude",
    ]
    generated = options_from_rows(
        [
            {
                schema.KEY_OPTION_ZENITH_ANGLE: 20.0,
                "ngs1_r": 5.0,
                "ngs1_theta": 45.0,
                "ngs1_magnitude": 14.0,
                "ngs2_r": None,
                "ngs2_theta": None,
                "ngs2_magnitude": None,
            },
            {
                schema.KEY_OPTION_ZENITH_ANGLE: 30.0,
                "ngs1_r": 10.0,
                "ngs1_theta": 90.0,
                "ngs1_magnitude": 15.0,
                "ngs2_r": 20.0,
                "ngs2_theta": 180.0,
                "ngs2_magnitude": 16.0,
            },
        ],
        columns=columns,
    )

    assert generated.columns == tuple(columns)
    assert generated.rows[0][-3:] == (None, None, None)
    assert generated.rows[1][-3:] == (20.0, 180.0, 16.0)


def test_options_from_rows_rejects_empty_rows() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        options_from_rows([])


def test_options_from_rows_rejects_inconsistent_implicit_row_keys() -> None:
    with pytest.raises(ValueError, match="identical key order"):
        options_from_rows(
            [
                {schema.KEY_OPTION_ZENITH_ANGLE: 20.0},
                {schema.KEY_OPTION_WAVELENGTH: 1.65},
            ]
        )


def test_options_from_rows_rejects_missing_explicit_column() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        options_from_rows(
            [{schema.KEY_OPTION_ZENITH_ANGLE: 20.0}],
            columns=[schema.KEY_OPTION_ZENITH_ANGLE, schema.KEY_OPTION_WAVELENGTH],
        )


def test_generated_options_can_initialize_mock_dataset(tmp_path) -> None:
    generated = options_from_rows(
        [
            {
                schema.KEY_OPTION_ZENITH_ANGLE: 15.0,
                "ngs1_r": 5.0,
                "ngs1_theta": 45.0,
                "ngs1_magnitude": 14.0,
                "ngs2_r": None,
                "ngs2_theta": None,
                "ngs2_magnitude": None,
            },
            {
                schema.KEY_OPTION_ZENITH_ANGLE: 25.0,
                "ngs1_r": 10.0,
                "ngs1_theta": 90.0,
                "ngs1_magnitude": 15.0,
                "ngs2_r": 20.0,
                "ngs2_theta": 180.0,
                "ngs2_magnitude": 16.0,
            },
        ],
        units={
            schema.KEY_OPTION_ZENITH_ANGLE: u.deg,
            "ngs1_r": u.arcsec,
            "ngs1_theta": u.deg,
            "ngs1_magnitude": u.mag,
            "ngs2_r": u.arcsec,
            "ngs2_theta": u.deg,
            "ngs2_magnitude": u.mag,
        },
        broadcast={schema.KEY_OPTION_WAVELENGTH: 1.65 * u.um},
    )
    dataset_path = tmp_path / "generated_options_mock.h5"

    num_sims = init_dataset(
        InitDatasetRequest(
            dataset_path=dataset_path,
            simulation=SimulationConfig(name="mock_simulation:MockSimulation"),
            setup=SetupConfig(ee_apertures=[50.0] * u.mas),
            options=generated.to_table_options_config(),
            save_psfs=True,
        )
    )

    assert num_sims == 2
    assert dataset_path.is_file()
