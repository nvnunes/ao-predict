import importlib

from ao_predict import (
    AnalysisDataset,
    AnalysisDatasetLoadPayload,
    AnalysisLoadContext,
    AnalysisLoadContribution,
    AnalysisSimulation,
    AnalysisSimulationLoadPayload,
    __version__,
    load_analysis_dataset,
)
from ao_predict.simulation import ConfigBackedSimulation, TiptopBaseConfig, TiptopConfigBackedSimulation, TiptopSimulation


def test_version_present():
    assert isinstance(__version__, str)
    assert __version__


def test_root_analysis_exports():
    assert AnalysisDataset.__name__ == "AnalysisDataset"
    assert AnalysisDatasetLoadPayload.__name__ == "AnalysisDatasetLoadPayload"
    assert AnalysisLoadContext.__name__ == "AnalysisLoadContext"
    assert AnalysisLoadContribution.__name__ == "AnalysisLoadContribution"
    assert AnalysisSimulation.__name__ == "AnalysisSimulation"
    assert AnalysisSimulationLoadPayload.__name__ == "AnalysisSimulationLoadPayload"
    assert load_analysis_dataset.__name__ == "load_analysis_dataset"


def test_package_and_analysis_module_exports() -> None:
    package = importlib.import_module("ao_predict")
    analysis = importlib.import_module("ao_predict.analysis")

    assert package.load_analysis_dataset is load_analysis_dataset
    assert package.AnalysisDataset is AnalysisDataset
    assert package.AnalysisDatasetLoadPayload is AnalysisDatasetLoadPayload
    assert package.AnalysisLoadContext is AnalysisLoadContext
    assert analysis.load_analysis_dataset is load_analysis_dataset
    assert analysis.AnalysisSimulation is AnalysisSimulation
    assert analysis.AnalysisLoadContribution is AnalysisLoadContribution


def test_simulation_module_config_backed_exports() -> None:
    simulation = importlib.import_module("ao_predict.simulation")

    assert simulation.ConfigBackedSimulation is ConfigBackedSimulation
    assert simulation.TiptopConfigBackedSimulation is TiptopConfigBackedSimulation
    assert simulation.TiptopBaseConfig is TiptopBaseConfig
    assert simulation.TiptopSimulation is TiptopSimulation
