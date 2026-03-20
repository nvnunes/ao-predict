from ao_predict import (
    AnalysisDataset,
    AnalysisSimulation,
    __version__,
    load_analysis_dataset,
)


def test_version_present():
    assert isinstance(__version__, str)
    assert __version__


def test_root_analysis_exports():
    assert AnalysisDataset.__name__ == "AnalysisDataset"
    assert AnalysisSimulation.__name__ == "AnalysisSimulation"
    assert load_analysis_dataset.__name__ == "load_analysis_dataset"
