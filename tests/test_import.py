from ao_predict import __version__


def test_version_present():
    assert isinstance(__version__, str)
    assert __version__
