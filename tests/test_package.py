import sonoforge


def test_version_contract():
    assert sonoforge.__version__ == "0.1.0"
    assert sonoforge.__all__ == ["__version__"]
