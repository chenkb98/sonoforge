import sonoforge


def test_version_contract():
    assert sonoforge.__version__ == "0.1.2"
    assert sonoforge.__all__ == ["__version__", "config", "generate_run", "render", "write_run"]
