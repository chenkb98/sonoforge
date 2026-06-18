import json

import pytest

from sonoforge import cli, diffusion


@pytest.fixture
def command(capsys):
    def invoke(*arguments):
        assert cli.main(list(arguments)) == 0
        return json.loads(capsys.readouterr().out)

    return invoke


def test_generate_inspect_and_no_overwrite(command, tmp_path):
    target = tmp_path / "tone"
    result = command(
        "generate",
        "--prompt",
        "sine 10000",
        "--sample-rate",
        "48000",
        "--duration",
        "0.001",
        "--output",
        str(target),
    )
    assert result["manifest"]["frames"] == 48
    inspected = command("inspect", result["files"][0])
    assert inspected["sample_rate"] == 48000 and inspected["frames"] == 48
    assert inspected["peak"] == result["manifest"]["peak"]
    before = {p.name: p.read_bytes() for p in target.iterdir()}
    with pytest.raises(SystemExit) as error:
        cli.main(["generate", "--output", str(target)])
    assert error.value.code == 2
    assert before == {p.name: p.read_bytes() for p in target.iterdir()}


def test_training_checkpoint_is_real_and_improves_loss(command, tmp_path):
    target = tmp_path / "model.npz"
    result = command("train-demo", "--steps", "100", "--output", str(target))
    assert result["synthetic"] is True and 0 <= result["final_loss"] < result["initial_loss"]
    model = diffusion.checkpoint_decode(target.read_bytes())
    assert model["weights"].shape == (17, 16)
    original = target.read_bytes()
    with pytest.raises(SystemExit):
        cli.main(["train-demo", "--output", str(target)])
    assert target.read_bytes() == original


def test_invalid_generation_fails_without_an_artifact(tmp_path):
    target = tmp_path / "invalid"
    with pytest.raises(SystemExit) as error:
        cli.main(
            ["generate", "--prompt", "sine 6000", "--sample-rate", "8000", "--output", str(target)]
        )
    assert error.value.code == 2 and not target.exists()
