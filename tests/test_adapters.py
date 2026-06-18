from types import SimpleNamespace

import numpy as np
import pytest

from sonoforge.adapters import stable_audio


class Pipeline:
    vae = SimpleNamespace(sampling_rate=16000)

    def __init__(self):
        self.calls = []

    def __call__(self, **arguments):
        self.calls.append(arguments)
        return SimpleNamespace(audios=np.array([[[0.0, 0.5], [0.0, -0.5]]]))


def test_stable_audio_upstream_call_and_channel_layout():
    pipeline, generator = Pipeline(), object()
    result = stable_audio("a bell", pipeline, generator, seconds=2, steps=12)
    assert pipeline.calls == [
        {
            "prompt": "a bell",
            "generator": generator,
            "audio_end_in_s": 2,
            "num_inference_steps": 12,
            "num_waveforms_per_prompt": 1,
        }
    ]
    assert result["sample_rate"] == 16000
    np.testing.assert_array_equal(result["audio"], [[0, 0], [0.5, -0.5]])


@pytest.mark.parametrize(
    "values",
    [{"prompt": ""}, {"seconds": 0}, {"seconds": np.nan}, {"steps": True}, {"generator": None}],
)
def test_invalid_requests_do_not_invoke_pipeline(values):
    pipeline = Pipeline()
    arguments = dict(prompt="sound", pipeline=pipeline, generator=object())
    arguments.update(values)
    with pytest.raises(ValueError):
        stable_audio(**arguments)
    assert pipeline.calls == []


def test_rejects_invalid_pipeline_result():
    pipeline = Pipeline()
    pipeline.vae = SimpleNamespace(sampling_rate=0)
    with pytest.raises(ValueError):
        stable_audio("sound", pipeline, object())
    assert pipeline.calls == []


def test_tensor_protocol_and_config_sample_rate():
    class Tensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.zeros((1, 1, 3))

    class TensorPipeline:
        vae = SimpleNamespace(config=SimpleNamespace(sampling_rate=24000))

        def __call__(self, **arguments):
            return SimpleNamespace(audios=Tensor())

    result = stable_audio("sound", TensorPipeline(), object())
    assert result["sample_rate"] == 24000 and result["audio"].shape == (3, 1)
