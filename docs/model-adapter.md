# 可选 Stable Audio 适配器

`sonoforge.adapters.stable_audio` 接受调用者已加载的 Diffusers
StableAudioPipeline，以及显式设置种子的 PyTorch Generator。
它不会下载模型、读取凭据或替调用者选择设备。

```python
import torch
from diffusers import StableAudioPipeline
from sonoforge.adapters import stable_audio

pipeline = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0")
pipeline = pipeline.to("cuda")
generator = torch.Generator(device="cuda").manual_seed(7)
result = stable_audio("A bell ringing", pipeline, generator, seconds=5)
```

返回帧×声道数组和模型实际采样率。上游模型决定可用时长与资源要求。
需自行安装 torch/diffusers 并遵守模型许可。本仓库仅运行了严格协议替身测试，
未下载或执行真实预训练权重，不宣称验证了生成音质。

[官方接口文档](https://huggingface.co/docs/diffusers/api/pipelines/stable_audio)
