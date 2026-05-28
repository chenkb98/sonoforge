# 使用方法

## 安装

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
```

激活虚拟环境后运行：

```sh
sonoforge generate --prompt "sine 440" --duration 0.1 --output outputs/tone
sonoforge train-demo --steps 100 --output outputs/denoiser.npz
sonoforge inspect outputs/tone/FILE.wav
```

输出目录应使用新路径；已有结果不会被 CLI 静默覆盖。各子命令的完整参数
可用 `--help` 查看。默认示例运行在 CPU 上，不要求外部模型或网络服务。

## 示例

三个可执行示例位于 `examples/`，在已安装包的环境中运行：

```sh
python examples/synthesis.py
```

音频数组采用帧×声道布局。WAV 仅支持未压缩 PCM16，线性重采样不包含
抗混叠滤波；高保真采样率转换应使用专门实现。所有生成或测试数据应标明来源。
