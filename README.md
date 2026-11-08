# spleeter-pytorch

A small implementation of the [Spleeter](https://github.com/deezer/spleeter) stem separation model in PyTorch. Using this model, audio files can be demixed into vocals, instrumentation etc.
Furthermore, this forked repository support 4- and 5-stem separations, and tuned to create Core ML model that can be easily used from Swift code.

## Example

Install the package using `pip3 install .`, then run

```sh
spleeter-pytorch audio-example.mp3
```

to separate the example file. The output will be located in `output/stems`.

## Prepare pretrained models

This repository includes only the checkpoint file for 2-stem separation.
The models for 4- and 5-stem separations are not contained because the sizes are large.
Currently, you can obtain the checkpoint files via GitHub Releases or by the following steps:

1. Clone the original [Spleeter](https://github.com/deezer/spleeter) repository.
2. Execute separation by some commands like `poetry run spleeter separate -p spleeter:5stems -o output audio_example.mp3`.
3. Reveal output files in the `pretrained_models` directory.

## Conversion to Core ML

The non-FFT parts of the Spleeter model can be converted to Core ML, for efficient inference on macOS/iOS devices. To perform the conversion, run

```sh
# Create MLPackage for 4-stem separation
python convert-to-coreml.py -n 4

# Compile Core ML model
swift compile_coreml_model.swift output/4stems/coreml/SpleeterModel.mlpackage
```

The `.mlpackage` and `.mlmodelc` will be located under `output/{n}stems/coreml` by default.

> Note: The converted model corresponds to the [`Separator`](spleeter_pytorch/separator.py) module and still requires the consumer of the model to manually perform the STFT conversion as performed in the [`Estimator`](spleeter_pytorch/estimator.py). This is due to Core ML [not supporting FFT operations yet](https://github.com/apple/coremltools/issues/1311).

The converted model outputs each stem's raw mask (`ratio_masks=False`), not Spleeter's ratio masks: the consumer
computes `m_i^2 / sum m_j^2` itself, in FP32. Squaring inside the model overflows FP16 — Core ML's default precision —
wherever a mask exceeds 256, which zeroed loud bins such as a kick drum's.

The 2- and 4-stem models stay close to the FP32 separator in FP16. Convert the 5-stem model with `-p float32`: its FP16
output drifts from the FP32 separator by about 20 dB SNR on the piano stem.

```sh
python convert-to-coreml.py -n 5 -p float32
```

## Export to ONNX

The same separator exports to ONNX (FP32, opset 17) for ONNX Runtime, which swift-spleeter's `SpleeterONNX` backend
runs on Android:

```sh
# Writes output/4stems/onnx/Spleeter4Model.onnx (about 157 MB) and compares it with PyTorch in onnxruntime
python convert-to-onnx.py -n 4 --check
```

The outputs are the raw masks under the Core ML models' names (`vocalsMask`, `drumsMask`, …), shaped
`2 × 1024 × frames × 1`, from a `magnitude` input shaped `2 × 1024 × frames`. The frame count is fixed — the separator
partitions its input with Python arithmetic on the traced shape, which the export bakes in — and defaults to 512, the
separator's own partition size: a shorter input is padded to 512 frames inside the graph anyway, so a 512-frame model
separates about 11.9 s per call for the cost of a 216-frame (5 s) one. `-f 216` exports the Core ML models' window.

`-w float16` stores the weights as FP16, each cast back to FP32 inside the graph (ONNX Runtime folds the casts at
load, so inference stays FP32): the 4-stem model drops to about 79 MB, and its stems stay within about 60 dB SNR of the
FP32 model's.

`convert-to-coreml.py -f 512` likewise converts a Core ML model for 512-frame windows.

Exporting needs `onnx` (and `onnxruntime` for `--check`) besides this package's dependencies.

## Note

- Currently this is only tested with the 2stems model. Feel free to [get one of the other models](https://github.com/deezer/spleeter/releases/tag/v1.4.0) and test it on them.
- There might be some bugs, the quality of output isn't as good as the original. If someone found the reason, please open a pull request. Thanks.

## Reference

- [Original Spleeter](https://github.com/deezer/spleeter) by [`deezer`](https://github.com/deezer)
- [Original `spleeter-pytorch`](https://github.com/tuan3w/spleeter-pytorch) by [`tuan3w`](https://github.com/tuan3w)
- [`spleeter-pytorch` with Core ML conversion](https://github.com/fwcd/spleeter-pytorch) by [`fwcd`](https://github.com/fwcd)

## License

**MIT**.
