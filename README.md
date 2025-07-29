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

## Note

- Currently this is only tested with the 2stems model. Feel free to [get one of the other models](https://github.com/deezer/spleeter/releases/tag/v1.4.0) and test it on them.
- There might be some bugs, the quality of output isn't as good as the original. If someone found the reason, please open a pull request. Thanks.

## Reference

- [Original Spleeter](https://github.com/deezer/spleeter) by [`deezer`](https://github.com/deezer)
- [Original `spleeter-pytorch`](https://github.com/tuan3w/spleeter-pytorch) by [`tuan3w`](https://github.com/tuan3w)
- [`spleeter-pytorch` with Core ML conversion](https://github.com/fwcd/spleeter-pytorch) by [`fwcd`](https://github.com/fwcd)

## License

**MIT**.
