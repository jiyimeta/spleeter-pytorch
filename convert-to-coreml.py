#!/usr/bin/env python3

# A script for converting the Spleeter separator to Core ML.

# Useful resources:
# - https://github.com/deezer/spleeter/issues/210
# - https://github.com/deezer/spleeter/issues/155
# - https://twitter.com/ExtractorVocal/status/1342643493227773952

import argparse
import coremltools as ct
import torch
from coremltools.models import MLModel

from pathlib import Path
from spleeter_pytorch.estimator import Estimator

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description='Converts Spleeter (minus the STFT preprocessing) to Core ML')
    parser.add_argument(
        '-n', '--num-instruments', type=int, choices=[2, 4, 5], default=2,
        help='The number of stems.')
    parser.add_argument(
        '-m', '--model', type=Path, default=None,
        help='The path to the model to use.')
    parser.add_argument(
        '-o', '--output', type=Path, default=None,
        help='The output directory to place the model in')
    parser.add_argument(
        '-l', '--length', type=float, default=5.0,
        help='The input length in seconds for the converted Core ML model (which will only take fixed-length inputs). Default: 5 seconds')

    args = parser.parse_args()

    if args.model is not None:
        model = args.model
    else:
        model = ROOT / 'checkpoints' / f'{args.num_instruments}stems' / 'model'

    if args.output is not None:
        output_dir = args.output
    else:
        output_dir = ROOT / 'output' / f'{args.num_instruments}stems' / 'coreml'

    samplerate = 44100

    if args.num_instruments == 2:
        conv_activation = "LeakyReLU"
        deconv_activation = "ReLU"
        softmax = False
    elif args.num_instruments == 4:
        conv_activation = "ELU"
        deconv_activation = "ELU"
        softmax = False
    elif args.num_instruments == 5:
        conv_activation = "ELU"
        deconv_activation = "ELU"
        softmax = True
    else:
        raise ValueError(f"Unsupported number of instruments: {args.num_instruments}")

    estimator = Estimator(
        num_instruments=args.num_instruments, checkpoint_path=model,
        conv_activation=conv_activation, deconv_activation=deconv_activation, softmax=softmax)
    estimator.eval()

    # Create sample 'audio' for tracing
    wav = torch.zeros(2, int(args.length * samplerate))

    # Reproduce the STFT step (which we cannot convert to Core ML, unfortunately)
    _, stft_mag = estimator.compute_stft(wav)

    print('==> Tracing model')
    traced_model = torch.jit.trace(estimator.separator, stft_mag)

    print('==> Converting to Core ML')
    mlmodel = ct.convert(
        traced_model,
        convert_to='mlprogram',
        # TODO: Investigate whether we'd want to make the input shape flexible
        # See https://coremltools.readme.io/docs/flexible-inputs
        inputs=[ct.TensorType(shape=stft_mag.shape)]
    )

    if not isinstance(mlmodel, MLModel):
        raise ValueError("Invalid model type.")

    spec = mlmodel.get_spec()

    ct.utils.rename_feature(spec, "stft_mag_1", "magnitude")
    if args.num_instruments == 2:
        ct.utils.rename_feature(spec, "var_614", "vocalsMask")
        ct.utils.rename_feature(spec, "var_648", "accompanimentMask")
    elif args.num_instruments == 4:
        ct.utils.rename_feature(spec, "var_1156", "vocalsMask")
        ct.utils.rename_feature(spec, "var_1190", "drumsMask")
        ct.utils.rename_feature(spec, "var_1224", "bassMask")
        ct.utils.rename_feature(spec, "var_1258", "otherMask")
    elif args.num_instruments == 5:
        ct.utils.rename_feature(spec, "var_1443", "vocalsMask")
        ct.utils.rename_feature(spec, "var_1477", "pianoMask")
        ct.utils.rename_feature(spec, "var_1511", "drumsMask")
        ct.utils.rename_feature(spec, "var_1545", "bassMask")
        ct.utils.rename_feature(spec, "var_1579", "otherMask")
    else:
        raise ValueError(f"Unsupported number of instruments: {args.num_instruments}")

    mlmodel = ct.models.MLModel(spec, weights_dir=mlmodel.weights_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f'SpleeterModel.mlpackage'

    print(f'==> Writing {output}')
    mlmodel.save(str(output))


if __name__ == '__main__':
    main()
