#!/usr/bin/env python3

# A script for exporting the Spleeter separator (minus the STFT) to ONNX, for ONNX Runtime — swift-spleeter's
# SpleeterONNX backend on Android. It mirrors convert-to-coreml.py: the same raw-mask outputs under the same names.

import argparse
import torch

from pathlib import Path
from spleeter_pytorch.estimator import Estimator

ROOT = Path(__file__).resolve().parent

# The output names, in the separator's stem order, that swift-spleeter reads.
MASK_NAMES = {
    2: ["vocalsMask", "accompanimentMask"],
    4: ["vocalsMask", "drumsMask", "bassMask", "otherMask"],
    5: ["vocalsMask", "pianoMask", "drumsMask", "bassMask", "otherMask"],
}

ACTIVATIONS = {
    # (conv activation, deconv activation, softmax)
    2: ("LeakyReLU", "ReLU", False),
    4: ("ELU", "ELU", False),
    5: ("ELU", "ELU", True),
}


class TupleSeparator(torch.nn.Module):
    """The separator returns a list of masks; return a tuple, so each mask is its own named graph output."""

    def __init__(self, separator):
        super().__init__()
        self.separator = separator

    def forward(self, magnitude):
        return tuple(self.separator(magnitude))


def main():
    parser = argparse.ArgumentParser(
        description='Exports Spleeter (minus the STFT preprocessing) to ONNX')
    parser.add_argument(
        '-n', '--num-instruments', type=int, choices=[2, 4, 5], default=2,
        help='The number of stems.')
    parser.add_argument(
        '-m', '--model', type=Path, default=None,
        help='The path to the checkpoint to use. Default: checkpoints/{n}stems/model')
    parser.add_argument(
        '-o', '--output', type=Path, default=None,
        help='The output file. Default: output/{n}stems/onnx/Spleeter{n}Model.onnx')
    parser.add_argument(
        '-f', '--frames', type=int, default=512,
        help='The STFT frames the model takes (fixed). Default: 512, the separator\'s own partition size, which a '
             'shorter input is padded to inside the graph anyway; 216 matches the Core ML models\' 5 s window.')
    parser.add_argument(
        '--opset', type=int, default=17, help='The ONNX opset. Default: 17')
    parser.add_argument(
        '--check', action='store_true',
        help='Run the exported model in onnxruntime on random magnitudes and compare it with PyTorch.')

    args = parser.parse_args()
    n = args.num_instruments
    model = args.model or ROOT / 'checkpoints' / f'{n}stems' / 'model'
    output = args.output or ROOT / 'output' / f'{n}stems' / 'onnx' / f'Spleeter{n}Model.onnx'

    conv_activation, deconv_activation, softmax = ACTIVATIONS[n]
    estimator = Estimator(
        num_instruments=n, checkpoint_path=model,
        conv_activation=conv_activation, deconv_activation=deconv_activation, softmax=softmax)
    estimator.eval()
    # Export each stem's raw mask; the consumer turns them into ratio masks itself, as with the Core ML models.
    estimator.separator.ratio_masks = False

    # The frame count is fixed: the separator pads and partitions its input with Python arithmetic on the traced
    # shape, which the export bakes in, so a dynamic frames axis would be wrong for any other length.
    magnitude = torch.zeros(2, estimator.F, args.frames)

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f'==> Exporting {output} ({args.frames} frames, opset {args.opset})')
    with torch.no_grad():
        torch.onnx.export(
            TupleSeparator(estimator.separator).eval(), (magnitude,), str(output),
            input_names=['magnitude'], output_names=MASK_NAMES[n], opset_version=args.opset,
            do_constant_folding=True)
    print(f'    {output.stat().st_size / 1e6:.1f} MB')

    if args.check:
        import numpy as np
        import onnxruntime as ort

        generator = torch.Generator().manual_seed(0)
        sample = torch.rand(2, estimator.F, args.frames, generator=generator) * 10
        with torch.no_grad():
            reference = [mask.numpy() for mask in estimator.separator(sample)]
        session = ort.InferenceSession(str(output), providers=['CPUExecutionProvider'])
        outputs = session.run(MASK_NAMES[n], {'magnitude': sample.numpy()})
        for name, ref, out in zip(MASK_NAMES[n], reference, outputs):
            assert ref.shape == out.shape, (name, ref.shape, out.shape)
            print(f'    {name}: shape {out.shape}, max |onnx - torch| = {np.abs(ref - out).max():.2e} '
                  f'(max |torch| = {np.abs(ref).max():.2e})')


if __name__ == '__main__':
    main()
