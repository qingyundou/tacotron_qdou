"""
Example usage:

python synthesize.py --checkpoint ~/tacotron/logs-tacotron-pml-x-lj/model.ckpt-131000 --hparams "sample_rate=16000,frame_length_ms=20,frame_shift_ms=5" --mode synthesis
"""

import argparse
import os
from hparams import hparams
from tacotron.synthesize import tacotron_synthesize


def get_sentences(args, hparams):
    if args.text_list != '':
        with open(args.text_list, 'rb') as f:
            sentences = list(map(lambda l: l.decode("utf-8")[:-1], f.readlines()))
    else:
        sentences = hparams.sentences

    return sentences


def main():
    accepted_modes = ['eval', 'synthesis']

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--variant', default='tacotron')
    parser.add_argument('--model', default='tacotron')
    parser.add_argument('--gta', default='True',
                        help='Ground truth aligned synthesis, defaults to True, only considered in synthesis mode')
    parser.add_argument('--eal', default='False',
                        help='Explicit alignment locking, defaults to False, only considered in synthesis mode')
    parser.add_argument('--mode', default='eval', help='Mode of synthesis run, can be one of {}'.format(accepted_modes))
    parser.add_argument('--text_list', default='',
                        help='Text file contains list of texts to be synthesized. Valid if mode=eval')
    parser.add_argument('--output_dir', default='output', help='Folder to contain synthesized PML features')
    parser.add_argument('--input_dir', default='training', help='folder to contain inputs sentences/targets')
    parser.add_argument('--base_dir', default=os.path.expanduser('~/tacotron'))
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    hparams.parse(args.hparams)

    accepted_models = ['tacotron', 'wavenet']

    if args.model not in accepted_models:
        raise ValueError('Accepted models are: {}, you entered: {}'.format(accepted_models, args.model))

    if args.mode not in accepted_modes:
        raise ValueError('Accepted modes are: {}, you entered: {}'.format(accepted_modes, args.model))

    if args.gta not in ('True', 'False'):
        raise ValueError('Ground truth alignment option must be either True or False')

    sentences = get_sentences(args, hparams)

    if args.model == 'tacotron':
        _ = tacotron_synthesize(args, hparams, args.checkpoint, sentences)


if __name__ == '__main__':
    main()
