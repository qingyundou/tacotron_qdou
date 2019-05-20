import argparse
import os
from multiprocessing import cpu_count
from tqdm import tqdm
from datasets import blizzard, ljspeech, nick
from hparams import hparams, hparams_debug_string
from infolog import log


def preprocess_blizzard(args, hparams):
    in_dir = os.path.join(args.base_dir, 'Blizzard2012')
    out_dir = os.path.join(args.base_dir, args.output)
    os.makedirs(out_dir, exist_ok=True)
    metadata = blizzard.build_from_path(in_dir, out_dir, hparams, args.num_workers, tqdm=tqdm)
    write_metadata(metadata[:-args.validation_size-args.test_size], out_dir)

    if args.validation_size > 0:
        write_validation(metadata[-args.validation_size-args.test_size:-args.test_size], out_dir)

    if args.test_size > 0:
        write_validation(metadata[-args.test_size:], out_dir, filename='test.txt')


def preprocess_ljspeech(args, hparams):
    in_dir = os.path.join(args.base_dir, 'LJSpeech-1.1')
    out_dir = os.path.join(args.base_dir, args.output)
    os.makedirs(out_dir, exist_ok=True)
    metadata = ljspeech.build_from_path(in_dir, out_dir, hparams, args.num_workers, tqdm=tqdm)
    write_metadata(metadata[:-args.validation_size-args.test_size], out_dir)

    if args.validation_size > 0:
        write_validation(metadata[-args.validation_size-args.test_size:-args.test_size], out_dir)

    if args.test_size > 0:
        write_validation(metadata[-args.test_size:], out_dir, filename='test.txt')


def preprocess_nick(args, hparams):
    in_dir = os.path.join(args.base_dir, 'Nick')
    out_dir = os.path.join(args.base_dir, args.output)
    os.makedirs(out_dir, exist_ok=True)
    metadata = nick.build_from_path(in_dir, out_dir, hparams, args.num_workers, tqdm=tqdm)
    write_metadata(metadata[:-args.validation_size-args.test_size], out_dir)

    if args.validation_size > 0:
        write_validation(metadata[-args.validation_size-args.test_size:-args.test_size], out_dir)

    if args.test_size > 0:
        write_validation(metadata[-args.test_size:], out_dir, filename='test.txt')


def write_metadata(metadata, out_dir, filename='train.txt'):
    with open(os.path.join(out_dir, filename), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    frames = sum([m[2] for m in metadata])
    hours = frames * hparams.frame_shift_ms / (3600 * 1000)
    log('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))
    log('Max input length:  %d' % max(len(m[5]) for m in metadata))
    log('Max output length: %d' % max(m[2] for m in metadata))


def write_validation(metadata, out_dir, filename='validation.txt'):
    with open(os.path.join(out_dir, filename), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.path.expanduser('~/tacotron'))
    parser.add_argument('--output', default='training')
    parser.add_argument('--dataset', required=True, choices=['blizzard', 'ljspeech', 'nick'])
    parser.add_argument('--num_workers', type=int, default=cpu_count())
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--validation_size', type=int, default=0)
    parser.add_argument('--test_size', type=int, default=0)

    args = parser.parse_args()
    hparams.parse(args.hparams)
    log(hparams_debug_string())

    if args.dataset == 'blizzard':
        preprocess_blizzard(args, hparams)
    elif args.dataset == 'ljspeech':
        preprocess_ljspeech(args, hparams)
    elif args.dataset == 'nick':
        preprocess_nick(args, hparams)


if __name__ == "__main__":
    main()
