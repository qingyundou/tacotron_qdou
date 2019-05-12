import argparse
import os
import re
from lib import sigproc as sp
from hparams import hparams, hparams_debug_string
from tacotron.synthesizer import Synthesizer
from tacotron.pml_synthesizer import PMLSynthesizer, cfg
from infolog import log
from tqdm import tqdm

sentences = [
    # From July 8, 2017 New York Times:
    'Scientists at the CERN laboratory say they have discovered a new particle.',
    'There’s a way to measure the acute emotional intelligence that has never gone out of style.',
    'President Trump met with other leaders at the Group of 20 conference.',
    'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
    # From Google's Tacotron example page:
    'Generative adversarial network or variational auto-encoder.',
    'The buses aren\'t the problem, they actually provide a solution.',
    'Does the quick brown fox jump over the lazy dog?',
    'Talib Kweli confirmed to AllHipHop that he will be releasing an album in the next year.',
]


def get_output_base_path(checkpoint_path):
    base_dir = os.path.dirname(checkpoint_path)
    m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
    name = 'eval-%d' % int(m.group(1)) if m else 'eval'
    return os.path.join(base_dir, name)


def run_eval(args):
    log(hparams_debug_string())

    # use the correct synthesizer for the model type
    if args.variant not in ['tacotron']:
        synth = PMLSynthesizer()
    else:
        synth = Synthesizer()

    synth.load(args.checkpoint, hparams, model_name=args.variant)
    base_path = get_output_base_path(args.checkpoint)

    for i, text in enumerate(sentences):
        path = '%s-%d.wav' % (base_path, i)
        print('Synthesizing: %s' % path)
        wav = synth.synthesize(text, to_wav=True)

        if args.variant not in ['tacotron']:
            sp.wavwrite(path, wav, cfg.wav_sr, norm_max_ifneeded=True, verbose=0)
        else:
            with open(path, 'wb') as f:
                f.write(wav)


def run_synthesis(args, checkpoint_path, output_dir, hparams):
    gta = True

    if gta:
        synth_dir = os.path.join(output_dir, 'gta')
        # create the output path if it does not exist
        os.makedirs(synth_dir, exist_ok=True)
    else:
        synth_dir = os.path.join(output_dir, 'natural')
        # create the output path if it does not exist
        os.makedirs(synth_dir, exist_ok=True)

    metadata_filename = os.path.join(args.input_dir, 'train.txt')
    log(hparams_debug_string())

    synth = PMLSynthesizer()
    synth.load(checkpoint_path, hparams, gta=gta, model_name=args.variant)

    with open(metadata_filename, encoding='utf-8') as f:
        metadata = [line.strip().split('|') for line in f]
        hours = sum((int(x[2]) for x in metadata)) * hparams.frame_shift_ms / (3600 * 1000)
        log('Loaded metadata for %d examples (%.2f hours)' % (len(metadata), hours))

    log('Starting Synthesis')
    pml_dir = os.path.join(args.input_dir, 'mels')
    wav_dir = os.path.join(args.input_dir, 'wavs')

    with open(os.path.join(synth_dir, 'map.txt'), 'w') as file:
        for i, meta in enumerate(tqdm(metadata)):
            texts = [m[5] for m in meta]
            mel_filenames = [os.path.join(pml_dir, m[1]) for m in meta]
            wav_filenames = [os.path.join(wav_dir, m[0]) for m in meta]
            basenames = [os.path.basename(m).replace('.npy', '').replace('mel-', '') for m in mel_filenames]
            mel_output_filenames, speaker_ids = synth.synthesize(texts, basenames, synth_dir, None, mel_filenames)

            for elems in zip(wav_filenames, mel_filenames, mel_output_filenames, speaker_ids, texts):
                file.write('|'.join([str(x) for x in elems]) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--variant', default='tacotron')
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    hparams.parse(args.hparams)
    run_eval(args)


if __name__ == '__main__':
    main()
