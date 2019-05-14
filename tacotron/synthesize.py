import os
import re
from lib import sigproc as sp
from hparams import hparams_debug_string
from tacotron.synthesizer import Synthesizer
from tacotron.pml_synthesizer import PMLSynthesizer, cfg
from infolog import log
from tqdm import tqdm
import tensorflow as tf


def get_output_base_path(checkpoint_path):
    base_dir = os.path.dirname(checkpoint_path)
    m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
    name = 'eval-%d' % int(m.group(1)) if m else 'eval'
    return os.path.join(base_dir, name)


def run_eval(args, checkpoint_path, output_dir, hparams, sentences):
    log(hparams_debug_string())

    # use the correct synthesizer for the model type
    if args.variant not in ['tacotron']:
        synth = PMLSynthesizer()
    else:
        synth = Synthesizer()

    synth.load(checkpoint_path, hparams, model_name=args.variant)
    base_path = get_output_base_path(checkpoint_path)

    for i, text in enumerate(sentences):
        path = os.path.join(output_dir, '%s-%d.wav' % (base_path, i))
        print('Synthesizing: %s' % path)
        wav = synth.synthesize(text, to_wav=True)

        if args.variant not in ['tacotron']:
            sp.wavwrite(path, wav, cfg.wav_sr, norm_max_ifneeded=True, verbose=0)
        else:
            with open(path, 'wb') as f:
                f.write(wav)


def run_synthesis(args, checkpoint_path, output_dir, hparams):
    gta = (args.gta == 'True')

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
    pml_dir = os.path.join(args.input_dir, 'pmls')
    wav_dir = os.path.join(args.input_dir, 'wavs')

    with open(os.path.join(synth_dir, 'map.txt'), 'w') as file:
        for i, meta in enumerate(tqdm(metadata)):
            texts = [m[5] for m in meta]
            pml_filenames = [os.path.join(pml_dir, m[1]) for m in meta]
            wav_filenames = [os.path.join(wav_dir, m[0]) for m in meta]
            basenames = [os.path.basename(p).replace('.npy', '').replace('pml-', '') for p in pml_filenames]
            pml_output_filenames, speaker_ids = synth.synthesize(texts, basenames, synth_dir, None, pml_filenames)

            for elems in zip(wav_filenames, pml_filenames, pml_output_filenames, speaker_ids, texts):
                file.write('|'.join([str(x) for x in elems]) + '\n')

    log('Synthesized PML features at {}'.format(synth_dir))
    return os.path.join(synth_dir, 'map.txt')


def tacotron_synthesize(args, hparams, checkpoint, sentences=None):
    output_dir = 'tacotron_' + args.output_dir

    # check the checkpoint path is working correctly
    try:
        checkpoint_path = tf.train.get_checkpoint_state(checkpoint).model_checkpoint_path
        log('Loaded model at {}'.format(checkpoint_path))
    except:
        raise RuntimeError('Failed to load checkpoint at {}'.format(checkpoint))

    if args.mode == 'synthesis':
        return run_synthesis(args, checkpoint, output_dir, hparams)
    else:
        return run_eval(args, checkpoint, output_dir, hparams, sentences)
