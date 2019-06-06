import os
from lib import sigproc as sp
from hparams import hparams_debug_string
from tacotron.alignment_synthesizer import AlignmentSynthesizer
from tacotron.synthesizer import Synthesizer
from tacotron.pml_synthesizer import Configuration, PMLSynthesizer
from infolog import log
from tqdm import tqdm
import numpy as np
from datasets import pml_dir, wav_dir
import warnings


def run_eval(args, checkpoint_path, output_dir, hparams, sentences):
    log(hparams_debug_string())

    synth_dir = os.path.join(args.base_dir, output_dir, 'eval')
    # create the output path if it does not exist
    os.makedirs(synth_dir, exist_ok=True)

    # use the correct synthesizer for the model type
    if args.variant not in ['tacotron_orig']:
        cfg = Configuration(hparams.sample_rate, hparams.pml_dimension)
        synth = PMLSynthesizer(cfg)
    else:
        synth = Synthesizer()

    synth.load(checkpoint_path, hparams, model_name=args.variant)

    # Set up denormalisation parameters for synthesis
    mean_path = os.path.abspath(os.path.join(args.base_dir, args.training_dir, 'pml_data/mean.dat'))
    std_path = os.path.abspath(os.path.join(args.base_dir, args.training_dir, 'pml_data/std.dat'))
    mean_norm = None
    std_norm = None

    if os.path.isfile(mean_path) and os.path.isfile(std_path):
        mean_norm = np.fromfile(mean_path, 'float32')
        std_norm = np.fromfile(std_path, 'float32')
    else:
        warnings.warn('No mean or standard deviation files found at locations {} and {}'.format(mean_path, std_path))

    print('Synthesizing to {}...'.format(synth_dir))
    wavs = synth.synthesize(sentences, to_wav=True, mean_norm=mean_norm, std_norm=std_norm,
                            spec_type=hparams.spec_type)

    for i, wav in enumerate(wavs):
        path = os.path.join(synth_dir, 'eval-%d.wav' % i)
        print('Writing {}...'.format(path))

        if args.variant not in ['tacotron_orig']:
            sp.wavwrite(path, wav, hparams.sample_rate, norm_max_ifneeded=True, verbose=0)
        else:
            with open(path, 'wb') as f:
                f.write(wav)


def run_synthesis(args, checkpoint_path, output_dir, hparams, synthesis_mode='train'):
    gta = (args.gta == 'True')
    eal = (args.eal == 'True')

    if eal:
        synth_dir = 'eal'
        gta = False # disable ground truth alignment mode if explicit alignment locking is enabled
    elif gta:
        synth_dir = 'gta'
    else:
        synth_dir = 'natural'

    synth_dir = os.path.join(args.base_dir, output_dir, synth_dir)
    # create the output path if it does not exist
    os.makedirs(synth_dir, exist_ok=True)

    metadata_filename = os.path.join(args.base_dir, args.training_dir, '{}.txt'.format(synthesis_mode))
    log(hparams_debug_string())

    cfg = Configuration(hparams.sample_rate, hparams.pml_dimension)
    synth = PMLSynthesizer(cfg)
    synth.load(checkpoint_path, hparams, gta=gta, model_name=args.variant)

    with open(metadata_filename, encoding='utf-8') as f:
        metadata = [line.strip().split('|') for line in f]
        hours = sum((int(x[2]) for x in metadata)) * hparams.frame_shift_ms / (3600 * 1000)
        log('Loaded metadata for %d examples (%.2f hours)' % (len(metadata), hours))

    if eal:
        align_synth = AlignmentSynthesizer()
        align_synth.load(checkpoint_path, hparams, cut_lengths=False, gta=True, model_name=args.variant)

    log('Starting synthesis')
    pml_path = os.path.join(args.base_dir, args.training_dir, pml_dir)
    wav_path = os.path.join(args.base_dir, args.training_dir, wav_dir)

    with open(os.path.join(synth_dir, 'map.txt'), 'w') as file:
        for i in tqdm(range(0, len(metadata), args.batch_size)):
            texts = []
            pml_filenames = []
            wav_filenames = []

            for meta in metadata[i:min(i + args.batch_size, len(metadata) - 1)]:
                texts.append(meta[5])
                pml_filenames.append(os.path.join(pml_path, meta[3]))
                wav_filenames.append(os.path.join(wav_path, meta[6]))

            basenames = [os.path.basename(p).replace('.npy', '').replace('pml-', '') for p in pml_filenames]
            locked_alignments = None

            if eal:
                locked_alignments = align_synth.synthesize(texts, pml_filenames=pml_filenames)

            log('Alignments synthesized with shape: {}'.format(locked_alignments.shape))
            synth.load(checkpoint_path, hparams, gta=gta, model_name=args.variant,
                       logs_enabled=False, locked_alignments=locked_alignments)
            pml_features = synth.synthesize(texts, pml_filenames)
            pml_output_filenames = []

            for j, basename in enumerate(basenames):
                pml_filename = os.path.join(synth_dir, 'pml-{}.npy'.format(basename))
                np.save(pml_filename, pml_features[j], allow_pickle=False)
                pml_output_filenames.append(pml_filename)

            for elems in zip(wav_filenames, pml_filenames, pml_output_filenames, texts):
                file.write('|'.join([str(x) for x in elems]) + '\n')

    log('Synthesized PML features at {}'.format(synth_dir))
    return os.path.join(synth_dir, 'map.txt')


def tacotron_synthesize(args, hparams, checkpoint, sentences=None):
    output_dir = 'tacotron_' + args.output_dir

    if args.mode == 'synthesis':
        return run_synthesis(args, checkpoint, output_dir, hparams, args.dataset)
    else:
        return run_eval(args, checkpoint, output_dir, hparams, sentences)
