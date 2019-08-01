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

from tacotron.utils import plot

def run_eval(args, checkpoint_path, output_dir, hparams, sentences, flag_to_wav=False, checkpoint_eal=None):
    import pdb
    pdb.set_trace()
#     sentences = sentences[:3]
    log(hparams_debug_string())

#     synth_dir = os.path.join(args.base_dir, output_dir, 'eval')
    synth_dir = os.path.join(output_dir, 'eval','wav') if flag_to_wav else os.path.join(output_dir, 'eval','npy')

    # create the output path if it does not exist
    os.makedirs(synth_dir, exist_ok=True)

    # use the correct synthesizer for the model type
    if args.variant not in ['tacotron_orig']:
        cfg = Configuration(hparams.sample_rate, hparams.pml_dimension)
        synth = PMLSynthesizer(cfg)
    else:
        synth = Synthesizer()

    synth.load(checkpoint_path, hparams, model_name=args.variant, checkpoint_eal=checkpoint_eal)
    
    pdb.set_trace()

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
    if flag_to_wav:
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
    
    else:
        pml_features_matrix = synth.synthesize(sentences, to_wav=False, mean_norm=mean_norm, std_norm=std_norm,
                                spec_type=hparams.spec_type)
        for i, pml_feature in enumerate(pml_features_matrix):
            path = os.path.join(synth_dir, 'eval-%d.npy' % i)
            print('Writing {}...'.format(path))

            if args.variant not in ['tacotron_orig']:
                np.save(path, pml_feature, allow_pickle=False)
            else:
                warnings.warn('Generating pml features is not an option for original Tacotron')


def run_synthesis(args, checkpoint_path, output_dir, hparams, synthesis_mode='train', flag_connect_NV=False, checkpoint_eal=None):
    gta = (args.gta == 'True')
    eal = (args.eal == 'True')

    if eal:
        synth_dir = 'eal'
        gta = False # disable ground truth alignment mode if explicit alignment locking is enabled
        if checkpoint_eal is None: checkpoint_eal = checkpoint_path
    elif gta:
        synth_dir = 'gta'
    else:
        synth_dir = 'natural'

    synth_dir = os.path.join(output_dir, synth_dir, 'npy') # qd212

    # create the output path if it does not exist
    os.makedirs(synth_dir, exist_ok=True)

    metadata_filename = os.path.join(args.base_dir, args.training_dir, '{}.txt'.format(synthesis_mode))
    log(hparams_debug_string())

    cfg = Configuration(hparams.sample_rate, hparams.pml_dimension)
    synth = PMLSynthesizer(cfg)

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

    import pdb
    pdb.set_trace()
    
    with open(os.path.join(synth_dir, 'map.txt'), 'w') as file:
        for i in tqdm(range(0, len(metadata), args.batch_size)):
            texts = []
            pml_filenames = []
            wav_filenames = []
            pml_filenames_NV = []

#             for meta in metadata[i:min(i + args.batch_size, len(metadata) - 1)]:
            for meta in metadata[i:min(i + args.batch_size, len(metadata))]:
                texts.append(meta[5])
                pml_filenames.append(os.path.join(pml_path, meta[3]))
                wav_filenames.append(os.path.join(wav_path, meta[6]))
                pml_filenames_NV.append(os.path.join(synth_dir, meta[6].replace('.wav','.npy')))

            basenames = [os.path.basename(p).replace('.npy', '').replace('pml-', '') for p in pml_filenames]
            
            print(i)
            pdb.set_trace()

            if eal:
                locked_alignments = align_synth.synthesize(texts, pml_filenames=pml_filenames)
                log('Alignments synthesized with shape: {}'.format(locked_alignments.shape))
                synth.load(checkpoint_eal, hparams, eal=True, model_name=args.variant,
                       logs_enabled=False, locked_alignments=locked_alignments)
            else:
                log('locked_alignments is None')
                synth.load(checkpoint_path, hparams, gta=True, model_name=args.variant,
                           logs_enabled=False, locked_alignments=None)

            pml_features = synth.synthesize(texts, pml_filenames)
            pml_output_filenames = []

            for j, basename in enumerate(basenames):
                pml_filename = os.path.join(synth_dir, 'pml-{}.npy'.format(basename))
#                 np.save(pml_filename, pml_features[j], allow_pickle=False)
                if flag_connect_NV: np.save(pml_filenames_NV[j], pml_features[j], allow_pickle=False)
                else: np.save(pml_filename, pml_features[j], allow_pickle=False)
                pml_output_filenames.append(pml_filename)

            for elems in zip(wav_filenames, pml_filenames, pml_output_filenames, texts):
                file.write('|'.join([str(x) for x in elems]) + '\n')

    log('Synthesized PML features at {}'.format(synth_dir))
    return os.path.join(synth_dir, 'map.txt')


def run_synthesis_check(args, checkpoint_path, output_dir, hparams, synthesis_mode='train', flag_connect_NV=False):
    gta = (args.gta == 'True')
    eal = (args.eal == 'True')

    if eal:
        synth_dir = 'eal'
        gta = False # disable ground truth alignment mode if explicit alignment locking is enabled
    elif gta:
        synth_dir = 'gta'
    else:
        synth_dir = 'natural'

#     synth_dir = os.path.join(args.base_dir, output_dir, synth_dir)
    synth_dir = os.path.join(output_dir, synth_dir, 'alignment') # qd212

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

    import pdb
    pdb.set_trace()
#     args.batch_size = 10
    
    with open(os.path.join(synth_dir, 'map.txt'), 'w') as file:
        for i in tqdm(range(0, len(metadata), args.batch_size)):
            texts = []
            pml_filenames = []
            wav_filenames = []
            pml_filenames_NV = []

#             for meta in metadata[i:min(i + args.batch_size, len(metadata) - 1)]:
            for meta in metadata[i:min(i + args.batch_size, len(metadata))]:
                texts.append(meta[5])
                pml_filenames.append(os.path.join(pml_path, meta[3]))
                wav_filenames.append(os.path.join(wav_path, meta[6]))
                pml_filenames_NV.append(os.path.join(synth_dir, meta[6].replace('.wav','.npy')))

            basenames = [os.path.basename(p).replace('.npy', '').replace('pml-', '') for p in pml_filenames]
            locked_alignments = None
            
            print(i)
#             pdb.set_trace()
#             idx_utt = 72
#             name_utt = pml_filenames_NV[idx_utt]
#             texts = # fixed utt
#             pml_filenames = # fixed utt
            
            if eal:
                locked_alignments = align_synth.synthesize(texts, pml_filenames=pml_filenames)
                log('Alignments synthesized with shape: {}'.format(locked_alignments.shape))
                # check / visualize alignment
                pdb.set_trace()
                
#                 synth.load(checkpoint_path, hparams, gta=gta, model_name=args.variant,
#                        logs_enabled=False, locked_alignments=locked_alignments)
#                 tmp_alignment, tmp_pml_intermediates = synth.synthesize_check(texts, pml_filenames)
#                 np.save(os.path.join(synth_dir,'step-%d-eal-pml.npy' % i),tmp_pml_intermediates)
        
#                 random_attention_plot = plot.plot_alignment(locked_alignments[72],
#                                                             os.path.join(synth_dir,'step-%d-eal-align-0073.png' % i),
#                                                                 info='%s, %s, %s, step=%d, loss=%.5f' % (
#                                                                 'args.variant', 'commit', 'time_string()', 0, 1.1))

                synth.load(checkpoint_path_eal, hparams, gta=gta, model_name=args.variant,
                       logs_enabled=False, locked_alignments=locked_alignments)
                tmp_alignment = synth.synthesize_check(texts, pml_filenames)
        
#                 random_attention_plot = plot.plot_alignment(tmp_alignment[72],
#                                                             os.path.join(synth_dir,'step-%d-eal-align-0073-ck.png' % i),
#                                                                 info='%s, %s, %s, step=%d, loss=%.5f' % (
#                                                                 'args.variant', 'commit', 'time_string()', 0, 1.1))

            else:
                log('locked_alignments is None')
                # get & check / visualize alignment
                synth.load(checkpoint_path, hparams, gta=gta, model_name=args.variant,
                       logs_enabled=False, locked_alignments=locked_alignments)
                pdb.set_trace()
                
#                 tmp_alignment, tmp_pml_intermediates = synth.synthesize_check(texts, pml_filenames)
#                 np.save(os.path.join(synth_dir,'step-%d-gta-pml.npy' % i),tmp_pml_intermediates)
            
                tmp_alignment = synth.synthesize_check(texts, pml_filenames)
                pdb.set_trace()
                random_attention_plot = plot.plot_alignment(tmp_alignment[72],
                                                            os.path.join(synth_dir,'step-%d-gta-align-0073.png' % i),
                                                                info='%s, %s, %s, step=%d, loss=%.5f' % (
                                                                'args.variant', 'commit', 'time_string()', 0, 1.1))

    return os.path.join(synth_dir, 'map.txt')

def run_eval_check(args, checkpoint_path, output_dir, hparams, sentences,flag_to_wav=False, checkpoint_eal=None):
    import pdb
    pdb.set_trace()
    
    log(hparams_debug_string())

    synth_dir = os.path.join(output_dir, 'eval','alignment')

    # create the output path if it does not exist
    os.makedirs(synth_dir, exist_ok=True)

    # use the correct synthesizer for the model type
    if args.variant not in ['tacotron_orig']:
        cfg = Configuration(hparams.sample_rate, hparams.pml_dimension)
        synth = PMLSynthesizer(cfg)
    else:
        synth = Synthesizer()

    synth.load(checkpoint_path, hparams, model_name=args.variant, checkpoint_eal=checkpoint_eal)
    
    tmp_alignment = synth.synthesize_check(sentences)
    pdb.set_trace()
    random_attention_plot = plot.plot_alignment(tmp_alignment[6],
                                                os.path.join(synth_dir,'ss-align-0073.png'),
                                                    info='%s, %s, %s, step=%d, loss=%.5f' % (
                                                    'args.variant', 'commit', 'time_string()', 0, 1.1))
    

def save_alignment(args, checkpoint_path, output_dir, hparams, synthesis_mode='train'):
    synth_dir = os.path.join(output_dir, 'gta', 'alignment/npy') # qd212
    os.makedirs(synth_dir, exist_ok=True)

    metadata_filename = os.path.join(args.base_dir, args.training_dir, '{}.txt'.format(synthesis_mode))
    log(hparams_debug_string())

    with open(metadata_filename, encoding='utf-8') as f:
        metadata = [line.strip().split('|') for line in f]
        hours = sum((int(x[2]) for x in metadata)) * hparams.frame_shift_ms / (3600 * 1000)
        log('Loaded metadata for %d examples (%.2f hours)' % (len(metadata), hours))

    align_synth = AlignmentSynthesizer()
    align_synth.load(checkpoint_path, hparams, cut_lengths=False, gta=True, model_name=args.variant)

    log('Starting synthesis')
    pml_path = os.path.join(args.base_dir, args.training_dir, pml_dir)
    wav_path = os.path.join(args.base_dir, args.training_dir, wav_dir)

#     import pdb
#     pdb.set_trace()
    
    for i in tqdm(range(0, len(metadata), args.batch_size)):
        texts = []
        pml_filenames = []
        wav_filenames = []
        alignment_filenames = []

        for meta in metadata[i:min(i + args.batch_size, len(metadata))]:
            texts.append(meta[5])
            pml_filenames.append(os.path.join(pml_path, meta[3]))
            wav_filenames.append(os.path.join(wav_path, meta[6]))
            alignment_filenames.append(os.path.join(synth_dir, meta[6].replace('.wav','_align.npy')))

        locked_alignments = align_synth.synthesize(texts, pml_filenames=pml_filenames)
        log('Alignments synthesized with shape: {}'.format(locked_alignments.shape))
        
        print(i)
#         pdb.set_trace()

        for j, a in enumerate(alignment_filenames):
            np.save(a, locked_alignments[j], allow_pickle=False)
    return


def tacotron_synthesize(args, hparams, checkpoint, sentences=None, checkpoint_eal=None):
#     output_dir = 'tacotron_' + args.output_dir
    output_dir = args.output_dir
    
#     import pdb
#     pdb.set_trace()
#     return run_synthesis_check(args, checkpoint, output_dir, hparams, args.dataset, flag_connect_NV=True)
#     return run_eval_check(args, checkpoint, output_dir, hparams, sentences,flag_to_wav=False, checkpoint_eal=checkpoint_eal)

    if args.mode == 'synthesis':
        return run_synthesis(args, checkpoint, output_dir, hparams, args.dataset, flag_connect_NV=True, checkpoint_eal=checkpoint_eal)
    elif args.mode == 'eval':
        return run_eval(args, checkpoint, output_dir, hparams, sentences, checkpoint_eal=checkpoint_eal)
    elif args.mode == 'alignment':
        save_alignment(args, checkpoint, output_dir, hparams, synthesis_mode='train')
        