import numpy as np
from hparams import hparams, hparams_debug_string
from tacotron.pml_synthesizer import Configuration, PMLSynthesizer
from lib import sigproc as sp
import os
import argparse
import infolog

log = infolog.log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    parser.add_argument('--hparams', default='sample_rate=16000,pml_dimension=163,spec_type=fwbnd',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    args = parser.parse_args()
    hparams.parse(args.hparams)
    log(hparams_debug_string())
    verbose_level = 0

    if args.checkpoint is not None:
        synth = PMLSynthesizer()
        log('Synthesizing Audio...')
        synth.load(args.checkpoint, model_name='tacotron_pml')
        fixed_sentence = 'and district attorney henry m. wade both testified that they saw it later that day.'
        wav = synth.synthesize(fixed_sentence, to_wav=True)
    else:
        # Set up denormalisation parameters for synthesis
        mean_path = '/home/josh/tacotron/LJSpeech-1.1/pml/mean.dat'
        std_path = '/home/josh/tacotron/LJSpeech-1.1/pml/std.dat'
        mean_norm = None
        std_norm = None

        if os.path.isfile(mean_path) and os.path.isfile(std_path):
            mean_norm = np.fromfile(mean_path, 'float32')
            std_norm = np.fromfile(std_path, 'float32')

        # pml_cmp = np.fromfile('/home/josh/tacotron/LJSpeech-1.1/pml/LJ010-0018.cmp', dtype=np.float32)
        pml_cmp = np.fromfile('/home/josh/tacotron/LJSpeech-1.1/pml/LJ029-0088.cmp', dtype=np.float32)
        pml_features = pml_cmp.reshape((-1, hparams.pml_dimension))
        cfg = Configuration(hparams.sample_rate, hparams.pml_dimension)
        synth = PMLSynthesizer(cfg)
        log('Synthesizing Audio...')
        wav = synth.pml_to_wav(pml_features, verbose_level=verbose_level, mean_norm=mean_norm, std_norm=std_norm,
                               spec_type=hparams.spec_type)

    # handle the file save
    path = 'test_pml_converter.wav'
    sp.wavwrite(path, wav, hparams.sample_rate, norm_max_ifneeded=True, verbose=verbose_level)


if __name__ == '__main__':
    main()
