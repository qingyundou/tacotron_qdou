import numpy as np
import tensorflow as tf
from tacotron.models import create_model
from tacotron.utils.text import text_to_sequence
from util import audio
from lib import sigproc as sp
from infolog import log


# simplified port of the configuration from Merlin proper
class Configuration(object):
    def __init__(self):
        self.acoustic_feature_type = 'PML'
        self.acoustic_features = ['mgc', 'lf0', 'bap']
        self.acoustic_in_dimension_dict = {'mgc': 60, 'lf0': 1, 'bap': 25}
        self.acoustic_out_dimension_dict = {'mgc': 60, 'lf0': 1, 'bap': 25}

        self.acoustic_start_index = {
            'mgc': 0,
            'lf0': self.acoustic_out_dimension_dict['mgc'],
            'bap': self.acoustic_out_dimension_dict['mgc'] + self.acoustic_out_dimension_dict['lf0']
        }

        self.acoustic_file_ext_dict = {
            'mgc': '.mcep', 'lf0': '.lf0', 'bap': '.bndnm'}

        self.acoustic_dir_dict = {}
        self.var_file_dict = {}

        self.wav_sr = 16000  # 48000 #16000

        self.nn_features = ['lab', 'cmp', 'wav']
        self.nn_feature_dims = {}
        self.nn_feature_dims['lab'] = 601
        self.nn_feature_dims['cmp'] = sum(
            self.acoustic_out_dimension_dict.values())
        self.nn_feature_dims['wav'] = self.wav_sr / 200

        self.cmp_dim = self.nn_feature_dims['cmp']  # 86


cfg = Configuration()

_pad = 0


class PMLSynthesizer:
    def load(self, checkpoint_path, hparams, gta=False, model_name='tacotron_pml'):
        print('Constructing model: %s' % model_name)
        inputs = tf.placeholder(tf.int32, [None, None], 'inputs')
        input_lengths = tf.placeholder(tf.int32, [None], 'input_lengths')
        targets = tf.placeholder(tf.float32, [None, None, hparams.pml_dimension], 'pml_targets')

        with tf.variable_scope('model') as scope:
            self.model = create_model(model_name, hparams)

            if gta:
                self.model.initialize(inputs, input_lengths, pml_targets=targets, gta=gta)
            else:
                self.model.initialize(inputs, input_lengths)

            self.pml_outputs = self.model.pml_outputs

        self.gta = gta
        self._hparams = hparams

        self.inputs = inputs
        self.input_lengths = input_lengths
        self.targets = targets

        log('Loading checkpoint: %s' % checkpoint_path)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint_path)

    def synthesize(self, texts, pml_filenames=None, to_wav=False):
        hparams = self._hparams
        cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]

        seqs = []

        for text in texts:
            seqs.append(text_to_sequence(text, cleaner_names))

        feed_dict = {
            self.model.inputs: np.asarray(seqs, dtype=np.int32),
            self.model.input_lengths: np.asarray([len(seq) for seq in seqs], dtype=np.int32)
        }

        if self.gta:
            np_targets = [np.load(pml_filename) for pml_filename in pml_filenames]
            prepared_targets = self._prepare_targets(np_targets, hparams.outputs_per_step)
            feed_dict[self.targets] = prepared_targets
            assert len(np_targets) == len(texts)

        pml_features_matrix = self.session.run(self.pml_outputs, feed_dict=feed_dict)

        if to_wav:
            wavs = []

            for pml_features in pml_features_matrix:
                wav = self.pml_to_wav(pml_features)
                wav = wav[:audio.find_endpoint(wav, threshold_db=0)]
                wavs.append(wav)

            return wavs

        return pml_features_matrix

    def pml_to_wav(self, pml_features, shift=0.005, dftlen=4096, nm_cont=False, verbose_level=0):
        from lib.pulsemodel.synthesis import synthesize

        # f0s is from flf0
        f0 = pml_features[:, cfg.acoustic_start_index['lf0']:
                             cfg.acoustic_start_index['lf0'] + cfg.acoustic_in_dimension_dict['lf0']]

        f0 = np.squeeze(f0)  # remove the extra 1 dimension here
        f0[f0 > 0] = np.exp(f0[f0 > 0])
        ts = shift * np.arange(len(f0))
        f0s = np.vstack((ts, f0)).T

        # spec comes from fmcep
        mcep = pml_features[:, cfg.acoustic_start_index['mgc']:
                               cfg.acoustic_start_index['mgc'] + cfg.acoustic_in_dimension_dict['mgc']]

        spec = sp.mcep2spec(mcep, sp.bark_alpha(cfg.wav_sr), dftlen)

        # NM comes from bap
        fwnm = pml_features[:, cfg.acoustic_start_index['bap']:
                               cfg.acoustic_start_index['bap'] + cfg.acoustic_in_dimension_dict['bap']]

        nm = sp.fwbnd2linbnd(fwnm, cfg.wav_sr, dftlen)

        # use standard PML vocoder
        wav = synthesize(cfg.wav_sr, f0s, spec, NM=nm, nm_cont=nm_cont, verbose=verbose_level)

        # return the raw wav data
        return wav

    def _prepare_targets(self, targets, outputs_per_step):
        max_len = max((len(t) for t in targets)) + 50
        data_len = self._round_up(max_len, outputs_per_step)
        return np.stack([self._pad_target(t, data_len) for t in targets])

    def _pad_target(self, t, length):
        return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=_pad)

    def _round_up(self, x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x + multiple - remainder


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    verbose_level = 0

    if 'checkpoint' in args:
        synth = PMLSynthesizer()
        print('Synthesizing Audio...')
        synth.load(args.checkpoint, model_name='tacotron_pml')
        fixed_sentence = 'and district attorney henry m. wade both testified that they saw it later that day.'
        wav = synth.synthesize(fixed_sentence)
    else:
        # pml_cmp = np.fromfile('/home/josh/tacotron/LJSpeech-1.1/pml/LJ010-0018.cmp', dtype=np.float32)
        pml_cmp = np.fromfile('/home/josh/tacotron/Nick/pml/herald_1993_1.cmp', dtype=np.float32)
        pml_dimension = 86
        pml_features = pml_cmp.reshape((-1, pml_dimension))
        synth = PMLSynthesizer()
        print('Synthesizing Audio...')
        wav = synth.pml_to_wav(pml_features, verbose_level=verbose_level)

    # handle the file save
    path = 'test_pml_converter.wav'
    sp.wavwrite(path, wav, cfg.wav_sr, norm_max_ifneeded=True, verbose=verbose_level)


if __name__ == '__main__':
    main()
