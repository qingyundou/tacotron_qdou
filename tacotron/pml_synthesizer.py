import numpy as np
import tensorflow as tf
from tacotron.models import create_model
from tacotron.utils.text import text_to_sequence
from util import audio
from lib import sigproc as sp
from infolog import log


# simplified port of the configuration from Merlin proper
class Configuration(object):
    def __init__(self, wav_sr=16000, pml_dimension=86):
        self.acoustic_feature_type = 'PML'
        self.acoustic_features = ['mgc', 'lf0', 'bap']

        if pml_dimension == 86:
            self.acoustic_in_dimension_dict = {'mgc': 60, 'lf0': 1, 'bap': 25}
            self.acoustic_out_dimension_dict = {'mgc': 60, 'lf0': 1, 'bap': 25}

            self.acoustic_start_index = {
                'mgc': 0,
                'lf0': self.acoustic_out_dimension_dict['mgc'],
                'bap': self.acoustic_out_dimension_dict['mgc'] + self.acoustic_out_dimension_dict['lf0']
            }
        elif pml_dimension == 163:
            self.acoustic_in_dimension_dict = {'mgc': 129, 'lf0': 1, 'bap': 33}
            self.acoustic_out_dimension_dict = {'mgc': 129, 'lf0': 1, 'bap': 33}

            self.acoustic_start_index = {
                'lf0': 0,
                'mgc': self.acoustic_out_dimension_dict['lf0'],
                'bap': self.acoustic_out_dimension_dict['mgc'] + self.acoustic_out_dimension_dict['lf0']
            }

        self.acoustic_file_ext_dict = {
            'mgc': '.mcep', 'lf0': '.lf0', 'bap': '.bndnm'}

        self.acoustic_dir_dict = {}
        self.var_file_dict = {}

        self.wav_sr = wav_sr  # 48000 #16000

        self.nn_features = ['lab', 'cmp', 'wav']
        self.nn_feature_dims = {}
        self.nn_feature_dims['lab'] = 601
        self.nn_feature_dims['cmp'] = sum(
            self.acoustic_out_dimension_dict.values())
        self.nn_feature_dims['wav'] = self.wav_sr / 200

        self.cmp_dim = self.nn_feature_dims['cmp']  # 86 or 163


_pad = 0


class PMLSynthesizer:
    def __init__(self, cfg=None):
        if cfg is None:
            self.cfg = Configuration()
        else:
            self.cfg = cfg

    def load(self, checkpoint_path, hparams, gta=False, model_name='tacotron_pml', locked_alignments=None,
             logs_enabled=False):
        if logs_enabled:
            log('Constructing model: %s' % model_name)

        inputs = tf.placeholder(tf.int32, [None, None], 'inputs')
        input_lengths = tf.placeholder(tf.int32, [None], 'input_lengths')
        targets = tf.placeholder(tf.float32, [None, None, hparams.pml_dimension], 'pml_targets')

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as scope:
            self.model = create_model(model_name, hparams)

            if gta:
                self.model.initialize(inputs, input_lengths, pml_targets=targets, gta=gta, logs_enabled=logs_enabled)
            elif locked_alignments is not None:
                self.model.initialize(inputs, input_lengths, locked_alignments=locked_alignments,
                                      logs_enabled=logs_enabled)
            else:
                self.model.initialize(inputs, input_lengths, logs_enabled=logs_enabled)

            self.pml_outputs = self.model.pml_outputs

        self.gta = gta
        self._hparams = hparams

        self.inputs = inputs
        self.input_lengths = input_lengths
        self.targets = targets

        if logs_enabled:
            log('Loading checkpoint: %s' % checkpoint_path)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint_path)

    def synthesize(self, texts, pml_filenames=None, to_wav=False, **kwargs):
        hp = self._hparams
        cleaner_names = [x.strip() for x in hp.cleaners.split(',')]
        seqs = [np.asarray(text_to_sequence(text, cleaner_names), dtype=np.int32) for text in texts]
        input_seqs = self._prepare_inputs(seqs)

        feed_dict = {
            self.model.inputs: np.asarray(input_seqs, dtype=np.int32),
            self.model.input_lengths: np.asarray([len(seq) for seq in seqs], dtype=np.int32)
        }

        if self.gta:
            np_targets = [np.load(pml_filename) for pml_filename in pml_filenames]
            prepared_targets = self._prepare_targets(np_targets, hp.outputs_per_step)
            feed_dict[self.targets] = prepared_targets
            assert len(np_targets) == len(texts)

        pml_features_matrix = self.session.run(self.pml_outputs, feed_dict=feed_dict)

        if to_wav:
            wavs = []

            for pml_features in pml_features_matrix:
                wav = self.pml_to_wav(pml_features, **kwargs)
                wav = wav[:audio.find_endpoint(wav, threshold_db=0)]
                wavs.append(wav)

            return wavs

        return pml_features_matrix

    def pml_to_wav(self, pml_features, shift=0.005, dftlen=4096, nm_cont=False, verbose_level=0, mean_norm=None,
                   std_norm=None, spec_type='mcep'):
        from lib.pulsemodel.synthesis import synthesize

        # get the mean and variance, and denormalise
        if mean_norm is not None and std_norm is not None:
            std_tiled = np.tile(std_norm, (pml_features.shape[0], 1))
            mean_tiled = np.tile(mean_norm, (pml_features.shape[0], 1))
            pml_features = pml_features * std_tiled + mean_tiled

        # f0s is from flf0
        f0 = pml_features[:, self.cfg.acoustic_start_index['lf0']:
                             self.cfg.acoustic_start_index['lf0'] + self.cfg.acoustic_in_dimension_dict['lf0']]

        f0 = np.squeeze(f0)  # remove the extra 1 dimension here
        f0[f0 > 0] = np.exp(f0[f0 > 0])
        ts = shift * np.arange(len(f0))
        f0s = np.vstack((ts, f0)).T

        # spec comes from fmcep or something else fwbnd
        if spec_type == 'mcep':
            mcep = pml_features[:, self.cfg.acoustic_start_index['mgc']:
                                   self.cfg.acoustic_start_index['mgc'] + self.cfg.acoustic_in_dimension_dict['mgc']]
            spec = sp.mcep2spec(mcep, sp.bark_alpha(self.cfg.wav_sr), dftlen)
        elif spec_type == 'fwbnd':
            compspec = pml_features[:, self.cfg.acoustic_start_index['mgc']:
                                   self.cfg.acoustic_start_index['mgc'] + self.cfg.acoustic_in_dimension_dict['mgc']]
            spec = np.exp(sp.fwbnd2linbnd(compspec, self.cfg.wav_sr, dftlen))

        # NM comes from bap
        fwnm = pml_features[:, self.cfg.acoustic_start_index['bap']:
                               self.cfg.acoustic_start_index['bap'] + self.cfg.acoustic_in_dimension_dict['bap']]

        nm = sp.fwbnd2linbnd(fwnm, self.cfg.wav_sr, dftlen)

        # use standard PML vocoder
        wav = synthesize(self.cfg.wav_sr, f0s, spec, NM=nm, nm_cont=nm_cont, verbose=verbose_level)

        # return the raw wav data
        return wav

    def _prepare_inputs(self, inputs):
        max_len = max((len(x) for x in inputs))
        return np.stack([self._pad_input(x, max_len) for x in inputs])

    def _prepare_targets(self, targets, outputs_per_step):
        max_len = max((len(t) for t in targets)) + 50
        data_len = self._round_up(max_len, outputs_per_step)
        return np.stack([self._pad_target(t, data_len) for t in targets])

    def _pad_input(self, x, length):
        return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)

    def _pad_target(self, t, length):
        return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=_pad)

    def _round_up(self, x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x + multiple - remainder
