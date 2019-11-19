import numpy as np
import tensorflow as tf
from tacotron.models import create_model
from tacotron.utils.text import text_to_sequence
from util import audio
from lib import sigproc as sp
from infolog import log
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from lib.pulsemodel.synthesis import synthesize


# simplified port of the configuration from Merlin proper
class Configuration(object):
    def __init__(self, wav_sr=16000, pml_dimension=86):
        self.acoustic_feature_type = 'PML'
        self.acoustic_features = ['mgc', 'lf0', 'bap']
        self.pp_mcep = False

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

            self.pp_mcep = True

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


class Synthesizer:
    def __init__(self, cfg=None):
        if cfg is None:
            self.cfg = Configuration()
        else:
            self.cfg = cfg

    def load(self, checkpoint_path, hparams, gta=False, eal=False, model_name='tacotron_pml', locked_alignments=None,
             logs_enabled=False, checkpoint_eal=None, flag_online=False):
        if locked_alignments is not None:
            eal = True
        if logs_enabled:
            log('Constructing model: %s' % model_name)

        inputs = tf.placeholder(tf.int32, [None, None], 'inputs')
        input_lengths = tf.placeholder(tf.int32, [None], 'input_lengths')
        targets = tf.placeholder(tf.float32, [None, None, hparams.num_mels], 'mel_targets')

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as scope:
            self.model = create_model(model_name, hparams)

            if gta:
                self.model.initialize(inputs, input_lengths, mel_targets=targets, gta=True, logs_enabled=logs_enabled)
            elif eal:
                self.model.initialize(inputs, input_lengths, mel_targets=targets, eal=True, 
                                      locked_alignments=locked_alignments, logs_enabled=logs_enabled)
            else:
                self.model.initialize(inputs, input_lengths, logs_enabled=logs_enabled)

            self.linear_outputs = self.model.linear_outputs

        self.gta, self.eal = gta, eal
        self._hparams = hparams

        self.inputs = inputs
        self.input_lengths = input_lengths
        self.targets = targets

        if logs_enabled:
            log('Loading checkpoint: %s' % checkpoint_path)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        if checkpoint_eal is None:
            log('Loading all vars from checkpoint: %s' % checkpoint_path)
            saver = tf.train.Saver()
            saver.restore(self.session, checkpoint_path)
        else:
            list_var = [var for var in tf.global_variables() if 'Location_Sensitive_Attention' in var.name and 'Adam' not in var.name]
            list_var += [var for var in tf.global_variables() if 'memory_layer' in var.name and 'Adam' not in var.name]
                
            log('Loading all vars from checkpoint: %s' % checkpoint_eal)
            saver_eal = tf.train.Saver()
            saver_eal.restore(self.session, checkpoint_eal)
            
            log('Overwriting attention mechanism weights from checkpoint: %s' % checkpoint_path)
            saver = tf.train.Saver(list_var)
            saver.restore(self.session, checkpoint_path)
            

    def synthesize(self, texts, pml_filenames=None, tgt_filenames=None, to_wav=False, num_workers=4, **kwargs):
        if tgt_filenames is None: tgt_filenames = pml_filenames
        hp = self._hparams

        kwargs.setdefault('pp_mcep', self.cfg.pp_mcep)
        kwargs.setdefault('spec_type', hp.spec_type)

        cleaner_names = [x.strip() for x in hp.cleaners.split(',')]
        seqs = [np.asarray(text_to_sequence(text, cleaner_names), dtype=np.int32) for text in texts]
        input_seqs = self._prepare_inputs(seqs)

        feed_dict = {
            self.model.inputs: np.asarray(input_seqs, dtype=np.int32),
            self.model.input_lengths: np.asarray([len(seq) for seq in seqs], dtype=np.int32)
        }

#         if self.gta:
        if self.gta or self.eal:
            np_targets = [np.load(tgt_filename) for tgt_filename in tgt_filenames]
            prepared_targets = self._prepare_targets(np_targets, hp.outputs_per_step)
            feed_dict[self.targets] = prepared_targets
            assert len(np_targets) == len(texts)

        tgt_features_matrix = self.session.run(self.linear_outputs, feed_dict=feed_dict)

        if to_wav:
            executor = ProcessPoolExecutor(max_workers=num_workers)
            futures = []

            for f in tgt_features_matrix:
                futures.append(executor.submit(partial(_pml_to_wav, f, self.cfg, **kwargs)))

            wavs = [future.result() for future in futures]
            return wavs

        return tgt_features_matrix
    
    def synthesize_check(self, texts, pml_filenames=None, tgt_filenames=None, to_wav=False, num_workers=4, **kwargs):
        if tgt_filenames is None: tgt_filenames = pml_filenames
        hp = self._hparams

        kwargs.setdefault('pp_mcep', self.cfg.pp_mcep)
        kwargs.setdefault('spec_type', hp.spec_type)

        cleaner_names = [x.strip() for x in hp.cleaners.split(',')]
        seqs = [np.asarray(text_to_sequence(text, cleaner_names), dtype=np.int32) for text in texts]
        input_seqs = self._prepare_inputs(seqs)

        feed_dict = {
            self.model.inputs: np.asarray(input_seqs, dtype=np.int32),
            self.model.input_lengths: np.asarray([len(seq) for seq in seqs], dtype=np.int32)
        }

#         if self.gta:
        if self.gta or self.eal:
            np_targets = [np.load(tgt_filename) for tgt_filename in tgt_filenames]
            prepared_targets = self._prepare_targets(np_targets, hp.outputs_per_step)
            feed_dict[self.targets] = prepared_targets
            assert len(np_targets) == len(texts)

        alignments, = self.session.run([self.model.alignments], feed_dict=feed_dict)
#         alignments, pml_intermediates = self.session.run([self.model.alignments, self.model.pml_intermediates], feed_dict=feed_dict)

        if True: # not self.cut_lengths
            max_length = hp.max_iters
            alignments = self.pad_along_axis(alignments, max_length, axis=2)

        if len(alignments) == 1:
            return alignments[0]

        return alignments
#         return alignments, pml_intermediates
    
    def pad_along_axis(self, matrix, target_length, axis=0):
        pad_size = target_length - matrix.shape[axis]
        axis_nb = len(matrix.shape)

        if pad_size < 0:
            return matrix

        npad = [(0, 0) for x in range(axis_nb)]
        npad[axis] = (0, pad_size)
        b = np.pad(matrix, pad_width=npad, mode='constant', constant_values=0)
        return b
    

    def pml_to_wav(self, pml_features, shift=0.005, dftlen=4096, nm_cont=False, verbose_level=0, mean_norm=None,
                   std_norm=None, spec_type='mcep', pp_mcep=False):
        # return the raw wav data
        return _pml_to_wav(pml_features, self.cfg, shift, dftlen, nm_cont, verbose_level,
                           mean_norm, std_norm, spec_type, pp_mcep)

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


def _pml_to_wav(pml_features, cfg, shift=0.005, dftlen=4096, nm_cont=False, verbose_level=0, mean_norm=None,
               std_norm=None, spec_type='mcep', pp_mcep=False, find_endpoint=False, threshold_db=0):
    # get the mean and variance, and denormalise
    if mean_norm is not None and std_norm is not None:
        std_tiled = np.tile(std_norm, (pml_features.shape[0], 1))
        mean_tiled = np.tile(mean_norm, (pml_features.shape[0], 1))
        pml_features = pml_features * std_tiled + mean_tiled

    # f0s is from flf0
    f0 = pml_features[:, cfg.acoustic_start_index['lf0']:
                         cfg.acoustic_start_index['lf0'] + cfg.acoustic_in_dimension_dict['lf0']]

    f0 = np.squeeze(f0)  # remove the extra 1 dimension here
    f0[f0 > 0] = np.exp(f0[f0 > 0])
    ts = shift * np.arange(len(f0))
    f0s = np.vstack((ts, f0)).T

    # spec comes from fmcep or something else fwbnd
    if spec_type == 'mcep':
        mcep = pml_features[:, cfg.acoustic_start_index['mgc']:
                               cfg.acoustic_start_index['mgc'] + cfg.acoustic_in_dimension_dict['mgc']]

        if pp_mcep:
            from lib.merlin import generate_pp
            mcep = generate_pp.mcep_postproc_sptk(mcep, cfg.wav_sr, dftlen=dftlen)

        spec = sp.mcep2spec(mcep, sp.bark_alpha(cfg.wav_sr), dftlen)
    elif spec_type == 'fwbnd':
        compspec = pml_features[:, cfg.acoustic_start_index['mgc']:
                               cfg.acoustic_start_index['mgc'] + cfg.acoustic_in_dimension_dict['mgc']]
        spec = np.exp(sp.fwbnd2linbnd(compspec, cfg.wav_sr, dftlen))

        if pp_mcep:
            from lib.merlin import generate_pp
            mcep = sp.spec2mcep(spec * cfg.wav_sr, sp.bark_alpha(cfg.wav_sr), 256)
            mcep_pp = generate_pp.mcep_postproc_sptk(mcep, cfg.wav_sr, dftlen=dftlen)
            spec = sp.mcep2spec(mcep_pp, sp.bark_alpha(cfg.wav_sr), dftlen=dftlen) / cfg.wav_sr

    # NM comes from bap
    fwnm = pml_features[:, cfg.acoustic_start_index['bap']:
                           cfg.acoustic_start_index['bap'] + cfg.acoustic_in_dimension_dict['bap']]

    nm = sp.fwbnd2linbnd(fwnm, cfg.wav_sr, dftlen)

    # use standard PML vocoder
    wav = synthesize(cfg.wav_sr, f0s, spec, NM=nm, nm_cont=nm_cont, verbose=verbose_level)

    # clip the wav to the endpoint if required
    if find_endpoint:
        wav = wav[:audio.find_endpoint(wav, threshold_db=threshold_db)]

    # return the raw wav data
    return wav
