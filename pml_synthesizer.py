import io
import numpy as np
import tensorflow as tf
from hparams import hparams
from librosa import effects
from models import create_model
from text import text_to_sequence
from util import audio
import sigproc as sp


# simplified port of the configuration from Merlin proper
class Configuration(object):
  def __init__(self):
    self.acoustic_feature_type = 'PML'
    self.acoustic_features = ['mgc', 'lf0', 'bap']
    self.acoustic_in_dimension_dict = {'mgc': 60,  'lf0': 1, 'bap': 25}
    self.acoustic_out_dimension_dict = {'mgc': 60, 'lf0': 1, 'bap': 25}

    self.acoustic_start_index = {
        'mgc': 0,
        'lf0': self.acoustic_out_dimension_dict['mgc'],
        'bap': self.acoustic_out_dimension_dict['mgc'] + self.acoustic_out_dimension_dict['lf0']
    }

    self.acoustic_file_ext_dict = {
        'mgc': '.mcep',  'lf0': '.lf0', 'bap': '.bndnm'}

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


class PMLSynthesizer:
  def load(self, checkpoint_path, model_name='tacotron_pml'):
    print('Constructing model: %s' % model_name)
    inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
    input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')

    with tf.variable_scope('model') as scope:
      self.model = create_model(model_name, hparams)
      self.model.initialize(inputs, input_lengths)
      self.pml_output = self.model.pml_outputs[0]

    print('Loading checkpoint: %s' % checkpoint_path)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(self.session, checkpoint_path)

  def synthesize(self, text):
    cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
    seq = text_to_sequence(text, cleaner_names)

    feed_dict = {
      self.model.inputs: [np.asarray(seq, dtype=np.int32)],
      self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32)
    }
    
    pml_features = self.session.run(self.pml_output, feed_dict=feed_dict)
    wav = self.pml_to_wav(pml_features)
    wav = wav[:audio.find_endpoint(wav, threshold_db=0)]
    return wav

  def pml_to_wav(self, pml_features, shift=0.005, dftlen=4096, nm_cont=False, verbose_level=0):
    from pulsemodel.synthesis import synthesize

    # f0s is from flf0
    f0 = pml_features[:, cfg.acoustic_start_index['lf0']:
                         cfg.acoustic_start_index['lf0'] + cfg.acoustic_in_dimension_dict['lf0']]

    f0 = np.squeeze(f0) # remove the extra 1 dimension here
    f0[f0>0] = np.exp(f0[f0>0])
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
