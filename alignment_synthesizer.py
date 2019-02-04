import numpy as np
import tensorflow as tf
from hparams import hparams
from models import create_model
from text import text_to_sequence
from util.rename import rename_scope


class AlignmentSynthesizer:
  def load(self, checkpoint_path, model_name='tacotron_pml', original_scope_name='model', scope_name='alignment_model'):
    print('Constructing model: %s' % model_name)
    inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
    input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')

    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
      self.model = create_model(model_name, hparams)
      self.model.initialize(inputs, input_lengths)
      self.alignment = self.model.alignments[0]

    print('Loading checkpoint: %s' % checkpoint_path)
    self.session = tf.Session()
    rename_scope(self.session, checkpoint_path, original_scope_name, scope_name)
    self.session.run(tf.global_variables_initializer())

  def synthesize(self, text):
    cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
    seq = text_to_sequence(text, cleaner_names)

    feed_dict = {
      self.model.inputs: [np.asarray(seq, dtype=np.int32)],
      self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32)
    }
    
    alignment = self.session.run(self.alignment, feed_dict=feed_dict)
    return alignment
