import numpy as np
import tensorflow as tf
from hparams import hparams
from models import create_model
from text import text_to_sequence


class AlignmentSynthesizer:
  def load(self, model_name='tacotron_pml'):
    print('Constructing model: %s' % model_name)
    inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
    input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')

    with tf.variable_scope('model', reuse=True) as scope:
      self.model = create_model(model_name, hparams)
      self.model.initialize(inputs, input_lengths)
      self.alignment = self.model.alignments[0]

    self.session = tf.Session()
    # Save the variables
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
