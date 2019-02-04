import numpy as np
import tensorflow as tf
from hparams import hparams
from models import create_model
from text import text_to_sequence


class AlignmentSynthesizer:
  def load(self, checkpoint, model_name='tacotron_pml'):
    print('Constructing model: %s' % model_name)
    inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
    input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')

    with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as scope:
      self.model = create_model(model_name, hparams)
      self.model.initialize(inputs, input_lengths)
      self.alignment = self.model.alignments[0]

    print('Loading checkpoint: %s' % checkpoint)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(self.session, checkpoint)

  def synthesize(self, text):
    cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
    seq = text_to_sequence(text, cleaner_names)

    feed_dict = {
      self.model.inputs: [np.asarray(seq, dtype=np.int32)],
      self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32)
    }
    
    alignment = self.session.run(self.alignment, feed_dict=feed_dict)
    return alignment

def main():
  import argparse
  import matplotlib.pyplot as plt

  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint')
  parser.add_argument('--model', default='tacotron')
  args = parser.parse_args()

  synth = AlignmentSynthesizer()
  synth.load(args.checkpoint, model_name=args.model)
  fixed_sentence = 'and district attorney henry m. wade both testified that they saw it later that day.'
  alignment = synth.synthesize(fixed_sentence)

  fig, ax = plt.subplots()
  im = ax.imshow(
    alignment,
    aspect='auto',
    cmap='hot',
    origin='lower',
    interpolation='none')
  fig.colorbar(im, ax=ax)
  xlabel = 'Decoder Step'
  plt.xlabel(xlabel)
  plt.ylabel('Encoder Step')
  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
  main()
