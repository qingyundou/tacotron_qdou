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

  def synthesize(self, text_or_sequence, is_sequence=False):
    if is_sequence:
      seq = text_or_sequence
    else:
      cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
      seq = text_to_sequence(text_or_sequence, cleaner_names)

    feed_dict = {
      self.model.inputs: [np.asarray(seq, dtype=np.int32)],
      self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32)
    }
    
    alignment = self.session.run(self.alignment, feed_dict=feed_dict)
    return alignment

def main():
  import argparse
  import matplotlib.pyplot as plt
  import matplotlib.patches as mpatches
  from matplotlib import cm
  from matplotlib.colors import ListedColormap
  from PIL import Image

  parser = argparse.ArgumentParser()
  parser.add_argument('--taco_pml_checkpoint', default='/home/josh/tacotron/remote-logs/tacotron-lj-pml-500k/model.ckpt-352000')
  parser.add_argument('--taco_checkpoint', default='/home/josh/tacotron/remote-logs/tacotron-lj-500k/model.ckpt-195000')
  args = parser.parse_args()

  synth = AlignmentSynthesizer()
  synth.load(args.taco_pml_checkpoint, model_name='tacotron_pml')
  fixed_sentence = 'and district attorney henry m. wade both testified that they saw it later that day.'
  first_alignment = synth.synthesize(fixed_sentence)

  cutoff = 220

  # Get the colormap colors
  cmap = cm.cool
  cool = cmap(np.arange(cmap.N))
  # Set alpha
  cool[:,-1] = np.linspace(0, 1, cmap.N)
  # Create new colormap
  cool = ListedColormap(cool)
  # generate the image
  first_image = Image.fromarray(np.uint8(cool(first_alignment)*255))
  height, width = first_alignment.shape
  first_image = np.array(first_image)[:, :cutoff]

  # reset the graph after the first synthesise call
  tf.reset_default_graph()

  synth.load(args.taco_checkpoint, model_name='tacotron')
  fixed_sentence = 'and district attorney henry m. wade both testified that they saw it later that day.'
  second_alignment = synth.synthesize(fixed_sentence)

  # Get the colormap colors
  cmap = cm.Wistia
  wistia = cmap(np.arange(cmap.N))
  # Set alpha
  wistia[:,-1] = np.linspace(0, 1, cmap.N)
  # Create new colormap
  wistia = ListedColormap(wistia)
  # generate the image
  second_image = Image.fromarray(np.uint8(wistia(second_alignment)*255))
  height, width = second_alignment.shape
  second_image = second_image.resize((round(width * 12.5 / 5), height))
  second_image = np.array(second_image)[:, :cutoff]

  fig, ax = plt.subplots()

  first_im = ax.imshow(
    first_image,
    aspect='auto',
    origin='lower',
    interpolation='none')

  second_im = ax.imshow(
    second_image,
    aspect='auto',
    origin='lower',
    interpolation='none')

  taco_pml_patch = mpatches.Patch(color=cool(0.8), label='Tacotron PML Alignment')
  taco_patch = mpatches.Patch(color=wistia(0.8), label='Tacotron Alignment')
  ax.legend(handles=[taco_pml_patch, taco_patch], loc='upper left')

  xstep = 50
  outputs_per_step = 5
  frame_shift = 0.005
  plt.xticks(np.arange(0, 250, step=xstep), np.arange(0, 250 * frame_shift * outputs_per_step, step=frame_shift * outputs_per_step * xstep))
  xlabel = 'Decoder Time (s)'
  plt.xlabel(xlabel)
  plt.ylabel('Encoder Step (character index)')
  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
  main()
