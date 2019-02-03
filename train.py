import argparse
from datetime import datetime
import math
import numpy as np
import os
import subprocess
import time
import tensorflow as tf
import traceback

from alignment_synthesizer import AlignmentSynthesizer
from datasets.datafeeder import DataFeeder
from hparams import hparams, hparams_debug_string
from models import create_model
from pml_synthesizer import PMLSynthesizer, cfg
import sigproc as sp
from text import sequence_to_text, text_to_sequence
from util import audio, infolog, plot, ValueWindow
log = infolog.log


def get_git_commit():
  subprocess.check_output(['git', 'diff-index', '--quiet', 'HEAD'])   # Verify client is clean
  commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()[:10]
  log('Git commit: %s' % commit)
  return commit


def add_stats(model):
  with tf.variable_scope('stats') as scope:
    if hasattr(model, 'linear_targets'):
      tf.summary.histogram('linear_outputs', model.linear_outputs)
      tf.summary.histogram('linear_targets', model.linear_targets)
      tf.summary.scalar('loss_linear', model.linear_loss)

    if hasattr(model, 'mel_targets'):
      tf.summary.histogram('mel_outputs', model.mel_outputs)
      tf.summary.histogram('mel_targets', model.mel_targets)
      tf.summary.scalar('loss_mel', model.mel_loss)

    if hasattr(model, 'pml_targets'):
      tf.summary.histogram('pml_outputs', model.pml_outputs)
      tf.summary.histogram('pml_targets', model.pml_targets)
      tf.summary.scalar('loss_pml', model.pml_loss)

    tf.summary.scalar('learning_rate', model.learning_rate)
    tf.summary.scalar('loss', model.loss)
    gradient_norms = [tf.norm(grad) for grad in model.gradients]
    tf.summary.histogram('gradient_norm', gradient_norms)
    tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms))
    return tf.summary.merge_all()


def time_string():
  return datetime.now().strftime('%Y-%m-%d %H:%M')


def train(log_dir, args):
  commit = get_git_commit() if args.git else 'None'
  checkpoint_path = os.path.join(log_dir, 'model.ckpt')
  input_path = os.path.join(args.base_dir, args.input)
  log('Checkpoint path: %s' % checkpoint_path)
  log('Loading training data from: %s' % input_path)
  log('Using model: %s' % args.model)
  log(hparams_debug_string())

  # Set up DataFeeder:
  coord = tf.train.Coordinator()
  with tf.variable_scope('datafeeder') as scope:
    feeder = DataFeeder(coord, input_path, hparams)

  # Set up model:
  global_step = tf.Variable(0, name='global_step', trainable=False)
  with tf.variable_scope('model') as scope:
    model = create_model(args.model, hparams)
    model.initialize(feeder.inputs, feeder.input_lengths, feeder.mel_targets, feeder.linear_targets, feeder.pml_targets)
    model.add_loss()
    model.add_optimizer(global_step)
    stats = add_stats(model)

  # Bookkeeping:
  step = 0
  time_window = ValueWindow(100)
  loss_window = ValueWindow(100)
  saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

  # Set up fixed alignment synthesizer
  alignment_synth = AlignmentSynthesizer()

  # Set up text for synthesis
  fixed_sentence = 'Scientists at the CERN laboratory say they have discovered a new particle.'

  # Train!
  with tf.Session() as sess:
    try:
      summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
      sess.run(tf.global_variables_initializer())

      if args.restore_step:
        # Restore from a checkpoint if the user requested it.
        restore_path = '%s-%d' % (checkpoint_path, args.restore_step)
        saver.restore(sess, restore_path)
        log('Resuming from checkpoint: %s at commit: %s' % (restore_path, commit), slack=True)
      else:
        log('Starting new training run at commit: %s' % commit, slack=True)

      feeder.start_in_session(sess)
      step = 0 # initialise step variable so can use in while condition

      while not coord.should_stop() and step <= args.num_steps:
        start_time = time.time()
        step, loss, opt = sess.run([global_step, model.loss, model.optimize])
        time_window.append(time.time() - start_time)
        loss_window.append(loss)
        message = 'Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f]' % (
          step, time_window.average, loss, loss_window.average)
        log(message, slack=(step % args.checkpoint_interval == 0))

        if loss > 100 or math.isnan(loss):
          log('Loss exploded to %.05f at step %d!' % (loss, step), slack=True)
          raise Exception('Loss Exploded')

        if step % args.summary_interval == 0:
          log('Writing summary at step: %d' % step)
          summary_writer.add_summary(sess.run(stats), step)

        if step % args.checkpoint_interval == 0:
          log('Saving checkpoint to: %s-%d' % (checkpoint_path, step))
          saver.save(sess, checkpoint_path, global_step=step)
          log('Saving audio and alignment...')
          input_seq = sess.run(model.inputs[0])
          summary_elements = []

          # if the model has linear spectrogram features, use them to synthesize audio
          if hasattr(model, 'linear_targets'):
            target_spectrogram, spectrogram = sess.run([model.linear_targets[0], model.linear_outputs[0]])
            output_waveform = audio.inv_spectrogram(spectrogram.T)
            target_waveform = audio.inv_spectrogram(target_spectrogram.T)
            audio.save_wav(output_waveform, os.path.join(log_dir, 'step-%d-audio.wav' % step))
          # otherwise, synthesize audio from PML vocoder features
          elif hasattr(model, 'pml_targets'):
            target_pml_features, pml_features = sess.run([model.pml_targets[0], model.pml_outputs[0]])
            synth = PMLSynthesizer()
            output_waveform = synth.pml_to_wav(pml_features)
            target_waveform = synth.pml_to_wav(target_pml_features)
            sp.wavwrite(os.path.join(log_dir, 'step-%d-audio.wav' % step), output_waveform, cfg.wav_sr, norm_max_ifneeded=True)

          # we need to adjust the output and target waveforms so the values lie in the interval [-1.0, 1.0]
          output_waveform *= 1 / np.max(np.abs(output_waveform))
          target_waveform *= 1 / np.max(np.abs(target_waveform))

          summary_elements.append(
            tf.summary.audio('ideal-%d' % step, np.expand_dims(target_waveform, 0), hparams.sample_rate),
          )

          summary_elements.append(
            tf.summary.audio('sample-%d' % step, np.expand_dims(output_waveform, 0), hparams.sample_rate),
          )

          # get the alignment for the top sentence in the batch
          random_alignment = sess.run(model.alignments[0])
          random_attention_plot = plot.plot_alignment(random_alignment, os.path.join(log_dir, 'step-%d-random-align.png' % step),
            info='%s, %s, %s, step=%d, loss=%.5f' % (args.model, commit, time_string(), step, loss))

          summary_elements.append(
            tf.summary.image('attention-%d' % step, random_attention_plot),
          )

          # also process the alignment for a fixed sentence for comparison
          alignment_synth.load('%s-%d' % (checkpoint_path, step), model_name=args.model)
          fixed_alignment = alignment_synth.synthesize(fixed_sentence)
          fixed_attention_plot = plot.plot_alignment(fixed_alignment, os.path.join(log_dir, 'step-%d-fixed-align.png' % step),
            info='%s, %s, %s, step=%d, loss=%.5f' % (args.model, commit, time_string(), step, loss))

          summary_elements.append(
            tf.summary.image('fixed-attention-%d' % step, fixed_attention_plot),
          )

          # save the audio and alignment to tensorboard (audio sample rate is hyperparameter)
          merged = sess.run(tf.summary.merge(summary_elements))

          summary_writer.add_summary(merged, step)

          log('Input: %s' % sequence_to_text(input_seq))

    except Exception as e:
      log('Exiting due to exception: %s' % e, slack=True)
      traceback.print_exc()
      coord.request_stop(e)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default=os.path.expanduser('~/tacotron'))
  parser.add_argument('--log_dir', default=os.path.expanduser('~/tacotron'))
  parser.add_argument('--input', default='training/train.txt')
  parser.add_argument('--model', default='tacotron')
  parser.add_argument('--name', help='Name of the run. Used for logging. Defaults to model name.')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--restore_step', type=int, help='Global step to restore from checkpoint.')
  parser.add_argument('--summary_interval', type=int, default=100,
    help='Steps between running summary ops.')
  parser.add_argument('--checkpoint_interval', type=int, default=1000,
    help='Steps between writing checkpoints.')
  parser.add_argument('--slack_url', help='Slack webhook URL to get periodic reports.')
  parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
  parser.add_argument('--git', action='store_true', help='If set, verify that the client is clean.')
  parser.add_argument('--num_steps', type=int, default=100000, help='Maximum number of steps to run training for.')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
  run_name = args.name or args.model
  log_dir = os.path.join(args.log_dir, 'logs-%s' % run_name)
  os.makedirs(log_dir, exist_ok=True)
  infolog.init(os.path.join(log_dir, 'train.log'), run_name, args.slack_url)
  hparams.parse(args.hparams)
  train(log_dir, args)


if __name__ == '__main__':
  main()
