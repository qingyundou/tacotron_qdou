import numpy as np
import tensorflow as tf
from hparams import hparams
from tacotron.models import create_model
from tacotron.utils.text import text_to_sequence


class AlignmentSynthesizer:
    def load(self, checkpoint, model_name='tacotron_pml', forced_alignments=None):
        print('Constructing model: %s' % model_name)
        inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
        input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as scope:
            self.model = create_model(model_name, hparams, forced_alignments)
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

        alignment, = self.session.run([self.alignment], feed_dict=feed_dict)
        return alignment
