import numpy as np
import tensorflow as tf
from tacotron.models import create_model
from tacotron.utils.text import text_to_sequence

_pad = 0


class AlignmentSynthesizer:
    def load(self, checkpoint, hparams, gta=False, model_name='tacotron_pml', locked_alignments=None, cut_lengths=True):
        print('Constructing model: %s' % model_name)
        inputs = tf.placeholder(tf.int32, [None, None], 'inputs')
        input_lengths = tf.placeholder(tf.int32, [None], 'input_lengths')
        targets = tf.placeholder(tf.float32, [None, None, hparams.pml_dimension], 'pml_targets')

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as scope:
            self.model = create_model(model_name, hparams)

            if gta:
                self.model.initialize(inputs, input_lengths, pml_targets=targets,
                                      gta=gta, locked_alignments=locked_alignments, cut_lengths=cut_lengths)
            else:
                self.model.initialize(inputs, input_lengths, locked_alignments=locked_alignments)

            self.alignments = self.model.alignments

        self.gta = gta
        self._hparams = hparams
        self.targets = targets

        print('Loading checkpoint: %s' % checkpoint)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint)

    def synthesize(self, texts, is_sequence=False, pml_filenames=None):
        hp = self._hparams
        cleaner_names = [x.strip() for x in hp.cleaners.split(',')]

        if isinstance(texts, str):
            seqs = [np.asarray(text_to_sequence(texts, cleaner_names), dtype=np.int32)]
        elif is_sequence:
            seqs = [np.asarray(texts, dtype=np.int32)]
        else:
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

        alignments, = self.session.run([self.alignments], feed_dict=feed_dict)

        if len(alignments) == 1:
            return alignments[0]

        return alignments

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
