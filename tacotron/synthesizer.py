import numpy as np
import tensorflow as tf
from hparams import hparams
from tacotron.models import create_model
from tacotron.utils.text import text_to_sequence
from util import audio
from tqdm import tqdm


_pad = 0


class Synthesizer:
    def load(self, checkpoint_path, model_name='tacotron'):
        print('Constructing model: %s' % model_name)
        inputs = tf.placeholder(tf.int32, [None, None], 'inputs')
        input_lengths = tf.placeholder(tf.int32, [None], 'input_lengths')

        with tf.variable_scope('model') as scope:
            self.model = create_model(model_name, hparams)
            self.model.initialize(inputs, input_lengths)
            self.linear_outputs = self.model.linear_outputs
            self.wav_outputs = tf.map_fn(audio.inv_spectrogram_tensorflow, self.model.linear_outputs)
            self.wav_output = audio.inv_spectrogram_tensorflow(self.model.linear_outputs[0])
            self.linear_output = self.model.linear_outputs[0]

        print('Loading checkpoint: %s' % checkpoint_path)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint_path)

    def synthesize(self, texts, to_wav=False, is_sequence=False):
        cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]

        if isinstance(texts, str):
            seqs = [np.asarray(text_to_sequence(texts, cleaner_names), dtype=np.int32)]
        elif is_sequence:
            seqs = [np.asarray(texts, dtype=np.int32)]
        else:
            seqs = [np.asarray(text_to_sequence(text, cleaner_names), dtype=np.int32) for text in texts]

        input_seqs = self._prepare_inputs(seqs)
        print('Prepared Inputs')

        feed_dict = {
            self.model.inputs: np.asarray(input_seqs, dtype=np.int32),
            self.model.input_lengths: np.asarray([len(seq) for seq in seqs], dtype=np.int32)
        }

        if to_wav:
            wav_outputs = self.session.run(self.wav_outputs, feed_dict=feed_dict)
            wavs = []

            for wav in tqdm(wav_outputs):
                wav = audio.inv_preemphasis(wav)
                wav = wav[:audio.find_endpoint(wav)]
                wavs.append(wav)

            return wavs
        else:
            return self.session.run(self.linear_output, feed_dict=feed_dict)

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
