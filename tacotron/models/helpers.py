import numpy as np
import tensorflow as tf
from tensorflow.contrib.seq2seq import Helper
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import bernoulli


# Adapted from tf.contrib.seq2seq.GreedyEmbeddingHelper
class TacoTestHelper(Helper):
    def __init__(self, batch_size, output_dim, r):
        with tf.name_scope("TacoTestHelper"):
            self._batch_size = batch_size
            self._output_dim = output_dim
            self._end_token = tf.tile([0.0], [output_dim * r])

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return np.int32

    def initialize(self, name=None):
        return (tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim))

    def sample(self, time, outputs, state, name=None):
        return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        '''Stop on EOS. Otherwise, pass the last output as the next input and pass through state.'''
        with tf.name_scope(name, "TacoTestHelper"):
            finished = tf.reduce_all(tf.equal(outputs, self._end_token), axis=1)
            # Feed last output frame as next input. outputs is [N, output_dim * r]
            next_inputs = outputs[:, -self._output_dim:]
            return finished, next_inputs, state


class TacoTrainingHelper(Helper):
    def __init__(self, inputs, targets, output_dim, r):
        # inputs is [N, T_in], targets is [N, T_out, D]
        with tf.name_scope("TacoTrainingHelper"):
            self._batch_size = tf.shape(inputs)[0]
            self._output_dim = output_dim
            self._end_token = tf.tile([0.0], [output_dim * r])

            # Feed every r-th target frame as input
            self._targets = targets[:, r - 1::r, :]

            # Use full length for every target because we don't want to mask the padding frames
            num_steps = tf.shape(self._targets)[1]
            self._lengths = tf.tile([num_steps], [self._batch_size])

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return np.int32

    def initialize(self, name=None):
        return (tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim))

    def sample(self, time, outputs, state, name=None):
        return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        with tf.name_scope(name, "TacoTrainingHelper"):
            finished = (time + 1 >= self._lengths)
            next_inputs = self._targets[:, time, :]
            return finished, next_inputs, state


class TacoScheduledOutputTrainingHelper(TacoTrainingHelper):
    def __init__(self, inputs, targets, output_dim, r, sampling_probability=0.5, seed=None):
        # inputs is [N, T_in], targets is [N, T_out, D]
        with tf.name_scope('TacoScheduledOutputTrainingHelper'):
            self._sampling_probability = tf.convert_to_tensor(
                sampling_probability, name="sampling_probability")

            if self._sampling_probability.get_shape().ndims not in (0, 1):
                raise ValueError(
                    "Parameter sampling_probability must be either a scalar or a vector. "
                    "Saw shape: %s" % (self._sampling_probability.get_shape()))

            self._seed = seed

            super(TacoScheduledOutputTrainingHelper, self).__init__(
                inputs=inputs,
                targets=targets,
                output_dim=output_dim,
                r=r,
            )

    def sample(self, time, outputs, state, name=None):
        with tf.name_scope(name, "TacoScheduledOutputTrainingHelperSample",
                           [time, outputs, state]):
            sampler = bernoulli.Bernoulli(probs=self._sampling_probability)
            return sampler.sample(sample_shape=self.batch_size, seed=self._seed)

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        with tf.name_scope(name, "TacoScheduledOutputTrainingHelperNextInputs",
                           [time, outputs, state, sample_ids]):
            (finished, base_next_inputs, state) = (
                super(TacoScheduledOutputTrainingHelper, self).next_inputs(
                    time=time,
                    outputs=outputs,
                    state=state,
                    sample_ids=sample_ids,
                    name=name,
                )
            )

            sample_ids = math_ops.cast(sample_ids, tf.dtypes.bool)

            def maybe_sample():
                """Perform scheduled sampling."""
                return array_ops.where(sample_ids, outputs, base_next_inputs)

            all_finished = math_ops.reduce_all(finished)
            no_samples = math_ops.logical_not(math_ops.reduce_any(sample_ids))

            next_inputs = control_flow_ops.cond(
                math_ops.logical_or(all_finished, no_samples),
                lambda: base_next_inputs, maybe_sample
            )

            return finished, next_inputs, state


def _go_frames(batch_size, output_dim):
    '''Returns all-zero <GO> frames for a given batch size and output dimension'''
    return tf.tile([[0.0]], [batch_size, output_dim])
