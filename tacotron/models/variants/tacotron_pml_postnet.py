import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper
from tensorflow.contrib.seq2seq import BasicDecoder, BahdanauAttention, AttentionWrapper
from tacotron.utils import symbols
from infolog import log
from tacotron.models.helpers import TacoTestHelper, TacoTrainingHelper
from tacotron.models.modules import encoder_cbhg, prenet, postnet
from tacotron.models.rnn_wrappers import DecoderPrenetWrapper, ConcatOutputAndAttentionWrapper



class TacotronPMLPostnet():
  def __init__(self, hparams):
    self._hparams = hparams


  def initialize(self, inputs, input_lengths, mel_targets=None, linear_targets=None, pml_targets=None):
    '''Initializes the model for inference.

    Sets "pml_outputs", and "alignments" fields.

    Args:
      inputs: int32 Tensor with shape [N, T_in] where N is batch size, T_in is number of
        steps in the input time series, and values are character IDs
      input_lengths: int32 Tensor with shape [N] where N is batch size and values are the lengths
        of each sequence in inputs.
      mel_targets: float32 Tensor with shape [N, T_out, M] where N is batch size, T_out is number
        of steps in the output time series, M is num_mels, and values are entries in the mel
        spectrogram. Only needed for training.
      linear_targets: float32 Tensor with shape [N, T_out, F] where N is batch_size, T_out is number
        of steps in the output time series, F is num_freq, and values are entries in the linear
        spectrogram. Only needed for training.
      pml_targets: float32 Tensor with shape [N, T_out, P] where N is batch_size, T_out is number of
        steps in the PML vocoder features trajectories, P is pml_dimension, and values are PML vocoder
        features. Only needed for training.
    '''
    with tf.variable_scope('inference') as scope:
      is_training = pml_targets is not None
      batch_size = tf.shape(inputs)[0]
      hp = self._hparams

      # Embeddings
      embedding_table = tf.get_variable(
        'embedding', [len(symbols), hp.embed_depth], dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.5))
      embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)          # [N, T_in, embed_depth=256]

      # Encoder
      prenet_outputs = prenet(embedded_inputs, is_training, hp.prenet_depths)    # [N, T_in, prenet_depths[-1]=128]
      encoder_outputs = encoder_cbhg(prenet_outputs, input_lengths, is_training, # [N, T_in, encoder_depth=256]
                                     hp.encoder_depth)

      # Attention
      attention_cell = AttentionWrapper(
        GRUCell(hp.attention_depth),
        BahdanauAttention(hp.attention_depth, encoder_outputs),
        alignment_history=True,
        output_attention=False)                                                  # [N, T_in, attention_depth=256]
      
      # Apply prenet before concatenation in AttentionWrapper.
      attention_cell = DecoderPrenetWrapper(attention_cell, is_training, hp.prenet_depths)

      # Concatenate attention context vector and RNN cell output into a 2*attention_depth=512D vector.
      concat_cell = ConcatOutputAndAttentionWrapper(attention_cell)              # [N, T_in, 2*attention_depth=512]

      # Decoder (layers specified bottom to top):
      decoder_cell = MultiRNNCell([
          OutputProjectionWrapper(concat_cell, hp.decoder_depth),
          ResidualWrapper(GRUCell(hp.decoder_depth)),
          ResidualWrapper(GRUCell(hp.decoder_depth))
        ], state_is_tuple=True)                                                  # [N, T_in, decoder_depth=256]

      # Project onto r PML feature vectors (predict r outputs at each RNN step):
      output_cell = OutputProjectionWrapper(decoder_cell, hp.pml_dimension * hp.outputs_per_step)
      decoder_init_state = output_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

      if is_training:
        helper = TacoTrainingHelper(inputs, pml_targets, hp.pml_dimension, hp.outputs_per_step)
      else:
        helper = TacoTestHelper(batch_size, hp.pml_dimension, hp.outputs_per_step)

      (multi_decoder_outputs, _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
        BasicDecoder(output_cell, helper, decoder_init_state),
        maximum_iterations=hp.max_iters)                                         # [N, T_out/r, P*r]

      # Reshape outputs to be one output per entry
      decoder_outputs = tf.reshape(multi_decoder_outputs, [batch_size, -1, hp.pml_dimension])   # [N, T_out, P]

      # Postnet: predicts a residual
      postnet_outputs = postnet(
        decoder_outputs,
        layers=hp.postnet_conv_layers,
        conv_width=hp.postnet_conv_width,
        channels=hp.postnet_conv_channels,
        is_training=is_training)
      
      pml_outputs = decoder_outputs + postnet_outputs

      # Grab alignments from the final decoder state:
      alignments = tf.transpose(final_decoder_state[0].alignment_history.stack(), [1, 2, 0])

      self.inputs = inputs
      self.input_lengths = input_lengths
      self.pml_outputs = pml_outputs
      self.alignments = alignments
      self.pml_targets = pml_targets
      log('Initialized Tacotron model. Dimensions: ')
      log('  embedding:               %d' % embedded_inputs.shape[-1])
      log('  prenet out:              %d' % prenet_outputs.shape[-1])
      log('  encoder out:             %d' % encoder_outputs.shape[-1])
      log('  attention out:           %d' % attention_cell.output_size)
      log('  concat attn & out:       %d' % concat_cell.output_size)
      log('  decoder cell out:        %d' % decoder_cell.output_size)
      log('  decoder out (%d frames):  %d' % (hp.outputs_per_step, multi_decoder_outputs.shape[-1]))
      log('  decoder out (1 frame):   %d' % pml_outputs.shape[-1])


  def add_loss(self):
    '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
    with tf.variable_scope('loss') as scope:
      l1 = tf.abs(self.pml_targets - self.pml_outputs)
      self.pml_loss = tf.reduce_mean(l1)
      self.loss = self.pml_loss


  def add_optimizer(self, global_step):
    '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.

    Args:
      global_step: int32 scalar Tensor representing current global step in training
    '''
    with tf.variable_scope('optimizer') as scope:
      hp = self._hparams
      if hp.decay_learning_rate:
        self.learning_rate = _learning_rate_decay(hp.initial_learning_rate, global_step)
      else:
        self.learning_rate = tf.convert_to_tensor(hp.initial_learning_rate)
      optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.adam_beta1, hp.adam_beta2)
      gradients, variables = zip(*optimizer.compute_gradients(self.loss))
      self.gradients = gradients
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

      # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
      # https://github.com/tensorflow/tensorflow/issues/1122
      with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
          global_step=global_step)


def _learning_rate_decay(init_lr, global_step):
  # Noam scheme from tensor2tensor:
  warmup_steps = 4000.0
  step = tf.cast(global_step + 1, dtype=tf.float32)
  return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)
