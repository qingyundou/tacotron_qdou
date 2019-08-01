import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper
from tensorflow.contrib.seq2seq import BasicDecoder
from tacotron.utils.symbols import symbols
from infolog import log
from .attention import LocationSensitiveAttention
from .helpers import TacoTestHelper, TacoTrainingHelper, TacoTrainingHelper_EAL, TacoScheduledOutputTrainingHelper
from .lockable_attention_wrapper import LockableAttentionWrapper
from .modules import encoder_cbhg, post_cbhg, prenet
from .rnn_wrappers import DecoderPrenetWrapper, ConcatOutputAndAttentionWrapper


class TacotronPMLExtendedLocSens:
    def __init__(self, hparams):
        self._hparams = hparams

    def initialize(self, inputs, input_lengths, mel_targets=None, linear_targets=None, pml_targets=None,
                   is_training=False, gta=False, eal=False, locked_alignments=None, logs_enabled=True, flag_trainAlign=False):
        '''Initializes the model for inference.

        Sets "mel_outputs", "linear_outputs", and "alignments" fields.

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
          is_training: boolean flag that is set to True during training
          gta: boolean flag that is set to True when ground truth alignment is required
          locked_alignments: when explicit attention alignment is required, the locked alignments are passed in this
            parameter and the attention alignments are locked to these values
          logs_enabled: boolean flag that defaults to True, if False no construction logs output
        '''
        # fix the alignments shape to (batch_size, encoder_steps, decoder_steps) if not already including
        # batch dimension
        locked_alignments_ = locked_alignments
        self.flag_trainAlign = flag_trainAlign
        
        if locked_alignments_ is not None:
            if is_training and eal:
                pass
            elif np.ndim(locked_alignments_) < 3:
                locked_alignments_ = np.expand_dims(locked_alignments_, 0)

        with tf.variable_scope('inference') as scope:
            batch_size = tf.shape(inputs)[0]
            hp = self._hparams

            # Embeddings
            embedding_table = tf.get_variable(
                'embedding', [len(symbols), hp.embed_depth], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.5))
            embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)  # [N, T_in, embed_depth=256]

            # Encoder
            prenet_outputs = prenet(embedded_inputs, is_training, hp.prenet_depths)  # [N, T_in, prenet_depths[-1]=128]
            encoder_outputs = encoder_cbhg(prenet_outputs, input_lengths, is_training,  # [N, T_in, encoder_depth=256]
                                           hp.encoder_depth)

            # Attention
            attention_cell = LockableAttentionWrapper(
                GRUCell(hp.attention_depth),
                LocationSensitiveAttention(hp.attention_depth, encoder_outputs),
                alignment_history=True,
                locked_alignments=locked_alignments_,
                output_attention=False,
                name='attention_wrapper',
                flag_trainAlign=self.flag_trainAlign)  # [N, T_in, attention_depth=256]

            # Apply prenet before concatenation in AttentionWrapper.
            prenet_cell = DecoderPrenetWrapper(attention_cell, is_training, hp.prenet_depths)

            # Concatenate attention context vector and RNN cell output into a 2*attention_depth=512D vector.
            concat_cell = ConcatOutputAndAttentionWrapper(prenet_cell)  # [N, T_in, 2*attention_depth=512]

            # Decoder (layers specified bottom to top):
            decoder_cell = MultiRNNCell([
                OutputProjectionWrapper(concat_cell, hp.decoder_depth),
                ResidualWrapper(GRUCell(hp.decoder_depth)),
                ResidualWrapper(GRUCell(hp.decoder_depth))
            ], state_is_tuple=True)  # [N, T_in, decoder_depth=256]

            # Project onto r PML feature vectors (predict r outputs at each RNN step):
            output_cell = OutputProjectionWrapper(decoder_cell, hp.pml_dimension * hp.outputs_per_step)
            decoder_init_state = output_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

            if is_training:
                if gta:
                    helper = TacoTrainingHelper(inputs, pml_targets, hp.pml_dimension, hp.outputs_per_step)
                elif eal:
                    helper = TacoTrainingHelper_EAL(inputs, pml_targets, hp.pml_dimension, hp.outputs_per_step)
                elif hp.scheduled_sampling:
                    helper = TacoScheduledOutputTrainingHelper(inputs, pml_targets, hp.pml_dimension, hp.outputs_per_step,
                                                               hp.scheduled_sampling_probability)
                else:
                    log('For training, one of these should be true: gta, eal, hp.scheduled_sampling')
            else:
                if gta:
                    helper = TacoTrainingHelper(inputs, pml_targets, hp.pml_dimension, hp.outputs_per_step)
                elif eal:
                    helper = TacoTrainingHelper_EAL(inputs, pml_targets, hp.pml_dimension, hp.outputs_per_step)
                else:
                    helper = TacoTestHelper(batch_size, hp.pml_dimension, hp.outputs_per_step)

            (decoder_outputs, _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
                BasicDecoder(output_cell, helper, decoder_init_state),
                maximum_iterations=hp.max_iters)  # [N, T_out/r, P*r]

            # Reshape outputs to be one output per entry
            pml_intermediates = tf.reshape(decoder_outputs, [batch_size, -1, hp.pml_dimension])  # [N, T_out, P]

            # Add post-processing CBHG:
            post_outputs = post_cbhg(pml_intermediates, hp.pml_dimension, is_training,  # [N, T_out, postnet_depth=256]
                                     hp.postnet_depth)
            pml_outputs = tf.layers.dense(post_outputs, hp.pml_dimension)  # [N, T_out, P]

            # Grab alignments from the final decoder state:
            alignments = tf.transpose(final_decoder_state[0].alignment_history.stack(), [1, 2, 0])

            self.inputs = inputs
            self.input_lengths = input_lengths
            self.pml_intermediates = pml_intermediates
            self.pml_outputs = pml_outputs
            self.alignments = alignments
            self.pml_targets = pml_targets
            self.attention_cell = attention_cell
            self.locked_alignments = locked_alignments_

            if logs_enabled:
                log('Initialized Tacotron model. Dimensions: ')
                log('  Train mode:              {}'.format(is_training))
                log('  GTA mode:                {}'.format(gta))
                log('  EAL mode:                {}'.format(eal))
                log('  Embedding:               {}'.format(embedded_inputs.shape[-1]))
                log('  Prenet out:              {}'.format(prenet_outputs.shape[-1]))
                log('  Encoder out:             {}'.format(encoder_outputs.shape[-1]))
                log('  Attention out:           {}'.format(attention_cell.output_size))
                log('  Concat attn & out:       {}'.format(concat_cell.output_size))
                log('  Decoder cell out:        {}'.format(decoder_cell.output_size))
                log('  Decoder out ({} frames):  {}'.format(hp.outputs_per_step, decoder_outputs.shape[-1]))
                log('  Decoder out (1 frame):   {}'.format(pml_intermediates.shape[-1]))
                log('  Postnet out:             {}'.format(post_outputs.shape[-1]))
                log('  PML out:                 {}'.format(pml_outputs.shape[-1]))

    def add_loss(self):
        '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
        with tf.variable_scope('loss') as scope:
            hp = self._hparams
            self.pml_intermediate_loss = tf.reduce_mean(tf.abs(self.pml_targets - self.pml_intermediates))
            self.pml_loss = tf.reduce_mean(tf.abs(self.pml_targets - self.pml_outputs))
            self.loss = self.pml_intermediate_loss + self.pml_loss
            
            # avg kl
            if self.locked_alignments is not None:
                tmp = self.locked_alignments[:,:,:tf.shape(self.alignments)[-1]]
                self.alignments_ref = tf.clip_by_value(tmp, np.finfo(np.float32).tiny, np.finfo(np.float32).max)
                self.loss_align = tf.reduce_mean(self.alignments_ref*tf.log(self.alignments_ref/self.alignments))
#                 self.loss_joint = self.loss + self.loss_align
#                 self.loss_align = tf.reduce_mean(self.alignments_ref*tf.log(self.alignments_ref) - 
#                                                  self.alignments_ref*tf.log(self.alignments))

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
            
            if self.flag_trainAlign:
                var_list = [v for v in tf.trainable_variables() if 'Location_Sensitive_Attention' in v.name or 'memory_layer' in v.name]
                gradients, variables = zip(*optimizer.compute_gradients(self.loss_align, var_list=var_list))
            else:
                gradients, variables = zip(*optimizer.compute_gradients(self.loss))
                
            self.gradients = gradients
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

            # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
            # https://github.com/tensorflow/tensorflow/issues/1122
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
                                                          global_step=global_step)

    def set_locked_alignments(self, locked_alignments=None):
        """
        Sets the lockable alignments to a different value.

        :param locked_alignments: Alignments to lock the attention mechanism to.
        :return: None
        """
        pass


def _learning_rate_decay(init_lr, global_step):
    # Noam scheme from tensor2tensor:
    warmup_steps = 4000.0
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)
