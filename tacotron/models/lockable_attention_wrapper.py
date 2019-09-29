import tensorflow as tf
from tensorflow.contrib.seq2seq import AttentionWrapper, AttentionWrapperState
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, math_ops

import numpy as np


class LockableAttentionWrapper(AttentionWrapper):
    def __init__(self,
                 cell,
                 attention_mechanism,
                 attention_layer_size=None,
                 alignment_history=False,
                 cell_input_fn=None,
                 output_attention=True,
                 initial_cell_state=None,
                 name=None,
                 attention_layer=None,
                 attention_fn=None,
                 locked_alignments=None,
                 flag_trainAlign=False,
                 flag_trainJoint=False):
        """Construct the `AttentionWrapper`.
        **NOTE** If you are using the `BeamSearchDecoder` with a cell wrapped in
        `AttentionWrapper`, then you must ensure that:
        - The encoder output has been tiled to `beam_width` via
          `tf.contrib.seq2seq.tile_batch` (NOT `tf.tile`).
        - The `batch_size` argument passed to the `zero_state` method of this
          wrapper is equal to `true_batch_size * beam_width`.
        - The initial state created with `zero_state` above contains a
          `cell_state` value containing properly tiled final state from the
          encoder.
        An example:
        ```
        tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
            encoder_outputs, multiplier=beam_width)
        tiled_encoder_final_state = tf.conrib.seq2seq.tile_batch(
            encoder_final_state, multiplier=beam_width)
        tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
            sequence_length, multiplier=beam_width)
        attention_mechanism = MyFavoriteAttentionMechanism(
            num_units=attention_depth,
            memory=tiled_inputs,
            memory_sequence_length=tiled_sequence_length)
        attention_cell = AttentionWrapper(cell, attention_mechanism, ...)
        decoder_initial_state = attention_cell.zero_state(
            dtype, batch_size=true_batch_size * beam_width)
        decoder_initial_state = decoder_initial_state.clone(
            cell_state=tiled_encoder_final_state)
        ```
        Args:
          cell: An instance of `RNNCell`.
          attention_mechanism: A list of `AttentionMechanism` instances or a single
            instance.
          attention_layer_size: A list of Python integers or a single Python
            integer, the depth of the attention (output) layer(s). If None
            (default), use the context as attention at each time step. Otherwise,
            feed the context and cell output into the attention layer to generate
            attention at each time step. If attention_mechanism is a list,
            attention_layer_size must be a list of the same length. If
            attention_layer is set, this must be None. If attention_fn is set, it
            must be guaranteed that the outputs of attention_fn also meet the above
            requirements.
          alignment_history: Python boolean, whether to store alignment history
            from all time steps in the final output state (currently stored as a
            time major `TensorArray` on which you must call `stack()`).
          cell_input_fn: (optional) A `callable`.  The default is:
            `lambda inputs, attention: array_ops.concat([inputs, attention], -1)`.
          output_attention: Python bool.  If `True` (default), the output at each
            time step is the attention value.  This is the behavior of Luong-style
            attention mechanisms.  If `False`, the output at each time step is
            the output of `cell`.  This is the behavior of Bhadanau-style
            attention mechanisms.  In both cases, the `attention` tensor is
            propagated to the next time step via the state and is used there.
            This flag only controls whether the attention mechanism is propagated
            up to the next cell in an RNN stack or to the top RNN output.
          initial_cell_state: The initial state value to use for the cell when
            the user calls `zero_state()`.  Note that if this value is provided
            now, and the user uses a `batch_size` argument of `zero_state` which
            does not match the batch size of `initial_cell_state`, proper
            behavior is not guaranteed.
          name: Name to use when creating ops.
          attention_layer: A list of `tf.layers.Layer` instances or a
            single `tf.layers.Layer` instance taking the context and cell output as
            inputs to generate attention at each time step. If None (default), use
            the context as attention at each time step. If attention_mechanism is a
            list, attention_layer must be a list of the same length. If
            attention_layers_size is set, this must be None.
          attention_fn: An optional callable function that allows users to provide
            their own customized attention function, which takes input
            (attention_mechanism, cell_output, attention_state, attention_layer, time_step) and
            outputs (attention, alignments, next_attention_state). If provided, the
            attention_layer_size should be the size of the outputs of attention_fn.
            From TensorFlow 2.0 implementation.
          locked_alignments: Alignments to lock the attention mechanism to.
        Raises:
          TypeError: `attention_layer_size` is not None and (`attention_mechanism`
            is a list but `attention_layer_size` is not; or vice versa).
          ValueError: if `attention_layer_size` is not None, `attention_mechanism`
            is a list, and its length does not match that of `attention_layer_size`;
            if `attention_layer_size` and `attention_layer` are set simultaneously.
        """
        super(LockableAttentionWrapper, self).__init__(cell,
                                                       attention_mechanism,
                                                       attention_layer_size,
                                                       alignment_history,
                                                       cell_input_fn,
                                                       output_attention,
                                                       initial_cell_state,
                                                       name,
                                                       attention_layer)

        if attention_fn is None:
            # by default we don't need the time step in the compute attention function
            attention_fn = self._compute_attention

        self._attention_fn = attention_fn
        self._locked_alignments = locked_alignments
        self._flag_trainAlign = flag_trainAlign
        self._flag_trainJoint = flag_trainJoint

    def call(self, inputs, state):
        """Perform a step of attention-wrapped RNN.
        - Step 1: Mix the `inputs` and previous step's `attention` output via
          `cell_input_fn`.
        - Step 2: Call the wrapped `cell` with this input and its previous state.
        - Step 3: Score the cell's output with `attention_mechanism`.
        - Step 4: Calculate the alignments by passing the score through the
          `normalizer`.
        - Step 5: Calculate the context vector as the inner product between the
          alignments and the attention_mechanism's values (memory).
        - Step 6: Calculate the attention output by concatenating the cell output
          and context through the attention layer (a linear layer with
          `attention_layer_size` outputs).
        Args:
          inputs: (Possibly nested tuple of) Tensor, the input at this time step.
          state: An instance of `AttentionWrapperState` containing
            tensors from the previous time step.
        Returns:
          A tuple `(attention_or_cell_output, next_state)`, where:
          - `attention_or_cell_output` depending on `output_attention`.
          - `next_state` is an instance of `AttentionWrapperState`
             containing the state calculated at this time step.
        Raises:
          TypeError: If `state` is not an instance of `AttentionWrapperState`.
        """
        if not isinstance(state, AttentionWrapperState):
            raise TypeError("Expected state to be instance of AttentionWrapperState. "
                            "Received type %s instead." % type(state))

        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        cell_batch_size = (
                cell_output.shape[0].value or array_ops.shape(cell_output)[0])
        error_message = (
                "When applying AttentionWrapper %s: " % self.name +
                "Non-matching batch sizes between the memory "
                "(encoder output) and the query (decoder output).  Are you using "
                "the BeamSearchDecoder?  You may need to tile your memory input via "
                "the tf.contrib.seq2seq.tile_batch function with argument "
                "multiple=beam_width.")
        with ops.control_dependencies(
                self._batch_size_checks(cell_batch_size, error_message)):
            cell_output = array_ops.identity(
                cell_output, name="checked_cell_output")

        if self._is_multi:
            previous_attention_state = state.attention_state
            previous_alignment_history = state.alignment_history
        else:
            previous_attention_state = [state.attention_state]
            previous_alignment_history = [state.alignment_history]

        all_alignments = []
        all_attentions = []
        all_attention_states = []
        maybe_all_histories = []
        for i, attention_mechanism in enumerate(self._attention_mechanisms):
            attention, alignments, next_attention_state = self._attention_fn(
                attention_mechanism, cell_output, previous_attention_state[i],
                self._attention_layers[i] if self._attention_layers else None, state.time)
            alignment_history = previous_alignment_history[i].write(
                state.time, alignments) if self._alignment_history else ()

            all_attention_states.append(next_attention_state)
            all_alignments.append(alignments)
            all_attentions.append(attention)
            maybe_all_histories.append(alignment_history)

        attention = array_ops.concat(all_attentions, 1)
        next_state = AttentionWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=attention,
            attention_state=self._item_or_tuple(all_attention_states),
            alignments=self._item_or_tuple(all_alignments),
            alignment_history=self._item_or_tuple(maybe_all_histories))

        if self._output_attention:
            return attention, next_state
        else:
            return cell_output, next_state

    def _compute_attention(self, attention_mechanism, cell_output, attention_state,
                           attention_layer, time_step):
        """Computes the attention and alignments for a given attention_mechanism."""
        alignments, next_attention_state = attention_mechanism(
            cell_output, state=attention_state)

        if self._locked_alignments is None:
            # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
            expanded_alignments = array_ops.expand_dims(alignments, 1)
        else:
            if type(self._locked_alignments) is np.ndarray:
                # alignments come in with shape: (batch_size, encoder_steps, decoder_steps)
                tmp_alignments = tf.constant(self._locked_alignments, dtype=tf.float32)
                # select the relevant time step
                tmp_alignments = tmp_alignments[:, :, time_step]
            else:
                tmp_alignments = self._locked_alignments[:, :, time_step]
            expanded_alignments = array_ops.expand_dims(tmp_alignments, 1)
            # not elegant but safe; this keeps the old eal implementation as it was, which helps checking the locked alignments
            if not (self._flag_trainAlign or self._flag_trainJoint):
                alignments = tmp_alignments
            
        # Context is the inner product of alignments and values along the
        # memory time dimension.
        # alignments shape is
        #   [batch_size, 1, memory_time]
        # attention_mechanism.values shape is
        #   [batch_size, memory_time, memory_size]
        # the batched matmul is over memory_time, so the output shape is
        #   [batch_size, 1, memory_size].
        # we then squeeze out the singleton dim.
        context = math_ops.matmul(expanded_alignments, attention_mechanism.values)
        context = array_ops.squeeze(context, [1])

        if attention_layer is not None:
            attention = attention_layer(array_ops.concat([cell_output, context], 1))
        else:
            attention = context

        return attention, alignments, next_attention_state
