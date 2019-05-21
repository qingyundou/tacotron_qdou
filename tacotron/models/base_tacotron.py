import numpy as np


class _BaseTacotron:
    def __init__(self, hparams, alignments=None):
        """
        Initialises _BaseTacotron instance

        :param hparams:
        :param alignments: expects alignments in shape (encoder_steps, decoder_steps) which is internal
        """
        self._hparams = hparams

        # fix the alignments shape to (batch_size, encoder_steps, decoder_steps) if not already including
        # batch dimension
        alignments_ = alignments

        if alignments_ is not None:
            if np.ndim(alignments_) < 3:
                alignments_ = np.expand_dims(alignments_, 0)

        self._alignments = alignments_
