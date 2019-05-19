import numpy as np
import tensorflow as tf

# Default hyperparameters:
hparams = tf.contrib.training.HParams(
    # Comma-separated list of cleaners to run on text prior to training and eval. For non-English
    # text, you may want to use "basic_cleaners" or "transliteration_cleaners" See TRAINING_DATA.md.
    cleaners='english_cleaners',

    # Audio:
    num_mels=80,
    num_freq=1025,
    sample_rate=20000,
    frame_length_ms=50,
    frame_shift_ms=12.5,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,
    pml_dimension=86,

    # Model:
    outputs_per_step=5,
    embed_depth=256,
    prenet_depths=[256, 128],
    encoder_depth=256,
    postnet_depth=256,
    attention_depth=256,
    decoder_depth=256,

    # Simplified Model Features
    # Embedding:
    embedding_dim=512,
    # Encoder:
    encoder_conv_layers=3,
    encoder_conv_width=5,
    encoder_conv_channels=512,
    encoder_gru_units=256,  # for each direction
    # Decoder:
    decoder_gru_layers=1,
    decoder_gru_units=1024,
    # Postnet:
    postnet_conv_layers=5,
    postnet_conv_width=5,
    postnet_conv_channels=512,
    # Expand Network
    expand_conv_layers=5,
    expand_conv_width=5,
    expand_conv_channels=512,
    expand_gru_units=256,  # for each direction

    # Training:
    batch_size=32,
    adam_beta1=0.9,
    adam_beta2=0.999,
    initial_learning_rate=0.002,
    decay_learning_rate=True,
    use_cmudict=False,  # Use CMUDict during training to learn pronunciation of ARPAbet phonemes

    # Eval:
    max_iters=800,
    griffin_lim_iters=60,
    power=1.5,  # Power to raise magnitudes to prior to Griffin-Lim

    sentences=[
        # From July 8, 2017 New York Times:
        'Scientists at the CERN laboratory say they have discovered a new particle.',
        'There’s a way to measure the acute emotional intelligence that has never gone out of style.',
        'President Trump met with other leaders at the Group of 20 conference.',
        'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
        # From Google's Tacotron example page:
        'Generative adversarial network or variational auto-encoder.',
        'The buses aren\'t the problem, they actually provide a solution.',
        'Does the quick brown fox jump over the lazy dog?',
        'Talib Kweli confirmed to AllHipHop that he will be releasing an album in the next year.',
    ],

    # Wavenet
    # Input type:
    # 1. raw [-1, 1]
    # 2. mulaw [-1, 1]
    # 3. mulaw-quantize [0, mu]
    # If input_type is raw or mulaw, network assumes scalar input and
    # discretized mixture of logistic distributions output, otherwise one-hot
    # input and softmax output are assumed.
    # Model general type
    input_type="raw",
    # Raw has better quality but harder to train. mulaw-quantize is easier to train but has lower quality.
    quantize_channels=2 ** 16,
    # 65536 (16-bit) (raw) or 256 (8-bit) (mulaw or mulaw-quantize) // number of classes = 256 <=> mu = 255
    use_bias=True,  # Whether to use bias in convolutional layers of the Wavenet
    legacy=True,
    # Whether to use legacy mode: Multiply all skip outputs but the first one with sqrt(0.5) (True for more early training stability, especially for large models)
    residual_legacy=True,
    # Whether to scale residual blocks outputs by a factor of sqrt(0.5) (True for input variance preservation early in training and better overall stability)

    # Model Losses parmeters
    # Minimal scales ranges for MoL and Gaussian modeling
    log_scale_min=float(np.log(1e-14)),  # Mixture of logistic distributions minimal log scale
    log_scale_min_gauss=float(np.log(1e-7)),  # Gaussian distribution minimal allowed log scale
    # Loss type
    cdf_loss=False,
    # Whether to use CDF loss in Gaussian modeling. Advantages: non-negative loss term and more training stability. (Automatically True for MoL)

    # model parameters
    # To use Gaussian distribution as output distribution instead of mixture of logistics, set "out_channels = 2" instead of "out_channels = 10 * 3". (UNDER TEST)
    out_channels=2,
    # This should be equal to quantize channels when input type is 'mulaw-quantize' else: num_distributions * 3 (prob, mean, log_scale).
    layers=20,  # Number of dilated convolutions (Default: Simplified Wavenet of Tacotron-2 paper)
    stacks=2,  # Number of dilated convolution stacks (Default: Simplified Wavenet of Tacotron-2 paper)
    residual_channels=128,  # Number of residual block input/output channels.
    gate_channels=256,  # split in 2 in gated convolutions
    skip_out_channels=128,  # Number of residual block skip convolution channels.
    kernel_size=3,  # The number of inputs to consider in dilated convolutions.

    # Upsampling parameters (local conditioning)
    cin_channels=86,  # Set this to -1 to disable local conditioning, else it must be equal to num_mels!!
    # Upsample types: ('1D', '2D', 'Resize', 'SubPixel', 'NearestNeighbor')
    # All upsampling initialization/kernel_size are chosen to omit checkerboard artifacts as much as possible. (Resize is designed to omit that by nature).
    # To be specific, all initial upsample weights/biases (when NN_init=True) ensure that the upsampling layers act as a "Nearest neighbor upsample" of size "hop_size" (checkerboard free).
    # 1D spans all frequency bands for each frame (channel-wise) while 2D spans "freq_axis_kernel_size" bands at a time. Both are vanilla transpose convolutions.
    # Resize is a 2D convolution that follows a Nearest Neighbor (NN) resize. For reference, this is: "NN resize->convolution".
    # SubPixel (2D) is the ICNR version (initialized to be equivalent to "convolution->NN resize") of Sub-Pixel convolutions. also called "checkered artifact free sub-pixel conv".
    # Finally, NearestNeighbor is a non-trainable upsampling layer that just expands each frame (or "pixel") to the equivalent hop size. Ignores all upsampling parameters.
    upsample_type='SubPixel',
    # Type of the upsampling deconvolution. Can be ('1D' or '2D', 'Resize', 'SubPixel' or simple 'NearestNeighbor').
    upsample_activation='Relu',  # Activation function used during upsampling. Can be ('LeakyRelu', 'Relu' or None)
    upsample_scales=[5, 16],  # prod(upsample_scales) should be equal to hop_size
    freq_axis_kernel_size=3,
    # Only used for 2D upsampling types. This is the number of requency bands that are spanned at a time for each frame.
    leaky_alpha=0.4,  # slope of the negative portion of LeakyRelu (LeakyRelu: y=x if x>0 else y=alpha * x)
    NN_init=True,
    # Determines whether we want to initialize upsampling kernels/biases in a way to ensure upsample is initialize to Nearest neighbor upsampling. (Mostly for debug)
    NN_scaler=0.3,
    # Determines the initial Nearest Neighbor upsample values scale. i.e: upscaled_input_values = input_values * NN_scaler (1. to disable)

    # global conditioning
    gin_channels=-1,
    # Set this to -1 to disable global conditioning, Only used for multi speaker dataset. It defines the depth of the embeddings (Recommended: 16)
    use_speaker_embedding=True,  # whether to make a speaker embedding
    n_speakers=5,  # number of speakers (rows of the embedding)
    speakers_path=None,
    # Defines path to speakers metadata. Can be either in "speaker\tglobal_id" (with header) tsv format, or a single column tsv with speaker names. If None, use "speakers".
    speakers=['speaker0', 'speaker1',
              # List of speakers used for embeddings visualization. (Consult "wavenet_vocoder/train.py" if you want to modify the speaker names source).
              'speaker2', 'speaker3', 'speaker4'],
    # Must be consistent with speaker ids specified for global conditioning for correct visualization.
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
