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
    ]
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
