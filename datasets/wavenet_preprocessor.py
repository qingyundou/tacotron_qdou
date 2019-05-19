import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from util import audio
from wavenet_vocoder.util import is_mulaw, is_mulaw_quantize, mulaw, mulaw_quantize


def build_from_path(hparams, audio_input_dir, vocoder_input_dir, pml_dir, wav_dir, n_jobs=12, tqdm=lambda x: x):
    """
    Preprocesses the speech dataset from a gven input path to given output directories

    Args:
        - hparams: hyper parameters
        - audio_input_directory: input directory that contains the wav files to preprocess
        - vocoder_input_directory: input directory that contains the pml vocoder features to process
        - mel_dir: output directory of the preprocessed speech pml vocoder features dataset
        - wav_dir: output directory of the preprocessed speech audio dataset
        - n_jobs: Optional, number of worker process to parallelize across
        - tqdm: Optional, provides a nice progress bar

    Returns:
        - A list of tuple describing the train examples. this should be written to train.txt
    """

    # We use ProcessPoolExecutor to parallelize across processes, this is just for
    # optimization purposes and it can be omited
    executor = ProcessPoolExecutor(max_workers=n_jobs)
    futures = []
    for file in os.listdir(audio_input_dir):
        wav_path = os.path.join(audio_input_dir, file)
        basename = os.path.basename(wav_path).replace('.wav', '')
        pml_path = os.path.join(vocoder_input_dir, '{}.cmp'.format(basename))
        futures.append(
            executor.submit(partial(_process_utterance, pml_dir, wav_dir, basename, wav_path, pml_path, hparams)))

    return [future.result() for future in tqdm(futures) if future.result() is not None]


def _process_utterance(pml_dir, wav_dir, index, wav_path, pml_path, hparams):
    """
    Preprocesses a single utterance wav/text pair

    this writes the mel scale spectogram to disk and return a tuple to write
    to the train.txt file

    Args:
        - mel_dir: the directory to write the mel spectograms into
        - linear_dir: the directory to write the linear spectrograms into
        - wav_dir: the directory to write the preprocessed wav into
        - index: the numeric index to use in the spectrogram filename
        - wav_path: path to the audio file containing the speech input
        - pml_path: path to the cmp file containing the pml vocoder features
        - text: text spoken in the input audio file
        - hparams: hyper parameters

    Returns:
        - A tuple: (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, linear_frames, text)
    """
    try:
        # Load the audio as numpy array
        wav = audio.load_wav(wav_path)
    except FileNotFoundError:  # catch missing wav exception
        print('file {} present in csv metadata is not present in wav folder. skipping!'.format(
            wav_path))
        return None

    # rescale wav
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

        # Assert all audio is in [-1, 1]
        if (wav > 1.).any() or (wav < -1.).any():
            raise RuntimeError('wav has invalid value: {}'.format(wav_path))

    # Mu-law quantize
    if is_mulaw_quantize(hparams.input_type):
        # [0, quantize_channels)
        out = mulaw_quantize(wav, hparams.quantize_channels)

        constant_values = mulaw_quantize(0, hparams.quantize_channels)
        out_dtype = np.int16

    elif is_mulaw(hparams.input_type):
        # [-1, 1]
        out = mulaw(wav, hparams.quantize_channels)
        constant_values = mulaw(0., hparams.quantize_channels)
        out_dtype = np.float32

    else:
        # [-1, 1]
        out = wav
        constant_values = 0.
        out_dtype = np.float32

    # Get the PML features from the cmp file
    pml_cmp = np.fromfile(pml_path, dtype=np.float32)
    pml_features = pml_cmp.reshape((-1, hparams.pml_dimension))
    pml_frames = pml_features.shape[0]

    if pml_frames > hparams.max_pml_frames and hparams.clip_pmls_length:
        return None

    # Find parameters
    n_fft = (hparams.num_freq - 1) * 2

    if hparams.use_lws:
        # Ensure time resolution adjustement between audio and mel-spectrogram
        l, r = audio.pad_lr(wav, n_fft, audio.get_hop_size(hparams))

        # Zero pad audio signal
        out = np.pad(out, (l, r), mode='constant', constant_values=constant_values)
    else:
        # Ensure time resolution adjustement between audio and mel-spectrogram
        l_pad, r_pad = audio.librosa_pad_lr(wav, n_fft, audio.get_hop_size(hparams))

        # Reflect pad audio signal (Just like it's done in Librosa to avoid frame inconsistency)
        out = np.pad(out, (l_pad, r_pad), mode='constant', constant_values=constant_values)

    # print(len(out), pml_frames, audio.get_hop_size(hparams), pml_frames * audio.get_hop_size(hparams))
    assert len(out) >= pml_frames * audio.get_hop_size(hparams)

    # time resolution adjustment
    # ensure length of raw audio is multiple of hop size so that we can use
    # transposed convolution to upsample
    out = out[:pml_frames * audio.get_hop_size(hparams)]
    assert len(out) % audio.get_hop_size(hparams) == 0
    time_steps = len(out)

    # Write the spectrogram and audio to disk
    audio_filename = os.path.join(wav_dir, 'audio-{}.npy'.format(index))
    pml_filename = os.path.join(pml_dir, 'pml-{}.npy'.format(index))
    np.save(audio_filename, out.astype(out_dtype), allow_pickle=False)
    np.save(pml_filename, pml_features, allow_pickle=False)

    # global condition features
    if hparams.gin_channels > 0:
        raise RuntimeError('When activating global conditions, please set your speaker_id rules in line 129 of '
                           'datasets/wavenet_preprocessor.py to use them during training')
    else:
        speaker_id = '<no_g>'

    # Return a tuple describing this training example
    return audio_filename, pml_path, pml_filename, speaker_id, time_steps, pml_frames
