from concurrent.futures import ProcessPoolExecutor
from functools import partial
import glob
import numpy as np
import os
from util import audio
import shutil
from datasets import linear_dir, mel_dir, pml_dir, pml_data_dir, wav_dir


def build_from_path(in_dir, out_dir, hparams, num_workers=1, tqdm=lambda x: x):
    '''Preprocesses the LJ Speech dataset from a given input path into a given output directory.

      Args:
        in_dir: The directory where you have downloaded the LJ Speech dataset
        out_dir: The directory to write the output into
        num_workers: Optional number of worker processes to parallelize across
        tqdm: You can optionally pass tqdm to get a nice progress bar

      Returns:
        A list of tuples describing the training examples. This should be written to train.txt
    '''

    # We use ProcessPoolExecutor to parallelize across processes. This is just an optimization and you
    # can omit it and just call _process_utterance on each input if you want.
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1

    with open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
            text = parts[2]

            # get the pml features
            pml_path = os.path.join(in_dir, 'pml', '%s.cmp' % parts[0])
            pml_features = np.fromfile(pml_path, dtype=np.float32)

            futures.append(executor.submit(partial(_process_utterance, out_dir, index, wav_path, text, pml_features, hparams)))
            index += 1

    # copy the data files, if they exist
    files = glob.iglob(os.path.join(in_dir, 'pml', '*.dat'))
    os.makedirs(os.path.join(out_dir, pml_data_dir), exist_ok=True)

    for file in files:
        if os.path.isfile(file):
            shutil.copy2(file, os.path.join(out_dir, pml_data_dir))

    return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, wav_path, text, pml_cmp, hparams):
    '''Preprocesses a single utterance audio/text pair.

    This writes the mel and linear scale spectrograms to disk and returns a tuple to write
    to the train.txt file.

    Args:
      out_dir: The directory to write the spectrograms into
      index: The numeric index to use in the spectrogram filenames.
      wav_path: Path to the audio file containing the speech input
      text: The text spoken in the input audio file
      pml_cmp: One dimensional array containing vocoder features read from .cmp file

    Returns:
      A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
    '''

    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)

    # Create directories if they do not exist
    os.makedirs(os.path.join(out_dir, wav_dir), exist_ok=True)
    os.makedirs(os.path.join(out_dir, pml_dir), exist_ok=True)
    os.makedirs(os.path.join(out_dir, mel_dir), exist_ok=True)
    os.makedirs(os.path.join(out_dir, linear_dir), exist_ok=True)

    # Copy the wav into the training directory
    shutil.copy2(wav_path, os.path.join(out_dir, wav_dir)) # just dir name needed

    # Write the PML features to disk
    pml_filename = 'ljspeech-pml-%05d.npy' % index
    pml_dimension = hparams.pml_dimension
    pml_features = pml_cmp.reshape((-1, pml_dimension))
    pml_frames = pml_features.shape[0]
    np.save(os.path.join(out_dir, pml_dir, pml_filename), pml_features, allow_pickle=False)

    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    n_frames = spectrogram.shape[1]

    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]

    # Ensure lengths of spectrograms and PML features are the same
    if n_frames > pml_frames:
        spectrogram = spectrogram[:, :pml_frames]

    # Check the shape of the mel target
    if mel_frames > pml_frames:
        mel_spectrogram = mel_spectrogram[:, :pml_frames]

    # Write the spectrograms to disk:
    spectrogram_filename = 'ljspeech-spec-%05d.npy' % index
    mel_filename = 'ljspeech-mel-%05d.npy' % index
    np.save(os.path.join(out_dir, linear_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, mel_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

    # Return a tuple describing this training example:
    return spectrogram_filename, mel_filename, n_frames, pml_filename, pml_frames, \
        text, os.path.basename(wav_path)
