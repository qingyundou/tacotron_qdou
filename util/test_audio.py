import librosa
import numpy as np
import os 
import unittest
from util import audio

class TestAudioModule(unittest.TestCase):

    def test_removes_silence(self):
        # Define the wav path
        wav_path = os.path.join(os.path.dirname(__file__), 'audio_files/herald_152_1.wav')

        # Load the audio to a numpy array:
        wav, sr = librosa.core.load(wav_path)

        # Remove silence from the wav
        silence_removed = audio.remove_silence(wav, top_db=15)

        # Get the power sequence
        rmse = librosa.feature.rmse(y=wav, frame_length=256, hop_length=64)[0]
        db = librosa.power_to_db(rmse**2, ref=np.max)

        # Output trimmed audio file
        librosa.output.write_wav(os.path.join(os.path.dirname(__file__), 'audio_files/test.wav'), silence_removed, sr)

        # assert that the silence removed waveform is shorter
        self.assertTrue(len(silence_removed) < len(wav))
        self.assertTrue(len(silence_removed) < 0.25 * len(wav))

if __name__ == '__main__':
    unittest.main()
