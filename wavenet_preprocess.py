import argparse
import os
from multiprocessing import cpu_count

from datasets import wavenet_preprocessor
from hparams import hparams
from tqdm import tqdm


def preprocess(args, audio_input_dir, vocoder_input_dir, out_dir, hparams):
	pml_dir = os.path.join(out_dir, 'pml')
	wav_dir = os.path.join(out_dir, 'audio')
	os.makedirs(pml_dir, exist_ok=True)
	os.makedirs(wav_dir, exist_ok=True)
	metadata = wavenet_preprocessor.build_from_path(hparams, audio_input_dir, vocoder_input_dir, pml_dir, wav_dir, args.n_jobs, tqdm=tqdm)
	write_metadata(metadata, out_dir)

def write_metadata(metadata, out_dir):
	with open(os.path.join(out_dir, 'map.txt'), 'w', encoding='utf-8') as f:
		for m in metadata:
			f.write('|'.join([str(x) for x in m]) + '\n')

	timesteps = sum([int(m[4]) for m in metadata])
	sr = hparams.sample_rate
	hours = timesteps / sr / 3600
	print('Write {} utterances, {} audio timesteps, ({:.2f} hours)'.format(
		len(metadata), timesteps, hours))
	print('Max pml frames length: {}'.format(max(int(m[5]) for m in metadata)))
	print('Max audio timesteps length: {}'.format(max(m[4] for m in metadata)))

def run_preprocess(args, hparams):
	output_folder = os.path.join(args.base_dir, args.output)
	audio_input_dir = os.path.join(args.base_dir, args.audio_input)
	vocoder_input_dir = os.path.join(args.base_dir, args.vocoder_input)

	preprocess(args, audio_input_dir, vocoder_input_dir, output_folder, hparams)

def main():
	print('initializing preprocessing..')
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_dir', default=os.path.expanduser('~/tacotron'))
	parser.add_argument('--hparams', default='',
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--audio_input', default='LJSpeech-1.1/wavs')
	parser.add_argument('--vocoder_input', default='LJSpeech-1.1/pml')
	parser.add_argument('--output', default='tacotron_output/analysis_synthesis/')
	parser.add_argument('--n_jobs', type=int, default=cpu_count())
	args = parser.parse_args()

	modified_hp = hparams.parse(args.hparams)

	run_preprocess(args, modified_hp)

if __name__ == '__main__':
	main()
