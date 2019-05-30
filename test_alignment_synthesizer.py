import argparse
from hparams import hparams
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image
from tacotron.alignment_synthesizer import AlignmentSynthesizer
import tensorflow as tf


def original_tacotron_vs_tacopml(taco_checkpoint, tacopml_checkpoint):
    synth = AlignmentSynthesizer()
    synth.load(tacopml_checkpoint, hparams, model_name='tacotron_pml')
    fixed_sentence = 'and district attorney henry m. wade both testified that they saw it later that day.'
    first_alignment = synth.synthesize(fixed_sentence)  # of shape (encoder_steps, decoder_steps)
    print('First Synthesized Alignment Shape: {}'.format(first_alignment.shape))

    cutoff = 220

    # Get the colormap colors
    cmap = cm.cool
    cool = cmap(np.arange(cmap.N))
    # Set alpha
    cool[:, -1] = np.linspace(0, 1, cmap.N)
    # Create new colormap
    cool = ListedColormap(cool)
    # generate the image
    first_image = Image.fromarray(np.uint8(cool(first_alignment) * 255))
    height, width = first_alignment.shape
    first_image = np.array(first_image)[:, :cutoff]

    # reset the graph after the first synthesise call
    tf.reset_default_graph()

    synth.load(taco_checkpoint, hparams, model_name='tacotron_orig', locked_alignments=first_alignment)
    fixed_sentence = 'and district attorney henry m. wade both testified that they saw it later that day.'
    second_alignment = synth.synthesize(fixed_sentence)
    print('First Synthesized Alignment: {}'.format(first_alignment))
    print('Second Synthesized Alignment: {}'.format(second_alignment))
    print('First Synthesized Alignment Shape: {}'.format(first_alignment.shape))
    print('Second Synthesized Alignment Shape: {}'.format(second_alignment.shape))

    # Get the colormap colors
    cmap = cm.Wistia
    wistia = cmap(np.arange(cmap.N))
    # Set alpha
    wistia[:, -1] = np.linspace(0, 1, cmap.N)
    # Create new colormap
    wistia = ListedColormap(wistia)
    # generate the image
    second_image = Image.fromarray(np.uint8(wistia(second_alignment) * 255))
    height, width = second_alignment.shape
    # second_image = second_image.resize((round(width * 12.5 / 5), height))
    second_image = np.array(second_image)[:, :cutoff]

    fig, ax = plt.subplots()

    first_im = ax.imshow(
        first_image,
        aspect='auto',
        origin='lower',
        interpolation='none')

    second_im = ax.imshow(
        second_image,
        aspect='auto',
        origin='lower',
        interpolation='none')

    taco_pml_patch = mpatches.Patch(color=cool(0.8), label='Tacotron PML Alignment')
    taco_patch = mpatches.Patch(color=wistia(0.8), label='Tacotron Alignment')
    ax.legend(handles=[taco_pml_patch, taco_patch], loc='upper left')

    xstep = 50
    outputs_per_step = 5
    frame_shift = 0.005
    plt.xticks(np.arange(0, 250, step=xstep),
               np.arange(0, 250 * frame_shift * outputs_per_step, step=frame_shift * outputs_per_step * xstep))
    xlabel = 'Decoder Time (s)'
    plt.xlabel(xlabel)
    plt.ylabel('Encoder Step (character index)')
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--first_checkpoint',
                        default='/media/josh/Store/tacotron-remote-logs/remote-logs/tacotron-lj-pml-500k/model.ckpt-352000')
    parser.add_argument('--second_checkpoint',
                        default='/media/josh/Store/tacotron-remote-logs/remote-logs/tacotron-lj-500k/model.ckpt-195000')
    parser.add_argument('--experiment', default='original')
    args = parser.parse_args()

    if args.experiment == 'original':
        original_tacotron_vs_tacopml(args.first_checkpoint, args.second_checkpoint)


if __name__ == '__main__':
    main()
