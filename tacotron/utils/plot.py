import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf


def plot_alignment(alignment, path, info=None):
  fig, ax = plt.subplots()
  im = ax.imshow(
    alignment,
    aspect='auto',
    cmap='hot',
    origin='lower',
    interpolation='none')
  fig.colorbar(im, ax=ax)
  xlabel = 'Decoder Step'
  if info is not None:
    xlabel += '\n\n' + info
  plt.xlabel(xlabel)
  plt.ylabel('Encoder Step')
  plt.tight_layout()

  # save the alignment to disk
  plt.savefig(path, format='png')

  # setup io buffer to hold image and return tensorflow version
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  buf.seek(0)
  plot = tf.image.decode_png(buf.getvalue(), channels=4)
  plot = tf.expand_dims(plot, 0)
  return plot
