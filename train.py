import argparse
import os
import infolog
import hparams
from tacotron.train import train as train_tacotron

log = infolog.log


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default=os.path.expanduser('~/tacotron'))
  parser.add_argument('--log_dir', default=os.path.expanduser('~/tacotron'))
  parser.add_argument('--input', default='training/train.txt')
  parser.add_argument('--model', default='tacotron')
  parser.add_argument('--variant', default='tacotron')
  parser.add_argument('--name', help='Name of the run. Used for logging. Defaults to model name.')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--restore_step', type=int, help='Global step to restore from checkpoint.')
  parser.add_argument('--summary_interval', type=int, default=100,
    help='Steps between running summary ops.')
  parser.add_argument('--checkpoint_interval', type=int, default=1000,
    help='Steps between writing checkpoints.')
  parser.add_argument('--slack_url', help='Slack webhook URL to get periodic reports.')
  parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
  parser.add_argument('--git', action='store_true', help='If set, verify that the client is clean.')
  parser.add_argument('--num_steps', type=int, default=100000, help='Maximum number of steps to run training for.')
  args = parser.parse_args()

  accepted_models = ['tacotron', 'wavenet']

  if args.model not in accepted_models:
      raise ValueError('please enter a valid model to train: {}'.format(accepted_models))

  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
  run_name = args.name or args.variant
  log_dir = os.path.join(args.log_dir, 'logs-%s' % run_name)
  os.makedirs(log_dir, exist_ok=True)
  infolog.init(os.path.join(log_dir, 'train.log'), run_name, args.slack_url)
  log('Initialised log file')
  hparams.parse(args.hparams)

  if args.model == 'tacotron':
      train_tacotron(log_dir, args)
  # elif args.model == 'wavenet':
      # wavenet_train(args, log_dir, hparams, args.wavenet_input)
  else:
      raise ValueError('Model provided {} unknown! {}'.format(args.model, accepted_models))


if __name__ == '__main__':
  main()
