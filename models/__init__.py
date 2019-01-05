from .tacotron import Tacotron
from .tacotron_pml import TacotronPML


def create_model(name, hparams):
  if name == 'tacotron':
    return Tacotron(hparams)
  elif name == 'tacotron_pml':
    return TacotronPML(hparams)
  else:
    raise Exception('Unknown model: ' + name)
