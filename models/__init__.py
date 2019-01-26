from .tacotron import Tacotron
from .tacotron_pml import TacotronPML
from .tacotron_pml_x import TacotronPMLExtended


def create_model(name, hparams):
  if name == 'tacotron':
    return Tacotron(hparams)
  elif name == 'tacotron_pml':
    return TacotronPML(hparams)
  elif name == 'tacotron_pml_x':
    return TacotronPMLExtended(hparams)
  else:
    raise Exception('Unknown model: ' + name)
