from .tacotron import Tacotron
from .tacotron_mel_pml import TacotronMelPML
from .tacotron_pml import TacotronPML
from .tacotron_pml_x import TacotronPMLExtended
from .tacotron_pml_postnet import TacotronPMLPostnet
from .tacotron_pml_locsens import TacotronPMLLocSens
from .tacotron_pml_x_locsens import TacotronPMLExtendedLocSens


def create_model(name, hparams):
  if name == 'tacotron':
    return Tacotron(hparams)
  elif name == 'tacotron_pml':
    return TacotronPML(hparams)
  elif name == 'tacotron_pml_x':
    return TacotronPMLExtended(hparams)
  elif name == 'tacotron_mel_pml':
    return TacotronMelPML(hparams)
  elif name == 'tacotron_pml_postnet':
    return TacotronPMLPostnet(hparams)
  elif name == 'tacotron_pml_locsens':
    return TacotronPMLLocSens(hparams)
  elif name == 'tacotron_pml_x_locsens':
    return TacotronPMLExtendedLocSens(hparams)
  else:
    raise Exception('Unknown model: ' + name)
