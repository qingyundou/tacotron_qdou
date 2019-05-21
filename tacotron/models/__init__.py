from tacotron.models.variants.tacotron import Tacotron
from tacotron.models.variants.tacotron_lstm_short import TacotronPMLShortLSTM
from tacotron.models.variants.tacotron_mel_pml import TacotronMelPML
from tacotron.models.variants.tacotron_pml import TacotronPML
from tacotron.models.variants.tacotron_pml_x import TacotronPMLExtended
from tacotron.models.variants.tacotron_pml_postnet import TacotronPMLPostnet
from tacotron.models.variants.tacotron_pml_locsens import TacotronPMLLocSens
from tacotron.models.variants.tacotron_simpl import TacotronPMLSimplifiedLocSens
from tacotron.models.variants.tacotron_simpl_lstm import TacotronPMLSimplifiedLocSensLSTM
from .tacotron import TacotronPMLExtendedLocSens


def create_model(name, hparams):
  if name == 'tacotron':
    return TacotronPMLExtendedLocSens(hparams)
  elif name == 'tacotron_lstm_short':
    return TacotronPMLShortLSTM(hparams)
  elif name == 'tacotron_simpl':
    return TacotronPMLSimplifiedLocSens(hparams)
  elif name == 'tacotron_orig':
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
  elif name == 'tacotron_simpl_lstm':
    return TacotronPMLSimplifiedLocSensLSTM(hparams)
  else:
    raise Exception('Unknown model: ' + name)
