import torch.nn as nn
from model.Encoder import Encoder
from model.Decoder import Decoder
from model.utils import awgn

class AutoEncoder(nn.Module):
  def __init__(self,config):
    super(AutoEncoder,self).__init__()
    self.enc=Encoder(config)
    self.dec=Decoder(config)

  def forward(self,x,snr_db=10):
    enc_output=self.enc(x)
    noisy_output=awgn(enc_output,snr_db=snr_db)
    final_output=self.dec(noisy_output)
    return final_output