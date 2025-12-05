import torch
import torch.nn as nn
from model.utils import DRblock

class Decoder(nn.Module):
  def __init__(self,config):
    super(Decoder,self).__init__()
    self.inbits=config.inbits
    self.inilayer=nn.Conv1d(in_channels=config.cbits,out_channels=config.inbits*config.neurons,kernel_size=1)
    self.dr_block1=DRblock(config)
    self.dr_block2=DRblock(config)
    self.cu=nn.Conv1d(in_channels=2*config.neurons,out_channels=config.neurons,kernel_size=1)
    self.final_conv=nn.Conv1d(in_channels=config.neurons,out_channels=1,padding=1,kernel_size=3)
    self.relu=nn.ReLU()
    self.sigmoid=nn.Sigmoid()

  def forward(self,x):
    x=x.permute(0,2,1)
    x=self.relu(self.inilayer(x))
    noisy_x = x.view(x.size(0), -1,self.inbits)
    x=self.dr_block1(noisy_x)
    x=self.dr_block2(x)
    x= torch.cat([noisy_x,x], dim=1)
    x=self.cu(x)
    x=noisy_x-x
    x=self.sigmoid(self.final_conv(x))
    x=x.permute(0,2,1)
    return x