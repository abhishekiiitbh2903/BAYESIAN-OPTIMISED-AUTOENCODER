import torch
import torch.nn as nn

class DRblock(nn.Module):
    def __init__(self, config):
        super(DRblock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=config.neurons, out_channels=config.neurons, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=config.neurons, out_channels=config.neurons, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=config.neurons, out_channels=config.neurons, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=config.neurons, out_channels=config.neurons, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x  
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        x = x + residual  
        x = self.relu(x)  
        return x
    

def awgn(input_signal, snr_db, rate=1.0):
    snr_linear = 10 ** (snr_db / 10.0)
    avg_energy = torch.mean(input_signal ** 2, dim=(1, 2), keepdim=True)
    noise_variance = avg_energy / snr_linear
    noise = torch.randn_like(input_signal) * torch.sqrt(noise_variance)
    return input_signal + noise