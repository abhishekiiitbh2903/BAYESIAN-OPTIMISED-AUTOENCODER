import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=config.neurons, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=config.neurons, out_channels=config.neurons, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=config.neurons, out_channels=config.neurons, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=config.neurons * config.inbits, out_channels=config.cbits, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1, 1)
        x = self.conv4(x)
        x = x.permute(0, 2, 1)
        return x
