from dataclasses import dataclass
import torch

@dataclass
class Config:
    inbits = 9
    cbits = 5
    neurons = 128
    batch_size= 128
    num_epochs=500
    learning_rate=0.001
    decay_rate=0.99
    decay_steps=1000
    train_path='dataset/data/train.mat'
    test_path='dataset/data/test.mat'
    val_path='dataset/data/validation.mat'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = 'best_model.pth'


