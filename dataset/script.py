from torch.utils.data import Dataset
import scipy as si
import numpy as np
import torch

class PhaseBitDataset(Dataset):
  def __init__(self,file_path):
    super().__init__()
    mat = si.io.loadmat(file_path)
    phasebit = mat['phasebit']
    phasebit = np.reshape(phasebit, (phasebit.shape[0], phasebit.shape[1], 1))
    self.data = torch.tensor(phasebit, dtype=torch.float32)

  def __len__(self):
      return self.data.shape[0]

  def __getitem__(self, idx):
      return self.data[idx]
  

def get_dataloader(config):
   train_path=config.train_path
   val_path=config.val_path
   test_path=config.test_path
   train_dataset = PhaseBitDataset(train_path)
   val_dataset = PhaseBitDataset(val_path)
   test_dataset = PhaseBitDataset(test_path)
   train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
   val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
   test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
   return train_dataloader, val_dataloader, test_dataloader

