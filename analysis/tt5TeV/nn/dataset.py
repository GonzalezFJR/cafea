from read import BuildDataset
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

path = '../histos/tt5TeV/forNN/'
train, test = BuildDataset(path, "TT, TTPS", "WJets", ['A_njets', 'A_nbtags', 'A_ht', 'A_sumAllPt', 'A_leta', 'A_j0pt', 'A_mjj', 'A_medianDRjj'])

class HepDataset(Dataset):
  def __init__(self, pd):
    self.labels = pd['label']
    self.data   = pd.drop('label', 1)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    dat = self.data  .iloc[idx].to_numpy()
    lab = self.labels.iloc[idx]
    return dat, lab


from torch.utils.data import DataLoader
training_data = MyDataset(train)
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    
