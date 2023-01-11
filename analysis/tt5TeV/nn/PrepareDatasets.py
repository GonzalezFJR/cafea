from cafea.plotter.plotter import *
import os, pandas
import numpy as np
branches = ['A_njets', 'A_nbtags', 'A_ht', 'A_st', 'A_sumAllPt', 'A_leta', 'A_j0pt', 'A_j0eta', 'A_u0pt', 'A_u0eta', 'A_ptjj', 'A_mjj', 'A_medianDRjj', 'A_minDRjj', 'A_mlb', 'A_mt', 'A_ptsumveclb', 'A_drlb']

def LoadColumns(path, sampleName, branches, isSignal=True):
  ''' open a pkl.gz sample and retrieve columns with names "branches", that must have the same length, and returns pandas dataset '''
  path = os.path.join(path, sampleName+'.pkl.gz')
  h = loadHistos(path)
  pd = pandas.DataFrame()
  for name in branches:
    data = h[name].value
    pd.insert(0, name, data)
  if isSignal:
    pd.insert(0, 'label', np.ones(len(pd)))
  else:
    pd.insert(0, 'label', np.zeros(len(pd)))
  return pd

def toList(t):
  ''' Transforms whatever string into a list '''
  if isinstance(t, str) and ',' in t:
    t = t.replace(' ', '').split(',')
  elif isinstance(t, str):
    t = [t]
  return t

def BuildDataset(path, signal, bkg, var=branches, trainFrac=0.8, nData=None, verbose=1):
  ''' Loads a dataset with labels for signal and bkg and returns train and test data samples '''
  signal = toList(signal); bkg = toList(bkg); var = toList(var)
  datasetsSignal = pandas.concat([LoadColumns(path, s, var, isSignal=True) for s in signal])
  datasetsBkg    = pandas.concat([LoadColumns(path, b, var, isSignal=False) for b in bkg], ignore_index=True)
  if nData is not None:
    if nData <0:
      nsig = len(datasetsSignal)
      nbkg = len(datasetsBkg)
      nmin = min([nsig, nbkg])
      nData = nmin
    datasetsSignal = datasetsSignal.sample(nData)
    datasetsBkg    = datasetsBkg   .sample(nData)
  datasets = [datasetsSignal, datasetsBkg]
  df = pandas.concat(datasets, ignore_index=True)
  #df = df.sample(frac=1) # suffle
  mask = np.random.rand(len(df)) < 0.8
  train = df[ mask]
  test  = df[~mask]
  ### Build info
  if verbose:
    nsig = sum(np.where(df['label']==1, 1, 0))
    nbkg = sum(np.where(df['label']==0, 1, 0))
    n    = len(df)
    ntrain = len(train)
    ntest  = len(test)
    nvar = len(var)
    print(f" >> Dataset with {n} events and {nvar} columns\n >> Contrains {nsig} signal events and {nbkg} background events\n >> The test sample has {ntest} envents and the train sample has {ntrain} events")
  
  return train, test

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class HepDataset(Dataset):
  ''' Create a torch dataloader '''
  def __init__(self, pd):
    self.labels = pd['label'] if 'label' in pd else np.ones(len(pd))
    self.data   = pd.drop('label', 1)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    dat = self.data  .iloc[idx].to_numpy(dtype=float)
    lab = self.labels.iloc[idx]
    #lab = np.transpose([lab, np.where(lab==1, 0, 1)])
    return dat, lab

def PrepareData(path, signal, bkg, var=branches, trainFrac=0.8, batch_size=64, nData=None, verbose=1):
  dtrain, dtest = BuildDataset(path, signal, bkg, var, trainFrac, nData, verbose)
  train    = HepDataset(dtrain)
  train_dl = DataLoader(train, batch_size=batch_size, shuffle=True)
  test     = HepDataset(dtest)
  test_dl  = DataLoader(test, batch_size=batch_size, shuffle=True)
  return train_dl, test_dl

