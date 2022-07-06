import torch, os, pandas
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PrepareDatasets import HepDataset
import numpy as np

class NeuralNet(nn.Module):
  ''' Definition of the model that includes the initilizer and the forward method '''
  def __init__(self, nfeatures):
    super(NeuralNet, self).__init__()

    self.l1 = nn.Linear(nfeatures, 300)
    self.l2 = nn.Linear(300, 20)
    #self.l3 = nn.Linear(100, 100)
    #self.l4 = nn.Linear(100, 100)
    self.l5 = nn.Linear(20, 10)
    self.l6 = nn.Linear(10, 2)

    #self.sm = nn.Softmax(dim=1)
    #self.sm = torch.sigmoid

  def forward(self, x):
    x = F.relu(self.l1(x))
    x = F.relu(self.l2(x))
    #x = F.relu(self.l3(x))
    #x = F.relu(self.l4(x))
    x = F.relu(self.l5(x))
    x = self.l6(x)
    return x #self.sm(x)

def Accuracy(model, dl):
   ''' Get the max probability of being signal or background '''
   model.eval()
   correct = 0.
   size = len(dl.dataset)
   with torch.no_grad():
     for batch_i, (xi, yi) in enumerate(dl):
       scores = (model(xi.float()))
       correct += (scores.argmax(1) == yi).type(torch.float).sum().item()
   return correct/size

def Train(model, dataloader, optimizer, mb_size, epochs=10, datatest=None):
  ''' Train function '''
  for epoch in range(epochs):
    for batch_i, (xi, yi) in enumerate(dataloader):
       #model.train()
       scores = model(xi.float())

       #loss = F.cross_entropy(input=scores, target=yi.long())
       lossfun = nn.CrossEntropyLoss() #BCEWithLogitsLoss() #CrossEntropyLoss()#BCELoss()
       loss = lossfun(input=scores, target=yi.long())#.view(-1, 1))

       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
       if ((batch_i+1) % 20 == 0) or (batch_i==0):
         acc = Accuracy(model, dataloader)
         if datatest is not None:
           acctest = Accuracy(model, datatest)
         print(f'Epoch: {epoch}[{batch_i+1}], loss: {loss.item()}, accuracy: {acc}' + (f', accuracy test {acctest}' if datatest is not None else ''))
    
def EvaluateModelForDataset(model, dataset):
  ''' Get weights and labels for applying model to dataset -- WARNING: applies softmax to model output!! '''
  model.eval()
  tot_prob = np.array([])
  tot_lab  = np.array([])
  with torch.no_grad():
    for data, lab in dataset:
      prob = model(data.float())
      prob = F.softmax(prob, dim=1)[:,1]
      tot_prob = np.append(tot_prob, prob)
      tot_lab  = np.append(tot_lab, lab)
  return tot_prob, tot_lab

def EvaluateModelForArrays(model, arr):
  ''' This function is meant to be used in coffea analysis. The order of the arrays is important! '''
  df = pandas.DataFrame()
  for i, br in enumerate(arr):
   df.insert(0, str(i), br.to_numpy())   
  df.insert(0, 'label', np.ones(len(df)))
  hepdata = DataLoader(HepDataset(df), 2048, shuffle=False)
  return EvaluateModelForDataset(model, hepdata)

def GetSigBkgProb(model, dataset):
  ''' Get signal and background scores for events in dataset '''
  prob, lab = EvaluateModelForDataset(model, dataset)
  print(prob)
  mask = np.where(lab==1, True, False)
  sigprob = prob[mask]
  bkgprob = prob[~mask]
  return sigprob, bkgprob

