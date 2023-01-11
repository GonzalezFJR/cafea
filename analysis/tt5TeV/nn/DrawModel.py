import matplotlib
matplotlib.use('template')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time, math, random
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, auc
import sklearn as sk
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from PrepareDatasets import *

def PlotROC(trainprob, trainlab, testprob, testlab, oname='ROCcurve', show=False):
  plt.figure()
  fpr, tpr, _   = roc_curve(testlab, testprob) #extract true positive rate and false positive rate
  fpr2, tpr2, _ = roc_curve(trainlab,trainprob) #extract true positive rate and false positive rate
  plt.figure(figsize=(8,7))
  plt.rcParams.update({'font.size': 15}) #Larger font size
  AUC  = roc_auc_score(testlab , testprob )
  AUC2 = roc_auc_score(trainlab, trainprob)
  plt.plot(fpr, tpr, color='crimson', lw=2, label='ROC curve test (area = {0:.4f})'.format(AUC))
  plt.plot(fpr2, tpr2, color='blue', lw=1, label='ROC curve train (area = {0:.4f})'.format(AUC2))
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.05])
  plt.ylim([0.0, 1.05])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.legend(loc="lower right")
  plt.grid(True)
  plt.savefig(oname+'.png',format='png')
  print("Saved figure: ", oname+'.png')
  if show: plt.show()

def PlotHisto(prob, lab, oname='histo', show=False):
  plt.figure()
  back = prob[lab==0]
  sign = prob[lab==1]
  plt.figure(figsize=(8,5))
  plt.rcParams.update({'font.size': 15}) 
  plt.hist(back, 20, color='blue', edgecolor='blue', lw=2, label='background', alpha=0.3, density=True)
  plt.hist(sign, 20, color='red', edgecolor='red', lw=2, label='signal', alpha=0.3, density=True)
  plt.xlim([0.,1.])
  plt.xlabel('Event probability of being classified as signal')
  plt.legend(loc="upper right")
  plt.grid(True)
  plt.savefig(oname+'.png',format='png')
  print("Saved figure: ", oname+'.png')
  if show: plt.show()


