import pickle
from DrawModel import *
from NN import EvaluateModelForDataset

model_name = 'models/model_04Jul22_07h04m.pkl'

### Open the model
with open(model_name, 'rb') as f:
  model = pickle.load(f)
  
path = '/mnt_pool/c3_users/user/juanr/cafea/histos5TeV/forTraining/'
signal = "TTPS"
bkg = "WJetsToLNu"
#var = ['A_ht', 'A_sumAllPt', 'A_leta', 'A_j0pt', 'A_mjj', 'A_medianDRjj', 'A_drlb']
ch = '3j2b'
var = [ch + '_ht', ch + '_sumAllPt', ch + '_leta', ch + '_j0pt', ch + '_mjj', ch + '_medianDRjj', ch + '_drlb']

trainFrac = 0.8
batch_size = 2048
learning_rate = 0.001
epochs = 200

traindl, testdl = PrepareData(path, signal, bkg, var, trainFrac, batch_size, nData=-1)
print(model)
print(traindl)
prob_test , lab_test  = EvaluateModelForDataset(model, testdl )
prob_train, lab_train = EvaluateModelForDataset(model, traindl)

PlotROC(prob_train, lab_train, prob_test, lab_test)
PlotHisto(prob_train, lab_train)
