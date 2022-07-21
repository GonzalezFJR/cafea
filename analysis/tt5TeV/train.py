from PrepareDatasets import *
from NN import *
import datetime
torch.set_num_threads(64)

### Define path, signa, bkg, variables...
path = '../../histos5TeV/'
signal = "TT, TTPS"
bkg = "WJetsToLNu, W0JetsToLNu, W1JetsToLNu, W2JetsToLNu, W3JetsToLNu"
#var = ['2j1b_ht', '2j1b_sumAllPt', '2j1b_leta', '2j1b_j0pt', '2j1b_mjj', '2j1b_medianDRjj', '2j1b_drlb']

var = ['3j1b_ht', '3j1b_st', '3j1b_sumAllPt', '3j1b_leta', '3j1b_j0pt', '3j1b_j0eta', '3j1b_medianDRjj', '3j1b_minDRjj', '3j1b_mt', '3j1b_ptsumveclb']#, '3j1b_mlb', '3j1b_mt', '3j1b_ptsumveclb']#, '3j1b_drlb']#, '3j1b_druu', '3j1b_druumedian', '3j1b_muu', '3j1b_ptuu'] '3j1b_u0pt', '3j1b_u0eta',


#path = '../histos/'
#signal = 'TT_incl'
#bkg = 'WJetsToLNu_incl, W0JetsToLNu_incl, W1JetsToLNu_incl, W2JetsToLNu_incl, W3JetsToLNu_incl'
#var = ['A_njets', 'A_nbtags', 'A_ht', 'A_sumAllPt', 'A_leta', 'A_j0pt', 'A_mjj', 'A_medianDRjj']

### Training parameters
trainFrac = 0.8
batch_size = 1024
learning_rate = 0.003
epochs = 200

### Get the data
traindl, testdl = PrepareData(path, signal, bkg, var, trainFrac, batch_size, nData=-1)

### Create the model
model = NeuralNet(len(var)).to('cpu')

### Create the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # stocastic grade descend

### train!
Train(model, traindl, optimizer, batch_size, epochs, datatest=testdl)

### Save the model
date = datetime.datetime.strftime(datetime.datetime.now(), "%d%h%y_%Hh%Mm")
outname = 'model_' + date + '.pkl'
if os.path.isfile(outname):  os.system("mv %s %s.old"%(outname, outname))
with open(outname, 'wb') as f:
  pickle.dump(model, f)
print(" >> Model saved to: ", outname)


#sig, bkg = GetSigBkgProb(model, testdl)


