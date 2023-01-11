from PrepareDatasets import *
from NN import *
import datetime

### Define path, signa, bkg, variables...
path = '../histos/tt5TeV/forNN/'
signal = "TT, TTPS"
bkg = "WJetsToLNu, W0JetsToLNu, W1JetsToLNu, W2JetsToLNu, W3JetsToLNu"
var = ['A_ht', 'A_sumAllPt', 'A_leta', 'A_j0pt', 'A_mjj', 'A_medianDRjj', 'A_drlb']


#path = '../histos/'
#signal = 'TT_incl'
#bkg = 'WJetsToLNu_incl, W0JetsToLNu_incl, W1JetsToLNu_incl, W2JetsToLNu_incl, W3JetsToLNu_incl'
#var = ['A_njets', 'A_nbtags', 'A_ht', 'A_sumAllPt', 'A_leta', 'A_j0pt', 'A_mjj', 'A_medianDRjj']

### Training parameters
trainFrac = 0.8
batch_size = 256
learning_rate = 0.005
epochs = 300

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


