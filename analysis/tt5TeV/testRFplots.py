from cafea.plotter.DataGraphs import *

path = '/home/xuan/python/cafea/cafea/forTraining/'
signal = ["TTPS"]
bkg = ["WJetsToLNu", "W0JetsToLNu", "W1JetsToLNu", "W2JetsToLNu", "W3JetsToLNu"]

datadict = {'signal': signal, 'bkg': bkg}

lev = '3j1b'
var = [f'{lev}_medianDRjj', f'{lev}_mlb', f'{lev}_mjj', f'{lev}_drlb']
outpath = './plots/'
if not os.path.exists(outpath):
    os.makedirs(outpath)

columns = ['3j1b_ht', '3j1b_st', '3j1b_sumAllPt', '3j1b_j0pt', '3j1b_u0pt', '3j1b_ptjj', '3j1b_mjj', '3j1b_medianDRjj', '3j1b_minDRjj', '3j1b_mlb', '3j1b_ptsumveclb', '3j1b_drlb', '3j1b_druu', '3j1b_druumedian', '3j1b_muu', '3j1b_ptuu']

### Get the dataframe and draw some characteristic plots
df = BuildPandasDF(path, datadict, columns, even=True)
DrawPairPlots(df, columns, savefig=f'{outpath}pairplots_{lev}.png')
DrawHistos(df, columns, savefig=f'{outpath}histos_{lev}.png')
DrawBoxPlots(df, columns, savefig=f'{outpath}boxplots_{lev}.png')
DrawCorrelationMatrix(df, columns, savefig=f'{outpath}correlation_{lev}.png')

# Load the model
import pickle as pkl
modelpath = '/home/xuan/python/cafea/cafea/models/rf3j1b_200_6_allvariables_p2v2.pkl'
model = pkl.load(open(modelpath, 'rb'))

### Draw ranking
DrawRanking(model, df, columns, savefig=f'{outpath}ranking_{lev}.png')

# Confusion matrix
y_true = df['label'].values
y_pred = model.predict(df[columns])
ConfusionMatrix(y_true, y_pred, savefig=f'{outpath}confusion_{lev}.png')

# Histogram of probabilities for signal and background and ROC curve
df_train, df_test = BuildPandasDF(path, datadict, columns, even=True, train_test_split=0.2)
train_true = df_train['label'].values
train_pred = model.predict_proba(df_train[columns])[:,1]
test_true = df_test['label'].values
test_pred = model.predict_proba(df_test[columns])[:,1]

DrawROC(train_true, train_pred, test_true, test_pred, savefig=f'{outpath}roc_{lev}.png')
DrawSigBkgHisto(train_true, train_pred, test_true, test_pred, savefig=f'{outpath}sigbkg_{lev}.png')