# python ljets/test.py --path histos/2jun2022_btagM_jet25/

from config import *
from QCD import *
from cafea.modules.fileReader import *

categories = {'level':level, 'channel':ch}
if var is None: var = 'minDRjj'

plt = plotter(path, prDic=processDic, bkgList=bkglist, colors=colordic, lumi=lumi)
name = GetFileNameFromPath(path)

qcd = QCD(path, prDic=processDic, bkglist=bkglist, lumi=lumi) #, categories=categories)
hQCD = qcd.GetQCD(var)
hdampup,hdampdo = GetModSystHistos(path, 'TT_hdamp', 'hdamp', var=var)
tuneup , tunedo = GetModSystHistos(path, 'TT_UE', 'UE', var=var)
plt.AddExtraBkgHist([hQCD, hdampup, hdampdo, tuneup, tunedo], add=True)
RebinVar(plt, var)
plt.SaveCombine(var, '%s_%s_%s'%(var, categories['channel'], categories['level']), categories=categories)



