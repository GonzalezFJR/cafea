# python ljets/test.py --path histos/2jun2022_btagM_jet25/

from config import *
from QCD import *

categories = {'level':level, 'channel':ch}
if var is None: var = 'ht'


plt = plotter(path, prDic=processDic, bkgList=bkglist, colors=colordic, lumi=lumi)
if doData and  str(*ch) in ['e', 'm']:
  qcd = QCD(path, prDic=processDic, bkglist=bkglist, lumi=lumi)
  plt.AddExtraBkgHist(qcd.GetQCD(var))
PrintHisto(plt.GetHistogram(var))
exit()




