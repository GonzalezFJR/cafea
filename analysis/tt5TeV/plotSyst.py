'''
 Usage: python ljets/plotSyst.py -p histos/2jun2022_btagM_jet25/TT.pkl.gz
'''

from config import *
from QCD import *

plt = plotter(path, prDic=processDic, bkgList=bkglist, colors=colordic, lumi=lumi)
plt.SetOutpath(outpath)
plt.SetLumi(lumi, "pb$^{-1}$", "5.02 TeV")
plt.SetYRatioTit('Ratio')
plt.SetOutput(output)
hdampup,hdampdo = GetModSystHistos(path, 'TT_hdamp', 'hdamp', var=var)
tuneup , tunedo = GetModSystHistos(path, 'TT_UE', 'UE', var=var)
if doData:
  hQCD = GetQCDbkg(var, categories)
  plt.AddExtraBkgHist([hQCD, hdampup, hdampdo, tuneup, tunedo], add=True)
else: 
  plt.AddExtraBkgHist([hdampup, hdampdo, tuneup, tunedo], add=True)


def DrawComp(var, process, categories, labels=[], colors=[], lineStyle=[], scale=[]):
  plt.DrawComparison(var, process, categories, labels, colors, lineStyle, scale)

def DrawSyst(var, process, syst): 
  colors = ['k', 'r', 'b']
  normdict = {'channel':ch, 'level':level, 'syst':'norm'}
  h = plt.GetHistogram(var, process, normdict)
  systlist = [x.name for x in h.identifiers('syst')]
  upsyst = syst+'Up'; dosyst = syst+'Down';
  if not upsyst in
  
  dic = [{'channel':ch, 'level':level, 'syst':'norm'}, {'channel':ch, 'level':level, 'syst':syst+'Up'}, {'channel':ch, 'level':level, 'syst':syst'Do} ]

outpath = '/nfs/fanae/user/juanr/www/public/tt5TeV/ljets/' + outpatho
cat = [{'channel':'m', 'level':'incl', 'syst':'norm'}, {'channel':'m', 'level':'incl', 'syst':'JESUp'}, {'channel':'m', 'level':'incl', 'syst':'JESDo'}]
if var is None: var = 'ht'
pr = 'tt'
labels = ['Nominal', 'up', 'down']
colors = ['k', 'r', 'b']

DrawComp(var, pr, cat, labels, colors)

