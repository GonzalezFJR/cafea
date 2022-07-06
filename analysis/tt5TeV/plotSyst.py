'''
 Usage: python ljets/plotSyst.py -p histos/2jun2022_btagM_jet25/TT.pkl.gz
'''

from config import *
from QCD import *

outpath = '/nfs/fanae/user/juanr/www/public/tt5TeV/ljets/systematics/' + outpatho

plt = plotter(path, prDic=processDic, bkgList=bkglist, colors=colordic, lumi=lumi, var=var)
plt.SetOutpath(outpath)
plt.SetLumi(lumi, "pb$^{-1}$", "5.02 TeV")
plt.SetYRatioTit('Ratio')
plt.SetOutput(output)
if doData:
  hQCD = GetQCDbkg(var, categories)
  plt.AddExtraBkgHist([hQCD])


def DrawAllForVar(var):
  outpath = '/nfs/fanae/user/juanr/www/public/tt5TeV/ljets/systematics/'
  plt.SetOutpath(outpath)
  channel = ['e', 'm']
  level = ['g4jets', '0b', '1b', '2b']
  syst = ['JES']

  b0 = None; bN = None
  if var == 'minDRjj':
    b0 = 0.4; bN = 2.0
  elif var == 'ht':
    b0 = 150; bN = 400;
  #elif var == 'njets':
  #  b0 = 4; bN = 

  if b0 is not None:
    plt.SetRebin(var, b0, bN, includeLower=True, includeUpper=True)

  for l in level:
    for c in channel:
      for s in syst:
        outname = '%s_%s_%s_%s_%s'%(pr, var, c, l, s)
        plt.SetOutput(outname)
        DrawSyst(var, s, pr, c, l)
 

def DrawComp(var, process, categories, labels=[], colors=[], lineStyle=[], scale=[]):
  plt.DrawComparison(var, process, categories, labels, colors, lineStyle, scale)

def DrawSyst(var, syst, process='tt', chan='m', level='g4jets'): 
  colors = ['k', 'r', 'b']
  labels = ['Nominal', '%s up'%syst, '%s down'%syst]
  h = plt.GetHistogram(var)
  systlist = [x.name for x in h.identifiers('syst')]
  upsyst = syst+'Up'; dosyst = syst+'Down' if syst+'Down' in systlist else syst+'Do';
  if not upsyst in systlist or not dosyst in systlist:
    print (" >> WARNING: No syst %s found -- List: "%(syst), systlist)
    return
  selec = [{'channel':chan, 'level':level, 'syst':'norm'}, {'channel':chan, 'level':level, 'syst':upsyst}, {'channel':chan, 'level':level, 'syst':dosyst} ]
  DrawComp(var, process, selec, labels, colors)

#cat = [{'channel':'m', 'level':'incl', 'syst':'norm'}, {'channel':'m', 'level':'incl', 'syst':'JESUp'}, {'channel':'m', 'level':'incl', 'syst':'JESDo'}]
#if var is None: var = 'ht'
#pr = 'tt'
#labels = ['Nominal', 'up', 'down']
#colors = ['k', 'r', 'b']

pr = 'tt'
if not var is None:
  if systch == 'hdamp':
    hdampup,hdampdo = GetModSystHistos(path, 'TT_hdamp', 'hdamp', var=var)
    plt.AddExtraBkgHist([hdampup, hdampdo], add=True)
  elif systch == 'UE':
    tuneup , tunedo = GetModSystHistos(path, 'TT_UE', 'UE', var=var)
    plt.AddExtraBkgHist([tuneup, tunedo], add=True)
  DrawSyst(var, systch, pr, ch, level)
  exit()

else:
  var = ['minDRjj', 'ht']
  for v in var:
    DrawAllForVar(v)
  exit()

