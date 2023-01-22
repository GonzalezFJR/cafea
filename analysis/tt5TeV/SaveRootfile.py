# python ljets/test.py --path histos/2jun2022_btagM_jet25/

from config import *
from cafea.modules.fileReader import *

if var is None: var = 'medianDRjj'

plt = plotter(path, prDic=processDic, bkgList=bkglist, colors=colordic, lumi=lumi, var=var)
name = GetFileNameFromPath(path)

hdampup,hdampdo = GetModSystHistos(path, 'TT_hdamp', 'hdamp', var=var)
tuneup , tunedo = GetModSystHistos(path, 'TT_UE', 'UE', var=var)
plt.AddExtraBkgHist([hdampup, hdampdo, tuneup, tunedo], add=True)
plt.SetOutpath(path + '/combineFiles/')
plt.SetVerbose(verbose)
if not doData:
  plt.SetDataName('Asimov')
RebinVar(plt, var)

def SaveFile(level, channel):
  categories = {'level':level, 'channel':channel}
  lev  = categories['level'  ] if not isinstance(categories['level'  ], list) else categories['level'  ][0];
  chan = categories['channel'] if not isinstance(categories['channel'], list) else categories['channel'][0];
  outname = '%s_%s_%s'%(var, chan, lev)
  plt.SaveCombine(var, outname, categories=categories)

if __name__=="__main__":
  if not isinstance(ch, list): ch = [ch]
  if not isinstance(level, list): level = [level]
  for l in level:
    for c in ch:
      SaveFile(l, c)
