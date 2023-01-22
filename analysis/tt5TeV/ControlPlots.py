from config import *
import warnings
from datetime import datetime
warnings.filterwarnings("ignore", category=RuntimeWarning)

if not 'QCD.pkl.gz' in list(os.listdir(path)):
  print('WARNING: QCD file not found in input folder!!!!')
  #exit()

def Draw(plt, var, level, channel, outname, verbose=False):
  categories =  {'level':level, 'channel':channel}
  plt.SetCategories(categories)
  if not CheckHistoCategories(plt.hists[var], categories):
    if verbose: print(f'  > Skipping [{var}] cat = ', categories)
    return
  label = GetChLab(categories['channel']) + GetLevLab(categories['level']) 
  xtit = RebinVar(plt, var, level)
  plt.SetRegion(label)
  plt.SetOutput(outname)
  #plt.SetLogY()
    
  aname = None
  if   var == 'l0pt': aname = 'lep0pt'
  elif var == 'l0eta': aname = 'lep0eta'
  plt.Stack(var, xtit=xtit, ytit=None, aname=aname, dosyst=True, verbose=verbose)

def DrawPar(L): 
  plt, var, lev, chan, outname = L
  Draw(plt, var, lev, chan, outname)

outpath = baseweb+datatoday+'/ContorlPlots/'
print(' >> Output = ', outpath)
if not os.path.isdir(outpath): os.makedirs(outpath)

print(' >> Loading histograms! This might take a while...')
plt = plotter(path, prDic=processDic, bkgList=bkglist, colors=colordic, lumi=lumi)
plt.SetLumi(lumi, "pb$^{-1}$", "5.02 TeV")
plt.SetRatio(True)
plt.plotData = True
plt.SetOutpath(outpath)

variables = ['met', 'medianDRjj', 'ht', 'st', 'counts', 'njets', 'nbtags', 'met', 'j0pt', 'j0eta', 'ept', 'eeta', 'mpt', 'meta','mjj', 'mt', 'ptjj', 'minDRjj', 'medianDRjj', 'u0pt', 'u0eta', 'minDRuu', 'medianDRuu', 'ptlb', 'ptuu', 'mlb', 'sumallpt', 'dRlb', 'MVAscore', 'metnocut']
levels = ['g4jets', '3j1b', '3j2b', '4j1b', '4j2b', 'g5j1b', 'g5j2b']
channels = ['e', 'm']
systematics = ['ISR', 'FSR', 'btagSF', 'eleSF', 'muonSF', 'JES', 'prefire']
plt.SetSystematics(systematics)
inputs = []

if __name__=="__main__":
  nplots = 0;
  if not var is None:
    clab = ch if not isinstance(ch, list) and not len(list)>1 else 'l'
    outname = "custom_%s_%s_%s"%(var, clab, level)
    Draw(plt, var, level, ch, outname)
  else:
    tot = len(channels)*len(levels)*len(variables)
    print('Creating a total of %i inputs...'%tot)
    progress = float(nplots)/tot*100
    for c in channels: 
      for l in levels: 
        cat = {'channel':c, 'level':l}
        clab = c if not isinstance(c, list) else 'l'
        for var in variables:
          nplots += 1
          if l=='incl' and var in ['j0pt', 'j0eta']: continue
          outname = "%s_%s_%s"%(var, clab, l)
          inputs.append([plt, var, l, c, outname])
          #Draw(plt, var, l, c, outname, verbose=False)
          #print("\r[{:<100}] {:.2f} % {:.0f}/{:.0f}".format('#' * int(progress), progress, nplots, tot),end='')

    if nSlots < 4: nSlots = 4
    if tot/nSlots > 40: print("WARNING: you are probably plotting to many plots for the amount of slots!! Total plots: %i, nSlots: %i. Increase the number of slots."%(tot,nSlots))
    print('Running with %i slots...'%nSlots)
    from multiprocessing import Pool, Manager
    pool = Pool(nSlots)
    pool.map(DrawPar, inputs)
    pool.close()
    pool.join()
