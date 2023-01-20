from config import *
from QCD import *
import warnings
from datetime import datetime
warnings.filterwarnings("ignore", category=RuntimeWarning)


def Draw(plt, var, level, channel):
  categories =  {'level':level, 'channel':channel}
  plt.SetCategories(categories)
  if not CheckHistoCategories(plt.hists[var], categories):
    print(f'Nop... [{var}] cat = ', categories)
    return
  label = GetChLab(categories['channel']) + GetLevLab(categories['level']) 
  xtit = RebinVar(plt, var, level)
  plt.SetRegion(label)
  plt.SetOutpath(outpath)
  plt.SetOutput(output)
  #plt.SetLogY()
    
  aname = None
  if   var == 'l0pt': aname = 'lep0pt'
  elif var == 'l0eta': aname = 'lep0eta'
  plt.Stack(var, xtit=xtit, ytit=None, aname=aname, dosyst=True)

now = datetime.now()
dat = str(now.strftime('%d')) + str(now.strftime('%B')).lower()[:3] + str(now.strftime('%Y'))[2:]
outpatho = dat+'/QCD/'
outpath = '/nfs/fanae/user/juanr/www/public/tt5TeV/ljets/' + outpatho
print(' >> Output = ', outpath)
if not os.path.isdir(outpath): os.makedirs(outpath)

print(' >> Loading histograms! This might take a while...')
plt = plotter(path, prDic=processDic_noQCD, bkgList=bkglist_noQCD, colors=colordic, lumi=lumi)
plt.SetLumi(lumi, "pb$^{-1}$", "5.02 TeV")
plt.SetRatio(True)
plt.plotData = True

variables = ['met', 'medianDRjj', 'DNNscore', 'ht', 'st', 'counts', 'njets', 'nbtags', 'met', 'j0pt', 'j0eta', 'ept', 'eeta', 'mpt', 'meta','mjj', 'mt', 'ptjj', 'minDRjj', 'medianDRjj', 'u0pt', 'u0eta', 'minDRuu', 'medianDRuu', 'ptlb', 'ptuu', 'mlb', 'sumallpt', 'dRlb']
levels = ['g4jets', '3j1b', '3j2b', '4j1b', '4j2b', 'g5j1b', 'g5j2b']
channels = ['e_fake', 'm_fake']


if __name__=="__main__":
  if not var is None:
    Draw(plt, var, level, ch)
  else:
    for c in channels: 
      for l in levels: 
        cat = {'channel':c, 'level':l}
        clab = c if not isinstance(c, list) else 'l'
        outp = outpath+'/1l/'+clab+'/'+l+'/'
        for var in variables:
          if l=='incl' and var in ['j0pt', 'j0eta']: continue
          outname = "%s_%s_%s"%(var, clab, l)
          Draw(var, cat, outname, outpath=outpath)

