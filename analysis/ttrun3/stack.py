from config import *

def Draw(var, categories, output=None, label='', outpath='temp/', doRatio=True):
  plt = plotter(path, prDic=processDic, bkgList=bkglist, colors=colordic, lumi=lumi, var=var)
  if not CheckHistoCategories(plt.hists[var], categories):
    print("Nope")
    return
  plt.SetRatio(doRatio)
  plt.SetOutpath(outpath)
  plt.plotData = doData
  plt.SetLumi(lumi, "pb$^{-1}$", '13.6 TeV')
  if var in ['counts', 'l0pt','ept', 'mpt', 'l0eta', 'eeta', 'meta', 'njets','nbtagsl','nbtagsm']:
    categories['sign'] = 'OS'
  plt.SetCategories(categories)
  #plt.SetDataName('Pseudodata')
  label = (GetChLab(categories['channel']) if isinstance(categories['channel'], str) else GetChLab(categories['channel'][0]) ) + GetLevLab(categories['level'])
  #AddLabel(self, x, #y, text, options={}):
  plt.SetRegion(label)
  plt.SetOutput(output)

  b0 = None; bN = None
  if   var == 'deltaphi':
    b0 = 2
  elif var == 'invmass':
    b0 = 2

  if b0 is not None:
    plt.SetRebin(var, b0, bN, includeLower=True, includeUpper=True)
  
  #plt.SetSystematics(syst=['LepSF_muon', 'LepSF_elec', 'FSR', 'ISR'])#, 'PU', 'JES', 'trigSF', 'FSR', 'ISR'])#, 'lepSF']) # FSR, ISR, JES, lepSF, trigSF
  plt.Stack(var, xtit='', ytit='', dosyst=True)

  #plt.PrintYields('counts')

def Print2lplots():
  for c in ['ee','em', 'mm']:
    for l in ['dilep', 'g2jets', 'g2jetsg1b']:
      outp = outpath+'/'+l+'/'
      cat = {'channel':c, 'level':l}#, 'syst':'norm'}
      for var in ['ht', 'met', 'j0pt', 'j0eta', 'invmass', 'invmass2', 'invmass3']:
        if l=='dilep' and var in ['j0pt', 'j0eta']: continue
        outname = "%s_%s_%s"%(var, c, l)
        Draw(var, cat, outname, outpath=outp)
      for var in ['counts', 'l0pt','ept', 'mpt', 'l0eta', 'eeta', 'meta', 'njets','nbtagsl','nbtagsm']:
        outname = "%s_%s_%s"%(var, c, l)
        Draw(var, cat, outname, outpath=outp)

outpath = '/nfs/fanae/user/andreatf/www/private/ttrun3/withLepSF_withoutJECPU/' + outpatho
if not var is None:
  ch='em'; level='g2jets'
  categories = { 'channel': ch, 'level' : level}#, 'syst':'norm'}
  outp = outpath+'/'+level+'/'
  outname = "%s_%s_%skk"%(var, ch, level)
  Draw(var, categories, outname, outpath=outp)


else:
  #Draw('invmass', {'channel':['em'], 'level':'g2jets', 'syst':'norm'}, output='invmass_em_g2jets', outpath=outpath)
  Print2lplots()

