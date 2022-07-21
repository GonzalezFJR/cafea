from config import *
from QCD import *

qcd = QCD(path, prDic=processDic, bkglist=bkglist, lumi=lumi)
plt = plotter(path, prDic=processDic, bkgList=bkglist, colors=colordic, lumi=lumi, var=var)
plt.SetLumi(lumi, "pb$^{-1}$", "5.02 TeV")

def Draw(var, categories, output=None, label='', outpath='temp/', doQCD=False, doRatio=True):
  #doQCD = False
  if not CheckHistoCategories(plt.hists[var], categories):
    print(f'Nop... [{var}] cat = ', categories)
    return
  plt.ResetExtraBkg()
  plt.SetRatio(doRatio)
  plt.plotData = doData
  plt.SetCategories(categories)
  label = GetChLab(categories['channel']) + GetLevLab(categories['level']) 
  plt.SetRegion(label)
  plt.SetOutpath(outpath)
  plt.SetOutput(output)
  plt.SetLogY()
  rebin = None; xtit = ''; ytit = ''
  if doQCD: 
    hqcd = qcd.GetQCD(var, categories)

  b0 = None; bN = None
  if var in ['minDRjj', 'minDRuu']:
    b0 = 0.4; bN = 2.0
  elif var in ['medianDRjj']:
    b0 = 1; bN = 4.0
  #elif var in ['ht']:
  #  b0 = 2
  elif var in ['st']:
    b0 = 100; bN = 600;
  elif var in ['sumallpt']:
    b0 = 0; bN = 200
    xtit = '$\sum_\mathrm{j,\ell}\,\mathrm{p}_{T}$ (GeV)'
  elif var in ['DNNscore']:
    b0 = 2;

  if b0 is not None:
    plt.SetRebin(var, b0, bN, includeLower=True, includeUpper=True)
    if doQCD: hqcd = Rebin(hqcd, var, b0, bN, includeLower=True, includeUpper=True)
    

  if doQCD: plt.AddExtraBkgHist(hqcd)
  aname = None
  if   var == 'l0pt': aname = 'lep0pt'
  elif var == 'l0eta': aname = 'lep0eta'
  plt.Stack(var, xtit=xtit, ytit=ytit, aname=aname, dosyst=True)
  #plt.PrintYields('counts')

def Print2lplots():
  for c in ['em', 'ee', 'mm']:
    for l in ['incl', 'g2jets']:
      outp = outpath+'/2l/'+c+'/'+l+'/'
      cat = {'channel':c, 'level':l, 'syst':'norm'}
      for var in ['l0pt','counts', 'njets', 'nbtags', 'ht', 'met', 'j0pt', 'j0eta', 'l0eta', 'invmass', 'invmass2']:
        if l=='incl' and var in ['j0pt', 'j0eta']: continue
        outname = "%s_%s_%s"%(var, c, l)
        Draw(var, cat, outname, outpath=outp)

def Print1lplots():
  outp = outpath+'/1l/'
  for c in ['e', 'm', ['e','m']]: #, 'e_fake', 'm_fake']:
    doQCD = not 'fake' in c
    doRatio = not 'fake' in c
    #for l in ['incl', 'g2jets', 'g4jets', '0b', '1b', '2b', '2j1b', '3j1b', '3j2b', '4j1b', '4j2b', 'g5j1b', 'g5j2b']:
    for l in ['2j1b', '3j1b', '3j2b', '4j1b', '4j2b', 'g5j1b', 'g5j2b']:
      cat = {'channel':c, 'level':l}#, 'syst':'norm'}
      clab = c if not isinstance(c, list) else 'l'
      outp = outpath+'/1l/'+clab+'/'+l+'/'
      for var in ['DNNscore']:#['ht', 'st', 'counts', 'njets', 'nbtags', 'met', 'j0pt', 'j0eta', 'ept', 'eeta', 'mpt', 'meta','mjj', 'mt', 'ptjj', 'minDRjj', 'medianDRjj', 'u0pt', 'u0eta', 'minDRuu', 'medianDRuu', 'ptlb', 'ptuu', 'mlb', 'sumallpt', 'dRlb']:#, 'DNNscore']: dRlb
        if l=='incl' and var in ['j0pt', 'j0eta']: continue
        if var == 'DNNscore' and not l in ['2j1b', '3j1b', '3j2b']: continue
        outname = "%s_%s_%s"%(var, clab, l)
        Draw(var, cat, outname, outpath=outp, doQCD=doQCD)


if not var is None:
  categories = { 'channel': ch, 'level' : level}#, 'syst':'norm'}
  Draw(var, categories, output, doQCD=True if ((len(ch) <= 1 and str(*ch) in ['e', 'm']) or (ch[0] in ['e', 'm']) ) else False)


else:
  outpatho = '5jul/'
  outpath = '/nfs/fanae/user/juanr/www/public/tt5TeV/ljets/' + outpatho
  #Print2lplots()
  Print1lplots()


