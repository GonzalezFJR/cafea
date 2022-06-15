from config import *
from QCD import *

qcd = QCD(path, prDic=processDic, bkglist=bkglist, lumi=lumi)
plt = plotter(path, prDic=processDic, bkgList=bkglist, colors=colordic, lumi=lumi, var=var)
plt.SetLumi(lumi, "pb$^{-1}$", "5.02 TeV")

def Draw(var, categories, output=None, label='', outpath='temp/', doQCD=False, doRatio=True):
  if not CheckHistoCategories(plt.hists[var], categories):
    return
  plt.ResetExtraBkg()
  plt.SetRatio(doRatio)
  plt.plotData = doData
  plt.SetCategories(categories)
  label = GetChLab(categories['channel'][0]) + GetLevLab(categories['level'])
  plt.SetRegion(label)
  plt.SetOutpath(outpath)
  plt.SetOutput(output)
  rebin = None
  if doQCD: 
    hqcd = qcd.GetQCD(var, categories)

  if var == 'minDRjj':
    b0 = 0.4; bN = 2.0
    plt.SetRebin(var, b0, bN, includeLower=True, includeUpper=True)
    if doQCD: hqcd = Rebin(hqcd, var, b0, bN, includeLower=True, includeUpper=True)

  if doQCD: plt.AddExtraBkgHist(hqcd)
  aname = None
  if   var == 'l0pt': aname = 'lep0pt'
  elif var == 'l0eta': aname = 'lep0eta'
  plt.Stack(var, xtit='', ytit='', aname=aname, dosyst=True)
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
  for c in ['e', 'm', 'e_fake', 'm_fake']:
    doQCD = not 'fake' in c
    doRatio = not 'fake' in c
    for l in ['g4jets', '0b', '1b', '2b']:
      cat = {'channel':c, 'level':l}#, 'syst':'norm'}
      outp = outpath+'/1l/'+c+'/'+l+'/'
      for var in ['ht', 'counts', 'njets', 'nbtags', 'met', 'j0pt', 'j0eta', 'ept', 'eeta', 'mpt', 'meta','mjj', 'mt', 'ptjj', 'minDRjj']:
        if l=='incl' and var in ['j0pt', 'j0eta']: continue
        outname = "%s_%s_%s"%(var, c, l)
        Draw(var, cat, outname, outpath=outp, doQCD=doQCD)


if not var is None:
  categories = { 'channel': ch, 'level' : level}#, 'syst':'norm'}
  Draw(var, categories, output, doQCD=True if str(*ch) in ['e', 'm'] else False)


else:
  outpath = '/nfs/fanae/user/juanr/www/public/tt5TeV/ljets/' + outpatho
  #Print2lplots()
  Print1lplots()


