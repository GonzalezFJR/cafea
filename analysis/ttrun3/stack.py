from config import *
from PDFscaleUncertainties import *

def Draw(var, categories, output=None, label='', outpath='temp/', doRatio=True):
  plt = plotter(path, prDic=processDic, bkgList=bkglist, colors=colordic, lumi=lumi, var=var)
  if not CheckHistoCategories(plt.hists[var], categories):
    print("Nope")
    return
  plt.SetRatio(doRatio)
  plt.SetOutpath(outpath)
  plt.plotData = doData
  plt.SetLumi(lumi, "pb$^{-1}$", '13.6 TeV')
  plt.SetRatioRange(0.75,1.25)
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
  

  #PrintHisto(plt.hists[var])

  ############## Uncertainties
  # Muon and electron efficiencies --> From SF, already included in the histograms, automatic
  # FSR, ISR -> From weights, already included in the histograms, automatic
  # Pileup -> Not yet...

  # JES -> From JES tt sample without JEC (but, for the moment, flat)
  jecs = GetJECSystHistos(path, 'variations/TTTo2L2Nu_withJEC', var='counts', categories=categories)[0]

  # hdamp -> Flat on tt 
  hdampup,hdampdo = GetModSystHistos(path, 'TTTo2L2Nu_hdamp', 'hdamp', var='counts')
  for c in categories:
    hdampup = hdampup.integrate(c, categories[c])
    hdampdo = hdampdo.integrate(c, categories[c])
  hdampup = hdampup.integrate('syst', 'hdampUp')
  hdampdo = hdampdo.integrate('syst', 'hdampDown')
  #print('hdamp', hdampMean)
  # For the moment, hard-coded:
  hdamp = 0.0082 # 0.82%


  # PDF and scales -> From weights, flat on tt
  categoriesPDF = {'channel':'em', 'level': 'g2jets'}
  pdf   = Get1bPDFUnc(  path, categories=categoriesPDF, sample='TTTo2L2Nu', doPrint=False)
  scale = Get1binScaleUnc(path, categories=categoriesPDF, sample='TTTo2L2Nu', doPrint=False)

  totFlatTTunc = np.sqrt(jecs**2 + hdamp**2 + pdf**2 + scale**2)

  # Background uncertainties
  #plt.SetNormUncDict({'tt':totFlatTTunc, 'tW':0.15,'semilep':0.2,'WJets':0.3, 'DY':0.2, 'Diboson':0.3})

  plt.SetSystematics(syst=['ISR', 'FSR','lepSF_muon', 'lepSF_elec','norm','stat'])
  plt.Stack(var, xtit='', ytit='', dosyst=True)


  plt.PrintYields('counts')

outpath = outpatho
outpath = '/nfs/fanae/user/andreatf/www/private/ttrun3/withLepSF_withoutJECPU_metfiltersOverlap_correctLepSF/' 

def Print2lplots():
  for c in ['ee','em', 'mm']:
    for l in ['dilep', 'g2jets', 'g2jetsg1b']:
      outp = outpath+'/'+l+'/'
      cat = {'channel':c, 'level':l}
      for var in ['ht', 'met', 'j0pt', 'j0eta', 'invmass', 'invmass2', 'invmass3']:
        if l=='dilep' and var in ['j0pt', 'j0eta']: continue
        outname = "%s_%s_%s"%(var, c, l)
        Draw(var, cat, output=outname, outpath=outp)
      for var in ['counts', 'l0pt','ept', 'mpt', 'l0eta', 'eeta', 'meta', 'njets','nbtagsl','nbtagsm']:
        outname = "%s_%s_%s"%(var, c, l)
        Draw(var, cat, outname, outpath=outp)



if not var is None:
  ch='em'; level='g2jets'
  categories = { 'channel': ch, 'level' : level}#, 'syst':'norm'}
  outp = outpath+'/'+level+'/'
  outname = "%s_%s_%skk"%(var, ch, level)
  Draw(var, categories, outname, outpath=outp)


else:
  Draw('njets', {'channel':['em'], 'level':'dilep','syst':'norm'}, output='njetsmalnrm', outpath=outpath)
  #Print2lplots()

