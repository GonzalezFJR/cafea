from config import *
#from PDFscaleUncertainties import *

from cafea.modules.CreateDatacardFromRootfile import Datacard
outpath = '/nfs/fanae/user/juanr/CMSSW_10_2_13/src/tt5TeV/ljets/15jun2022/'
fname = path

oname = output 
if oname is None:
  oname = path
  if oname.endswith('.root'): oname = oname[:-5]
  if '/' in oname: oname[oname.rfind('/')+1:]
if not oname.endswith('.txt'): oname += '.txt'

lumiUnc = 0.015
bkg =  ['tW', 'WJets', 'QCD', 'DY']
norm = [0.2, 0.2, 0.2, 0.2]
signal = 'tt'
systList = ['lepSF', 'btagSF', 'FSR', 'ISR', 'hdamp', 'UE', 'JES']#, 'trigSF', 'Scales', 'PDF', 'Prefire']
d = Datacard(fname, signal, bkg, lumiUnc, norm, systList, nSpaces=12)
#d.AddExtraUnc('prefiring', 0.014, ['tt', 'tW', 'WJets', 'DY'])

#pdf   = Get1bPDFUnc(  path, categories=categoriesPDF, sample='TTTo2L2Nu', doPrint=False)
#scale = Get1binScaleUnc(path, categories=categoriesPDF, sample='TTTo2L2Nu', doPrint=False)

d.AddExtraUnc('PDF', 0.007, signal)
d.AddExtraUnc('Scales', 0.002, signal)
d.AddExtraUnc('trigSF', 0.01, signal)
d.AddExtraUnc('Prefire', 0.01, signal)
d.SetOutPath(outpath)
d.Save(oname)
