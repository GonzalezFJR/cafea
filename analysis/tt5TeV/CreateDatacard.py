from config import *
#from PDFscaleUncertainties import *

from cafea.modules.CreateDatacardFromRootfile import Datacard
from cafea.plotter.plotter import GetHisto

if  not '/' in inputFile: outpath = './'
else: outpath = inputFile[:inputFile.rfind('/')]

channels = ['e', 'm']
levels = ['3j1b', '3j2b', '4j1b', '4j2b', 'g5j1b', 'g5j2b']
def GetChanLevFromName(fname):
  # medianDRjj_e_g5j1b.root
  inputs = fname[fname.rfind('/')+1:].replace('.root', '').split('_')
  chan = None; lev = None
  for i in inputs:
    if i in channels: chan = i
    if i in levels: lev = i
  if chan is None or lev is None:
    print("WARNING: could not get channel or level from file name: %s"%fname)
  return chan, lev

def GetModUnc(path, chan, lev):
  nbin = channels.index(chan) *len(levels) + levels.index(lev)
  if os.path.isfile(path + 'masterhistos/master.pkl.gz'):
    # print('Loading masterhistos from %s'%path)
    histo = GetHisto(path + 'masterhistos/master.pkl.gz', 'master').integrate('process', 'tt')
    nominal = histo.integrate('syst', 'norm').values()[()][nbin]
    # PDF and scale
    pdfUp = histo.integrate('syst', 'PDFUp').values()[()][nbin]
    pdfDown = histo.integrate('syst', 'PDFDown').values()[()][nbin]
    pdf = (abs(pdfUp-nominal) + abs(pdfDown-nominal))/2/nominal
    scaleUp = histo.integrate('syst', 'ScaleUp').values()[()][nbin]
    scaleDown = histo.integrate('syst', 'ScaleDown').values()[()][nbin]
    scales = (abs(scaleUp-nominal) + abs(scaleDown-nominal))/2/nominal
    # hdamp and UE
    tot = sum(histo.integrate('syst', 'norm').values()[()])
    hdampUp = sum(histo.integrate('syst', 'hdampUp').values()[()])
    hdampDown = sum(histo.integrate('syst', 'hdampDown').values()[()])
    hdamp = max(abs(hdampUp-tot), abs(hdampDown-tot))/tot
    UEUp = sum(histo.integrate('syst', 'UEUp').values()[()])
    UEDown = sum(histo.integrate('syst', 'UEDown').values()[()])
    UE = max(abs(UEUp-tot),abs(UEDown-tot))/tot
    return pdf, scales, hdamp, UE
  else:
    print("WARNING: please provide master histograms to take modeling uncertaintes... for now, returning hardcoded values")
    pdf = 0.007
    scales = 0.002
    hdamp = 0.007
    UE = 0.005
  return pdf, scales, hdamp, UE


def CreateDatacard(fname, outpath=outpath, oname=output):
  chan, lev = GetChanLevFromName(fname)
  if oname is None:
    oname = fname[fname.rfind('/')+1:] if '/' in fname else fname
    if oname.endswith('.root'): oname = oname[:-5]
    if '/' in oname: oname[oname.rfind('/')+1:]
  oname = 'dat_'+oname
  if not oname.endswith('.txt'): oname += '.txt'
  
  lumiUnc = 0.015
  bkg =  ['tW', 'WJets', 'QCD', 'DY']
  norm = [0.2, 0.2, 0.3, 0.2]
  signal = 'tt'
  systList = ['muonSF', 'elecSF', 'btagSF', 'FSR', 'ISR', 'JES', 'prefire']# 'hdamp', 'UE', 'trigSF', 'Scales', 'PDF', 'Prefire']
  d = Datacard(fname, signal, bkg, lumiUnc, norm, systList, nSpaces=12, verbose=verbose)
  #d.AddExtraUnc('prefiring', 0.014, ['tt', 'tW', 'WJets', 'DY'])
  
  #pdf   = Get1bPDFUnc(  fname, categories=categoriesPDF, sample='TTTo2L2Nu', doPrint=False)
  #scale = Get1binScaleUnc(fname, categories=categoriesPDF, sample='TTTo2L2Nu', doPrint=False)
  
  pdf, scales, hdamp, UE = GetModUnc(path, chan, lev)
  d.AddExtraUnc('PDF', pdf, signal)
  d.AddExtraUnc('Scales', scales, signal)
  if chan == 'e': d.AddExtraUnc('trigElecSF', 0.015, signal)
  else          : d.AddExtraUnc('trigMuonSF', 0.01, signal)
  d.AddExtraUnc('hdamp', hdamp, signal)
  d.AddExtraUnc('UE', UE, signal)
  #d.AddExtraUnc('Prefire', 0.01, signal)
  d.SetOutPath(outpath)
  print('outpath = %s'%outpath)
  d.Save(oname)

if inputFile == '': 
  print('Please provide a root file to create datacards from using --inputFile /path/to/inputs/');
  exit(1)

if os.path.isdir(inputFile):
  for d in os.listdir(inputFile):
    if not d.endswith('.root'): continue
    fname = os.path.join(inputFile, d)
    CreateDatacard(fname)
else:
  CreateDatacard(inputFile)
