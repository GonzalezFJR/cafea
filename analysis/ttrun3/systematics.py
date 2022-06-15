from config import *
from cafea.plotter.xsec import *
from cafea.modules.fileReader import *
from PDFscaleUncertainties import *

names = {
  'lepSF' : 'Lepton efficiences',
  'trigSF' : 'Trigger efficiencies',
  'JES' : 'Jet energy scale',
  'UE' : 'Underlying event',
  'hdamp' : '$h_\mathrm{damp}$',
  'ISR' : 'Initial-state radiation',
  'FSR' : 'Final-state radiation',
  'DY' : 'Drell--Yan',
}

#path = 'histos/run3/5jun2022/'
### Fix categories
categories = {'channel':'em', 'level': 'g2jets', 'sign':'OS'}#, 'syst':'norm'}
categoriesPDF = {'channel':'em', 'level': 'g2jets'}#, 'syst':'norm'}

### Create plotter
p = plotter(path, prDic=processDic, bkgList=bkglist, colors=colordic, lumi=lumi, var='counts')

### Add hdamp and tune uncertainties
hdampup,hdampdo = GetModSystHistos(path, 'TTTo2L2Nu_hdamp', 'hdamp', var='counts')
tuneup , tunedo = GetModSystHistos(path, 'TTTo2L2Nu_UE', 'UE', var='counts')
p.AddExtraBkgHist([hdampup, hdampdo, tuneup, tunedo], add=True)

### Create xsec object
x = xsec('tt', 0.01, {'tW':0.1, 'Nonprompt':0.2, 'DY':0.1, 'Diboson':0.3}, plotter=p, verbose=1, thxsec=922.45, experimental=['lepSF', 'trigSF', 'JES'], modeling=['UE', 'hdamp', 'ISR', 'FSR'], categories=categories)
x.SetNames(names)
pdf   = Get1bPDFUnc(  path, categories=categoriesPDF, sample='TTTo2L2Nu', doPrint=False)
scale = Get1binScaleUnc(path, categories=categoriesPDF, sample='TTTo2L2Nu', doPrint=False)
x.AddModUnc('PDF$+\\alpha_{S}$', pdf, isRelative=True)
x.AddModUnc('$\mu_R, \mu_F$ scales', scale, isRelative=True)
x.ComputeXsecUncertainties()
x.GetYieldsTable()
x.GetUncTable()


