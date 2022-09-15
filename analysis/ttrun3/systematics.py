from config import *
from cafea.plotter.xsec import *
from cafea.modules.fileReader import *
from PDFscaleUncertainties import *

names = {
  'lepSF_muon' : 'Muon efficiences',
  'lepSF_elec' : 'Electron efficiences',
  #'eleceff' : 'Electron efficiences',
  #'muoneff' : 'Muon efficiences',
  #'trigSF' : 'Trigger efficiencies',
  'JES' : 'Jet energy scale',
  #'UE' : 'Underlying event',
  #'hdamp' : '$h_\mathrm{damp}$',
  'ISR' : 'Initial-state radiation',
  'FSR' : 'Final-state radiation',
  #'DY' : 'Drell--Yan',
  'PU' : 'Pileup reweighting',
}

#path = 'histos/run3/5jun2022/'
### Fix categories
categories = {'channel':'em', 'level': 'dilep', 'sign':'OS'}#, 'syst':'norm'}
categoriesPDF = {'channel':'em', 'level': 'g2jets'}#, 'syst':'norm'}
processDic = {
  'tt': 'TTTo2L2Nu',
  'tW': 'tbarW, tW',
  'semilep':'TTToSemiLeptoni',
  'WJets':'WJetsToLNu',
  'DY': 'DYJetsToLL_M50, DYJetsToLL_M10to50', 
  'Diboson' : 'WW, WZ, ZZ',#'WWTo2L2Nu, WZTo3LNu',#'WW, WZ, ZZTo2L2Nu',
  'data' : 'MuonEG,EGamma,DoubleMuon,SingleMuon,Muon'
}
bkglist    = ['tt', 'tW','semilep', 'WJets', 'DY', 'Diboson']
### Create plotter
p = plotter(path, prDic=processDic, bkgList=bkglist, colors=colordic, lumi=lumi, var='counts')
'''
### Add hdamp and tune uncertainties
hdampup,hdampdo = GetModSystHistos(path, 'TTTo2L2Nu_hdamp', 'hdamp', var='counts')
tuneup , tunedo = GetModSystHistos(path, 'TTTo2L2Nu_UE', 'UE', var='counts')
p.AddExtraBkgHist([hdampup, hdampdo, tuneup, tunedo], add=True)
'''

### Create xsec object
experimental = ['lepSF_muon', 'lepSF_elec']
modeling = ['ISR', 'FSR'] # ['UE', 'hdamp', 'ISR', 'FSR']
x = xsec('tt', 0.06, {'tW':0.15,'semilep':0.2,'WJets':0.3, 'DY':0.2, 'Diboson':0.3}, plotter=p, verbose=4, thxsec=921, experimental=experimental, modeling=modeling, categories=categories)
x.SetNames(names)
pdf   = Get1bPDFUnc(  path, categories=categoriesPDF, sample='TTTo2L2Nu', doPrint=False)
scale = Get1binScaleUnc(path, categories=categoriesPDF, sample='TTTo2L2Nu', doPrint=False)
x.AddModUnc('PDF$+\\alpha_{S}$', pdf, isRelative=True)
x.AddModUnc('$\mu_R, \mu_F$ scales', scale, isRelative=True)

jecs = GetJECSystHistos(path, 'variations/TTTo2L2Nu_withJEC', var='counts', categories=categories)
x.AddExpUnc('JEC', jecs, isRelative=True)
x.ComputeXsecUncertainties()


# Update muon eff adding 0.5 %
#mueff1 = x.xsecunc['muoneff'] # Muon eff unc from SFs
#mueff2 = x.xsecnom*0.005      # Extra 0.5% from phase space extrapolation
#x.xsecunc['muoneff'] = np.sqrt(mueff1*mueff1 + mueff2*mueff2)

x.GetYieldsTable()
x.GetUncTable(form='%1.2f')
x.ComputeCrossSection()

