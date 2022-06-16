'''
 This script is used to transform scale factors, which are tipically provided as 2D histograms within root files,
 into coffea format of corrections.
'''

#import uproot, uproot_methods
import uproot
from coffea import hist, lookup_tools
import os, sys
from cafea.modules.paths import cafea_path
import numpy as np
import awkward as ak
import gzip
import pickle
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory, CorrectedMETFactory
from coffea.btag_tools.btagscalefactor import BTagScaleFactor
from cafea.plotter.plotter import GetHisto, GetSFfromCountsHisto, DrawEff, DrawEff2D, GetH2DfromXY
from cafea.modules.GetValuesFromJsons import get_param

basepathFromTTH = 'data/fromTTH/'

###### Lepton scale factors
################################################################
extLepSF = lookup_tools.extractor()

# Electron reco
extLepSF.add_weight_sets(["ElecRecoSFb20_2016 EGamma_SF2D %s"%cafea_path(basepathFromTTH+'lepSF/reco/elec/2016/el_scaleFactors_gsf_ptLt20.root')])
extLepSF.add_weight_sets(["ElecRecoSF_2016 EGamma_SF2D %s"%cafea_path(basepathFromTTH+'lepSF/reco/elec/2016/el_scaleFactors_gsf_ptGt20.root')])
extLepSF.add_weight_sets(["ElecRecoSFb20_2017 EGamma_SF2D %s"%cafea_path(basepathFromTTH+'lepSF/reco/elec/2017/el_scaleFactors_gsf_ptLt20.root')])
extLepSF.add_weight_sets(["ElecRecoSF_2017 EGamma_SF2D %s"%cafea_path(basepathFromTTH+'lepSF/reco/elec/2017/el_scaleFactors_gsf_ptGt20.root')])
extLepSF.add_weight_sets(["ElecRecoSF_2018 EGamma_SF2D %s"%cafea_path(basepathFromTTH+'lepSF/reco/elec/2018/el_scaleFactors_gsf.root')])
extLepSF.add_weight_sets(["ElecRecoSFb20_2016_er EGamma_SF2D_error %s"%cafea_path(basepathFromTTH+'lepSF/reco/elec/2016/el_scaleFactors_gsf_ptLt20.root')])
extLepSF.add_weight_sets(["ElecRecoSF_2016_er EGamma_SF2D_error %s"%cafea_path(basepathFromTTH+'lepSF/reco/elec/2016/el_scaleFactors_gsf_ptGt20.root')])
extLepSF.add_weight_sets(["ElecRecoSFb20_2017_er EGamma_SF2D_error %s"%cafea_path(basepathFromTTH+'lepSF/reco/elec/2017/el_scaleFactors_gsf_ptLt20.root')])
extLepSF.add_weight_sets(["ElecRecoSF_2017_er EGamma_SF2D_error %s"%cafea_path(basepathFromTTH+'lepSF/reco/elec/2017/el_scaleFactors_gsf_ptGt20.root')])
extLepSF.add_weight_sets(["ElecRecoSF_2018_er EGamma_SF2D_error %s"%cafea_path(basepathFromTTH+'lepSF/reco/elec/2018/el_scaleFactors_gsf.root')])

# Electron loose
extLepSF.add_weight_sets(["ElecLooseSF_2016 EGamma_SF2D %s"%cafea_path(basepathFromTTH+'lepSF/loose/elec/TnP_loose_ele_2016.root')])
extLepSF.add_weight_sets(["ElecLooseSF_2017 EGamma_SF2D %s"%cafea_path(basepathFromTTH+'lepSF/loose/elec/TnP_loose_ele_2017.root')])
extLepSF.add_weight_sets(["ElecLooseSF_2018 EGamma_SF2D %s"%cafea_path(basepathFromTTH+'lepSF/loose/elec/TnP_loose_ele_2018.root')])
extLepSF.add_weight_sets(["ElecLoosettHSF_2016 EGamma_SF2D %s"%cafea_path(basepathFromTTH+'lepSF/loose/elec/TnP_loosettH_ele_2016.root')])
extLepSF.add_weight_sets(["ElecLoosettHSF_2017 EGamma_SF2D %s"%cafea_path(basepathFromTTH+'lepSF/loose/elec/TnP_loosettH_ele_2017.root')])
extLepSF.add_weight_sets(["ElecLoosettHSF_2018 EGamma_SF2D %s"%cafea_path(basepathFromTTH+'lepSF/loose/elec/TnP_loosettH_ele_2018.root')])
extLepSF.add_weight_sets(["ElecLooseSF_2016_er EGamma_SF2D_error %s"%cafea_path(basepathFromTTH+'lepSF/loose/elec/TnP_loose_ele_2016.root')])
extLepSF.add_weight_sets(["ElecLooseSF_2017_er EGamma_SF2D_error %s"%cafea_path(basepathFromTTH+'lepSF/loose/elec/TnP_loose_ele_2017.root')])
extLepSF.add_weight_sets(["ElecLooseSF_2018_er EGamma_SF2D_error %s"%cafea_path(basepathFromTTH+'lepSF/loose/elec/TnP_loose_ele_2018.root')])
extLepSF.add_weight_sets(["ElecLoosettHSF_2016_er EGamma_SF2D_error %s"%cafea_path(basepathFromTTH+'lepSF/loose/elec/TnP_loosettH_ele_2016.root')])
extLepSF.add_weight_sets(["ElecLoosettHSF_2017_er EGamma_SF2D_error %s"%cafea_path(basepathFromTTH+'lepSF/loose/elec/TnP_loosettH_ele_2017.root')])
extLepSF.add_weight_sets(["ElecLoosettHSF_2018_er EGamma_SF2D_error %s"%cafea_path(basepathFromTTH+'lepSF/loose/elec/TnP_loosettH_ele_2018.root')])

# Electron tight
extLepSF.add_weight_sets(["ElecTightSF_2016 EGamma_SF2D %s"%cafea_path(basepathFromTTH+'lepSF/tight/elec/egammaEff2016_EGM2D.root')])
extLepSF.add_weight_sets(["ElecTightSF_2017 EGamma_SF2D %s"%cafea_path(basepathFromTTH+'lepSF/tight/elec/egammaEff2017_EGM2D.root')])
extLepSF.add_weight_sets(["ElecTightSF_2018 EGamma_SF2D %s"%cafea_path(basepathFromTTH+'lepSF/tight/elec/egammaEff2018_EGM2D.root')])
extLepSF.add_weight_sets(["ElecTightSF_2016_er EGamma_SF2D_error %s"%cafea_path(basepathFromTTH+'lepSF/tight/elec/egammaEff2016_EGM2D.root')])
extLepSF.add_weight_sets(["ElecTightSF_2017_er EGamma_SF2D_error %s"%cafea_path(basepathFromTTH+'lepSF/tight/elec/egammaEff2017_EGM2D.root')])
extLepSF.add_weight_sets(["ElecTightSF_2018_er EGamma_SF2D_error %s"%cafea_path(basepathFromTTH+'lepSF/tight/elec/egammaEff2018_EGM2D.root')])

# Muon loose
extLepSF.add_weight_sets(["MuonLooseSF_2016 EGamma_SF2D %s"%cafea_path(basepathFromTTH+'lepSF/loose/muon/TnP_loose_muon_2016.root')])
extLepSF.add_weight_sets(["MuonLooseSF_2017 EGamma_SF2D %s"%cafea_path(basepathFromTTH+'lepSF/loose/muon/TnP_loose_muon_2017.root')])
extLepSF.add_weight_sets(["MuonLooseSF_2018 EGamma_SF2D %s"%cafea_path(basepathFromTTH+'lepSF/loose/muon/TnP_loose_muon_2018.root')])
extLepSF.add_weight_sets(["MuonLooseSF_2016_er EGamma_SF2D_error %s"%cafea_path(basepathFromTTH+'lepSF/loose/muon/TnP_loose_muon_2016.root')])
extLepSF.add_weight_sets(["MuonLooseSF_2017_er EGamma_SF2D_error %s"%cafea_path(basepathFromTTH+'lepSF/loose/muon/TnP_loose_muon_2017.root')])
extLepSF.add_weight_sets(["MuonLooseSF_2018_er EGamma_SF2D_error %s"%cafea_path(basepathFromTTH+'lepSF/loose/muon/TnP_loose_muon_2018.root')])

# Muon tight
extLepSF.add_weight_sets(["MuonTightSF_2016 EGamma_SF2D %s"%cafea_path(basepathFromTTH+'lepSF/tight/muon/egammaEff2016_EGM2D.root')])
extLepSF.add_weight_sets(["MuonTightSF_2017 EGamma_SF2D %s"%cafea_path(basepathFromTTH+'lepSF/tight/muon/egammaEff2017_EGM2D.root')])
extLepSF.add_weight_sets(["MuonTightSF_2018 EGamma_SF2D %s"%cafea_path(basepathFromTTH+'lepSF/tight/muon/egammaEff2018_EGM2D.root')])
extLepSF.add_weight_sets(["MuonTightSF_2016_er EGamma_SF2D_error %s"%cafea_path(basepathFromTTH+'lepSF/tight/muon/egammaEff2016_EGM2D.root')])
extLepSF.add_weight_sets(["MuonTightSF_2017_er EGamma_SF2D_error %s"%cafea_path(basepathFromTTH+'lepSF/tight/muon/egammaEff2017_EGM2D.root')])
extLepSF.add_weight_sets(["MuonTightSF_2018_er EGamma_SF2D_error %s"%cafea_path(basepathFromTTH+'lepSF/tight/muon/egammaEff2018_EGM2D.root')])

# 5.02 TeV
extLepSF.add_weight_sets(["MuonTightSF_5TeV EGamma_SF2D %s"%cafea_path(basepathFromTTH+'lepSF/5TeV/final_mu_loosetotightSF.root')])
extLepSF.add_weight_sets(["MuonTightSF_5TeV_er EGamma_SF2D_error %s"%cafea_path(basepathFromTTH+'lepSF/5TeV/final_mu_loosetotightSF.root')])
extLepSF.add_weight_sets(["MuonLooseSF_5TeV EGamma_SF2D %s"%cafea_path(basepathFromTTH+'lepSF/5TeV/final_mu_recotolooseSF.root')])
extLepSF.add_weight_sets(["MuonLooseSF_5TeV_er EGamma_SF2D_error %s"%cafea_path(basepathFromTTH+'lepSF/5TeV/final_mu_recotolooseSF.root')])
extLepSF.add_weight_sets(["ElecTightSF_5TeV EGamma_SF2D %s"%cafea_path(basepathFromTTH+'lepSF/5TeV/final_ele_loosetotightSF.root')])
extLepSF.add_weight_sets(["ElecTightSF_5TeV_er EGamma_SF2D_error %s"%cafea_path(basepathFromTTH+'lepSF/5TeV/final_ele_loosetotightSF.root')])
extLepSF.add_weight_sets(["ElecLooseSF_5TeV EGamma_SF2D %s"%cafea_path(basepathFromTTH+'lepSF/5TeV/final_ele_recotolooseSF.root')])
extLepSF.add_weight_sets(["ElecLooseSF_5TeV_er EGamma_SF2D_error %s"%cafea_path(basepathFromTTH+'lepSF/5TeV/final_ele_recotolooseSF.root')])

# Fake rate 
# todo: check that these are the same as the "recorrected"
for year in [2016, 2017, 2018]:
  for syst in ['','_up','_down','_be1','_be2','_pt1','_pt2']:
    extLepSF.add_weight_sets([("MuonFR_{year}{syst} FR_mva085_mu_data_comb_recorrected{syst} %s"%cafea_path(basepathFromTTH+'fakerate/fr_{year}_recorrected.root')).format(year=year,syst=syst)])
    extLepSF.add_weight_sets([("ElecFR_{year}{syst} FR_mva080_el_data_comb_NC_recorrected{syst} %s"%cafea_path(basepathFromTTH+'fakerate/fr_{year}_recorrected.root')).format(year=year,syst=syst)])

# Flip rates                                                                                                                                                                                                       
for year in [2016, 2017, 2018]:
  extLepSF.add_weight_sets([("EleFlip_{year} chargeMisId %s"%cafea_path(basepathFromTTH+'fliprates/ElectronChargeMisIdRates_era{year}_2020Feb13.root')).format(year=year,syst=syst)])

extLepSF.finalize()
SFevaluator = extLepSF.make_evaluator()

def AttachMuonSF(muons, year=2018):
  '''
    Description:
      Inserts 'sf_nom', 'sf_hi', and 'sf_lo' into the muons array passed to this function. These
      values correspond to the nominal, up, and down muon scalefactor values respectively.
  '''
  eta = np.abs(muons.eta)
  pt = muons.pt
  if year == '2016APV': year = '2016'
  loose_sf  = SFevaluator['MuonLooseSF_{year}'.format(year=year)](eta,pt)
  tight_sf  = SFevaluator['MuonTightSF_{year}'.format(year=year)](eta,pt)
  loose_err = SFevaluator['MuonLooseSF_{year}_er'.format(year=year)](eta,pt)
  tight_err = SFevaluator['MuonTightSF_{year}_er'.format(year=year)](eta,pt)

  muons['sf_nom'] = loose_sf * tight_sf
  muons['sf_hi']  = (loose_sf + loose_err) * (tight_sf + tight_err)
  muons['sf_lo']  = (loose_sf - loose_err) * (tight_sf - tight_err)

def AttachElectronSF(electrons, year=2018):
  '''
    Description:
      Inserts 'sf_nom', 'sf_hi', and 'sf_lo' into the electrons array passed to this function. These
      values correspond to the nominal, up, and down electron scalefactor values respectively.
  '''
  # eta = np.abs(electrons.eta)
  eta = electrons.eta
  pt = electrons.pt
  if year == '2016APV': year = 2016
  if year != '5TeV':
    # For the ElecRecoSF we dont take the absolute value of eta!
    reco_sf          = SFevaluator['ElecRecoSF_{year}'.format(year=year)](eta,pt)
    reco_sf_err      = SFevaluator['ElecRecoSF_{year}_er'.format(year=year)](eta,pt)
    loose_ttH_sf     = SFevaluator['ElecLoosettHSF_{year}'.format(year=year)](np.abs(eta),pt)
    loose_ttH_sf_err = SFevaluator['ElecLoosettHSF_{year}_er'.format(year=year)](np.abs(eta),pt)

  loose_sf         = SFevaluator['ElecLooseSF_{year}'.format(year=year)](np.abs(eta),pt)
  loose_sf_err     = SFevaluator['ElecLooseSF_{year}_er'.format(year=year)](np.abs(eta),pt)
  tight_sf         = SFevaluator['ElecTightSF_{year}'.format(year=year)](np.abs(eta),pt)
  tight_sf_err     = SFevaluator['ElecTightSF_{year}_er'.format(year=year)](np.abs(eta),pt)

  if year == '5TeV':
    electrons['sf_nom'] = loose_sf * tight_sf
    electrons['sf_hi']  = (loose_sf + loose_sf_err) * (tight_sf + tight_sf_err)
    electrons['sf_lo']  = (loose_sf - loose_sf_err) * (tight_sf - tight_sf_err)
  else:
    electrons['sf_nom'] = reco_sf * loose_sf * loose_ttH_sf * tight_sf
    electrons['sf_hi']  = (reco_sf + reco_sf_err) * (loose_sf + loose_sf_err) * (loose_ttH_sf + loose_ttH_sf_err) * (tight_sf + tight_sf_err)
    electrons['sf_lo']  = (reco_sf - reco_sf_err) * (loose_sf - loose_sf_err) * (loose_ttH_sf - loose_ttH_sf_err) * (tight_sf - tight_sf_err)








def LoadTriggerSF(year, ch='2l', flav='em'):
  pathToTriggerSF = cafea_path('data/triggerSF/triggerSF_%s.pkl.gz'%year)
  with gzip.open(pathToTriggerSF) as fin: hin = pickle.load(fin)
  if ch=='2l': axisY='l1pt'
  else: axisY='l0eta'
  h = hin[ch][flav]
  ratio, do, up = GetSFfromCountsHisto(h['hmn'], h['hmd'], h['hdn'], h['hdd'])
  ratio[np.isnan(ratio)]=1.0; do[np.isnan(do)]=0.0;up[np.isnan(up)]=0.0
  GetTrig   = lookup_tools.dense_lookup.dense_lookup(ratio, [h['hmn'].axis('l0pt').edges(), h['hmn'].axis(axisY).edges()])
  GetTrigUp = lookup_tools.dense_lookup.dense_lookup(up   , [h['hmn'].axis('l0pt').edges(), h['hmn'].axis(axisY).edges()])
  GetTrigDo = lookup_tools.dense_lookup.dense_lookup(do   , [h['hmn'].axis('l0pt').edges(), h['hmn'].axis(axisY).edges()])
  return [GetTrig, GetTrigDo, GetTrigUp]

def GetTriggerSF(year, events, lep0, lep1):
  ls=[]
  for syst in [0,1,2]:
    #2l
    SF_ee=np.where(events.isee==True, LoadTriggerSF(year,ch='2l',flav='ee')[syst](lep0.pt,lep1.pt),1.0)
    SF_em=np.where(events.isem==True, LoadTriggerSF(year,ch='2l',flav='em')[syst](lep0.pt,lep1.pt),1.0)
    SF_mm=np.where(events.ismm==True, LoadTriggerSF(year,ch='2l',flav='mm')[syst](lep0.pt,lep1.pt),1.0)
    ls.append(SF_ee*SF_em*SF_mm)
  ls[1]=np.where(ls[1]==1.0,0.0,ls[1]) # stat unc. down
  ls[2]=np.where(ls[2]==1.0,0.0,ls[2]) # stat unc. up
  events['trigger_sf']=ls[0] #nominal
  events['trigger_sfDown']=ls[0]-np.sqrt(ls[1]*ls[1]+0.01*0.01) # place holder: 1% systematic unc.
  events['trigger_sfUp']=ls[0]+np.sqrt(ls[2]*ls[2]+0.01*0.01) # place holder: 1% systematic unc.










###### Trigger SFs (for 5.02 TeV, l+jets)
################################################################

def GetFTriggerSF5TeV(path, ch='m'):
  integrate=['pr']
  var = 'pteta'; mcSample = 'tt'
  dataSample = 'HighEGJet' if ch =='m' else 'SingleMuon'
  hu = GetHisto(path, var, {'channel':ch, 'val':'num'}, group=['sample', 'pr', {'pr':mcSample}], integrate=integrate)
  hd = GetHisto(path, var, {'channel':ch, 'val':'den'}, group=['sample', 'pr', {'pr':mcSample}], integrate=integrate)
  du = GetHisto(path, var, {'channel':ch, 'val':'num'}, group=['sample', 'pr', {'pr':dataSample}], integrate=integrate)
  dd = GetHisto(path, var, {'channel':ch, 'val':'den'}, group=['sample', 'pr', {'pr':dataSample}], integrate=integrate)
  ratio, do, up = GetSFfromCountsHisto(hu, hd, du, dd)
  GetTrig   = lookup_tools.dense_lookup.dense_lookup(ratio, [hu.axis('pt').edges(), hu.axis('abseta').edges()])
  GetTrigUp = lookup_tools.dense_lookup.dense_lookup(up   , [hu.axis('pt').edges(), hu.axis('abseta').edges()])
  GetTrigDo = lookup_tools.dense_lookup.dense_lookup(do   , [hu.axis('pt').edges(), hu.axis('abseta').edges()])
  return [GetTrig, GetTrigDo, GetTrigUp]

path_trigSF5TeV = cafea_path('data/5TeV/triggerSFs.pkl.gz')
GetElecTrigSF5TeV, GetElecTrigSF5TeVDown, GetElecTrigSF5TeVUp = GetFTriggerSF5TeV(path_trigSF5TeV, 'e')
GetMuonTrigSF5TeV, GetMuonTrigSF5TeVDown, GetMuonTrigSF5TeVUp = GetFTriggerSF5TeV(path_trigSF5TeV, 'm')
def GetTriggerSF5TeV(pt, eta, ch='e'):
  eta = np.abs(eta)
  SFs  = GetElecTrigSF5TeV    (pt, eta) if ch=='e' else GetMuonTrigSF5TeV    (pt, eta)
  SFdo = GetElecTrigSF5TeVDown(pt, eta) if ch=='e' else GetMuonTrigSF5TeVDown(pt, eta)
  SFup = GetElecTrigSF5TeVUp  (pt, eta) if ch=='e' else GetMuonTrigSF5TeVUp  (pt, eta)
  return [SFs, SFs+SFdo, SFs+SFup]

###### Btag scale factors
################################################################
# Hard-coded to DeepJet algorithm, medium WP

# MC efficiencies
def GetMCeffFunc(WP='medium', year=2018, flav='b'):
  pathToBtagMCeff = cafea_path('data/btagSF/UL/btagMCeff_%s.pkl.gz'%str(year))
  hists = {}
  with gzip.open(pathToBtagMCeff) as fin:
    hin = pickle.load(fin)
    for k in hin.keys():
      if k in hists: hists[k]+=hin[k]
      else:          hists[k]=hin[k]
  h = hists['jetptetaflav']
  hnum = h.integrate('WP', WP)
  hden = h.integrate('WP', 'all')
  getnum = lookup_tools.dense_lookup.dense_lookup(hnum.values(overflow='over')[()], [hnum.axis('pt').edges(), hnum.axis('abseta').edges(), hnum.axis('flav').edges()])
  getden = lookup_tools.dense_lookup.dense_lookup(hden.values(overflow='over')[()], [hden.axis('pt').edges(), hnum.axis('abseta').edges(), hden.axis('flav').edges()])
  #values = hnum.values(overflow='over')[()]
  #edges = [hnum.axis('pt').edges(), hnum.axis('abseta').edges(), hnum.axis('flav').edges()]
  fun = lambda pt, abseta, flav : getnum(pt,abs(abseta),flav)/getden(pt,abs(abseta),flav)
  return fun

wplabel = 'medium'#'medium'
MCeffFunc_2018 = GetMCeffFunc(wplabel.lower(), 2018)
MCeffFunc_2017 = GetMCeffFunc(wplabel.lower(), 2017)
MCeffFunc_5TeV = GetMCeffFunc(wplabel.lower(), '5TeV')

def GetBtagEff(eta, pt, flavor, year=2018):
  if   year==2017: return MCeffFunc_2017(pt, eta, flavor)
  elif year==2018: return MCeffFunc_2018(pt, eta, flavor)
  elif year=='5TeV': return MCeffFunc_5TeV(pt, eta, flavor)

def GetBTagSF(eta, pt, flavor, year=2018, sys=0):

  # Efficiencies and SFs for UL only available for 2017 and 2018
  if year == '2016APV': year = 2016
  if   year == 2016: SFevaluatorBtag = BTagScaleFactor(cafea_path("data/btagSF/DeepFlav_2016.csv"),wplabel.upper())
  elif year == 2017: SFevaluatorBtag = BTagScaleFactor(cafea_path("data/btagSF/UL/DeepJet_UL17.csv"),wplabel.upper())
  elif year == 2018: SFevaluatorBtag = BTagScaleFactor(cafea_path("data/btagSF/UL/DeepJet_UL18.csv"),wplabel.upper())
  elif year == '5TeV': SFevaluatorBtag = BTagScaleFactor(cafea_path("data/btagSF/DeepCSV_94XSF_V5_B_F.csv"),wplabel.upper())

  if   sys==0 : SF=SFevaluatorBtag.eval("central",flavor,eta,pt)
  elif sys==1 : SF=SFevaluatorBtag.eval("up",flavor,eta,pt)
  elif sys==-1: SF=SFevaluatorBtag.eval("down",flavor,eta,pt)

  return (SF)

def GetBtagSF5TeV(pt, eta, flav, isBtagJets, doSys=True):
  abseta = np.abs(eta);
  bJetSF   = GetBTagSF(abseta, pt, flav, year='5TeV')
  bJetEff  = GetBtagEff(abseta, pt, flav, year='5TeV')
  bJetEff_data = bJetEff*bJetSF
  isNotBtagJets = np.invert(isBtagJets) 

  pMC     = ak.prod(bJetEff       [isBtagJets], axis=-1) * ak.prod((1-bJetEff       [isNotBtagJets]), axis=-1)
  pData   = ak.prod(bJetEff_data  [isBtagJets], axis=-1) * ak.prod((1-bJetEff_data  [isNotBtagJets]), axis=-1)
  pMC      = ak.where(pMC==0,1,pMC) # removeing zeroes from denominator...
  btagSF   = pData  /pMC
  if not doSys: return btagSF

  bJetSFUp = GetBTagSF(abseta, pt, flav, year='5TeV', sys=1)
  bJetSFDo = GetBTagSF(abseta, pt, flav, year='5TeV', sys=-1)
  bJetEff_dataUp = bJetEff*bJetSFUp
  bJetEff_dataDo = bJetEff*bJetSFDo
  pDataUp = ak.prod(bJetEff_dataUp[isBtagJets], axis=-1) * ak.prod((1-bJetEff_dataUp[isNotBtagJets]), axis=-1)
  pDataDo = ak.prod(bJetEff_dataDo[isBtagJets], axis=-1) * ak.prod((1-bJetEff_dataDo[isNotBtagJets]), axis=-1)
  btagSFUp = pDataUp/pMC
  btagSFDo = pDataDo/pMC
  return btagSF, btagSFUp, btagSFDo
   
'''
def LoadTriggerSFs5TeV(samplename = 'tt'):
  pathToTrigger = cafea_path('data/5TeV/triggerSFs.pkl.gz')
  if isinstance(samplename, str): samplename = samplename.replace(' ', '').split(',') if ',' in samplename else [samplename]
  #if isinstance(samplename, list): samplename = tuple(samplename)
  hists = loadHistos(pathToTrigger)
  hnum_e = hists['pteta'].integrate('val', 'num').integrate('channel', 'e'); hnum_e = hnum_e.group(hist.Cat('sample', 'sample'), hist.Cat('pr', 'pr'), {'pr': samplename})
  hden_e = hists['pteta'].integrate('val', 'den').integrate('channel', 'e'); hden_e = hden_e.group(hist.Cat('sample', 'sample'), hist.Cat('pr', 'pr'), {'pr': samplename})
  getnum_e = lookup_tools.dense_lookup.dense_lookup(hnum_e.values(overflow='over')[('pr',)], [hnum_e.axis('pt').edges(), hnum_e.axis('abseta').edges()])
  getden_e = lookup_tools.dense_lookup.dense_lookup(hden_e.values(overflow='over')[('pr',)], [hden_e.axis('pt').edges(), hden_e.axis('abseta').edges()])
  hnum_m = hists['pteta'].integrate('val', 'num').integrate('channel', 'm'); hnum_m = hnum_m.group(hist.Cat('sample', 'sample'), hist.Cat('pr', 'pr'), {'pr': samplename})
  hden_m = hists['pteta'].integrate('val', 'den').integrate('channel', 'm'); hden_m = hden_m.group(hist.Cat('sample', 'sample'), hist.Cat('pr', 'pr'), {'pr': samplename})
  getnum_m = lookup_tools.dense_lookup.dense_lookup(hnum_m.values(overflow='over')[('pr',)], [hnum_m.axis('pt').edges(), hnum_m.axis('abseta').edges()])
  getden_m = lookup_tools.dense_lookup.dense_lookup(hden_m.values(overflow='over')[('pr',)], [hden_m.axis('pt').edges(), hden_m.axis('abseta').edges()])
  fun = lambda pt, eta, chan : [np.array(getnum_m(pt,abs(eta)), dtype=float)/np.array(getden_m(pt,abs(eta)), dtype=float), *proportion_confint(np.array(getnum_m(pt,abs(eta)), dtype=int), np.array(getden_m(pt,abs(eta)), dtype=int), 1-0.68)] if chan=='m' else [np.array(getnum_e(pt,abs(eta)), dtype=float)/np.array(getden_e(pt,abs(eta)), dtype=float), *proportion_confint(np.array(getnum_e(pt,abs(eta)), dtype=int), np.array(getden_e(pt,abs(eta)), dtype=int), 1-0.68)]
  return fun

def GetTriggerSF5TeV(pt, eta, chan):
  GetMCEff  = LoadTriggerSFs5TeV('tt')
  GetDataEff = LoadTriggerSFs5TeV('SingleMuon, HighEGJet')
  return np.array(GetDataEff(pt, eta, chan))/np.array(GetMCEff(pt, eta, chan))
'''
pt = np.array([25, 60.3, 154.4])
eta = np.array([-2.3, 0.2, 2.1])

###### Pileup reweighing
##############################################
## Get central PU data and MC profiles and calculate reweighting
## Using the current UL recommendations in:
##   https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData
##   - 2018: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/PileUp/UltraLegacy/
##   - 2017: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/
##   - 2016: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/PileUp/UltraLegacy/
##
## MC histograms from:
##    https://github.com/CMS-LUMI-POG/PileupTools/

pudirpath = cafea_path('data/pileup/')

def GetDataPUname(year='2017', var=0):
  ''' Returns the name of the file to read pu observed distribution '''
  if year == '2016APV': year = 2016
  if   var== 0: ppxsec = get_param("pu_w")
  elif var== 1: ppxsec = get_param("pu_w_up")
  elif var==-1: ppxsec = get_param("pu_w_down")
  year = str(year)
  return 'PileupHistogram-goldenJSON-13tev-%s-%sub-99bins.root'%((year), str(ppxsec))

MCPUfile = {'2016APV':'pileup_2016BF.root', '2016':'pileup_2016GH.root', '2017':'pileup_2017_shifts.root', '2018':'pileup_2018_shifts.root'}
def GetMCPUname(year='2017'):
  ''' Returns the name of the file to read pu MC profile '''
  return MCPUfile[str(year)]

PUfunc = {}
### Load histograms and get lookup tables (extractors are not working here...)
for year in ['2016', '2016APV', '2017', '2018']:
  PUfunc[year] = {}
  with uproot.open(pudirpath+GetMCPUname(year)) as fMC:
    hMC = fMC['pileup']
    PUfunc[year]['MC'] = lookup_tools.dense_lookup.dense_lookup(hMC .values(), hMC.axis(0).edges())
  with uproot.open(pudirpath+GetDataPUname(year,  0)) as fData:
    hD   = fData  ['pileup']
    PUfunc[year]['Data'  ] = lookup_tools.dense_lookup.dense_lookup(hD  .values(), hD.axis(0).edges())
  with uproot.open(pudirpath+GetDataPUname(year,  1)) as fDataUp:
    hDUp = fDataUp['pileup']
    PUfunc[year]['DataUp'] = lookup_tools.dense_lookup.dense_lookup(hDUp.values(), hD.axis(0).edges())
  with uproot.open(pudirpath+GetDataPUname(year, -1)) as fDataDo:
    hDDo = fDataDo['pileup']
    PUfunc[year]['DataDo'] = lookup_tools.dense_lookup.dense_lookup(hDDo.values(), hD.axis(0).edges())

def GetPUSF(nTrueInt, year, var=0):
  year = str(year)
  nMC  =PUfunc[year]['MC'](nTrueInt+1)
  nData=PUfunc[year]['DataUp' if var == 1 else ('DataDo' if var == -1 else 'Data')](nTrueInt)
  weights = np.divide(nData,nMC)
  return weights

###### JEC corrections 5 TeV
##############################################
extJEC_data = lookup_tools.extractor()
extJEC_data.add_weight_sets([
  "* * "+cafea_path('data/JEC/Spring18_ppRef5TeV_V4_DATA_L1FastJet_AK4PFchs.txt'),
  "* * "+cafea_path('data/JEC/Spring18_ppRef5TeV_V4_DATA_L2L3Residual_AK4PFchs.txt'),
  "* * "+cafea_path('data/JEC/Spring18_ppRef5TeV_V4_DATA_L2Relative_AK4PFchs.txt'),
  "* * "+cafea_path('data/JEC/Spring18_ppRef5TeV_V4_DATA_L2Residual_AK4PFchs.txt'),
  "* * "+cafea_path('data/JEC/Spring18_ppRef5TeV_V4_DATA_L3Absolute_AK4PFchs.txt'),
  "* * "+cafea_path('data/JEC/Spring18_ppRef5TeV_V4_DATA_Uncertainty_AK4PFchs.txt'),
  ])
extJEC_data.finalize() 
JECevaluator_data = extJEC_data.make_evaluator()
jec_names_data = ["Spring18_ppRef5TeV_V4_DATA_L1FastJet_AK4PFchs", "Spring18_ppRef5TeV_V4_DATA_L2L3Residual_AK4PFchs", "Spring18_ppRef5TeV_V4_DATA_L2Relative_AK4PFchs", "Spring18_ppRef5TeV_V4_DATA_L2Residual_AK4PFchs", "Spring18_ppRef5TeV_V4_DATA_L3Absolute_AK4PFchs"]#, "Spring18_ppRef5TeV_V4_DATA_Uncertainty_AK4PFchs"]
jec_inputs_data = {name: JECevaluator_data[name] for name in jec_names_data}
jec_stack_data = JECStack(jec_inputs_data)
name_map = jec_stack_data.blank_name_map
name_map['JetPt'] = 'pt'
name_map['JetMass'] = 'mass'
name_map['JetEta'] = 'eta'
name_map['JetPhi'] = 'phi'
name_map['JetA'] = 'area'
name_map['ptGenJet'] = 'pt_gen'
name_map['ptRaw'] = 'pt_raw'
name_map['massRaw'] = 'mass_raw'
name_map['Rho'] = 'rho'
name_map['METpt'] = 'pt'
name_map['METphi'] = 'phi'
name_map['UnClusteredEnergyDeltaX'] = 'MetUnclustEnUpDeltaX'
name_map['UnClusteredEnergyDeltaY'] = 'MetUnclustEnUpDeltaY'

jet_factory_data = CorrectedJetsFactory(name_map, jec_stack_data)
#
extJEC = lookup_tools.extractor()
#extJEC.add_weight_sets(["* * "+cafea_path('data/JEC/Summer19UL18_V5_MC_L2Relative_AK4PFchs.txt'),"* * "+cafea_path('data/JEC/Summer19UL18_V5_MC_L2Residual_AK4PFchs.txt'),"* * "+cafea_path('data/JEC/Summer19UL18_V5_MC_L1FastJet_AK4PFchs.txt'),"* * "+cafea_path('data/JEC/Summer19UL18_V5_MC_L3Absolute_AK4PFchs.txt'),"* * "+cafea_path('data/JEC/Summer19UL18_V5_MC_L1RC_AK4PFchs.txt'),"* * "+cafea_path('data/JEC/Summer19UL18_V5_MC_Uncertainty_AK4PFchs.junc.txt'),"* * "+cafea_path('data/JEC/Summer19UL18_V5_MC_L2L3Residual_AK4PFchs.txt')])
extJEC.add_weight_sets([
  "* * "+cafea_path('data/JEC/Spring18_ppRef5TeV_V4_MC_L1FastJet_AK4PFchs.txt'),
  "* * "+cafea_path('data/JEC/Spring18_ppRef5TeV_V4_MC_L2L3Residual_AK4PFchs.txt'),
  "* * "+cafea_path('data/JEC/Spring18_ppRef5TeV_V4_MC_L2Relative_AK4PFchs.txt'),
  "* * "+cafea_path('data/JEC/Spring18_ppRef5TeV_V4_MC_L2Residual_AK4PFchs.txt'),
  "* * "+cafea_path('data/JEC/Spring18_ppRef5TeV_V4_MC_L3Absolute_AK4PFchs.txt'),
  "* * "+cafea_path('data/JEC/Spring18_ppRef5TeV_V4_MC_Uncertainty_AK4PFchs.junc.txt'),
  #"* * "+cafea_path('data/JEC/Spring18_ppRef5TeV_V4_MC_UncertaintySources_AK4PFchs.junc.txt'),
  ])
extJEC.finalize()

JECevaluator = extJEC.make_evaluator()
#jec_names = ["Spring18_ppRef5TeV_V4_MC_L1FastJet_AK4PFchs","Spring18_ppRef5TeV_V4_MC_L2L3Residual_AK4PFchs","Spring18_ppRef5TeV_V4_MC_L2Relative_AK4PFchs","Spring18_ppRef5TeV_V4_MC_L2Residual_AK4PFchs","Spring18_ppRef5TeV_V4_MC_L3Absolute_AK4PFchs"]#,"Spring18_ppRef5TeV_V4_MC_Uncertainty_AK4PFchs"] 
jec_names = ["Spring18_ppRef5TeV_V4_MC_L1FastJet_AK4PFchs","Spring18_ppRef5TeV_V4_MC_L2L3Residual_AK4PFchs","Spring18_ppRef5TeV_V4_MC_L2Relative_AK4PFchs","Spring18_ppRef5TeV_V4_MC_L2Residual_AK4PFchs","Spring18_ppRef5TeV_V4_MC_L3Absolute_AK4PFchs","Spring18_ppRef5TeV_V4_MC_Uncertainty_AK4PFchs"] 
jec_inputs = {name: JECevaluator[name] for name in jec_names}
jec_stack = JECStack(jec_inputs)
jet_factory = CorrectedJetsFactory(name_map, jec_stack)
met_factory = CorrectedMETFactory(name_map)
# test
#val = evaluator['MuonTightSF_2016'](np.array([1.2, 0.3]),np.array([24.5, 51.3]))
#print('val = ', val)










#############################################################################
# Electron ES
# 
# Usin json, https://coffeateam.github.io/coffea/notebooks/applying_corrections.html#Opening-a-root-file-and-using-it-as-a-lookup-table
# smear: abseta, r9
# scale: abseta, r9, runNumber
# Files from: https://github.com/GonzalezFJR/nanoAOD-tools/tree/master/python/postprocessing/data/elecES
pathEScales   = cafea_path('data/ElecES/Run2017_LowPU_v2_scales.json')
pathESmearing = cafea_path('data/ElecES/Run2017_LowPU_v2_smearings.json')
pathESmearRho = cafea_path('data/ElecES/Run2017_LowPU_v2_smear_rho.json')
ESextractor = lookup_tools.extractor()
ESextractor.add_weight_sets(["* * "+pathEScales, "* * "+pathESmearing, "* * "+pathESmearRho])
ESextractor.finalize()
ESevaluator= ESextractor.make_evaluator()

def ESsigma(et, eMean, rho, phi=3.141592/2, nrSigmaRho=0, nrSigmaPhi=0):
  ''' Get Sigma from ES smearing
      From: https://github.com/GonzalezFJR/nanoAOD-tools/blob/master/python/postprocessing/data/elecES/EnergyScaleCorrection.h#L90
  '''
  phiErr = 0; rhoErr = 0;
  rhoVal = rho + rhoErr * nrSigmaRho;
  phiVal = phi + phiErr * nrSigmaPhi;
  constTerm =  rhoVal * np.sin(phiVal);
  alpha =  rhoVal *  eMean * np.cos(phiVal);
  return np.sqrt(constTerm * constTerm + alpha * alpha / et);

def GetRandomWithSigma(sigma):
    ''' For getting the random numbers for smearing with the shape of the electron array
        From: https://github.com/scikit-hep/awkward-1.0/issues/489
    '''
    #Convert it to a 1D numpy array and perform smearing
    numpy_arr = np.asarray(sigma.layout.content)
    smeared_arr = np.random.normal(ak.ones_like(numpy_arr), numpy_arr)
    
    #Convert it back to awkward form
    return ak.Array(ak.layout.ListOffsetArray64(sigma.layout.offsets, ak.Array(smeared_arr).layout))


def GetElecScale5TeV(elec, run=306936, isData=False):
  ''' Get corrected electron pt/mass
    From: https://github.com/GonzalezFJR/nanoAOD-tools/blob/master/python/postprocessing/modules/common/ElectronScaleSmear.py
    Et() from https://root.cern.ch/doc/master/GenVector_2PtEtaPhiM4D_8h_source.html#l00247
  '''
  ecor = elec.eCorr
  eraw = elec*(1./ecor)
  elec['pt_orig'] = elec['pt']
  et = eraw.energy / np.cosh(eraw.eta)
  elec['pt_raw'] = eraw.pt
  elec['mass_raw'] = eraw.mass
  elec['et_raw'] = et
  abseta = abs(elec.eta+elec.deltaEtaSC)
  r9 = elec.r9
  if isData:
    # data --> only scale
    if isinstance(run, int): run = ak.ones_like(r9,dtype=int)*run
    escale = ESevaluator['correction/scale_value'](abseta, r9, run)
    vEle = eraw*escale
  else:
    # mc --> only smear
    eMean = ESevaluator['correction/smear_value'](abseta, r9)
    rho   = ESevaluator['correction/smearrho_value'](abseta, r9)
    sigma = ESsigma(et, eMean, rho)
    smear = GetRandomWithSigma(sigma) #np.random.normal(ak.ones_like(sigma), sigma, len(sigma))
    vEle = eraw*smear #(1+eleSmear*nrandom)
  # Modify initial array
  elec['pt'] = vEle.pt
  elec['mass'] = vEle.mass
  elec['energy'] = vEle.energy
  elec['eta'] = vEle.eta
  elec['phi'] = vEle.phi

#escale = ESevaluator['correction/scale_value'](np.array([0.2, 1.4, 1.5]), np.array([0.55, 0.95, 0.99]), np.array([306935,306935,306935]))
#print(escale)