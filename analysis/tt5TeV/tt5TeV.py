#!/usr/bin/env python
import lz4.frame as lz4f
import cloudpickle
import json
import pprint
import numpy as np
import awkward as ak
import coffea
np.seterr(divide='ignore', invalid='ignore', over='ignore')
from coffea import hist, processor
from coffea.util import load, save
from optparse import OptionParser
from coffea.analysis_tools import PackedSelection
from coffea.lumi_tools import LumiMask

from cafea.analysis.objects import *
from cafea.analysis.corrections import GetBTagSF, GetBtagEff, AttachMuonSF, AttachElectronSF, GetPUSF, GetTriggerSF5TeV, GetElecScale5TeV, jet_factory, jet_factory_data, met_factory, GetBtagSF5TeV
from cafea.analysis.selection import *
from cafea.modules.paths import cafea_path

'''
def GetElecPt(pt, eta, ecorr = 1, isdata = False):
  eta_sep = (abs(eta) < 1.479)
  fact_leta = (1.016-0.0035) if isdata else 1.005
  fact_heta = (1.052 -0.036) if isdata else 0.992
  facts_leta = np.ones_like(pt)*fact_leta
  facts_heta = np.ones_like(pt)*fact_heta
  facts = np.where(eta_sep, facts_leta, facts_heta)
  return pt*facts

def GetElecPtSmear(pt, eta, isdata = False):
  if(isdata): return pt
  mass = 91.1876
  sigma_lowEta = 1.786/mass
  sigma_highEta = 3.451/mass
  rnd_lowEta = np.random.normal(1, sigma_lowEta, len(pt))
  rnd_highEta = np.random.normal(1, sigma_lowEta, len(pt))
  eta_sep = (abs(eta) < 1.479)
  smear = np.where(eta_sep, rnd_lowEta, rnd_highEta)
  return pt*smear
'''

fillAll = True
doSyst = True

def AttachTrigSF(e0, m0, events):
  TrigSFe, TrigSFedo, TrigSFeup = GetTriggerSF5TeV(e0.pt, np.abs(e0.eta), 'e')
  TrigSFm, TrigSFmdo, TrigSFmup = GetTriggerSF5TeV(m0.pt, np.abs(m0.eta), 'm')
  TrigSFe   = np.nan_to_num(ak.flatten(ak.fill_none(TrigSFe, 1.)), nan=1)
  TrigSFedo = np.nan_to_num(ak.flatten(ak.fill_none(TrigSFedo, 1.)), nan=1)
  TrigSFeup = np.nan_to_num(ak.flatten(ak.fill_none(TrigSFeup, 1.)), nan=1)
  TrigSFm   = np.nan_to_num(ak.flatten(ak.fill_none(TrigSFm, 1.)), nan=1)
  TrigSFmdo = np.nan_to_num(ak.flatten(ak.fill_none(TrigSFmdo, 1.)), nan=1)
  TrigSFmup = np.nan_to_num(ak.flatten(ak.fill_none(TrigSFmup, 1.)), nan=1)
  events['sf_trig']    = TrigSFe*TrigSFm
  events['sf_trig_hi'] = TrigSFeup*TrigSFmup
  events['sf_trig_lo'] = TrigSFedo*TrigSFmdo

class AnalysisProcessor(processor.ProcessorABC):
    def __init__(self, samples):

        self._samples = samples

        # Create the histograms
        # 'name' : hist.Hist("Ytitle", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat("syst", "syst"), hist.Bin("name", "X axis (GeV)", 20, 0, 100)),
        self._accumulator = processor.dict_accumulator({
        'dummy'      : hist.Hist("Dummy", hist.Cat("sample", "sample"), hist.Bin("dummy", "Number of events", 1, 0, 1)),
        'counts'     : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("counts",  "Counts", 1, 0, 10)),
        'l0pt'       : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("lep0pt",  "Leading lepton $p_{T}$ (GeV)", 10, 20, 120)),
        'PDF'        : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("PDF",     "Counts", 33, 0, 33)),
        'Scales'     : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("Scales",  "Counts", 9, 0, 9)),
        'l0eta'      : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("lep0eta", "Leading lepton $\eta$ ", 10, -2.5, 2.50)),
        'ept'        : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("ept",  "Electron $p_{T}$ (GeV)", 10, 20, 120)),
        'eeta'       : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("eeta", "Electron $\eta$ ", 10, -2.5, 2.50)),
        'mpt'        : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("mpt",  "Muon $p_{T}$ (GeV)", 10, 20, 120)),
        'meta'       : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("meta", "Muon $\eta$ ", 10, -2.5, 2.50)),
        'j0pt'       : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("j0pt",  "Leading jet $p_{T}$ (GeV)", 10, 0, 300)),
        'j0eta'      : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("j0eta", "Leading jet $\eta$ ", 12, -2.5, 2.50)),
        'invmass'    : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("invmass", "$m_{\ell\ell}$ (GeV) ", 20, 0, 300)),
        'invmass2'   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("invmass2", "$m_{\ell\ell}$ (GeV) ", 30, 70, 110)),
        'invmass_bb' : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("invmass", "$m_{\ell\ell}$ (GeV) ", 30, 70, 110)),
        'invmass_be' : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("invmass", "$m_{\ell\ell}$ (GeV) ", 30, 70, 110)),
        'invmass_ee' : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("invmass", "$m_{\ell\ell}$ (GeV) ", 30, 70, 110)),
        'njets'      : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("njets",   "Jet multiplicity", 6, 0, 6)),
        'nbtags'     : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("nbtags",  "b-tag multiplicity", 4, 0, 4)),
        'met'        : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("met",     "MET (GeV)", 10, 0, 200)),
        'ht'         : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("ht",      "H$_{T}$ (GeV)", 10, 0, 400)),
        'mt'         : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("mt",      "m$_{T}$ (GeV)", 10, 0, 150)),
        'mlb'        : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("mlb",     "m(l,b) (GeV)", 12, 0, 400)),
        'minDRjj'    : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("minDRjj", "min$\Delta$R(jj) ", 10, 0, 3)),
        'mjj'        : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("mjj",     "m(jj)( (GeV)", 10, 0, 200)),
        'ptjj'       : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("ptjj",    "p$_{T}$(jj) (GeV)", 15, 0, 300)),
        'counts_metg20': hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("counts",  "Counts", 1, 0, 10)),
        'counts_metl20': hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("counts",  "Counts", 1, 0, 10)),
        'counts_metg15': hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("counts",  "Counts", 1, 0, 10)),
        'counts_metl15': hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("counts",  "Counts", 1, 0, 10)),
        'counts_metg30': hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("counts",  "Counts", 1, 0, 10)),
        'counts_metl30': hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("counts",  "Counts", 1, 0, 10)),
        'counts_metg40': hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("counts",  "Counts", 1, 0, 10)),
        'counts_metl40': hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("counts",  "Counts", 1, 0, 10)),
        })

    @property
    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns

    # Main function: run on a given dataset
    def process(self, events):
        # Dataset parameters
        dataset = events.metadata["dataset"]
        histAxisName = self._samples[dataset]["histAxisName"]
        year         = self._samples[dataset]["year"]
        xsec         = self._samples[dataset]["xsec"]
        sow          = self._samples[dataset]["nSumOfWeights"]
        isData       = self._samples[dataset]["isData"]
        isSystSample = ('mtop' in histAxisName) or ('hdamp' in histAxisName) or ('UE' in histAxisName)
        doPS         = (histAxisName in ['tt', 'ttPS']) and events.PSWeight is not None and len(events.PSWeight[0])>=4
        doPDFunc = "sumPDFWeights" in self._samples[dataset]

        # Get the lumi mask for 5 TeV data
        golden_json_path = cafea_path("data/goldenJsons/Cert_306546-306826_5TeV_EOY2017ReReco_Collisions17_JSON.txt")

        if doPDFunc:
          sowPDF       = self._samples[dataset]["sumPDFWeights"]
          sowScale     = self._samples[dataset]["sumScaleWeights"]
          PDFnorm = 1./np.array(sowPDF)
          Scalenorm = 1./np.array(sowScale)
          scaleweights      = events.LHEScaleWeight.to_numpy()
          scaleweights_bins = ak.local_index(events.LHEScaleWeight)
          pdfweights        = events.LHEPdfWeight.to_numpy()
          pdfweights_bins   = ak.local_index(events.LHEPdfWeight)
          scaleweights      = scaleweights * Scalenorm
          pdfweights        = pdfweights * PDFnorm



        # Initialize objects
        met  = events.MET
        e    = events.Electron
        mu   = events.Muon
        tau  = events.Tau
        jets = events.Jet

        # Pre-selection (must be updated with 5TeV definitions)
        #e["idEmu"] = ttH_idEmu_cuts_E3(e.hoe, e.eta, e.deltaEtaSC, e.eInvMinusPInv, e.sieie)
        e["conept"] = coneptElec(e.pt, e.mvaTTH, e.jetRelIso)
        mu["conept"] = coneptMuon(mu.pt, mu.mvaTTH, mu.jetRelIso, mu.mediumId)
        e["btagDeepB"] = ak.fill_none(e.matched_jet.btagDeepB, -99)
        mu["btagDeepB"] = ak.fill_none(mu.matched_jet.btagDeepB, -99)

        # Muon selection
        mu["isLoose"] = MuonLoose(mu.pt, mu.eta, mu.dxy, mu.dz, mu.sip3d, mu.mediumPromptId, mu.btagDeepB, ptCut=20, etaCut=2.4)
        mu["isMVA"]= MuonMVA(mu.miniPFRelIso_all, mu.mvaTTH)

        # Electron selection
        #e['pt'] = GetElecPt(e.pt, e.eta, isdata = isData)
        #e['pt'] = GetElecPtSmear(e.pt, e.eta, isdata = isData)
        GetElecScale5TeV(e, run=306936, isData=isData)
        e['isLoose'] = ElecLoose(e.pt, e.eta, e.lostHits, e.sip3d, e.dxy, e.dz, e.btagDeepB, e.convVeto, e.mvaFall17V2noIso_WPL, 20, 2.4)
        e['isMVA']   = ElecMVA(e.miniPFRelIso_all, e.mvaTTH)

        # Build loose collections
        m_sel = mu[mu.isLoose & mu.isMVA]
        e_sel = e[e.isLoose & e.isMVA]
        m_fake = mu[mu.isLoose & (mu.isMVA == 0)]
        e_fake = e[e.isLoose & (e.isMVA == 0)]
        e0 = e_sel[ak.argmax(e_sel.pt,axis=-1,keepdims=True)]
        m0 = m_sel[ak.argmax(m_sel.pt,axis=-1,keepdims=True)]
  
        if not isData:
          AttachElectronSF(e_sel,year='5TeV')
          AttachMuonSF(m_sel,year='5TeV')
          AttachTrigSF(e0, m0, events)

        l_sel = ak.with_name(ak.concatenate([e_sel, m_sel], axis=1), 'PtEtaPhiMCandidate')

        events.MET['pt_raw'] = events.RawMET.pt

        if not isData:
          AddSFs(events, l_sel)

        events['isem'] = (ak.num(m_sel) == 1) & (ak.num(e_sel) == 1)
        events['ismm'] = (ak.num(m_sel) == 2) & (ak.num(e_sel) == 0)
        events['isee'] = (ak.num(m_sel) == 0) & (ak.num(e_sel) == 2)
        events['ise' ] = (ak.num(m_sel) == 0) & (ak.num(e_sel) == 1)
        events['ism' ] = (ak.num(m_sel) == 1) & (ak.num(e_sel) == 0)
        events['ise_fake' ] = (ak.num(m_sel) == 0) & ((ak.num(e_sel)) == 0) & (ak.num(e_fake) == 1)
        events['ism_fake' ] = (ak.num(m_sel) == 0) & ((ak.num(e_sel)) == 0) & (ak.num(m_fake) == 1)

        # Jet cleaning, before any jet selection
        vetos_tocleanjets = ak.with_name( l_sel, "PtEtaPhiMCandidate")
        tmp = ak.cartesian([ak.local_index(jets.pt), vetos_tocleanjets.jetIdx], nested=True)
        cleanedJets = jets[~ak.any(tmp.slot0 == tmp.slot1, axis=-1)] # this line should go before *any selection*, otherwise lep.jetIdx is not aligned with the jet index

        # Selecting jets and cleaning them
        jetptname = "pt_nom" if hasattr(cleanedJets, "pt_nom") else "pt"

        # Jet energy corrections
        if not isData:
          cleanedJets["pt_raw"] = (1 - cleanedJets.rawFactor)*cleanedJets.pt
          cleanedJets["mass_raw"] = (1 - cleanedJets.rawFactor)*cleanedJets.mass
          cleanedJets["pt_gen"] = ak.values_astype(ak.fill_none(cleanedJets.matched_gen.pt, 0), np.float32)
          cleanedJets["rho"] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, cleanedJets.pt)[0]
          events_cache = events.caches[0]
          corrected_jets = cleanedJets
          corrected_jets = jet_factory.build(cleanedJets, lazy_cache=events_cache)
          cleanedJets = corrected_jets
          cleanedJets_JESUp   = corrected_jets.JES_jes.up
          cleanedJets_JESDown = corrected_jets.JES_jes.down
          jetptname = "pt_nom" if hasattr(cleanedJets, "pt_nom") else "pt"
          met = met_factory.build(events.MET, corrected_jets, events_cache)
        else:
          cleanedJets["pt_raw"] = (1 - cleanedJets.rawFactor)*cleanedJets.pt
          cleanedJets["mass_raw"] = (1 - cleanedJets.rawFactor)*cleanedJets.mass
          cleanedJets["rho"] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, cleanedJets.pt)[0]
          events_cache = events.caches[0]
          corrected_jets = jet_factory_data.build(cleanedJets, lazy_cache=events_cache)
          cleanedJets = corrected_jets
          jetptname = "pt"#"pt_nom" if hasattr(cleanedJets, "pt_nom") else "pt"
          met = met_factory.build(events.MET, corrected_jets, events_cache)
        
        ################################ All this depends on jet pt
        jetptcut = 25
        cleanedJets["isGood"] = isTightJet(getattr(cleanedJets, jetptname), cleanedJets.eta, cleanedJets.jetId, jetPtCut=jetptcut)
        goodJets = cleanedJets[cleanedJets.isGood]
        if not isData:
          cleanedJets_JESUp["isGood"] = isTightJet(getattr(cleanedJets_JESUp, jetptname), cleanedJets_JESUp.eta, cleanedJets_JESUp.jetId, jetPtCut=jetptcut)
          cleanedJets_JESDown["isGood"] = isTightJet(getattr(cleanedJets_JESDown, jetptname), cleanedJets_JESDown.eta, cleanedJets_JESDown.jetId, jetPtCut=jetptcut)
          goodJetsJESUp = cleanedJets_JESUp[cleanedJets_JESUp.isGood]
          goodJetsJESDo = cleanedJets_JESDown[cleanedJets_JESDown.isGood]

        # Masks for the number of jets
        # Loose DeepJet WP
        # Recommendations for 94X: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation94X
        # Medium DeepCSV for 94X wp = 0.4941 , tight = 0.8001, loose = 0.1522
        # Medium DeepJet for 94X wp = 0.3033 
        # We're using DeepCSV...
        wp = 0.4941#0.8001 #0.4941 # medium
        isBtagJets = (goodJets.btagDeepB > wp)
        goodJets["isBtag"] = isBtagJets

        nbtags = ak.num(goodJets[isBtagJets])


        ######### SFs, weights, systematics ##########
        # Btag SF following 1a) in https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods ??? TODO
        if not isData:
          btagSF, btagSFUp, btagSFDo = GetBtagSF5TeV(goodJets.pt, goodJets.eta, goodJets.hadronFlavour, isBtagJets, True)
          isBtagJetsJESUp = (goodJetsJESUp.btagDeepB > wp)
          isBtagJetsJESDo = (goodJetsJESDo.btagDeepB > wp)
          nbtagsJESUp = ak.num(goodJetsJESUp[isBtagJetsJESUp])
          nbtagsJESDo = ak.num(goodJetsJESDo[isBtagJetsJESDo])
          btagSFJESUp = GetBtagSF5TeV(goodJetsJESUp.pt, goodJetsJESUp.eta, goodJetsJESUp.hadronFlavour, isBtagJetsJESUp, False)
          btagSFJESDo = GetBtagSF5TeV(goodJetsJESDo.pt, goodJetsJESDo.eta, goodJetsJESDo.hadronFlavour, isBtagJetsJESDo, False)
          goodJetsJESUp["isBtag"] = isBtagJetsJESUp
          goodJetsJESDo["isBtag"] = isBtagJetsJESDo

        ### Trigger
        trigem = (events.HLT.HIMu17) | (events.HLT.HIEle15_WPLoose_Gsf)
        trigee = (events.HLT.HIEle15_WPLoose_Gsf) | (events.HLT.HIEle17_WPLoose_Gsf)
        trigmm = (events.HLT.HIMu17)
        trige  = (events.HLT.HIEle20_WPLoose_Gsf)
        trigm  = events.HLT.HIMu17#HIL3Mu20

        # Single electron events: trige, only from HighEGJet
        # Single muon events: trigm, only from SingleMuon
        # ee events: trigee, only from HighEGJet
        # mm events: trigmm, only from SingleMuon
        # em events: in SingleMuon: pass trigmm, in HighEGJet: pass trigee and not trigmm

        if isData:
          if   histAxisName=='HighEGJet': 
            trigem = ((events.HLT.HIEle15_WPLoose_Gsf) | (events.HLT.HIEle17_WPLoose_Gsf)) & ((events.HLT.HIMu17)==0)
            trigee = (events.HLT.HIEle15_WPLoose_Gsf) | (events.HLT.HIEle17_WPLoose_Gsf)
            trigmm = np.zeros_like(events['event'], dtype=bool)
            trige  = events.HLT.HIEle20_WPLoose_Gsf
            trigm  = np.zeros_like(events['event'], dtype=bool)
          elif histAxisName=='SingleMuon': 
            trigem = events.HLT.HIMu17
            trigee = np.zeros_like(events['event'], dtype=bool)
            trigmm = (events.HLT.HIMu17)
            trige  = np.zeros_like(events['event'], dtype=bool)
            trigm  = events.HLT.HIMu17#HIL3Mu20

        # We need weights for: normalization, lepSF, triggerSF, pileup, btagSF...
        weights_dict = {}
        if (isData): genw = np.ones_like(events["event"])
        else:        genw = events["genWeight"]
        for ch_name in ["em", "e", "m", 'ee', 'mm']:
          weights_dict[ch_name] = coffea.analysis_tools.Weights(len(events),storeIndividual=True)
          weights_dict[ch_name].add("norm",genw if isData else (xsec/sow)*genw)
          if not isData: # Apply SFs
            if ch_name in ["em", "ee", "mm"]:
              weights_dict[ch_name].add("lepSF", events.sf_2l, events.sf_2l_hi, events.sf_2l_lo)
            else:
              weights_dict[ch_name].add("lepSF", ak.copy(events.sf_1l), ak.copy(events.sf_1l_hi), ak.copy(events.sf_1l_lo))
            weights_dict[ch_name].add("trigSF", ak.copy(events.sf_trig), ak.copy(events.sf_trig_hi), ak.copy(events.sf_trig_lo))
            weights_dict[ch_name].add("btagSF", ak.copy(btagSF), ak.copy(btagSFUp), ak.copy(btagSFDo))
          # PS = ISR, FSR (on ttPS only)
          if doPS: 
            i_ISRdown = 0; i_FSRdown = 1; i_ISRup = 2; i_FSRup = 3
            ISRUp = ak.flatten(events.PSWeight[ak.Array(ak.local_index(events.PSWeight)==i_ISRup)])
            ISRDo = ak.flatten(events.PSWeight[ak.Array(ak.local_index(events.PSWeight)==i_ISRdown)])
            FSRUp = ak.flatten(events.PSWeight[ak.Array(ak.local_index(events.PSWeight)==i_FSRup)])
            FSRDo = ak.flatten(events.PSWeight[ak.Array(ak.local_index(events.PSWeight)==i_FSRdown)])
            weights_dict[ch_name].add('ISR', np.ones_like(events["event"]), ISRUp, ISRDo)
            weights_dict[ch_name].add('FSR', np.ones_like(events["event"]), FSRUp, FSRDo)
          ## Factorization and renormalization (on ttbar only)
          #if doPDF:
          ## Prefire weights
          #weights_dict[ch_name].add("Prefiring", np.ones_like(events["event"]), prefUp, prefDown)
        emask = events.ise

        # Add systematics
        systList = ["norm"]
        systJets = ['JESUp', 'JESDo']#, 'JERUp', 'JERDown']
        if not isData and not isSystSample: systList = systList + ["lepSFUp","lepSFDown","btagSFUp", "btagSFDown"]+systJets#, "trigSFUp", "trigSFDown"] + systJets
        if doPS: systList += ['ISRUp', 'ISRDown', 'FSRUp', 'FSRDown']
        if not doSyst: systList = ["norm"]

        # Add selections...
        selections = PackedSelection(dtype='uint64')
        selections.add("em", ( (events.isem)&(trigem)))
        selections.add("ee", ( (events.isee)&(trigee)))
        selections.add("mm", ( (events.ismm)&(trigmm)))
        selections.add("e", ( (events.ise)&(trige)))
        selections.add("m", ( (events.ism)&(trigm)))
        selections.add("e_fake", ( (events.ise_fake)&(trige)))
        selections.add("m_fake", ( (events.ism_fake)&(trigm)))
        selections.add("metg20", (met.pt>=20))
        selections.add("metl20", (met.pt<20))
        selections.add("metg15", (met.pt>=20))
        selections.add("metl15", (met.pt<20))
        selections.add("metg30", (met.pt>=20))
        selections.add("metl30", (met.pt<20))
        selections.add("metg40", (met.pt>=20))
        selections.add("metl40", (met.pt<20))
        selections.add("incl", ak.ones_like(met.pt, dtype=bool))
 
        # Counts
        counts = np.ones_like(events['event'], dtype=float)
 
        # Initialize the out object
        hout = self.accumulator.identity()
        channels =['em', 'e', 'm', 'ee', 'mm', 'e_fake', 'm_fake'] 
        levels = ['incl', 'g1jet', 'g2jets', 'g4jets', '0b', '1b', '2b']


        # Count jets
        njets = ak.num(goodJets)
        ht = ak.sum(goodJets.pt,axis=-1)

        selections.add("g1jet",  (njets>=1))
        selections.add("g2jets", (njets >= 2))
        selections.add("g4jets", (njets >= 4))
        selections.add("0b", ((njets >= 4) & (nbtags==0)) )
        selections.add("1b", ((njets >= 4) & (nbtags==1)) )
        selections.add("2b", ((njets >= 4) & (nbtags>=2)) )

        if not isData: # JES systematics
          njetsJESUp = (ak.num(goodJetsJESUp))
          njetsJESDo = (ak.num(goodJetsJESDo))
          htJESUp = ak.sum(goodJetsJESUp.pt,axis=-1)
          htJESDo = ak.sum(goodJetsJESDo.pt,axis=-1)
          selections.add("g1jetJESUp",  (njetsJESUp >=1))
          selections.add("g2jetsJESUp", (njetsJESUp >= 2))
          selections.add("g4jetsJESUp", (njetsJESUp >= 4))
          selections.add("0bJESUp", ((njetsJESUp >= 4) & (nbtagsJESUp ==0)) )
          selections.add("1bJESUp", ((njetsJESUp >= 4) & (nbtagsJESUp ==1)) )
          selections.add("2bJESUp", ((njetsJESUp >= 4) & (nbtagsJESUp >=2)) )
          selections.add("g1jetJESDo",  (njetsJESDo >=1))
          selections.add("g2jetsJESDo", (njetsJESDo >= 2))
          selections.add("g4jetsJESDo", (njetsJESDo >= 4))
          selections.add("0bJESDo", ((njetsJESDo >= 4) & (nbtagsJESDo ==0)) )
          selections.add("1bJESDo", ((njetsJESDo >= 4) & (nbtagsJESDo ==1)) )
          selections.add("2bJESDo", ((njetsJESDo >= 4) & (nbtagsJESDo >=2)) )

        # Loop over the hists we want to fill
        for syst in systList:
          j0, drjj, mjj, ptjj = GetJetVariables(goodJets) if syst not in systJets else (GetJetVariables(goodJetsJESUp) if syst == 'JESUp' else GetJetVariables(goodJetsJESDo))
          for ch in channels:
            for lev in levels:
              #if syst in systJets and lev != 'incl': lev += syst
              cuts = [ch] + [lev + (syst if (syst in systJets and lev != 'incl') else '')]
              cut = selections.all(*cuts)
              weights = weights_dict[ch if not 'fake' in ch else ch[0]].weight(syst if not syst in (['norm']+systJets) else None)

              if syst == "JESUp":
                njets_var = njetsJESUp
                nbtags_var = nbtagsJESUp
                ht_var = htJESUp
              elif syst == "JESDo":
                njets_var = njetsJESDo
                nbtags_var = nbtagsJESDo
                ht_var = htJESDo
              else:
                njets_var = njets
                nbtags_var = nbtags
                ht_var = ht

              # Fill met norm histos
              cuts_metg20 = cuts+['metg20']; cuts_metl20 = cuts+['metl20']
              cuts_metg15 = cuts+['metg15']; cuts_metl15 = cuts+['metl15']
              cuts_metg30 = cuts+['metg30']; cuts_metl30 = cuts+['metl30']
              cuts_metg40 = cuts+['metg40']; cuts_metl40 = cuts+['metl40']
              cut_metg20 = selections.all(*cuts_metg20); cut_metl20 = selections.all(*cuts_metl20)
              cut_metg15 = selections.all(*cuts_metg15); cut_metl15 = selections.all(*cuts_metl15)
              cut_metg30 = selections.all(*cuts_metg30); cut_metl30 = selections.all(*cuts_metl30)
              cut_metg40 = selections.all(*cuts_metg40); cut_metl40 = selections.all(*cuts_metl40)
              weights_metg20 = weights[cut_metg20]; weights_metl20 = weights[cut_metl20]
              weights_metg15 = weights[cut_metg15]; weights_metl15 = weights[cut_metl15]
              weights_metg30 = weights[cut_metg30]; weights_metl30 = weights[cut_metl30]
              weights_metg40 = weights[cut_metg40]; weights_metl40 = weights[cut_metl40]
              hout['counts_metg20'].fill(sample=histAxisName, channel=ch, level=lev, counts=counts[cut_metg20], syst=syst, weight=weights_metg20)
              hout['counts_metl20'].fill(sample=histAxisName, channel=ch, level=lev, counts=counts[cut_metl20], syst=syst, weight=weights_metl20)
              if fillAll:
                hout['counts_metg15'].fill(sample=histAxisName, channel=ch, level=lev, counts=counts[cut_metg15], syst=syst, weight=weights_metg15)
                hout['counts_metl15'].fill(sample=histAxisName, channel=ch, level=lev, counts=counts[cut_metl15], syst=syst, weight=weights_metl15)
                hout['counts_metg30'].fill(sample=histAxisName, channel=ch, level=lev, counts=counts[cut_metg30], syst=syst, weight=weights_metg30)
                hout['counts_metl30'].fill(sample=histAxisName, channel=ch, level=lev, counts=counts[cut_metl30], syst=syst, weight=weights_metl30)
                hout['counts_metg40'].fill(sample=histAxisName, channel=ch, level=lev, counts=counts[cut_metg40], syst=syst, weight=weights_metg40)
                hout['counts_metl40'].fill(sample=histAxisName, channel=ch, level=lev, counts=counts[cut_metl40], syst=syst, weight=weights_metl40)
  
              ### We need to have 2 jets in order to calculate dijet observables
              dijet_cuts = cuts + ['g2jets' + (syst if syst in systJets else '')]
              dijet_cut = selections.all(*dijet_cuts)
              weights_dijet = weights[dijet_cut]
              if fillAll:
                hout['minDRjj'].fill(sample=histAxisName, channel=ch, level=lev, minDRjj=ak.flatten(drjj[dijet_cut]), syst=syst, weight=weights_dijet)
                hout['mjj'].fill(sample=histAxisName, channel=ch, level=lev, mjj=ak.flatten(mjj[dijet_cut]), syst=syst, weight=weights_dijet)
                hout['ptjj'].fill(sample=histAxisName, channel=ch, level=lev, ptjj=ak.flatten(ptjj[dijet_cut]), syst=syst, weight=weights_dijet)
  
              # Fill all the variables
              weights = weights[cut]
              jet0pt  = ak.flatten(j0.pt)
              jet0eta = ak.flatten(j0.eta)
              hout['counts'].fill(sample=histAxisName, channel=ch, level=lev, counts=counts[cut], syst=syst, weight=weights)
              hout['njets'].fill(sample=histAxisName, channel=ch, level=lev, njets=njets_var[cut], syst=syst, weight=weights)
              hout['nbtags'].fill(sample=histAxisName, channel=ch, level=lev, nbtags=nbtags_var[cut], syst=syst, weight=weights)
              hout['ht'].fill(sample=histAxisName, channel=ch, level=lev, ht=ht_var[cut], syst=syst, weight=weights)
              hout['met'].fill(sample=histAxisName, channel=ch, level=lev, met=met.pt[cut], syst=syst, weight=weights)
              if fillAll:
                if lev != 'incl': # Fill jet related variables when there is at least one jet
                  hout['j0pt'].fill(sample=histAxisName, channel=ch, level=lev, j0pt=jet0pt[cut], syst=syst, weight=weights)
                  hout['j0eta'].fill(sample=histAxisName, channel=ch, level=lev, j0eta=jet0eta[cut], syst=syst, weight=weights)
                if ch in ['e', 'e_fake']:
                  e = e_sel if ch == 'e' else e_fake
                  ept  = ak.flatten(e.pt [cut])
                  eeta = ak.flatten(e.eta[cut])
                  mt = ak.flatten(GetMT(e, met)[cut])
                  mlb = ak.flatten(GetMlb(e[cut], goodJets[cut]))
                  hout['ept' ].fill(sample=histAxisName, channel=ch, level=lev, ept=ept, syst=syst, weight=weights)
                  hout['eeta'].fill(sample=histAxisName, channel=ch, level=lev, eeta=eeta, syst=syst, weight=weights)
                  hout['mt'].fill(sample=histAxisName, channel=ch, level=lev, mt=mt, syst=syst, weight=weights)
                  #hout['mlb'].fill(sample=histAxisName, channel=ch, level=lev, mlb=mlb, syst=syst, weight=weights)
                elif ch in ['m', 'm_fake']:
                  m = m_sel if ch == 'm' else m_fake
                  mpt  = ak.flatten(m.pt[cut])
                  meta = ak.flatten(m.eta[cut])
                  mlb = ak.flatten(GetMlb(m[cut], goodJets[cut]))
                  mt = ak.flatten(GetMT(m, met)[cut])
                  hout['mpt'].fill(sample=histAxisName, channel=ch, level=lev, mpt=mpt, syst=syst, weight=weights)
                  hout['meta'].fill(sample=histAxisName, channel=ch, level=lev,meta = meta, syst=syst, weight=weights)
                  hout['mt'].fill(sample=histAxisName, channel=ch, level=lev, mt=mt, syst=syst, weight=weights)
                  #hout['mlb'].fill(sample=histAxisName, channel=ch, level=lev, mlb=mlb, syst=syst, weight=weights)
                elif ch in ['em', 'ee', 'mm']:
                  llpairs = ak.combinations(l_sel[cut], 2, fields=["l0","l1"])
                  mll = (llpairs.l0+llpairs.l1).mass # Invmass for leading two leps
                  mll_flat = ak.flatten(mll)
                  lep0pt = ak.flatten(llpairs.l0.pt)
                  lep0eta = ak.flatten(llpairs.l0.eta)
                  hout['invmass'].fill(sample=histAxisName, channel=ch, level=lev, invmass=mll_flat, syst=syst, weight=weights)
                  hout['invmass2'].fill(sample=histAxisName, channel=ch, level=lev, invmass2=mll_flat, syst=syst, weight=weights)
                  hout['l0pt'].fill(sample=histAxisName, channel=ch, level=lev, lep0pt=lep0pt, syst=syst, weight=weights)
                  hout['l0eta'].fill(sample=histAxisName, channel=ch, level=lev, lep0eta=lep0eta, syst=syst, weight=weights)
                  if ch == 'ee':
                    b0 = (abs(llpairs.l0.eta) < 1.479)
                    e0 = (abs(llpairs.l0.eta) > 1.479)
                    b1 = (abs(llpairs.l1.eta) < 1.479)
                    e1 = (abs(llpairs.l1.eta) > 1.479)
                    mll_bb = mll[(b0&b1)]
                    mll_be = mll[(b0&e1)|(e0&b1)]
                    mll_ee = mll[(e0&e1)]
                    mll_bb = ak.flatten(mll_bb)
                    mll_be = ak.flatten(mll_be)
                    mll_ee = ak.flatten(mll_ee)
                    weights_bb = weights[ak.flatten(b0&b1)]
                    weights_be = weights[ak.flatten((b0&e1)|(e0&b1))]
                    weights_ee = weights[ak.flatten(e0&e1)]
                    hout['invmass_bb'].fill(sample=histAxisName, channel=ch, level=lev, invmass=mll_bb, syst=syst, weight=weights_bb)
                    hout['invmass_be'].fill(sample=histAxisName, channel=ch, level=lev, invmass=mll_be, syst=syst, weight=weights_be)
                    hout['invmass_ee'].fill(sample=histAxisName, channel=ch, level=lev, invmass=mll_ee, syst=syst, weight=weights_ee)

              # Fill scale and pdf uncertainties
              if doPDFunc and syst == 'norm':
                scale_w = np.transpose(scaleweights[cut])*(weights)
                pdf_w   = np.transpose(pdfweights  [cut])*(weights)
                hout['Scales'].fill(sample=histAxisName, channel=ch, level=lev, Scales=ak.flatten(scaleweights_bins[cut]), syst="norm", weight=ak.flatten(scale_w))
                hout['PDF']   .fill(sample=histAxisName, channel=ch, level=lev, PDF   =ak.flatten(pdfweights_bins[cut]),   syst="norm", weight=ak.flatten(pdf_w))

        return hout
  
    def postprocess(self, accumulator):
        return accumulator

if __name__ == '__main__':
    # Load the .coffea files
    outpath= './coffeaFiles/'
    samples     = load(outpath+'samples.coffea')
    topprocessor = AnalysisProcessor(samples)

