#!/usr/bin/env python
import lz4.frame as lz4f
import cloudpickle
import json
import pprint
import copy
import numpy as np
import awkward as ak
import coffea
import sys
np.seterr(divide='ignore', invalid='ignore', over='ignore')
from coffea import hist, processor
from coffea.util import load, save
from optparse import OptionParser
from coffea.analysis_tools import PackedSelection
from coffea.lumi_tools import LumiMask

from cafea.modules.GetValuesFromJsons import get_param
from cafea.analysis.objects import *
from cafea.analysis.corrections import GetBTagSF, GetBtagEff, AttachMuonSF, AttachElectronSF, GetPUSF, GetTriggerSF5TeV, jet_factory, jet_factory_data, met_factory, GetBtagSF5TeV, GetPUSF, AttachMuonSFsRun3, AttachElecSFsRun3, GetTriggerSF, GetTrigSFttbar
from cafea.analysis.selection import *
from cafea.modules.paths import cafea_path

doSyst = True
doJES = False

def AttachTrigSF(e0, m0, events):
  TrigSFe, TrigSFedo, TrigSFeup = GetTriggerSF5TeV(e0.pt, np.abs(e0.eta), 'e')
  TrigSFm, TrigSFmdo, TrigSFmup = GetTriggerSF5TeV(m0.pt, np.abs(m0.eta), 'm')
  TrigSFe   = np.nan_to_num(ak.flatten(ak.fill_none(TrigSFe, 1.)), nan=1)
  TrigSFedo = np.nan_to_num(ak.flatten(ak.fill_none(TrigSFedo, 1.)), nan=1)
  TrigSFeup = np.nan_to_num(ak.flatten(ak.fill_none(TrigSFeup, 1.)), nan=1)
  TrigSFm   = np.nan_to_num(ak.flatten(ak.fill_none(TrigSFm, 1.)), nan=1)
  TrigSFmdo = np.nan_to_num(ak.flatten(ak.fill_none(TrigSFmdo, 1.)), nan=1)
  TrigSFmup = np.nan_to_num(ak.flatten(ak.fill_none(TrigSFmup, 1.)), nan=1)
  events['trigger_sf']    = TrigSFe*TrigSFm
  events['trigger_sfUp'] = TrigSFeup*TrigSFmup
  events['trigger_sfDown'] = TrigSFedo*TrigSFmdo

class AnalysisProcessor(processor.ProcessorABC):
    def __init__(self, samples):

        self._samples = samples

        # Create the histograms
        # 'name' : hist.Hist("Ytitle", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat("syst", "syst"), hist.Bin("name", "X axis (GeV)", 20, 0, 100)),
        self._accumulator = processor.dict_accumulator({
        'dummy'      : hist.Hist("Dummy", hist.Cat("sample", "sample"), hist.Bin("dummy", "Number of events", 1, 0, 1)),
        'PDF'        : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("PDF",     "Counts", 103, 0, 103)),
        'Scales'     : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("Scales",  "Counts", 9, 0, 9)),
        'counts'     : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Cat('sign', 'sign'), hist.Bin("counts",  "Counts", 1, 0, 10)),
        'l0pt'       : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Cat('sign', 'sign'), hist.Bin("l0pt",  "Leading lepton $p_{T}$ (GeV)", 10, 20, 120)),
        'l0eta'      : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Cat('sign', 'sign'), hist.Bin("l0eta", "Leading lepton $\eta$ ", 10, -2.5, 2.50)),
        'l1pt'       : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Cat('sign', 'sign'), hist.Bin("l1pt",  "Subleading lepton $p_{T}$ (GeV)", 10, 20, 120)),
        'l1eta'      : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Cat('sign', 'sign'), hist.Bin("l1eta", "Subleading lepton $\eta$ ", 10, -2.5, 2.50)),
        'ept'        : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Cat('sign', 'sign'), hist.Bin("ept",  "Electron $p_{T}$ (GeV)", 10, 20, 120)),
        'eeta'       : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Cat('sign', 'sign'), hist.Bin("eeta", "Electron $\eta$ ", 10, -2.5, 2.50)),
        'mpt'        : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Cat('sign', 'sign'), hist.Bin("mpt",  "Muon $p_{T}$ (GeV)", 10, 20, 120)),
        'meta'       : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Cat('sign', 'sign'), hist.Bin("meta", "Muon $\eta$ ", 10, -2.5, 2.50)),
        'j0pt'       : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("j0pt",  "Leading jet $p_{T}$ (GeV)", 10, 0, 300)),
        'j0eta'      : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("j0eta", "Leading jet $\eta$ ", 12, -2.5, 2.50)),
        'deltaphi'   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("deltaphi","$\Delta\\varphi (e\mu)$ (rad/$\pi$)", 10, 0, 1)),
        'invmass'    : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("invmass", "$m_{\ell\ell}$ (GeV) ", 20, 0, 300)),
        'invmass2'   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("invmass2", "$m_{\ell\ell}$ (GeV) ", 30, 75, 105)),
        'invmass3'   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("invmass3", "$m_{\ell\ell}$ (GeV) ", 30, 85, 95)),
        'invmass_bb' : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("invmass", "$m_{\ell\ell}$ (GeV) ", 30, 75, 105)),
        'invmass_be' : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("invmass", "$m_{\ell\ell}$ (GeV) ", 30, 75, 105)),
        'invmass_ee' : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("invmass", "$m_{\ell\ell}$ (GeV) ", 30, 75, 105)),
        'njets'      : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Cat('sign', 'sign'), hist.Bin("njets",   "Jet multiplicity", 6, 0, 6)),
        "nbtagsl"    : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Cat('sign', 'sign'), hist.Bin("nbtagsl", "Loose btag multiplicity ", 5, 0, 5)),
        "nbtagsm"    : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Cat('sign', 'sign'), hist.Bin("nbtagsm", "Medium btag multiplicity ", 5, 0, 5)),
        'met'        : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("met",     "MET (GeV)", 10, 0, 200)),
        'ht'         : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("ht",      "H$_{T}$ (GeV)", 10, 0, 400)),
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
        doPS         = (histAxisName in ['tt', 'ttPS', 'TTTo2L2Nu']) and events.PSWeight is not None and len(events.PSWeight[0])>=4
        doPDFunc = "sumPDFWeights" in self._samples[dataset]
        if histAxisName in ['tt', 'ttPS', 'TTTo2L2Nu']: doSyst = True
        else: doSyst = False
        # Golden JSON !
        golden_json_path = cafea_path("data/goldenJsons/Cert_Collisions2022_356309_356615_Golden.json")
        lumi_mask = np.ones_like(events['event'], dtype=bool)
        if isData:
          lumi_mask = LumiMask(golden_json_path)(events.run,events.luminosityBlock)

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

        # Pre-selection 
        #e["idEmu"] = ttH_idEmu_cuts_E3(e.hoe, e.eta, e.deltaEtaSC, e.eInvMinusPInv, e.sieie)
        e["conept"] = coneptElec(e.pt, e.mvaTTH, e.jetRelIso)
        mu["conept"] = coneptMuon(mu.pt, mu.mvaTTH, mu.jetRelIso, mu.mediumId)
        e["btagDeepB"] = ak.fill_none(e.matched_jet.btagDeepB, -99)
        mu["btagDeepB"] = ak.fill_none(mu.matched_jet.btagDeepB, -99)

        # Muon selection
        mu["isLoose"] = MuonLoose(mu.pt, mu.eta, mu.dxy, mu.dz, mu.sip3d, mu.mediumId, mu.btagDeepB, ptCut=20, etaCut=2.4)
        mu["isMVA"]   = MuonMVA(mu.miniPFRelIso_all, mu.mvaTTH)
        mu['isGood'] = isMuonPOGT(mu, ptCut=20)

        # Electron selection
        e['isLoose'] = ElecLoose(e.pt, e.eta, e.lostHits, e.sip3d, e.dxy, e.dz, e.btagDeepB, e.convVeto, e.mvaNoIso_WPL, 20, 2.4)
        e['isMVA']   = ElecMVA(e.miniPFRelIso_all, e.mvaTTH)
        if not hasattr(events, "fixedGridRhoFastjetAll"): events["fixedGridRhoFastjetAll"] = np.zeros_like(events, dtype=float)
        AttachCutBasedTight(e, events.fixedGridRhoFastjetAll)
        e['isGood'] = isElectronTight(e, ptCut=20, etaCut=2.4)
     

        # Build loose collections
        #m_sel = mu[mu.isLoose & mu.isMVA]
        #e_sel = e[e.isLoose & e.isMVA]
        m_sel = mu[mu.isGood]
        e_sel = e[e.isGood]

        e0 = e_sel[ak.argmax(e_sel.pt,axis=-1,keepdims=True)]
        m0 = m_sel[ak.argmax(m_sel.pt,axis=-1,keepdims=True)]

        #print('Num elecs = ', np.sum(ak.num(e_sel)), '\n\n')

        l_sel = ak.with_name(ak.concatenate([e_sel, m_sel], axis=1), 'PtEtaPhiMCandidate')
        llpairs = ak.combinations(l_sel, 2, fields=["l0","l1"])
        mll = (llpairs.l0+llpairs.l1).mass # Invmass for leading two leps
        deltaphi = (llpairs.l0.delta_phi(llpairs.l1))/np.pi

        l_sel_padded = ak.pad_none(l_sel, 2)
        l0 = l_sel_padded[:,0]
        l1 = l_sel_padded[:,1]

        leadinglep = l_sel[ak.argmax(l_sel.pt, axis=-1, keepdims=True)]
        subleadinglep = l_sel[ak.argmin(l_sel.pt, axis=-1, keepdims=True)]
        leadingpt = ak.flatten(leadinglep.pt) #ak.pad_none(l_sel.pt, 1)
        subleadingpt = ak.flatten(subleadinglep.pt) #ak.pad_none(l_sel.pt, 1)
        ### Attach scale factors
        if not isData:
          #AttachElectronSF(e_sel,year='2018')
          #AttachMuonSF(m_sel,year='2018') #5TeV
          AttachMuonSFsRun3(m_sel)
          AttachElecSFsRun3(e_sel)
          #AttachTrigSF(e0, m0, events)

        l_sel = ak.with_name(ak.concatenate([e_sel, m_sel], axis=1), 'PtEtaPhiMCandidate')
        if not isData:
          AddSFsRun3(events, l_sel)
          #PadSFs2leps(events, e_sel, "elecsf")
          #PadSFs2leps(events, m_sel, "muonsf")

        events['isem'] = (ak.num(m_sel) == 1) & (ak.num(e_sel) == 1)
        events['ismm'] = (ak.num(m_sel) == 2) & (ak.num(e_sel) == 0)
        events['isee'] = (ak.num(m_sel) == 0) & (ak.num(e_sel) == 2)
        events['isOS'] = (ak.prod(l_sel.charge, axis=1) == -1)
        events['isSS'] = (ak.prod(l_sel.charge, axis=1) ==  1)
        #GetTriggerSF(2018, events, l0, l1) # from top EFT

        if not isData:
          e_padded = ak.pad_none(e_sel, 1)
          m_padded = ak.pad_none(m_sel, 1)
          ept = e_padded[:,0].pt
          mpt = m_padded[:,0].pt
          trigSF, trigUp, trigDo = GetTrigSFttbar(ept, mpt)
          events['trigger_sf'    ] = trigSF
          events['trigger_sfUp'  ] = trigUp
          events['trigger_sfDown'] = trigDo

        # Jet cleaning, before any jet selection
        vetos_tocleanjets = ak.with_name( l_sel, "PtEtaPhiMCandidate")
        tmp = ak.cartesian([ak.local_index(jets.pt), vetos_tocleanjets.jetIdx], nested=True)
        cleanedJets = jets[~ak.any(tmp.slot0 == tmp.slot1, axis=-1)] # this line should go before *any selection*, otherwise lep.jetIdx is not aligned with the jet index

        # Selecting jets and cleaning them
        jetptname = "pt_nom" if hasattr(cleanedJets, "pt_nom") else "pt"
        
        # Without JEC
        if doJES == False:
          cleanedJets["pt"]=(1 - cleanedJets.rawFactor)*cleanedJets.pt
          cleanedJets["E"]=(1 - cleanedJets.rawFactor)*cleanedJets.E
          cleanedJets["mass"]=(1 - cleanedJets.rawFactor)*cleanedJets.mass
       
        # Jet energy corrections
        met = events.MET

        
        ################################ Jet selection
        jetptcut = 30
        metcut = 30
        cleanedJets["isGood"] = isTightJet(cleanedJets.pt, cleanedJets.eta, cleanedJets.jetId, jetPtCut=jetptcut)
        goodJets = cleanedJets[cleanedJets.isGood]

        # Count jets
        njets = ak.num(goodJets)
        ht = ak.sum(goodJets.pt,axis=-1)
        j0 = jets[ak.argmax(jets.pt,axis=-1,keepdims=True)]

        btagwpl = get_param("btag_wp_loose_UL18")
        isBtagJetsLoose = (goodJets.btagDeepFlavB > btagwpl)
        isNotBtagJetsLoose = np.invert(isBtagJetsLoose)
        nbtagsl = ak.num(goodJets[isBtagJetsLoose])
        btagwpm = get_param("btag_wp_medium_UL18")
        isBtagJetsMedium = (goodJets.btagDeepFlavB > btagwpm)
        isNotBtagJetsMedium = np.invert(isBtagJetsMedium)
        nbtagsm = ak.num(goodJets[isBtagJetsMedium])       
        
        trig = trgPassNoOverlap(events,isData,dataset,year)  
        METfilters = PassMETfilters(events,isData)
        # We need weights for: normalization, lepSF, triggerSF, pileup, btagSF...
        if (isData): genw = np.ones_like(events["event"])
        else:        genw = events["genWeight"]
        weights_dict = coffea.analysis_tools.Weights(len(events),storeIndividual=True)
        weights_dict.add("norm",genw if isData else (xsec/sow)*genw)
        if not isData: # Apply SFs
          #weights_dict.add("lepSF", events.sf_2l, events.sf_2l_hi, events.sf_2l_lo)
          #weights_dict.add("eleceff", ak.copy(events.elecsf), ak.copy(events.elecsf_hi), ak.copy(events.elecsf_lo))
          #weights_dict.add("muoneff", ak.copy(events.muonsf), ak.copy(events.muonsf_hi), ak.copy(events.muonsf_lo))
          #weights_dict.add("trigSF", ak.copy(events.trigger_sf), ak.copy(events.trigger_sfUp), ak.copy(events.trigger_sfDown))
          #weights_dict.add('PU', GetPUSF( (events.Pileup.nTrueInt), '2018'),  GetPUSF( (events.Pileup.nTrueInt), '2018', 1), GetPUSF( (events.Pileup.nTrueInt), '2018', -1) ) 
          weights_dict.add("lepSF_muon", events.sf_muon, copy.deepcopy(events.sf_hi_muon), copy.deepcopy(events.sf_lo_muon))
          weights_dict.add("lepSF_elec", events.sf_elec, copy.deepcopy(events.sf_hi_elec), copy.deepcopy(events.sf_lo_elec))

        # PS = ISR, FSR (on ttPS only)
        if doPS: 
          i_ISRdown = 0; i_FSRdown = 1; i_ISRup = 2; i_FSRup = 3
          ISRUp = ak.flatten(events.PSWeight[ak.Array(ak.local_index(events.PSWeight)==i_ISRup)])
          ISRDo = ak.flatten(events.PSWeight[ak.Array(ak.local_index(events.PSWeight)==i_ISRdown)])
          FSRUp = ak.flatten(events.PSWeight[ak.Array(ak.local_index(events.PSWeight)==i_FSRup)])
          FSRDo = ak.flatten(events.PSWeight[ak.Array(ak.local_index(events.PSWeight)==i_FSRdown)])
          weights_dict.add('ISR', np.ones_like(events["event"]), ISRUp, ISRDo)
          weights_dict.add('FSR', np.ones_like(events["event"]), FSRUp, FSRDo)

        # Add systematics
        systList = ["norm"]
        systJets = []#['JESUp', 'JESDo'] if doJES else []
        #if not isData and not isSystSample: systList = systList + ["lepSFUp","lepSFDown", "trigSFUp", "trigSFDown", "PUUp", "PUDown"]+systJets
        #if not isData and not isSystSample: systList = systList + ["eleceffUp","eleceffDown", "muoneffUp", "muoneffDown", "trigSFUp", "trigSFDown", "PUUp", "PUDown"]+systJets

        if not isData and not isSystSample: systList = systList + [ "lepSF_elecUp","lepSF_elecDown","lepSF_muonUp","lepSF_muonDown"]
        if doPS: systList += ['ISRUp', 'ISRDown', 'FSRUp', 'FSRDown']

        if not doSyst: systList = ["norm"]

        # Counts
        counts = np.ones_like(events['event'], dtype=float)

        # Initialize the out object, channels and levels
        hout = self.accumulator.identity()
        channels = ['em', 'ee', 'mm'] 
        levels = ['dilep', 'g2jets', 'offZ', 'metcut','g2jetsg1b']

        # Add selections...

        #Adding secuancial preselection for debugging
        printevents = True
        if printevents == True:
           np.set_printoptions(threshold=sys.maxsize)
           printarray = np.array([events.event,events.luminosityBlock,events.run,trig,events.isem,events.isee,events.ismm,]) 
           #print(printarray.transpose())
           selections = PackedSelection(dtype='uint64')
           print("counts per selec level")
           selections.add("lumimask", lumi_mask)
           selections.add("trigger", trig)
           selections.add("metfilter", METfilters)
           cutlum = selections.all(*["lumimask"])
           print("lumimask",len(counts[cutlum]))
           cuttrig = selections.all(*["lumimask","trigger"])
           print("trigger",len(counts[cuttrig]))
           cutmetfilter = selections.all(*["lumimask","trigger","metfilter"])
           print("metfilter",len(counts[cutmetfilter]))
           selections.add("OS", ( (events.isOS)))
           cutos = selections.all(*["lumimask","trigger","em","OS"])
           print("em_os",len(counts[cutos]))
           selections.add("mll", ( (mllvalues>20)))
           selections.add("ptl1l2", ( (leadingpt>35) & (subleadingpt>35)))
           cutpt = selections.all(*["lumimask","trigger","em","OS","ptl1l2"])
           cutmll = selections.all(*["lumimask","trigger","em","OS","ptl1l2","mll"])
           print("pt",len(counts[cutpt]))
           print("mll",len(counts[cutmll]))
        selections.add("em", ( (events.isem)&(trig)))
        selections.add("ee", ( (events.isee)&(trig)))
        selections.add("mm", ( (events.ismm)&(trig)))
        selections.add("OS", ( (events.isOS)))
        selections.add("SS", ( (events.isSS)))
        selections.add("dilep",  (njets >= 0)&(leadingpt>35)&(lumi_mask))
        selections.add("g2jets", (njets >= 2))
        selections.add("g2jetsg1b", (njets >= 2)&(nbtagsm>=1))
        selections.add("0jet", (njets == 0))
        selections.add("1jet", (njets == 1))
        selections.add("2jet", (njets == 2))
        selections.add("3jet", (njets == 3)) 
        selections.add("g4jet", (njets >= 4))
        mllvalues = np.where(ak.num(mll)==0, [[0]], mll)
        mllvalues = np.where(ak.num(mllvalues)>1, [[0]], mllvalues)
        mllvalues = ak.flatten(mllvalues, axis=1)
        selections.add("offZ",   ( np.abs(mllvalues-90) > 15)&(njets >= 2))
        selections.add("metcut", (met.pt >= metcut)&( np.abs(mllvalues-90) > 15)&(njets >= 2))
        selections.add("mll", ( (mllvalues>20)))
        #printarray = np.array(events.event[cut])
        #print(printarray)

        ##### Loop over the hists we want to fill
        #for syst in systList:
        syst = "norm"
        for syst in systList:
         njets_var = njets
         ht_var = ht

         for ch in channels:
          if syst == "norm":
            for lev in ['0jet', '1jet', '2jet', '3jet', 'g4jet','g2jets','g2jetsg1b']:
              cuts = [ch] + [lev] + ['mll', 'dilep'] + ['OS']
              cut   = selections.all(*cuts)
              weights = weights_dict.weight(None)
              weights = weights[cut]
              mll_flat = mllvalues[cut]
              hout['invmass'].fill(sample=histAxisName, channel=ch, level=lev, invmass=mll_flat, syst=syst, weight=weights)
              hout['invmass2'].fill(sample=histAxisName, channel=ch, level=lev, invmass2=mll_flat, syst=syst, weight=weights)
              hout['invmass3'].fill(sample=histAxisName, channel=ch, level=lev, invmass3=mll_flat, syst=syst, weight=weights)
          for lev in levels:
            cuts = [ch] + [lev + (syst if (syst in systJets and lev == 'g2jets') else '')] + ['mll', 'dilep']  
            cutsOS = cuts + ['OS']
            cutsSS = cuts + ['SS']
            cut   = selections.all(*cutsOS)
            cutSS = selections.all(*cutsSS)
            weights = weights_dict.weight(syst if not syst in (['norm']+systJets) else None)

            # Fill all the variables
            weightsSS = weights[cutSS]
            weights = weights[cut]
            mll_flat = mllvalues[cut]
            #deltaphi_cut = deltaphi[cut]
            lep0pt = ak.flatten(llpairs.l0.pt[cut])
            lep0eta = ak.flatten(llpairs.l0.eta[cut])
            jet0pt  = ak.flatten(j0.pt[cut])
            jet0eta = ak.flatten(j0.eta[cut])
            hout['counts'].fill(sample=histAxisName, channel=ch, level=lev, counts=counts[cut],  syst=syst, sign='OS', weight=weights)
            hout['counts'].fill(sample=histAxisName, channel=ch, level=lev, counts=counts[cutSS], syst=syst, sign='SS', weight=weightsSS)
            hout['njets'].fill(sample=histAxisName, channel=ch, level=lev, njets=njets[cut], syst=syst, sign='OS', weight=weights)
            hout['njets'].fill(sample=histAxisName, channel=ch, level=lev, njets=njets[cutSS], syst=syst, sign='SS', weight=weightsSS)
            hout['nbtagsl'].fill(sample=histAxisName, channel=ch, level=lev, nbtagsl=nbtagsl[cut], syst=syst, sign='OS', weight=weights)
            hout['nbtagsl'].fill(sample=histAxisName, channel=ch, level=lev, nbtagsl=nbtagsl[cutSS], syst=syst, sign='SS', weight=weightsSS)
            hout['nbtagsm'].fill(sample=histAxisName, channel=ch, level=lev, nbtagsm=nbtagsm[cut], syst=syst, sign='OS', weight=weights)
            hout['nbtagsm'].fill(sample=histAxisName, channel=ch, level=lev, nbtagsm=nbtagsm[cutSS], syst=syst, sign='SS', weight=weightsSS)
            hout['ht'].fill(sample=histAxisName, channel=ch, level=lev, ht=ht[cut], syst=syst, weight=weights)
            hout['deltaphi'].fill(sample=histAxisName, channel=ch, level=lev, deltaphi=ak.flatten(deltaphi[cut]), syst=syst, weight=weights)
            hout['met'].fill(sample=histAxisName, channel=ch, level=lev, met=met.pt[cut], syst=syst, weight=weights)
            hout['l0pt'] .fill(sample=histAxisName, channel=ch, level=lev, l0pt=ak.flatten(llpairs.l0.pt[cut]), syst=syst, sign='OS', weight=weights)
            hout['l0eta'].fill(sample=histAxisName, channel=ch, level=lev, l0eta=ak.flatten(llpairs.l0.eta[cut]), syst=syst, sign='OS', weight=weights)
            hout['l1pt'] .fill(sample=histAxisName, channel=ch, level=lev, l1pt=ak.flatten(llpairs.l1.pt[cut]), syst=syst, sign='OS', weight=weights)
            hout['l1eta'].fill(sample=histAxisName, channel=ch, level=lev, l1eta=ak.flatten(llpairs.l1.pt[cut]), syst=syst, sign='OS', weight=weights)
            hout['l0pt'] .fill(sample=histAxisName, channel=ch, level=lev, l0pt=ak.flatten(llpairs.l0.pt[cutSS]), syst=syst, sign='SS', weight=weightsSS)
            hout['l0eta'].fill(sample=histAxisName, channel=ch, level=lev, l0eta=ak.flatten(llpairs.l0.eta[cutSS]), syst=syst, sign='SS', weight=weightsSS)
            hout['l1pt'].fill(sample=histAxisName, channel=ch, level=lev, l1pt=ak.flatten(llpairs.l1.pt[cutSS]), syst=syst, sign='SS', weight=weightsSS)
            hout['l1eta'].fill(sample=histAxisName, channel=ch, level=lev, l1eta=ak.flatten(llpairs.l1.pt[cutSS]), syst=syst, sign='SS', weight=weightsSS)
            hout['invmass'].fill(sample=histAxisName, channel=ch, level=lev, invmass=mll_flat, syst=syst, weight=weights)
            hout['invmass2'].fill(sample=histAxisName, channel=ch, level=lev, invmass2=mll_flat, syst=syst, weight=weights)
            hout['invmass3'].fill(sample=histAxisName, channel=ch, level=lev, invmass3=mll_flat, syst=syst, weight=weights)
            if lev != 'dilep':
              #jet1pt  = ak.flatten(j1.pt)
              #jet1eta = ak.flatten(j1.eta)
              hout['j0pt'].fill(sample=histAxisName, channel=ch, level=lev, j0pt=jet0pt, syst=syst, weight=weights)
              hout['j0eta'].fill(sample=histAxisName, channel=ch, level=lev, j0eta=jet0eta, syst=syst, weight=weights)
            if ch == 'em':
              e = e_sel; m = m_sel
              ept  = ak.flatten(e.pt [cut])
              eeta = ak.flatten(e.eta[cut])
              mpt  = ak.flatten(m.pt [cut])
              meta = ak.flatten(m.eta[cut])
              hout['ept' ].fill(sample=histAxisName, channel=ch, level=lev, ept =ak.flatten(e.pt [cut  ]), sign='OS', syst=syst, weight=weights)
              hout['eeta'].fill(sample=histAxisName, channel=ch, level=lev, eeta=ak.flatten(e.eta[cut  ]), sign='OS', syst=syst, weight=weights)
              hout['mpt' ].fill(sample=histAxisName, channel=ch, level=lev, mpt =ak.flatten(m.pt [cut  ]), sign='OS', syst=syst, weight=weights)
              hout['meta'].fill(sample=histAxisName, channel=ch, level=lev, meta=ak.flatten(m.eta[cut  ]), sign='OS', syst=syst, weight=weights)
              hout['ept' ].fill(sample=histAxisName, channel=ch, level=lev, ept =ak.flatten(e.pt [cutSS]), sign='SS', syst=syst, weight=weightsSS)
              hout['eeta'].fill(sample=histAxisName, channel=ch, level=lev, eeta=ak.flatten(e.eta[cutSS]), sign='SS', syst=syst, weight=weightsSS)
              hout['mpt' ].fill(sample=histAxisName, channel=ch, level=lev, mpt =ak.flatten(m.pt [cutSS]), sign='SS', syst=syst, weight=weightsSS)
              hout['meta'].fill(sample=histAxisName, channel=ch, level=lev, meta=ak.flatten(m.eta[cutSS]), sign='SS', syst=syst, weight=weightsSS)
            '''
            elif ch == 'ee':
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
              weights_bb = weights[(b0&b1)]
              weights_be = weights[((b0&e1)|(e0&b1))]
              weights_ee = weights[(e0&e1)]
              hout['invmass_bb'].fill(sample=histAxisName, channel=ch, level=lev, invmass=mll_bb, syst=syst, weight=weights_bb)
              hout['invmass_be'].fill(sample=histAxisName, channel=ch, level=lev, invmass=mll_be, syst=syst, weight=weights_be)
              hout['invmass_ee'].fill(sample=histAxisName, channel=ch, level=lev, invmass=mll_ee, syst=syst, weight=weights_ee)
            '''

            # Fill scale and pdf uncertainties
            if doPDFunc:
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

