#!/usr/bin/env python
import lz4.frame as lz4f
import copy
import numpy as np
import awkward as ak
import coffea
np.seterr(divide='ignore', invalid='ignore', over='ignore')
from coffea import hist, processor
from coffea.analysis_tools import PackedSelection
from coffea.lumi_tools import LumiMask

from cafea.modules.GetValuesFromJsons import get_param
from cafea.analysis.objects import *
from cafea.analysis.corrections import  AttachMuonSFsRun3, AttachElecSFsRun3, AttachTrigSFsRun3
from cafea.analysis.selection import *
from cafea.modules.paths import cafea_path

class AnalysisProcessor(processor.ProcessorABC):
    def __init__(self, samples):

        self._samples = samples

        # Create the histograms
        # 'name' : hist.Hist("Ytitle", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat("syst", "syst"), hist.Bin("name", "X axis (GeV)", 20, 0, 100)),
        self._accumulator = processor.dict_accumulator({
        'counts'     : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Cat('sign', 'sign'), hist.Bin("counts",  "Counts", 1, 0, 10)),
        'l0pt'       : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Cat('sign', 'sign'), hist.Bin("l0pt",  "Leading lepton $p_{T}$ (GeV)", 10, 20, 120)),
        'l0eta'      : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Cat('sign', 'sign'), hist.Bin("l0eta", "Leading lepton $\eta$ ", 10, -2.5, 2.50)),
        'l1pt'       : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Cat('sign', 'sign'), hist.Bin("l1pt",  "Subleading lepton $p_{T}$ (GeV)", 10, 20, 120)),
        'l1eta'      : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Cat('sign', 'sign'), hist.Bin("l1eta", "Subleading lepton $\eta$ ", 10, -2.5, 2.50)),
        'ept'        : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Cat('sign', 'sign'), hist.Bin("ept",  "Electron $p_{T}$ (GeV)", 10, 20, 120)),
        'eeta'       : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Cat('sign', 'sign'), hist.Bin("eeta", "Electron $\eta$ ", 10, -2.5, 2.50)),
        'mpt'        : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Cat('sign', 'sign'), hist.Bin("mpt",  "Muon $p_{T}$ (GeV)", 10, 20, 120)),
        'meta'       : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Cat('sign', 'sign'), hist.Bin("meta", "Muon $\eta$ ", 10, -2.5, 2.50)),
        'invmass'    : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Cat('sign', 'sign'), hist.Bin("invmass", "$m_{\ell\ell}$ (GeV) ", 20, 0, 300)),

        'j0pt'       : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("j0pt",  "Leading jet $p_{T}$ (GeV)", 10, 0, 300)),
        'j0eta'      : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("j0eta", "Leading jet $\eta$ ", 12, -2.5, 2.50)),
        'deltaphi'   : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("deltaphi","$\Delta\\varphi (e\mu)$ (rad/$\pi$)", 10, 0, 1)),
        'njets'      : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Cat('sign', 'sign'), hist.Bin("njets",   "Jet multiplicity", 6, 0, 6)),
        "nbtags"    : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Cat('sign', 'sign'), hist.Bin("nbtags", "b-tag multiplicity ", 5, 0, 5)),
        'met'        : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("met",     "MET (GeV)", 10, 0, 200)),
        'ht'         : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("ht",      "H$_{T}$ (GeV)", 10, 0, 400)),
        'nvtxPU'         : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Bin("nvtxPU",      "Number of vertex", 40, 0, 80)),
        'ptll'       : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("level", "level"), hist.Cat('syst', 'syst'), hist.Cat('sign', 'sign'), hist.Bin("ptll",  "$p_{T}^{\ell\ell}$ (GeV)", 26, 40, 300)),

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

        # Golden JSON for the given dataset/year
        golden_json_path = cafea_path("data/goldenJsons/Cert_Collisions2022_356309_356615_Golden.json")
        lumi_mask = np.ones_like(events['event'], dtype=bool)
        if isData:
          lumi_mask = LumiMask(golden_json_path)(events.run,events.luminosityBlock)

        # Initialize objects
        met  = events.MET
        e    = events.Electron
        mu   = events.Muon
        jets = events.Jet

        # Pre-selection 

        # Muon selection
        mu['isGood'] = isMuonPOGT(mu, ptCut=20)
        mu['isExtra'] = isMuonPOGT(mu, ptCut=10) #forveto

        # Electron selection
        e['isGood'] = isElectronTight(e, ptCut=20, etaCut=2.4)
        e['isExtra'] = isElectronTight(e, ptCut=10, etaCut=2.4) #forveto

        # Build good collections
        m_sel = mu[mu.isGood]
        e_sel = e[e.isGood]

        # Leading electron and muon
        e0 = e_sel[ak.argmax(e_sel.pt,axis=-1,keepdims=True)]
        m0 = m_sel[ak.argmax(m_sel.pt,axis=-1,keepdims=True)]

        # Build loose collections
        m_extra = mu[mu.isExtra]
        e_extra = e[e.isExtra]

        # Build dilepton collections
        l_sel = ak.with_name(ak.concatenate([e_sel, m_sel], axis=1), 'PtEtaPhiMCandidate')
        l_sel_extra = ak.with_name(ak.concatenate([e_extra, m_extra], axis=1), 'PtEtaPhiMCandidate')
        llpairs = ak.combinations(l_sel, 2, fields=["l0","l1"])
        mll = (llpairs.l0+llpairs.l1).mass # Invmass for leading two leps
        deltaphi = (llpairs.l0.delta_phi(llpairs.l1))/np.pi

        leadinglep = l_sel[ak.argmax(l_sel.pt, axis=-1, keepdims=True)]
        subleadinglep = l_sel[ak.argmin(l_sel.pt, axis=-1, keepdims=True)]
        leadingpt = ak.flatten(leadinglep.pt) #ak.pad_none(l_sel.pt, 1)
        subleadingpt = ak.flatten(subleadinglep.pt) #ak.pad_none(l_sel.pt, 1)

        ### Attach scale factors -- This is for Run3 SFs !!! 
        if not isData:
          AttachMuonSFsRun3(m_sel)
          AttachElecSFsRun3(e_sel)
        l_sel = ak.with_name(ak.concatenate([e_sel, m_sel], axis=1), 'PtEtaPhiMCandidate')
        if not isData:
          AddSFsRun3(events, l_sel)

        # Event categories
        events['isem'] = (ak.num(m_sel) == 1) & (ak.num(e_sel) == 1) & (ak.num(l_sel_extra) <= 2) 
        events['ismm'] = (ak.num(m_sel) == 2) & (ak.num(e_sel) == 0) & (ak.num(l_sel_extra) <= 2)
        events['isee'] = (ak.num(m_sel) == 0) & (ak.num(e_sel) == 2) & (ak.num(l_sel_extra) <= 2)
        events['isOS'] = (ak.prod(l_sel.charge, axis=1) == -1) & (ak.num(l_sel_extra) <= 2)
        events['isSS'] = (ak.prod(l_sel.charge, axis=1) ==  1) & (ak.num(l_sel_extra) <= 2)

        # Trigger SFs !!
        if not isData: AttachTrigSFsRun3(events, e0, m0)

        # Initialize the containers
        hout = self.accumulator.identity()

        ## Add systematic uncertainties that depend on weights
        if (isData): genw = np.ones_like(events["event"])
        else:          genw = events["genWeight"]
        weights_dict = coffea.analysis_tools.Weights(len(events),storeIndividual=True)
        weights_dict.add("norm",genw if isData else (xsec/sow)*genw)
        if not isData: # Apply SFs
          weights_dict.add("trigSF", copy.deepcopy(events.SFtrigger), copy.deepcopy(events.SFtrigger_Up), copy.deepcopy(events.SFtrigger_Down))
          weights_dict.add("lepSF_muon", copy.deepcopy(events.sf_muon), copy.deepcopy(events.sf_hi_muon), copy.deepcopy(events.sf_lo_muon))
          weights_dict.add("lepSF_elec", copy.deepcopy(events.sf_elec), copy.deepcopy(events.sf_hi_elec), copy.deepcopy(events.sf_lo_elec))
          

        ################################ Jet selection
        jetptcut = 30
        metcut = 30
        j_isclean=ak.all(jets.metric_table(l_sel) > 0.4, axis=2)
        cleanedJets=jets[j_isclean]
        cleanedJets["isGood"] = isTightJet(cleanedJets.pt, cleanedJets.eta, cleanedJets.jetId, jetPtCut=jetptcut)
        goodJets = cleanedJets[cleanedJets.isGood]

        # Count jets
        njets = ak.num(goodJets)

        # HT is the sum of the jet pt for all jets with pt > 30 GeV
        ht = ak.sum(goodJets.pt,axis=-1)

        # j0 is the leading jet
        j0 = goodJets[ak.argmax(goodJets.pt,axis=-1,keepdims=True)]

        # b-tagging using Deep Jet algorithm and medium WP
        btagwpl = get_param("btag_wp_loose_UL18")
        btagwpm = get_param("btag_wp_medium_UL18")
        isBtagJetsMedium = (goodJets.btagDeepFlavB > btagwpm)
        nbtags = ak.num(goodJets[isBtagJetsMedium])       

        # Compute invariant mass
        mllvalues = np.where(ak.num(mll)==0, [[0]], mll)
        mllvalues = np.where(ak.num(mllvalues)>1, [[0]], mllvalues)
        mllvalues = ak.flatten(mllvalues, axis=1)
          
        # Trigger and met filters
        trig = trgPassNoOverlap(events,isData,dataset,year)  
        METfilters = PassMETfilters(events,isData)

        # Systematic uncertainties. For now, only related to lepton and trigger SFs
        systList = ["norm"]
        if not isData and not isSystSample: 
          systList = systList + [ "lepSF_elecUp","lepSF_elecDown","lepSF_muonUp","lepSF_muonDown","trigSFUp", "trigSFDown"]# 

        counts = np.ones_like(events['event'], dtype=float)

        # Initialize the out object, channels and levels
        channels = ['em', 'ee', 'mm'] 
        levels = ['dilep', 'g1jets','g2jets', 'offZ', 'metcut','g2jetsg1b', 'metcutg1b']

        #Adding secuancial preselection for debugging
        selections = PackedSelection(dtype='uint64')
        selections.add("em", ( (events.isem)&(trig)&(METfilters)))
        selections.add("ee", ( (events.isee)&(trig)&(METfilters)))
        selections.add("mm", ( (events.ismm)&(trig)&(METfilters)))
        selections.add("OS", ( (events.isOS)))
        selections.add("SS", ( (events.isSS)))
        selections.add("dilep",  (njets >= 0)&(leadingpt>35)&(subleadingpt>35)&(lumi_mask))
        selections.add("g1jets", (njets >= 1))
        selections.add("g2jets", (njets >= 2))
        selections.add("g2jetsg1b", (njets >= 2)&(nbtags>=1))
        selections.add("0jet", (njets == 0))
        selections.add("1jet", (njets == 1))
        selections.add("2jet", (njets == 2))
        selections.add("3jet", (njets == 3)) 
        selections.add("g4jet", (njets >= 4))
        selections.add("offZ",   ( np.abs(mllvalues-90) > 15)&(njets >= 2))
        selections.add("metcut", (met.pt >= metcut)&( np.abs(mllvalues-90) > 15)&(njets >= 2))
        selections.add("metcutg1b", (met.pt >= metcut)&( np.abs(mllvalues-90) > 15)&(njets >= 2)&(nbtags>=1))
        selections.add("mll", ( (mllvalues>20)))

        # Fill the histograms
        for syst in systList:
          # Weights for this systematic variation or nominal
          if syst == "norm":  weights = weights_dict.weight(None)
          else:               weights = weights_dict.weight(syst)
          for ch in channels:
            for lev in levels:
              # Some histograms are only filled for OS and SS
              for sign in ['OS', 'SS']:
                cuts = [ch] + [lev] + ['mll', 'dilep'] + [sign]
                cut   = selections.all(*cuts)
                weight   = weights[cut]

                # Just counts 
                hout['counts'].fill(sample=histAxisName, channel=ch, level=lev, counts=counts[cut],  syst=syst, sign=sign, weight=weight)
                hout['invmass'].fill(sample=histAxisName, channel=ch, level=lev, invmass=mllvalues[cut], syst=syst, sign=sign, weight=weight)
                hout['njets'].fill(sample=histAxisName, channel=ch, level=lev, njets=njets[cut], syst=syst, sign=sign, weight=weight)
                hout['nbtags'].fill(sample=histAxisName, channel=ch, level=lev, nbtags=nbtags[cut], syst=syst, sign=sign, weight=weight)
                hout['l0pt'] .fill(sample=histAxisName, channel=ch, level=lev, l0pt=ak.flatten(llpairs.l0.pt[cut]), syst=syst, sign=sign, weight=weight)
                hout['l0eta'].fill(sample=histAxisName, channel=ch, level=lev, l0eta=ak.flatten(llpairs.l0.eta[cut]), syst=syst, sign=sign, weight=weight)
                hout['l1pt'] .fill(sample=histAxisName, channel=ch, level=lev, l1pt=ak.flatten(llpairs.l1.pt[cut]), syst=syst, sign=sign, weight=weight)
                hout['l1eta'].fill(sample=histAxisName, channel=ch, level=lev, l1eta=ak.flatten(llpairs.l1.pt[cut]), syst=syst, sign=sign, weight=weight)
                hout['ptll'].fill(sample=histAxisName, channel=ch, level=lev, ptll=ak.flatten(llpairs.l0.pt[cut]+llpairs.l1.pt[cut]), syst=syst, sign=sign, weight=weight)

                if ch == 'em': 
                  hout['ept' ].fill(sample=histAxisName, channel=ch, level=lev, ept =ak.flatten(e_sel.pt [cut  ]), sign=sign, syst=syst, weight=weight)
                  hout['eeta'].fill(sample=histAxisName, channel=ch, level=lev, eeta=ak.flatten(e_sel.eta[cut  ]), sign=sign, syst=syst, weight=weight)
                  hout['mpt' ].fill(sample=histAxisName, channel=ch, level=lev, mpt =ak.flatten(m_sel.pt [cut  ]), sign=sign, syst=syst, weight=weight)
                  hout['meta'].fill(sample=histAxisName, channel=ch, level=lev, meta=ak.flatten(m_sel.eta[cut  ]), sign=sign, syst=syst, weight=weight)

              # All other histograms are filled only of OS
              cuts = [ch] + [lev] + ['mll', 'dilep']  
              cut   = selections.all(*cuts)
              weight   = weights[cut]

              hout['ht'].fill(sample=histAxisName, channel=ch, level=lev, ht=ht[cut], syst=syst, weight=weight)
              hout['deltaphi'].fill(sample=histAxisName, channel=ch, level=lev, deltaphi=ak.flatten(deltaphi[cut]), syst=syst, weight=weight)
              hout['met'].fill(sample=histAxisName, channel=ch, level=lev, met=met.pt[cut], syst=syst, weight=weight)

              if lev != 'dilep': # We must have at least 1 jet
                jet0pt  = ak.flatten(j0.pt[cut])
                jet0eta = ak.flatten(j0.eta[cut])
                hout['j0pt'].fill(sample=histAxisName, channel=ch, level=lev, j0pt=jet0pt, syst=syst, weight=weight)
                hout['j0eta'].fill(sample=histAxisName, channel=ch, level=lev, j0eta=jet0eta, syst=syst, weight=weight)

        return hout
  
    def postprocess(self, accumulator):
        return accumulator
