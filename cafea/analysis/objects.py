'''
 objects.py
 This script contains several functions that implement the object selection according to different object definitions.
 The functions are called with (jagged)arrays as imputs and return a boolean mask.
'''

import numpy as np
import awkward as ak
from cafea.modules.GetValuesFromJsons import get_param

### These functions have been synchronized with ttH ###

def isPresTau(pt, eta, dxy, dz, idDeepTau2017v2p1VSjet, minpt=20.0):
    return  (pt>minpt)&(abs(eta)<get_param("eta_t_cut"))&(abs(dxy)<get_param("dxy_tau_cut"))&(abs(dz)<get_param("dz_tau_cut"))&(idDeepTau2017v2p1VSjet>>1 & 1 ==1)

def isTightTau(idDeepTau2017v2p1VSjet):
    return (idDeepTau2017v2p1VSjet>>2 & 1)

def isTightJet(pt, eta, jet_id, jetPtCut=25.0):
    mask = ((pt>jetPtCut) & (abs(eta)<2.4) & (jet_id>1))
    return mask

def ttH_idEmu_cuts_E3(hoe, eta, deltaEtaSC, eInvMinusPInv, sieie):
    return (hoe<(0.10-0.00*(abs(eta+deltaEtaSC)>1.479))) & (eInvMinusPInv>-0.04) & (sieie<(0.011+0.019*(abs(eta+deltaEtaSC)>1.479)))

def smoothBFlav(jetpt,ptmin,ptmax,year,scale_loose=1.0):

    # Get the btag wp for the year
    if ((year == "2016") or (year == "2016APV")):
        wploose  = get_param("btag_wp_loose_L16")
        wpmedium = get_param("btag_wp_medium_L16")
    elif (year == "2017"):
        wploose  = get_param("btag_wp_loose_UL17")
        wpmedium = get_param("btag_wp_medium_UL17")
    elif (year == "2018"):
        wploose  = get_param("btag_wp_loose_UL18")
        wpmedium = get_param("btag_wp_medium_UL18")
    else:
        raise Exception(f"Error: Unknown year \"{year}\". Exiting...")

    x = np.minimum(np.maximum(0, jetpt - ptmin)/(ptmax-ptmin), 1.0)
    return x*wploose*scale_loose + (1-x)*wpmedium

def coneptElec(pt, mvaTTH, jetRelIso):
    conePt = (0.90 * pt * (1 + jetRelIso))
    return ak.where((mvaTTH>get_param("mva_TTH_e_cut")),pt,conePt)

def coneptMuon(pt, mvaTTH, jetRelIso, mediumId):
    conePt = (0.90 * pt * (1 + jetRelIso))
    return ak.where(((mvaTTH>get_param("mva_TTH_m_cut"))&(mediumId>0)),pt,conePt)

def isPresElec(pt, eta, dxy, dz, miniIso, sip3d, eleId):
    pt_mask    = (pt       > get_param("pres_e_pt_cut"))
    eta_mask   = (abs(eta) < get_param("eta_e_cut"))
    dxy_mask   = (abs(dxy) < get_param("dxy_cut"))
    dz_mask    = (abs(dz)  < get_param("dz_cut"))
    iso_mask   = (miniIso  < get_param("iso_cut"))
    sip3d_mask = (sip3d    < get_param("sip3d_cut"))
    return (pt_mask & eta_mask & dxy_mask & dz_mask & iso_mask & sip3d_mask & eleId)

def ElecLoose(pt, eta, lostHits, sip3d, dxy, dz, btagDeepFlavB, convVeto, elecMVA, ptCut = 20., etaCut=2.5):
  # Loose electrons for tt 5 TeV (includes everything except the mva ISO cuts)
  pt_mask = (pt > ptCut)
  eta_mask = (abs(eta) < etaCut)
  dxy_mask   = (abs(dxy) < 0.05)
  dz_mask    = (abs(dz)  < 0.1)
  sip3d_mask = (sip3d < 8)
  btag_mask = (btagDeepFlavB < 0.1522)
  lostHits_mask = (lostHits<1)
  convVeto_mask = (convVeto)
  elecMVA_mask = (elecMVA)
  etaSC_mask = ( (abs(eta) < 1.479) | (abs(eta) > 1.566) )
  return (pt_mask)&(eta_mask)&(etaSC_mask)&(dxy_mask)&(dz_mask)&(sip3d_mask)&(btag_mask)&(elecMVA_mask)&(lostHits_mask)&(convVeto_mask)

def ElecMVA(miniPFRelIso, lepMVA):
  return ( (miniPFRelIso<0.085) & (lepMVA>0.125) )
  
def MuonLoose(pt, eta, dxy, dz, sip3d, mediumPrompt, btagDeepFlavB, ptCut = 20., etaCut = 2.4):
  pt_mask = (pt > ptCut)
  eta_mask = (abs(eta) < etaCut)
  dxy_mask   = (abs(dxy) < 0.05)
  dz_mask    = (abs(dz)  < 0.1)
  sip3d_mask = (sip3d < 8)
  btag_mask = (btagDeepFlavB < 0.1522)
  id_mask = (mediumPrompt)
  return ( (id_mask)&(pt_mask)&(eta_mask)&(dxy_mask)&(dz_mask)&(sip3d_mask)&(btag_mask) )

def MuonMVA(miniPFRelIso, lepMVA):
   return ( (miniPFRelIso<0.325) & (lepMVA>0.55) )


def isPresMuon(dxy, dz, sip3d, eta, pt, miniRelIso):
    pt_mask    = (pt         > get_param("pres_m_pt_cut"))
    eta_mask   = (abs(eta)   < get_param("eta_m_cut"))
    dxy_mask   = (abs(dxy)   < get_param("dxy_cut"))
    dz_mask    = (abs(dz)    < get_param("dz_cut"))
    iso_mask   = (miniRelIso < get_param("iso_cut"))
    sip3d_mask = (sip3d      < get_param("sip3d_cut"))
    return (pt_mask & eta_mask & dxy_mask & dz_mask & iso_mask & sip3d_mask)

def isLooseElec(miniPFRelIso_all,sip3d,lostHits):
    return (miniPFRelIso_all<get_param("iso_cut")) & (sip3d<get_param("sip3d_cut")) & (lostHits<=1)

def isLooseMuon(miniPFRelIso_all,sip3d,looseId):
    return (miniPFRelIso_all<get_param("iso_cut")) & (sip3d<get_param("sip3d_cut")) & (looseId)

def isFOElec(conept, jetBTagDeepFlav, ttH_idEmu_cuts_E3, convVeto, lostHits, mvaTTH, jetRelIso, mvaFall17V2noIso_WP80, year):

    # Get the btag cut for the year
    if ((year == "2016") or (year == "2016APV")):
        bTagCut = get_param("btag_wp_medium_L16")
    elif (year == "2017"):
        bTagCut = get_param("btag_wp_medium_UL17")
    elif (year == "2018"):
        bTagCut = get_param("btag_wp_medium_UL18")
    else:
        raise Exception(f"Error: Unknown year \"{year}\". Exiting...")

    btabReq    = (jetBTagDeepFlav<bTagCut)
    ptReq      = (conept>get_param("fo_pt_cut"))
    qualityReq = (ttH_idEmu_cuts_E3 & convVeto & (lostHits==0))
    mvaReq     = ((mvaTTH>get_param("mva_TTH_e_cut")) | ((mvaFall17V2noIso_WP80) & (jetRelIso<get_param("fo_e_jetRelIso_cut"))))

    return ptReq & btabReq & qualityReq & mvaReq

def isFOMuon(pt, conept, jetBTagDeepFlav, mvaTTH, jetRelIso, year):

    # Get the btag cut for the year
    if ((year == "2016") or (year == "2016APV")):
        bTagCut = get_param("btag_wp_medium_L16")
    elif (year == "2017"):
        bTagCut = get_param("btag_wp_medium_UL17")
    elif (year == "2018"):
        bTagCut = get_param("btag_wp_medium_UL18")
    else:
        raise Exception(f"Error: Unknown year \"{year}\". Exiting...")

    btagReq = (jetBTagDeepFlav<bTagCut)
    ptReq   = (conept>get_param("fo_pt_cut"))
    mvaReq  = ((mvaTTH>get_param("mva_TTH_m_cut")) | ((jetBTagDeepFlav<smoothBFlav(0.9*pt*(1+jetRelIso),20,45,year)) & (jetRelIso < get_param("fo_m_jetRelIso_cut"))))
    return ptReq & btagReq & mvaReq

def tightSelElec(clean_and_FO_selection_TTH, mvaTTH):
    return (clean_and_FO_selection_TTH) & (mvaTTH > get_param("mva_TTH_e_cut"))

def tightSelMuon(clean_and_FO_selection_TTH, mediumId, mvaTTH):
    return (clean_and_FO_selection_TTH) & (mediumId>0) & (mvaTTH > get_param("mva_TTH_m_cut"))

def isClean(obj_A, obj_B, drmin=0.4):
    objB_near, objB_DR = obj_A.nearest(obj_B, return_metric=True)
    mask = ak.fill_none(objB_DR > drmin, True)
    return (mask)


### Run 3 definitions
###############################################################

###############################################################
