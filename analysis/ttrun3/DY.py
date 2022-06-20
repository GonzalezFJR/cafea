from config import *

prdic = {'other': 'TTTo2L2Nu, tbarW, tW, WJetsToLNu, TTToSemiLep, WWTo2L2Nu, WZTo3LNu',   'DY': 'DYJetsToLL_M50, DY_M10to50' }

cat = {'level':level, 'syst':'norm'}
mass = 'invmass'
peak = 'invmass2'

plt = plotter(path, prDic=prdic, colors=colordic, lumi=lumi)
plt.SetDataName('pseudodata')

def GetYieldPlt(plotter, chan='ee', level='dilep', process='DY', var='invmass'):
  categories = {'channel':chan, 'level':level, 'syst':'norm'}
  if process == 'data': 
    h = plotter.GetHistogram(var, process=None, categories=categories)
    h.scale(plotter.lumi)
    h,_ = plt.GetData(var, h)
  else:
    h = plotter.GetHistogram(var, process=process, categories=categories)
    h.scale(plotter.lumi)
  y, e = h.values(sumw2=True)[()]
  y = sum(y); e = sum(e)
  if process=='data' and plotter.dataName.lower() == 'pseudodata': e = np.sqrt(y)
  return y, e


def GetRoutin(plt, chan='ee', level='dilep'):
  DYtot, DYtot_e = GetYieldPlt(plt, chan, level, process='DY', var='invmass')
  DYin , DYin_e  = GetYieldPlt(plt, chan, level, process='DY', var='invmass2')
  DYout, DYout_e = (DYtot-DYin, DYtot_e-DYin_e)
  Routin = DYout/DYin
  Routin_e = Routin*np.sqrt( (DYout_e/DYout)**2 + (DYin_e/DYin)**2 )
  return Routin, Routin_e

def DYDD(plt, level='dilep', save=False):
  Nin_ee, Nin_ee_e = GetYieldPlt(plt, 'ee', level, 'data', 'invmass2')
  Nin_mm, Nin_mm_e = GetYieldPlt(plt, 'mm', level, 'data', 'invmass2')
  Nin_em, Nin_em_e = GetYieldPlt(plt, 'em', level, 'data', 'invmass2')
  Nto_ee, Nto_ee_e = GetYieldPlt(plt, 'ee', level, 'data', 'invmass')
  Nto_mm, Nto_mm_e = GetYieldPlt(plt, 'mm', level, 'data', 'invmass')
  Nto_em, Nto_em_e = GetYieldPlt(plt, 'em', level, 'data', 'invmass')
  Nou_ee, Nou_ee_e = (Nto_ee - Nin_ee, Nto_ee_e - Nin_ee_e)
  Nou_mm, Nou_mm_e = (Nto_mm - Nin_mm, Nto_mm_e - Nin_mm_e)
  Nou_em, Nou_em_e = (Nto_em - Nin_em, Nto_em_e - Nin_em_e)

  kee = np.sqrt(Nin_ee / Nin_mm); kee_e = (1./2)*kee* np.sqrt( (Nin_ee_e/Nin_ee)**2 + (Nin_mm_e/Nin_mm)**2 )
  kmm = 1/kee ; kmm_e = kee_e

  Routin_ee, Routin_ee_e = GetRoutin(plt, 'ee', level)
  Routin_mm, Routin_mm_e = GetRoutin(plt, 'mm', level)

  Nout_ee  = Routin_ee * (Nin_ee - 0.5 * kee * Nin_em)
  Nout_ee_e = Nout_ee * np.sqrt( (Routin_ee_e/Routin_ee)**2 + (Nin_ee_e/Nin_ee)**2 + (kee_e/kee)**2 + (Nin_em_e/Nin_em)**2 )

  Nout_mm  = Routin_mm * (Nin_mm - 0.5 * kmm * Nin_em)
  Nout_mm_e = Nout_mm * np.sqrt( (Routin_mm_e/Routin_mm)**2 + (Nin_mm_e/Nin_mm)**2 + (kmm_e/kmm)**2 + (Nin_em_e/Nin_em)**2 )

  DYto_ee, DYto_ee_e = GetYieldPlt(plt, 'ee', level, 'DY', 'invmass')
  DYin_ee, DYin_ee_e = GetYieldPlt(plt, 'ee', level, 'DY', 'invmass2')
  DYout_ee, DYout_ee_e = (DYto_ee - DYin_ee, DYto_ee_e - DYin_ee_e)

  DYto_mm, DYto_mm_e = GetYieldPlt(plt, 'mm', level, 'DY', 'invmass')
  DYin_mm, DYin_mm_e = GetYieldPlt(plt, 'mm', level, 'DY', 'invmass2')
  DYout_mm, DYout_mm_e = (DYto_mm - DYin_mm, DYto_mm_e - DYin_mm_e)

  SFee   = Nout_ee/DYout_ee
  SFee_e = SFee * np.sqrt( (Nout_ee_e/Nout_ee)**2 + (DYout_ee_e/DYout_ee)**2 )
  SFmm   = Nout_mm/DYout_mm
  SFmm_e = SFmm * np.sqrt( (Nout_mm_e/Nout_mm)**2 + (DYout_mm_e/DYout_mm)**2 )

  SFem   = np.sqrt(SFmm * SFee)
  SFem_e = (1./2)*SFem*np.sqrt( (SFee_e/SFee)**2 + (SFmm_e/SFmm)**2 )
  
  if save:
    t = OutText(outpatho, 'DYDD_'+level, 'new', 'tex')
    t.bar()
    t.SetTexAlign('l c c c')
    t.line( ""+ t.vsep() + "$\mathrm{e}^+\mathrm{e}^-$" + t.vsep() + "$\mu^+\mu^-$" + t.vsep() + "$\mathrm{e}^\pm\mu^\mp$"); t.sep()
    t.line("N$_{in}$  (MC)"    + t.vsep() + "%1.1f $\pm$ %1.1f"%(DYin_ee,   DYin_ee_e  ) + t.vsep() + "%1.1f $\pm$ %1.1f"%(DYin_mm,   DYin_mm_e  ) + t.vsep() + "" ); 
    t.line("N$_{out}$ (MC)"    + t.vsep() + "%1.1f $\pm$ %1.1f"%(DYout_ee,  DYout_ee_e ) + t.vsep() + "%1.1f $\pm$ %1.1f"%(DYout_mm,  DYout_mm_e ) + t.vsep() + "" ); 
    t.line("R$_{out/im}$"      + t.vsep() + "%1.3f $\pm$ %1.3f"%(Routin_ee, Routin_ee_e) + t.vsep() + "%1.3f $\pm$ %1.3f"%(Routin_mm, Routin_mm_e) + t.vsep() + "" ); 
    t.line("$k_{\ell\ell}$"    + t.vsep() + "%1.3f $\pm$ %1.3f"%(kee, kee_e)             + t.vsep() + "%1.3f $\pm$ %1.3f"%(kmm, kmm_e) + t.vsep() + "" ); 
    t.line("N$_{in}$ (data)"   + t.vsep() + "%1.1f $\pm$ %1.1f"%(Nin_ee, Nin_ee_e)       + t.vsep() + "%1.1f $\pm$ %1.1f"%(Nin_mm, Nin_mm_e) + t.vsep() + "%1.1f $\pm$ %1.1f"%(Nin_em, Nin_em_e)  ); t.sep()
    t.line("N$_{out}$"         + t.vsep() + "%1.1f $\pm$ %1.1f"%(Nout_ee, Nout_ee_e)     + t.vsep() + "%1.1f $\pm$ %1.1f"%(Nout_mm, Nout_mm_e) + t.vsep() + "" ); t.sep()
    t.line("SF (data/MC) out"  + t.vsep() + "%1.3f $\pm$ %1.3f"%(SFee, SFee_e)           + t.vsep() + "%1.3f $\pm$ %1.3f"%(SFmm, SFmm_e) + t.vsep() + "%1.3f $\pm$ %1.3f"%(SFem, SFem_e)  ); t.sep()
    #t.line("Drell--Yan (MC)  " + t.vsep() + "%1.1f $\pm$ %1.1f"%(, ) + t.vsep() + "%1.1f $\pm$ %1.1f"%( , ) + t.vsep() + "%1.1f $\pm$ %1.1f"%( , )  );
    #t.line("Drell--Yan (Data)" + t.vsep() + "%1.1f $\pm$ %1.1f"%( , ) + t.vsep() + "%1.1f $\pm$ %1.1f"%( , ) + t.vsep() + "%1.1f $\pm$ %1.1f"%( , )  ); t.sep()
    t.write()
  return SFem, SFem_e

if __name__ == '__main__':
  DYDD(plt, level=level, save=True)


