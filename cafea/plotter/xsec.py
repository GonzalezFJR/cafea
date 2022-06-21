from numpy import sqrt
from cafea.plotter.plotter import GetSystListForHisto
from cafea.plotter.OutText import OutText
from cafea.plotter.Uncertainties import GetStringNumUnc as NU

def squaresum(*vals):
  return sqrt(sum([x*x for x in vals]))

def meanunc(nom, up, do):
  var1 = abs(nom-up)
  var2 = abs(nom-do)
  return (var1+var2)/2

def relative(nom, unc, do=None):
  if do is not None: unc = meanunc(nom, unc, do)
  return unc/nom

def percent(nom, unc, do=None):
  if do is not None: unc = meanunc(nom, unc, do)
  return relative(nom, unc)*100


class xsec:

  def __init__(self, signalName='', lumiunc=0, bkgunc={}, thxsec=0, plotter=None, categories={}, verbose=1, experimental=[], modeling=[]):
    self.expunc = {} # Uncertainty on signal
    self.modunc = {} # Uncertainty on signal
    self.bkg    = {} # Yields and uncertainty on bkg yields
    self.signalName = signalName
    self.signal = 0
    self.signalStat = 0
    self.signalSyst = 0
    self.lumi = 0
    self.lumiunc = lumiunc
    self.xsecnom = 0
    self.xsecunc = {} # Uncertainties on xsec
    self.totBkg = 0
    self.totBkgUnc = 0
    self.verbose = verbose
    self.SetThXsec(thxsec)
    self.SetOutPath('.')
    self.SetNames()
    if plotter is not None:
      self.FromPlotter(plotter, signalName, lumiunc, bkgunc, experimental, modeling, categories)

  def SetNames(self, dic={}):
    self.dictNames = dic

  def AddBkg(self, name, val, normunc=0, systunc=0, statunc=0):
    ''' Add a bkg name, yield and uncertainty '''
    if normunc != 0: normunc = normunc*val
    totunc = squaresum(normunc, systunc, statunc)
    self.bkg[name] = [val, totunc]

  def SetSignal(self, name, val, statunc=0, systunc=0):
    ''' Set signal events '''
    self.signalName = name
    self.signal = val
    self.signalStat = statunc
    self.signalSyst = systunc

  def SetData(self, val):
    ''' Set data evets '''
    self.data = val
    self.dataunc = sqrt(val)
    if self.verbose >=3: print("    data -- %1.2f  (%1.2f %s)"%(self.data, percent(self.data, self.dataunc), '%'))

  def AddExpUnc(self, name, val, isRelative=False):
    ''' Add a experimental uncertainty on signal '''
    if isRelative: val = val*self.signal
    if self.verbose >= 3: print('   New exp unc [%s]  -- %1.3f'%(name, val))
    self.expunc[name] = val

  def AddModUnc(self, name, val, isRelative=False):
    ''' Add a modeling uncertainty on signal '''
    if isRelative: val = val*self.signal
    if self.verbose >= 3: print('   New mod unc [%s]  -- %1.3f'%(name, val))
    self.modunc[name] = val

  def SetThXsec(self, val):
    ''' Set the theoretical cross section using to normalize signal events '''
    self.thxsec = val

  def SetBR(self, val):
    ''' Set branching ratio '''
    self.BR = val

  def SetLumi(self, lumi, lumiunc=None):
    ''' Set lumi and uncertainty '''
    self.lumi = lumi
    if lumiunc is not None: self.lumiunc = lumiunc

  def SetDoFiducial(self, val = True):
    ''' Boolean to activate printing the fiducial cross section '''
    self.doFiducial = val

  def SetFiduEvents(self,f):
    ''' Number of fiducual (unweighted) events '''
    self.nfidu = f

  def SetGenEvents(self, g):
    ''' Number of generated events in the sample '''
    self.ngen = g

  def SetOutPath(self, p):
    ''' Set the output path '''
    self.outpath = p


  #############################################################################
  def GetUncTable(self, name='uncertainties', form='%1.2f'):
    t = OutText(self.outpath, name, 'new', 'tex')
    t.bar() 
    t.SetTexAlign('l c c')
    nom = self.xsecnom
    t.line('Source ' + t.vsep() + '$\Delta\sigma_{t\\bar{t}}$ (pb)' + t.vsep() + ' $\Delta\sigma_{t\\bar{t}}/\sigma_{t\\bar{t}}$ (\%)'); t.sep()
    for s in self.expunc:
      u = self.xsecunc[s]
      lab = self.dictNames[s] if s in self.dictNames.keys() else s
      t.line('%s '%lab + t.vsep() + form%u + t.vsep() + form%(u/nom*100) + ' \%')
    t.bar() 
    for s in self.modunc:
      lab = self.dictNames[s] if s in self.dictNames.keys() else s
      u = self.xsecunc[s]
      t.line('%s '%lab + t.vsep() + form%u + t.vsep() + form%(u/nom*100) + ' \%')
    t.bar() 
    for b in self.bkg.keys():
      lab = self.dictNames[b] if b in self.dictNames.keys() else b
      u = self.xsecunc[b]
      t.line('%s '%lab + t.vsep() + form%u + t.vsep() + form%(u/nom*100) + ' \%')
    t.bar() 
    tots = self.GetTotalSystematic()
    lumi = self.xsecunc['lumi']
    stat = self.xsecunc['stat']
    total = sqrt(sum([x*x for x in [tots, lumi, stat]]))
    t.line('Total systematic       ' + t.vsep() + form%(tots ) + t.vsep() + form%(tots /nom*100) + ' \%'); t.sep()
    t.line('Integrated luminosity  ' + t.vsep() + form%(lumi ) + t.vsep() + form%(lumi /nom*100) + ' \%'); t.sep()
    t.line('Statistical uncertainty' + t.vsep() + form%(stat ) + t.vsep() + form%(stat /nom*100) + ' \%'); t.sep()
    t.line('Total                  ' + t.vsep() + form%(total) + t.vsep() + form%(total/nom*100) + ' \%'); t.sep()
    t.write()

  def GetYieldsTable(self, name='yields', form='%1.2f'):
    t = OutText(self.outpath, name, 'new', 'tex')
    t.bar()
    t.SetTexAlign('l c')
    t.line('Source' + t.vsep() + 'Number of events'); t.sep()
    for k in self.bkg.keys():
      y, e = self.bkg[k]
      lab = self.dictNames[k] if k in self.dictNames.keys() else k
      y, e = NU(y,e)
      t.line(lab + t.vsep() + y + t.pm() + e)
      #t.line(lab + t.vsep() + (form + t.pm() + form)%(y, e));
    t.sep()
    t.line('Total background' + t.vsep() +  ('%s'+ t.pm() + '%s')%(NU(self.totBkg, self.totBkgUnc))); t.sep()
    t.line('Expected $t\\bar{t}$ signal' + t.vsep() + ('%s' + t.pm() + '%s')%(NU(self.signal, self.signalStat))); t.sep();
    t.line('Observed' + t.vsep() + ('%i'+ t.pm() + '%i')%(self.data, self.dataunc)); t.sep()
    t.write()

  def GetCrossSectionTable(self):
    pass

  def GetFiducialTable(self):
    pass

  #############################################################################
  def GetXsec(self, data, bkg, signal):
    ''' Calculate the cross section given some inputs '''
    return (data-bkg)/signal * self.thxsec

  def ComputeTotalBackground(self):
    ''' Compute total background uncertainties and yield '''
    unc = []
    if self.verbose >= 3:
      print(">> Computing total background...")
    for k in self.bkg.keys():
      y, e = self.bkg[k]
      if self.verbose >= 3: print("     -- [%s] : %1.2f +/- %1.2f"%(k, y, e))
      self.totBkg += y
      unc.append(e)
    self.totBkgUnc = squaresum(*unc)
    if self.verbose >= 3: print("     -- [Total background] : %1.2f +/- %1.2f"%(self.totBkg, self.totBkgUnc))

  def ComputeXsecUncertainties(self):
    ''' Compute uncertainties on cross section '''
    if self.totBkg == 0: self.ComputeTotalBackground()
    if self.xsecnom == 0: self.ComputeCrossSection()
    if self.verbose >=3: print('>> Calculating uncertainties...')
    xsec = self.xsecnom

    # Stat
    if self.verbose >=3: print("  # Stat uncertainty")
    xsecup = self.GetXsec(self.data+self.dataunc, self.totBkg, self.signal)
    xsecdo = self.GetXsec(self.data-self.dataunc, self.totBkg, self.signal)
    statunc = meanunc(xsec, xsecup, xsecdo)
    if self.verbose >=3: print("    stat -- %1.2f  (%1.2f %s)"%(statunc, percent(xsec, statunc), '%'))
    self.xsecunc['stat'] = statunc

    # Lumi
    if self.lumiunc > 0:
      if self.verbose >=3: print("  # Lumi uncertainty")
      lumiunc = xsec*self.lumiunc 
      self.xsecunc['lumi'] = lumiunc
      if self.verbose >=3: print("    lumi -- %1.2f  (%1.2f %s)"%(lumiunc, percent(xsec, lumiunc), '%'))

    # Backgrounds
    if self.verbose >=3: print("  # Bkg uncertainties")
    bkg = self.bkg.keys()
    for b in bkg:
      y, u = self.bkg[b]
      xsecup = self.GetXsec(self.data, self.totBkg+u, self.signal)
      xsecdo = self.GetXsec(self.data, self.totBkg-u, self.signal)
      bunc = meanunc(xsec, xsecup, xsecdo)
      if self.verbose >=3: print("    %s -- %1.2f  (%1.2f %s)"%(b, bunc, percent(xsec, bunc), '%'))
      self.xsecunc[b] = bunc

    # Modeling
    mod = self.modunc.keys()
    if len(mod) > 0:
      if self.verbose >=3: print("  # Modeling uncertainties")
      for s in mod:
        xsecup = self.GetXsec(self.data, self.totBkg, self.signal + self.modunc[s])
        xsecdo = self.GetXsec(self.data, self.totBkg, self.signal - self.modunc[s])
        sunc = meanunc(xsec, xsecup, xsecdo)
        if self.verbose >=3: print("    %s -- %1.2f  (%1.2f %s)"%(s, sunc, percent(xsec, sunc), '%'))
        self.xsecunc[s] = sunc
    # Experimental 
    exp = self.expunc.keys()
    if len(exp) > 0:
      if self.verbose >=3: print("  # Experimental uncertainties")
      for s in exp:
        xsecup = self.GetXsec(self.data, self.totBkg, self.signal + self.expunc[s])
        xsecdo = self.GetXsec(self.data, self.totBkg, self.signal - self.expunc[s])
        sunc = meanunc(xsec, xsecup, xsecdo)
        if self.verbose >=3: print("    %s -- %1.2f  (%1.2f %s)"%(s, sunc, percent(xsec, sunc), '%'))
        self.xsecunc[s] = sunc

  def ComputeCrossSection(self):
    ''' Compute inclusive cross section '''
    if self.verbose >= 3: print(">> Getting the cross section")
    self.xsecnom = self.GetXsec(self.data, self.totBkg, self.signal)
    if self.verbose >= 3: print("     -- Cross section: %1.2f"%self.xsecnom)

  def ComputeFiducial(self):
    ''' Compute fiducial cross section, efficiency and acceptance '''
    self.acceptance = 0
    self.efficiency = 0 
    self.fiducial = 0
    self.fiducialStat = 0
    self.fiducialSyst = 0
    self.fiducialLumi = 0

  def GetTotalSystematic(self):
    return sqrt(sum([self.xsecunc[x]*self.xsecunc[x] for x in (list(self.bkg.keys()) + list(self.modunc.keys()) + list(self.expunc.keys())) ]))

  def GetTotalUncertainty(self):
    tots = self.GetTotalSystematic()
    lumi = self.xsecunc['lumi']
    stat = self.xsecunc['stat']
    total = sqrt(sum([x*x for x in [tots, lumi, stat]]))
    return total


  ##############################################################################################
  def FromPlotter(self, plotter, signalName, lumiunc=0, bkgunc={}, experimental=[], modeling=[], categories={}, hname='counts'):
    bkg = plotter.bkglist
    yields = plotter.GetYields(hname, cat=categories)
    histo = plotter.GetHistogram(hname, categories=categories)
    for p in bkg:
      if p == signalName:
        self.signal = yields[p]
      else:
        self.AddBkg(p, yields[p], normunc=bkgunc[p] if p in bkgunc.keys() else 0)
    self.SetLumi(plotter.lumi, lumiunc)
    self.ComputeTotalBackground()
    doData = plotter.doData('counts')
    if 'data' in yields: data = yields['data']
    else: data = self.totBkg + self.signal
    self.SetData(data)
    ### Systematic uncertainties
    systlist = GetSystListForHisto(histo)
    if len(modeling) == 0 and len(experimental) == 0:
      modeling = systlist
    elif len(modeling) == 0 and len(experimental) > 0:
      experimental2 = experimental.copy()
      for s in experimental:
        if not s in systlist:
          print('WARNING: systematic %s not found...'%s)
        else: experimental2.append(s)
      experimental = experimental2.copy()
    elif len(experimental) == 0 and len(modeling) > 0:
      modeling2 = modeling.copy()
      for s in modeling:
        if not s in systlist:
          print('WARNING: systematic %s not found...'%s)
        else: modeling2.append(s)
      modeling = modeling2.copy()
    for s in modeling:
      up, do = plotter.GetYieldsSystUpDown(s, signalName, var=hname, cat=categories)
      unc = ( abs(self.signal - up) + abs(self.signal - do))/2
      self.AddModUnc(s, unc) 
    for s in experimental:
      up, do = plotter.GetYieldsSystUpDown(s, signalName, var=hname, cat=categories)
      unc = ( abs(self.signal - up) + abs(self.signal - do))/2
      self.AddExpUnc(s, unc) 

