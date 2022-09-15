'''
 Estimate the QCD background
 Usagel: 
   qcd = QCD(path, prDic=processDic, bkglist=bkglist, lumi=lumi)
   hqcd = qcd.GetQCD(var, categories)
'''

from config import *

class QCD:

  def __init__(self, path, prDic, bkglist, lumi, categories={}, var=None):
    self.path = path
    self.prDic = prDic
    self.bkglist = bkglist
    self.lumi = lumi
    self.varlist = [var] if isinstance(var, str) else var

    self.categories = categories
    self.cat_fake = self.categories.copy()
    self.doAllChan = False
    if not 'channel' in self.categories:
      self.doAllChan = True
      self.categories['channel'] = ['e', 'm']
      self.cat_fake  ['channel'] = ['e_fake', 'm_fake']
    else:
      self.cat_fake['channel'] = [(c+'_fake' if not 'fake' in c else c) for c in self.cat_fake['channel']]

    # Initialize histograms and normalization 
    self.QCDhist = {}
    self.norm = {-1 : 1, 0:1, -1:1}

    # Load histos and normalizations
    self.LoadHistos()

  def LoadQCDForVar(self, var):
    ''' Compute (data - MC) for a given distribution (dense_axis) '''
    if self.doAllChan:
      h_data_fake = self.plt.GetHistogram(var, ['data'],{}).group('channel', hist.Cat("channel", "channel"), {'e':'e_fake', 'm':'m_fake'}).group('process', hist.Cat("process", "process"), {'QCD':'data'} )
      h_mc_fake   = self.plt.GetHistogram(var, bkglist, {}).group('channel', hist.Cat("channel", "channel"), {'e':'e_fake', 'm':'m_fake'}).group('process', hist.Cat("process", "process"), {'QCD':bkglist})
    else:
      h_data_fake = self.plt.GetHistogram(var, ['data'], self.cat_fake).group('process', hist.Cat("process", "process"), {'QCD':'data'})
      h_mc_fake   = self.plt.GetHistogram(var, bkglist, self.cat_fake).group('process', hist.Cat("process", "process"), {'QCD':bkglist})
    h_mc_fake.scale(-1*self.lumi)
    systlist = [x.name for x in list(h_mc_fake.identifiers('syst'))]
    h_data_fake = self.FillDataSystCategories(h_data_fake, systlist, var)
    htot = (h_data_fake + h_mc_fake)
    return htot

  def LoadHistos(self):
    ''' Load all the histograms '''
    self.plt = plotter(self.path, prDic=self.prDic,  bkgList=self.bkglist, lumi=self.lumi)
    self.cat_fake = self.categories.copy()
    if self.varlist is None: self.varlist = self.plt.GetListOfVars()
    for var in self.varlist:
      if not CheckHistoCategories(self.plt.GetHistogram(var), {'channel' : 'e_fake', 'process':['data']+bkglist}, checkValues=True) and not CheckHistoCategories(self.plt.GetHistogram(var), {'channel' : 'm_fake', 'process':['data']+bkglist}, checkValues=True):
        continue
      dense_axes = [x.name for x in self.plt.GetHistogram(var).dense_axes()]
      if len(dense_axes) > 1: continue
      if not var == dense_axes[0]: continue
      self.QCDhist[var] = self.LoadQCDForVar(var)

  def FillDataSystCategories(self, hdata, systlist, var):
    ''' Fill systematics for fake data distribution (using nominal) '''
    hnorm = hdata.integrate('process', 'QCD')
    if CheckHistoCategories(hnorm, {'syst':'norm'}):
      hnorm = hnorm.integrate('syst', 'norm')
    if not 'level' in self.categories:
      levels = [x.name for x in list(hdata.identifiers('level'))]
      for l in levels:
        for c in ['e', 'm']:
          if not c in [x.name for x in hnorm.identifiers('channel')]: continue
          hnorm2 = hnorm.integrate('level', l).integrate('channel', c)
          for s in systlist:
            if s == 'norm': continue
            bins, vals = GetXYfromH1D(hnorm2, axis=var, mode='centers', errors=False, overflow=False)
            hdata.fill(**{'syst':s, 'weight':vals, 'process':'QCD', 'channel':c, 'level':l, var:bins})
    else:
      bins, vals = GetXYfromH1D(hnorm, axis=var, mode='centers', errors=False, overflow=False)
      for s in systlist:
        if s == 'norm': continue
        hdata.fill(**{'syst':s, 'weight':vals, 'process':'QCD', var:bins})
    return hdata

  def GetNormalization(self, categories, sys=0):
    ''' Load normalization '''
    cat_fake = categories.copy()
    if self.doAllChan:
      cat_fake  ['channel'] = ['e_fake', 'm_fake']
    else:
      cat_fake['channel'] = [(c+'_fake' if not 'fake' in c else c) for c in cat_fake['channel']]
    countsData = 'counts_metl20'
    if   sys== 1: countsData = 'counts_metl30'
    elif sys==-1: countsData = 'counts_metl15'
    data_metl20    = self.plt.GetYields(countsData, categories, pr='data')
    mc_metl20      = self.plt.GetYields(countsData, categories, pr=self.bkglist)
    data_metl20_fk = self.plt.GetYields(countsData, cat_fake,   pr='data')
    mc_metl20_fk   = self.plt.GetYields(countsData, cat_fake,   pr=self.bkglist)
    fact = (data_metl20 - mc_metl20)/(data_metl20_fk - mc_metl20_fk)
    return fact

  def GetNorm(self, sys=0):
    return self.norm[sys]

  def GetQCD(self, var, categories={}, sys=0):
    fact = self.GetNormalization(categories, sys)
    print('sys = ', sys, 'fact = ', fact)
    h = self.QCDhist[var].copy()
    for cat in categories: 
      h = h.integrate(cat, categories[cat])
    h.scale(fact)
    if sys==1: # Up
      #h = GroupKeepOrder(h, [['syst', 'syst', {'QCDUp':'norm'}, 'keep']])
      h = GroupKeepOrder(h, [['syst', 'syst', {'QCDUp':'norm'}], ['process', 'process', {'QCD':'QCD'}]])
    elif sys==-1: # Down
      #h = GroupKeepOrder(h, ['syst', 'syst', {'QCDDown':'norm'}])
      h = GroupKeepOrder(h, [['syst', 'syst', {'QCDDown':'norm'}], ['process', 'process', {'QCD':'QCD'}]])
    return h


