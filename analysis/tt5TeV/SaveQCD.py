'''
 Estimate the QCD background
 Usagel: 
   qcd = QCD(path, prDic=processDic, bkglist=bkglist, lumi=lumi)
   hqcd = qcd.GetQCD(var, categories)
'''

from analysis.tt5TeV.config import *
import time

if os.path.exists(path + 'QCD.pkl.gz'):
  print('WARNING: QCD file already exists in path... moving to ', path + 'old/QCD.pkl.gz.old')
  if not os.path.exists(path + 'old/'):
    os.makedirs(path + 'old/')
  os.rename(path + 'QCD.pkl.gz', path + 'old/QCD.pkl.gz.old')

print('Loading files from ', path, '... (this may take a while)') 
plt = plotter(path, prDic=processDic_noQCD,  bkgList=bkglist_noQCD, lumi=lumi)

# Get list of variables
varlist = plt.GetListOfVars()
variables = []
print('Getting list of variables for which QCD can be estimated...')
print("  -- skipping variables: ", var, end='')
for var in varlist:
  if not CheckHistoCategories(plt.GetHistogram(var), {'channel' : 'e_fake', 'process':['data']+bkglist_noQCD}, checkValues=True) or not CheckHistoCategories(plt.GetHistogram(var), {'channel' : 'm_fake', 'process':['data']+bkglist_noQCD}, checkValues=True):
    print(', ', var, end='')
    continue
  variables.append(var)

# Channels and levels
print('Getting levels and systematics...')
channels    = ['e', 'm']
levels      = [x.name for x in list(plt.GetHistogram(variables[0]).identifiers('level'))]
systematics = [x.name for x in list(plt.GetHistogram(variables[0]).identifiers('syst'))]
if 'norm' in systematics: systematics.remove('norm')

print('--------------------------------------------------')
print('Variables:', variables)
print('Channels:', channels)
print('Levels:', levels)
print('Systematics:', systematics)
print('--------------------------------------------------')

def GetQCDforVar(var):
  ''' Compute (data - MC) for a given variable (dense_axis) -- all channels and levels!! '''
  catfake = {'channel' : ['e_fake', 'm_fake']}
  h_data_fake = plt.GetHistogram(var, ['data']     , catfake, keepCats=True)#.group('channel', hist.Cat("channel", "channel"), {'e_fake':'e', 'm_fake':'m'}).group('process', hist.Cat("process", "process"), {'QCD':'data'} )
  h_mc_fake   = plt.GetHistogram(var, bkglist_noQCD, catfake, keepCats=True)#.group('channel', hist.Cat("channel", "channel"), {'e_fake':'e', 'm_fake':'m'}).group('process', hist.Cat("process", "process"), {'QCD':bkglist_noQCD})
  h_data_fake = GroupKeepOrder(h_data_fake, [['channel', 'channel', {'e':'e_fake', 'm':'m_fake'}], ['process', 'process', {'QCD':'data'       }]])
  h_mc_fake   = GroupKeepOrder(h_mc_fake  , [['channel', 'channel', {'e':'e_fake', 'm':'m_fake'}], ['process', 'process', {'QCD':bkglist_noQCD}]])
  h_mc_fake.scale(-1*lumi)
  FillDataSystCategories(h_data_fake, var)
  htot = (h_data_fake + h_mc_fake)
  return htot

def GroupCats(h, catdic):
  ''' Group categories '''
  for cat in catdic:
    h = h.group(cat, hist.Cat(cat, cat), {catdic[cat]:catdic[cat]})
  return h

def FillDataSystCategories(hdata, var):
  ''' Fill systematics for fake data distribution (using nominal) '''
  hnorm = hdata.copy()
  if CheckHistoCategories(hnorm, {'syst':'norm'}):
    hnorm = hnorm.integrate('syst', 'norm')
  if CheckHistoCategories(hnorm, {'process':'QCD'}):
    hnorm = hnorm.integrate('process', 'QCD')
  elif CheckHistoCategories(hnorm, {'process':'data'}):
    hnorm = hnorm.integrate('process', 'data')
  for l in levels:
    for c in channels:
      hnorm2 = hnorm.integrate('level', l).integrate('channel', c)
      if hnorm2.values() == {}: continue
      bins, vals = GetXYfromH1D(hnorm2, axis=var, mode='centers', errors=False, overflow=False)
      for s in systematics:
        if s == 'norm': continue
        hdata.fill(**{'syst':s, 'weight':vals, 'process':'QCD', 'channel':c, 'level':l, var:bins})

def GetQCDnorm(chan, level, sys=0):
  cat     = {'channel':chan, 'level':level}
  catfake = {'channel':chan+'_fake', 'level':level}
  if   sys== 1: countsData = 'counts_metl25'
  elif sys==-1: countsData = 'counts_metl15'
  else:         countsData = 'counts_metl20'
  data_metl20    = plt.GetYields(countsData, cat    , pr='data'       , overflow='none')
  mc_metl20      = plt.GetYields(countsData, cat    , pr=bkglist_noQCD, overflow='none')
  data_metl20_fk = plt.GetYields(countsData, catfake, pr='data'       , overflow='none')
  mc_metl20_fk   = plt.GetYields(countsData, catfake, pr=bkglist_noQCD, overflow='none')
  fact = (data_metl20 - mc_metl20)/(data_metl20_fk - mc_metl20_fk)
  return fact

def NormQCD(hqcd, chan, level):
  fact   = GetQCDnorm(chan, level)
  factUp = GetQCDnorm(chan, level, sys=1)
  factDo = GetQCDnorm(chan, level, sys=-1)
  cat = {'channel':chan, 'level':level}
  #for c in cat:
  #  hqcd = hqcd.integrate(c, cat[c])
  GroupKeepOrder(hqcd, [['channel', 'channel', {cat['channel']:cat['channel']}], ['level', 'level', {cat['level']:cat['level']}]])
  hqcdUp = hqcd.copy()
  hqcdDo = hqcd.copy()
  hqcd  .scale(fact)
  hqcdUp.scale(factUp)
  hqcdDo.scale(factDo)
  GroupKeepOrder(hqcdUp, [['syst', 'syst', {'QCDUp':'norm'}], ['process', 'process', {'QCD':'QCD'}]])
  GroupKeepOrder(hqcdDo, [['syst', 'syst', {'QCDDown':'norm'}], ['process', 'process', {'QCD':'QCD'}]])
  hqcd += hqcdUp
  hqcd += hqcdDo
  return hqcd

def GetQCD(qcdHist, level, chan):
  hqcd = qcdHist.copy()
  #hqcd = h.group('level', hist.Cat("level", "level"), {level:level}).group('channel', hist.Cat("channel", "channel"), {chan:chan})
  GroupKeepOrder(hqcd, [['level', 'level', {level:level}], ['channel', 'channel', {chan:chan}] ])
  hqcd = NormQCD(hqcd, chan, level)
  return hqcd

def GetQCDpar(inputs):
  qcdHist, var, level, chan, outdict = inputs
  h = GetQCD(qcdHist, level, chan)
  GroupKeepOrder(h, [['process', 'sample', {'QCD':'QCD'}]])
  if var not in outdict: outdict[var] = h
  else: outdict[var] += h
  outdict['progress'] += 1
  total = outdict['total']
  print("\r[{:<50}] {:.2f} %".format('#' * int(outdict['progress']/total*100/2), float(outdict['progress'])/total*100), end='')

################################################ RUN
# Initialize histograms
histos = {}
d0 = time.time()
progress = 0
total = len(variables)*len(levels)*len(channels)
QCDhistos = {}
print('Calculating distributions...')
ivar = 0; 
for var in variables:
  progress100 = (ivar / len(variables)) * 100
  print("\r[{:<50}] {:.2f} %".format('#' * int(progress100/2), progress100), end='')
  QCDhistos[var] = GetQCDforVar(var)
  # Normalize back to 1/pb
  QCDhistos[var].scale(1./lumi)
  ivar += 1

print('Grouping and normalizing...')
from multiprocessing import Pool, Manager
pool = Pool(nSlots)
manager = Manager()
outdict = manager.dict()
outdict['progress'] = 0
outdict['total'] = total

# Create inputs
inputs = []
for var in variables:
  for l in levels:
    for c in channels:
      inputs.append([QCDhistos[var], var, l, c, outdict])

# Run in parallel
print('Running in parallel with {} slots...'.format(nSlots))
pool.map(GetQCDpar, inputs)
pool.close()
pool.join()

# Save histograms
outdict = dict(outdict)
saveHistos(path, 'QCD', outdict)

print('\nQCD saved to ', path+'/QCD.pkl.gz')
#progress100 = (progress / total) * 100
#dt = time.time() - d0
#minutes = dt / 60
#seconds = dt % 60
#expected = (dt / progress * total) if progress != 0 else 0
#expected_min = expected / 60
#expected_sec = expected % 60
#print("\r[{:<50}] {:.2f} % ({:.0f}/{:.0f}) Time: {:.0f}m {:.0f}s (expected {:.0f}m {:.0f}s)".format('#' * int(progress100/2), progress100, progress, total, minutes, seconds, expected_min, expected_sec), end="")

