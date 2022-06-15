from config import *
from cafea.plotter.xsec import *

#path = 'histos/run3/5jun2022/'
categories = {'channel':'mm', 'level': 'g2jets', 'sign':'OS', 'syst':'norm'}

def GetStatUncForLumi(lumi):
  p = plotter(path, prDic=processDic, bkgList=bkglist, colors=colordic, lumi=lumi, var='counts')
  systlist = [x.name for x in p.hists['counts'].identifiers('syst')]
  nom   = (p.GetYields(cat=categories, pr='tt'))
  for x in systlist:
    cat = categories.copy()
    cat['syst'] = x
    var = (p.GetYields(cat=cat, pr='tt'))
    print('  [%s] -- %1.2f %s'%(x, abs(nom-var)/nom*100, '%'))
  exit()
  x = xsec('tt', 0.01, {'tW':0.1, 'WJets':0.2, 'DY':0.1, 'Diboson':0.3}, plotter=p, verbose=0, thxsec=922.45)
  x.ComputeXsecUncertainties()
  return percent(x.xsecnom, x.xsecunc['stat'])


GetStatUncForLumi(50)
exit()

vlumi = np.linspace(10, 1000, 100)
statunc = np.array([GetStatUncForLumi(x) for x in vlumi])

fig, ax = plt.subplots(1, 1, figsize=(7,7))
plt.plot(vlumi, statunc, 'r-')
ax.set_xlabel('Integrated luminosity (pb)')
ax.set_ylabel('Expected stat uncertainty on the $t\bar{t}$ cross section (%)')
fig.savefig('statcurve.png')
