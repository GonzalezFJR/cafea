from config import *
from cafea.plotter.xsec import *
from PDFscaleUncertainties import *

#path = 'histos/run3/5jun2022/'
categories = {'channel':'em', 'level': 'g2jets', 'sign':'OS'}
categoriesPDF = {'channel':'em', 'level': 'g2jets'}

hdampup,hdampdo = GetModSystHistos(path, 'TTTo2L2Nu_hdamp', 'hdamp', var='counts')
tuneup , tunedo = GetModSystHistos(path, 'TTTo2L2Nu_UE', 'UE', var='counts')
pdf   = Get1bPDFUnc(  path, categories=categoriesPDF, sample='TTTo2L2Nu', doPrint=False)
scale = Get1binScaleUnc(path, categories=categoriesPDF, sample='TTTo2L2Nu', doPrint=False)

def GetUncForLumi(lumi):
  p = plotter(path, prDic=processDic, bkgList=bkglist, colors=colordic, lumi=lumi, var='counts')
  p.AddExtraBkgHist([hdampup, hdampdo, tuneup, tunedo], add=True)
  x = xsec('tt', 0.03, {'tW':0.15, 'Nonprompt':0.2, 'DY':0.2, 'Diboson':0.2}, plotter=p, verbose=1, thxsec=922.45, experimental=['eleceff', 'muoneff', 'trigSF', 'JES', 'PU'], modeling=['UE', 'hdamp', 'ISR', 'FSR'], categories=categories) 
  x.AddModUnc('PDF$+\\alpha_{S}$', pdf, isRelative=True)
  x.AddModUnc('$\mu_R, \mu_F$ scales', scale, isRelative=True)
  x.ComputeXsecUncertainties()

  x.xsecunc['eleceff'] = x.xsecnom*0.018

  mueff1 = x.xsecunc['muoneff'] # Muon eff unc from SFs
  mueff2 = x.xsecnom*0.005      # Extra 0.5% from phase space extrapolation
  x.xsecunc['muoneff'] = np.sqrt(mueff1*mueff1 + mueff2*mueff2)


  nom = x.xsecnom
  syst = percent(nom, x.GetTotalSystematic())
  stat = percent(nom, x.xsecunc['stat'])
  total = np.sqrt(syst*syst + stat*stat)
  return stat, syst, total

def GetStatUncForLumi(lumi, syst):
  p = plotter(path, prDic=processDic, bkgList=bkglist, colors=colordic, lumi=lumi, var='counts')
  x = xsec('tt', 0.03, {'tW':0.15, 'Nonprompt':0.2, 'DY':0.2, 'Diboson':0.2}, plotter=p, verbose=1, thxsec=922.45, experimental=[], modeling=[], categories=categories) 
  x.ComputeXsecUncertainties()
  nom = x.xsecnom
  stat = percent(nom, x.xsecunc['stat'])
  total = np.sqrt(syst*syst + stat*stat)
  return stat, syst, total
  
  

#GetStatUncForLumi(50)
#exit()

stat, syst, total = GetUncForLumi(100)

vlumi = np.linspace(10, 210, 100)
unc = ([GetStatUncForLumi(x, syst) for x in vlumi])
unc = np.array(unc)
stat = unc[:,0]
syst = unc[:,1]
tot  = unc[:,2]

fig, ax = plt.subplots(1, 1, figsize=(7,7))
plt.plot(vlumi, stat, 'r-', label='Systematic unc.')
plt.plot(vlumi, syst, 'b-', label='Statistic unc.')
plt.plot(vlumi, tot,  'k-', label='Total unc.')

legend = ax.legend(loc='upper right')


ax.set_xlabel('Integrated luminosity (pb$^{-1}$)')
ax.set_ylabel('Expected uncertainty on the $\mathrm{t\\bar{t}}$ cross section (%)')
fig.savefig('uncCurve.png')
