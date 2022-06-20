from config import *
from topcoffea.plotter.xsec import *
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
  x = xsec('tt', 0.03, {'tW':0.15, 'Nonprompt':0.2, 'DY':0.2, 'Diboson':0.2}, plotter=p, verbose=1, thxsec=922.45, experimental=['lepSF', 'trigSF', 'JES'], modeling=['UE', 'hdamp', 'ISR', 'FSR'], categories=categories) 
  x.AddModUnc('PDF$+\\alpha_{S}$', pdf, isRelative=True)
  x.AddModUnc('$\mu_R, \mu_F$ scales', scale, isRelative=True)
  x.ComputeXsecUncertainties()
  nom = x.xsecnom
  syst = percent(nom, x.GetTotalSystematic())
  stat = percent(nom, x.xsecunc['stat'])
  total = np.sqrt(syst*syst + stat*stat)
  return stat, syst, total


#GetStatUncForLumi(50)
#exit()

vlumi = np.linspace(10, 1000, 100)
unc = ([GetUncForLumi(x) for x in vlumi])
unc = np.array(unc)
stat = unc[:,0]
syst = unc[:,1]
tot  = unc[:,2]

fig, ax = plt.subplots(1, 1, figsize=(7,7))
plt.plot(vlumi, stat, 'r-')
plt.plot(vlumi, syst, 'b-')
plt.plot(vlumi, tot, 'k-')
ax.set_xlabel('Integrated luminosity (pb)')
ax.set_ylabel('Expected uncertainty on the $t\\bar{t}$ cross section (%)')
fig.savefig('curve.png')
