'''
 Usage: python ljets/plotSyst.py -p histos/2jun2022_btagM_jet25/TT.pkl.gz
'''

from config import *

variation='lepSF_elec'
plt = plotter(path, prDic=processDic, bkgList=bkglist, colors=colordic, lumi=lumi)
outpath = '/nfs/fanae/user/andreatf/www/private/ttrun3/withLepSF_withoutJECPU/syst/' 
plt.SetOutpath(outpath)
plt.SetLumi(lumi, "pb$^{-1}$", "13.6 TeV")
plt.SetYRatioTit('Ratio')
output = "%s_%s_%s_%s"%(var, ch[0], level,variation)
plt.SetOutput(output)
plt.SetRatioRange(0.95,1.05)

#hdampup,hdampdo = GetModSystHistos(path, 'TT_hdamp', 'hdamp', var=var)
#tuneup , tunedo = GetModSystHistos(path, 'TT_UE', 'UE', var=var)
#plt.AddExtraBkgHist([hdampup, hdampdo, tuneup, tunedo], add=True)


def DrawComp(var, process, categories, labels=[], colors=[], lineStyle=[], scale=[]):
  plt.DrawComparison(var, process, categories, labels, colors, lineStyle, scale)

def DrawSyst(var, process, syst): 
  colors = ['k', 'r', 'b']
  normdict = {'channel':ch, 'level':level, 'syst':'norm'}
  h = plt.GetHistogram(var, process, normdict)
  systlist = [x.name for x in h.identifiers('syst')]
  upsyst = syst+'Up'; dosyst = syst+'Down';
  #if not upsyst in
  #dic = [{'channel':ch, 'level':level, 'syst':'norm'}, {'channel':ch, 'level':level, 'syst':syst+'Up'}, {'channel':ch, 'level':level, 'syst':syst'Do} ]

cat = [{'channel':ch, 'level':level, 'sign':'OS','syst':'norm'}, {'channel':ch, 'level':level, 'sign':'OS','syst':variation+'Up'}, {'channel':ch, 'level':level, 'sign':'OS', 'syst':variation+'Down'}]
if var is None: var = 'l0pt'
pr = 'tt'
labels = ['Nominal', 'up', 'down']
colors = ['k', 'r', 'b']

DrawComp(var, pr, cat, labels, colors)

