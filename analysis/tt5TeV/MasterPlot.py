''' 
 This script produce a n-jet n-btag plot per category of the analysis
'''

from coffea import hist
from config import *
from cafea.plotter.plotter import saveHistos, loadHistos

channels = ['e', 'm']
levels = ['3j1b', '3j2b', '4j1b', '4j2b', 'g5j1b', 'g5j2b']
var = "counts"
outname = 'master'
outpath = path + 'masterhistos/'
if not os.path.exists(outpath):
    os.makedirs(outpath)

ncats = len(channels)*len(levels)
ncatsperchan = len(levels)
hmaster  = hist.Hist("Events", hist.Cat("process", "process"), hist.Cat('syst', 'syst'), hist.Bin("master", "Category", ncats, -0.5, ncats-0.5))
hperchan = hist.Hist("Events", hist.Cat("process", "process"), hist.Cat('syst', 'syst'), hist.Cat("channel", "channel"), hist.Bin("perchan", "Category", ncatsperchan, -0.5, ncatsperchan-0.5))

def CreateHistos(plt, systematics, process, channels, levels):
    global var, hmaster, hperchan, outname, outpath
    iteration = 0
    total = len(channels)*len(levels)*len(process)*len(systematics)
    for c in channels:
        for l in levels:
            for pr in process:
                bins     = np.array([(levels.index(l) + channels.index(c)*len(levels))], dtype=np.float64)
                binsChan = np.array([levels.index(l)                                  ], dtype=np.float64)
                for s in systematics:
                    print("\r[{:<100}] {:.2f} % ".format('#' * int( float(iteration)/total*100), float(iteration)/total*100),end='')
                    iteration += 1
                    if s.startswith('QCD'): continue
                    h = counts.integrate('process', pr).integrate('level', l).integrate('channel', c).integrate('syst', s)
                    if h.values() == {}: continue
                    _, vals = GetXYfromH1D(h, axis=var, mode='centers', errors=False, overflow=False)
                    if pr == 'data':
                        ndata = int(vals[0])
                        vals = np.ones(ndata, dtype=np.float64)
                        bins = np.array([bins[0]]*ndata, dtype=np.float64)
                        binsChan = np.array([binsChan[0]]*ndata, dtype=np.float64)
                    hmaster .fill(**{'syst':s, 'weight':vals, 'process':pr, 'master':bins})
                    hperchan.fill(**{'syst':s, 'weight':vals, 'process':pr, 'perchan':binsChan, 'channel':c})
    print("\r[{:<100}] {:.2f} % ".format('#' * int( float(iteration)/total*100), float(iteration)/total*100))
    saveHistos(outpath, outname, {'master':hmaster, 'perchan':hperchan}, verbose=True)


def DrawMasterHistogram(fname):
    plt = plotter(fname, prDic={},  bkgList=bkglist, colors=colordic, lumi=lumi, var='master')
    fig, ax, rax = plt.Stack('master', xtit='test', ytit='Events', dosyst=True, verbose=1, doNotSave=True)
    fig.savefig('test.png')

if __name__ == "__main__":
    fname = outpath + outname + '.pkl.gz'
    if not os.path.exists(fname) or force:
        plt = plotter(path, prDic=processDic,  bkgList=bkglist, lumi=lumi, var=var)
        counts = plt.GetHistogram(var)
        systematics = [x.name for x in list(counts.identifiers('syst'))]
        process     = [x.name for x in list(counts.identifiers('process'))]
        print('Saving histograms to file: ', fname)
        CreateHistos(plt, systematics, process, channels, levels)
        DrawMasterHistogram(fname)
    else:
        DrawMasterHistogram(fname)

