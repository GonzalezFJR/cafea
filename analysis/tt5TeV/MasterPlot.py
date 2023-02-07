''' 
 This script produce a n-jet n-btag plot per category of the analysis
'''

from coffea import hist
from config import *
from cafea.plotter.plotter import saveHistos, loadHistos, DrawUncPerBin, DivideHistWithErrors
from PDFscaleUncertainties import Get1binScaleUnc, Get1bPDFUnc
import matplotlib.pyplot as matplt
import matplotlib

channels = ['e', 'm']
levels = ['3j1b', '3j2b', '4j1b', '4j2b', 'g5j1b', 'g5j2b']
var = "counts"
outname = 'master'

colorchan = 'blue'

outpath = path + 'masterhistos/'
if not os.path.exists(outpath):
    os.makedirs(outpath)

ncats = len(channels)*len(levels)
ncatsperchan = len(levels)
hmastQCD = hist.Hist("Events", hist.Cat("process", "process"), hist.Cat('syst', 'syst'), hist.Bin("mastQCD", "Category", ncats, -0.5, ncats-0.5))
hmaster  = hist.Hist("Events", hist.Cat("process", "process"), hist.Cat('syst', 'syst'), hist.Bin("master", "Category", ncats, -0.5, ncats-0.5))
hdists   = hist.Hist("Events", hist.Cat("process", "process"), hist.Cat('syst', 'syst'), hist.Cat("channel", "channel"), hist.Bin("shapes", "Category", 6 + 7*5, -0.5, (6 + 7*5)-0.5))
hperchan = hist.Hist("Events", hist.Cat("process", "process"), hist.Cat('syst', 'syst'), hist.Cat("channel", "channel"), hist.Bin("perchan", "Category", ncatsperchan, -0.5, ncatsperchan-0.5))

def CreateHistos(plt, systematics, process, channels, levels):
    global var, hmaster, hperchan, outname, outpath
    mvascore = plt.GetHistogram('MVAscore')
    drjj     = plt.GetHistogram('medianDRjj')
    iteration = 0
    total = len(channels)*len(levels)*len(process)*len(systematics)
    for c in channels:
        for l in levels:
            for pr in process:
                bins     = np.array([(levels.index(l) + channels.index(c)*len(levels))], dtype=np.float64)
                binsChan = np.array([levels.index(l)                                  ], dtype=np.float64)
                # Bin shapes
                nbins = 6 if l == '3j1b' else 7
                bin0  = (7*levels.index(l) - 1) if l != '3j1b' else 0
                binshape = np.linspace(bin0, bin0 + nbins -1, nbins, dtype=np.float64)
                for s in systematics:
                    print("\r[{:<100}] {:.2f} % ".format('#' * int( float(iteration)/total*100), float(iteration)/total*100),end='')
                    iteration += 1
                    # Counts
                    h = counts.integrate('process', pr).integrate('level', l).integrate('channel', c).integrate('syst', s)
                    hshap = mvascore.integrate('process', pr).integrate('level', l).integrate('channel', c).integrate('syst', s) if l == '3j1b' else drjj.integrate('process', pr).integrate('level', l).integrate('channel', c).integrate('syst', s)
                    if h.values() == {}: continue
                    _, vals, staterr = GetXYfromH1D(h, axis=var, mode='centers', errors=True, overflow=False)
                    if hshap.values() == {}: 
                        print('\nWARNING: No shape for ', pr, c, l, s)
                        vshap = np.zeros_like(binshape, dtype=np.float64)
                    else:
                        _, vshap = GetXYfromH1D(hshap, axis=hshap.dense_axes()[0].name, mode='centers', errors=False, overflow=False)
                        #vshap = vshap[1:]
                    # In principle, do not use QCD uncertainty
                    if s.startswith('QCD'): 
                        hmastQCD.fill(**{'syst':s, 'weight':vals, 'process':pr, 'mastQCD':bins})
                        continue
                    # Data is different to keep stat unc
                    if pr == 'data':
                        ndata = int(vals[0])
                        vals = np.ones(ndata, dtype=np.float64)
                        bins = np.array([bins[0]]*ndata, dtype=np.float64)
                        binsChan = np.array([binsChan[0]]*ndata, dtype=np.float64)
                        # binshape
                        binshape_data = []
                        vshap_data = []
                        for w, b in zip(vshap, binshape):
                            for i in range(int(w)):
                                binshape_data.append(b)
                                vshap_data.append(1)
                        binshape_data = np.array(binshape_data, dtype=np.float64)
                        vshap_data = np.array(vshap_data, dtype=np.float64)
                        hdists  .fill(**{'syst':s, 'weight':vshap_data, 'process':pr, 'shapes':binshape_data, 'channel':c})
                    else:
                        hdists  .fill(**{'syst':s, 'weight':vshap, 'process':pr, 'shapes':binshape, 'channel':c})
                    if s == 'norm': # Fill stat unc
                        statUp = vals + staterr
                        statDn = vals - staterr
                        for sname, svals in {'statUp':statUp, 'statDown':statDn}.items():
                            hperchan.fill(**{'syst':sname, 'weight':svals, 'process':pr, 'perchan':binsChan, 'channel':c})
                            hmaster .fill(**{'syst':sname, 'weight':svals, 'process':pr, 'master':bins})
                            hmastQCD.fill(**{'syst':sname, 'weight':svals, 'process':pr, 'mastQCD':bins})
                    hperchan.fill(**{'syst':s, 'weight':vals, 'process':pr, 'perchan':binsChan, 'channel':c})
                    hmaster .fill(**{'syst':s, 'weight':vals, 'process':pr, 'master':bins})
                    hmastQCD.fill(**{'syst':s, 'weight':vals, 'process':pr, 'mastQCD':bins})
            ##### Add tt modeling
            bins     = np.array([(levels.index(l) + channels.index(c)*len(levels))], dtype=np.float64)
            binsChan = np.array([levels.index(l)                                  ], dtype=np.float64)
            # hdamp, UE
            hdampup,hdampdo = GetModSystHistos(path, 'TT_hdamp', 'hdamp', var='counts')
            tuneup , tunedo = GetModSystHistos(path, 'TT_UE',    'UE', var='counts')
            hdampup_val, hdampup_err = hdampup.integrate('process').integrate('level', l).integrate('channel', c).integrate('syst').values(overflow='none', sumw2=True)[()]
            hdampdo_val, hdampdo_err = hdampdo.integrate('process').integrate('level', l).integrate('channel', c).integrate('syst').values(overflow='none', sumw2=True)[()]
            tuneup_val , tuneup_err  = tuneup .integrate('process').integrate('level', l).integrate('channel', c).integrate('syst').values(overflow='none', sumw2=True)[()]
            tunedo_val , tunedo_err  = tunedo .integrate('process').integrate('level', l).integrate('channel', c).integrate('syst').values(overflow='none', sumw2=True)[()]
            dictstaterr = {'hdampUpStatUp' : hdampup_val+np.sqrt(hdampup_err), 'hdampUpStatDown' : hdampup_val-np.sqrt(hdampup_err), 'hdampDownStatUp' : hdampdo_val+np.sqrt(hdampdo_err), 'hdampDownStatDown' : hdampdo_val-np.sqrt(hdampdo_err), 'UEUpStatUp' : tuneup_val+np.sqrt(tuneup_err), 'UEUpStatDown' : tuneup_val-np.sqrt(tuneup_err), 'UEDownStatUp' : tunedo_val+np.sqrt(tunedo_err), 'UEDownStatDown' : tunedo_val-np.sqrt(tunedo_err)}
            # PDF and scale uncertainties
            pr = 'tt'; ttSampleName = 'TTPS.pkl.gz'
            pdf_rel   = Get1bPDFUnc(  path+ttSampleName, categories={'sample':processDic['tt'], 'channel':c, 'level':l}, doPrint=False)
            scale_rel = Get1binScaleUnc(path+ttSampleName, categories={'sample':processDic['tt'], 'channel':c, 'level':l}, doPrint=False)
            nom = counts.integrate('process', pr).integrate('level', l).integrate('channel', c).integrate('syst', 'norm').values()[()][0]
            pdfup = nom*(1 + pdf_rel); pdfdw = nom*(1 - pdf_rel); scaleup = nom*(1 + scale_rel); scaledw = nom*(1 - scale_rel)
            newsyst = {'PDFUp':np.array([pdfup]), 'PDFDown':np.array([pdfdw]), 'ScaleUp':np.array([scaleup]), 'ScaleDown':np.array([scaledw]), 'hdampUp':hdampup_val, 'hdampDown':hdampdo_val, 'UEUp':tuneup_val, 'UEDown':tunedo_val}
            for syst, sval in (newsyst|dictstaterr).items():
              hmaster .fill(**{'syst':syst,   'weight':sval, 'process':pr, 'master':bins})
              hmastQCD.fill(**{'syst':syst,   'weight':sval, 'process':pr, 'mastQCD':bins})
              hperchan.fill(**{'syst':syst,   'weight':sval, 'process':pr, 'perchan':binsChan, 'channel':c})

             
    print("\r[{:<100}] {:.2f} % ".format('#' * int( float(iteration)/total*100), float(iteration)/total*100))
    saveHistos(outpath, outname, {'master':hmaster, 'perchan':hperchan, 'mastQCD':hmastQCD, 'shapes':hdists}, verbose=True)


def DrawMasterHistogram(fname):
    ''' Drwa the master histogram '''
    outpath = baseweb+datatoday+'/'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    outname = 'master'
    plt = plotter(fname, prDic={},  bkgList=bkglist, colors=colordic, lumi=lumi, var='master')
    fig, ax, rax = plt.Stack('master', xtit='', ytit='Events', dosyst=True, verbose=1, doNotSave=True)
    # Legend 
    handles, labels = ax.get_legend_handles_labels()
    print('diclegendlabels = ', diclegendlabels)
    for k, lab in diclegendlabels.items():
        if k in labels:
            labels[labels.index(k)] = lab
    ax.legend(handles, labels, loc='upper left', ncol=2, fontsize=10)
    # X axis
    binlabels = ['3j,1b', '3j,$\geq$2b', '4j,1b', '4j,$\geq$2b', '$\geq$5j1b', '$\geq$5j,$\geq$2b'] * 2
    rax.set_xticks(np.arange(0, len(binlabels)))
    rax.set_xticklabels(binlabels, rotation=90)
    rax.set_xlabel('')
    # Labels and lines
    rax.axvline(x=5.5, color=colorchan, linestyle='--')
    ax.axvline(x=5.5, color=colorchan, linestyle='--')
    fig.subplots_adjust(right=0.97, top=0.94, bottom=0.13)
    fig.set_size_inches(8, 8)
    ax.text(0.1, 0.70, '$e+$jets', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=15, color=colorchan)
    ax.text(0.75, 0.70, '$\mu+$jets', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=15, color=colorchan)
    for sufix in ['png', 'pdf']: fig.savefig(outpath + outname + '.' + sufix)
    print('Saved to: ', outpath + outname + '.png')

def DrawMasterUnc(fname, syst=['hdamp', 'UE', 'PDF', 'Scale'], process='tt', var='master', outname='systematics', prDic={}):
    outpath = baseweb+datatoday+'/Modeling/'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    binLabels = ['3j,1b', '3j,$\geq$2b', '4j,1b', '4j,$\geq$2b', '$\geq$5j1b', '$\geq$5j,$\geq$2b'] * 2
    fig, ax = DrawUncPerBin(fname, syst, process, var=var, outpath=outpath, outname=outname, binLabels=binLabels, savefig=False)
    for i in range(len(ax)):
      ax[i].axvline(x=5.5, color=colorchan, linestyle='--')
    ax[0].text(0.25, 1.20, '$e+$jets', horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes, fontsize=15, color=colorchan)
    ax[0].text(0.75, 1.20, '$\mu+$jets', horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes, fontsize=15, color=colorchan)
    fig.savefig(outpath + outname + '.png')
    fig.savefig(outpath + outname + '.pdf')
    print('Saved to: ', outpath + outname + '.png')

def DrawMasterUncWithStatBands(fname, syst='hdamp', process='tt', var='master', outname='systematics', prDic={}):
    outpath = baseweb+datatoday+'/Modeling/'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    binLabels = ['3j,1b', '3j,$\geq$2b', '4j,1b', '4j,$\geq$2b', '$\geq$5j1b', '$\geq$5j,$\geq$2b'] * 2
    h = GetHisto(fname, var, categories={'process':'tt'})
    hnom = h.integrate('syst', 'norm')
    hnom_Up = h.integrate('syst', 'statUp')
    hnom_Do = h.integrate('syst', 'statDown')
    hnom_sysUp = h.integrate('syst', syst+'Up')
    hnom_sysUp_Up = h.integrate('syst', syst+'UpStatUp')
    hnom_sysUp_Do = h.integrate('syst', syst+'UpStatDown')
    hnom_sysDo = h.integrate('syst', syst+'Down')
    hnom_sysDo_Up = h.integrate('syst', syst+'DownStatUp')
    hnom_sysDo_Do = h.integrate('syst', syst+'DownStatDown')
    fig, ax = matplt.subplots(1, 1, figsize=(10,8))
    nbins = len(binLabels)
    x = np.arange(nbins+1) - 0.5
    hist.plotratio(hnom, hnom, clear=False, ax=ax, error_opts={'marker': 'o', 'markersize': 0, 'elinewidth': 0, 'capsize': 0, 'capthick': 0, 'color': 'k'})
    hist.plotratio(hnom_sysUp, hnom, clear=False, ax=ax, error_opts={'marker': 'o', 'markersize': 0, 'elinewidth': 0, 'capsize': 0, 'capthick': 0, 'color': 'r'}, unc='num')
    hist.plotratio(hnom_sysDo, hnom, clear=False, ax=ax, error_opts={'marker': 'o', 'markersize': 0, 'elinewidth': 0, 'capsize': 0, 'capthick': 0, 'color': 'b'}, unc='num')
    hrat, hratUp, hratDo = DivideHistWithErrors(hnom, hnom, hnom_Up, hnom_Do)    
    rax, _ = DrawUncBands(ax, hnom, hratUp.values()[()], hratDo.values()[()], ratioax=None, hatch="\/\/", color="gray", label='Nominal')
    hratUp, hratUpUp, hratUpDo = DivideHistWithErrors(hnom_sysUp, hnom, hnom_sysUp_Up, hnom_sysUp_Do)
    rax, _ = DrawUncBands(ax, hnom_sysUp, hratUpUp.values()[()], hratUpDo.values()[()], ratioax=None, hatch='\/\/\/\/', color="red", alpha=0.4, label='Up variation')
    hratDo, hratDoUp, hratDoDo = DivideHistWithErrors(hnom_sysDo, hnom, hnom_sysDo_Up, hnom_sysDo_Do)
    rax, _ = DrawUncBands(ax, hnom_sysDo, hratDoUp.values()[()], hratDoDo.values()[()], ratioax=None, hatch='\/\/\/\/', color="blue", alpha=0.4, label='Down variation')
    binlabels = ['3j,1b', '3j,$\geq$2b', '4j,1b', '4j,$\geq$2b', '$\geq$5j1b', '$\geq$5j,$\geq$2b'] * 2
    ax.set_xticks(np.arange(0, len(binlabels)))
    ax.set_xticklabels(binlabels, rotation=90)
    ax.set_xlabel('')
    ax.set_ylabel('Variaion / nominal', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_ylim(0.96, 1.04)
    colorchan = 'k'
    ax.axvline(x=5.5, color=colorchan, linestyle='--')
    fig.subplots_adjust(right=0.97, top=0.94, bottom=0.13)
    ax.text(0.07, 0.94, '$e+$jets', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=15, color=colorchan)
    ax.text(0.57, 0.94, '$\mu+$jets', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=15, color=colorchan)
    ax.text(0.07, 1.04, 'Statistical uncertainty for systematic estimate of: ' + syst, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=15, color=colorchan)
    #hist.plotratio(hData, h.sum("process"), clear=False,ax=rax, error_opts=data_err_opts, denom_fill_opts= {'alpha':0, 'color':'none'} if drawSystBand else {}, guide_opts={}, unc='num')
    # legend in the lower right
    leg = matplt.legend(loc='lower right', frameon=False, fontsize=15)
    fig.savefig(outpath + outname + '.png')
    fig.savefig(outpath + outname + '.pdf')
    print('Saved to: ', outpath + outname + '.png')


def DrawMasterQCDunc(fname, outname='qcdunc'):
    outpath = baseweb+datatoday+'/QCD/'
    binLabels = ['3j,1b', '3j,$\geq$2b', '4j,1b', '4j,$\geq$2b', '$\geq$5j1b', '$\geq$5j,$\geq$2b'] * 2
    h = GetHisto(fname, 'mastQCD', categories={'process':'QCD'})
    n   = h.integrate('syst', 'norm').values()[()]
    err = h.integrate('syst', 'statUp').values()[()]
    hup = h.integrate('syst', 'QCDUp').values()[()]
    hdo = h.integrate('syst', 'QCDDown').values()[()]
    statRel = np.where(n > 0., abs(n-err) / n, 0.)
    print('err = ', err)
    print('n = ', n)
    hup = np.abs(hup - n) 
    hdo = np.abs(hdo - n) 
    # max difference
    hnom = np.maximum(hup, hdo)
    hnom = np.where(n > 0., hnom / n, 0.)
    nomRel = np.where(hnom > 1., 1, hnom)
    print('statRel = ', statRel)
    print('nomRel = ', nomRel)
    return


def DrawShapesMaser(fname, chan='m'):
    ''' Drwa the master histogram '''
    outpath = baseweb+datatoday+'/'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    outname = 'shapes_'+chan
    plt = plotter(fname, prDic={},  bkgList=bkglist, colors=colordic, lumi=lumi, var='shapes')
    plt.SetCategories({"channel":chan})
    fig, ax, rax = plt.Stack('shapes', xtit='', ytit='Events', dosyst=True, verbose=1, doNotSave=True)
    # Legend 
    handles, labels = ax.get_legend_handles_labels()
    print('diclegendlabels = ', diclegendlabels)
    for k, lab in diclegendlabels.items():
        if k in labels:
            labels[labels.index(k)] = lab
    ax.legend(handles, labels, loc='upper right', ncol=2, fontsize=15)
    rax.set_ylim(0.2, 1.8)
    # X axis
    #rax.set_xticks(np.arange(0, len(binlabels)))
    #rax.set_xticklabels(binlabels, rotation=90)
    #raxe.set_xlabel('')
    rax.set_xticks([])
    # Labels and lines
    sep = [5.5];
    for i in range(1, 5): sep.append(sep[-1]+7)
    for axx in [ax, rax]:
      for x in sep:
        axx.axvline(x=x, color=colorchan, linestyle='--')
    fig.subplots_adjust(right=0.97, top=0.94, bottom=0.06, left=0.08)
    fig.set_size_inches(12, 8)
    ax.text(0.4, 1.035, '$e+$jets' if chan=='e' else '$\mu+$jets', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=21, color='k')
    catlabels = ['3j,1b', '3j,$\geq$2b', '4j,1b', '4j,$\geq$2b', '$\geq$5j1b', '$\geq$5j,$\geq$2b'] * 2
    xpos = [0.04, 0.19, 0.35, 0.55, 0.74, 0.89]
    for cat, x in zip(catlabels, xpos):
      ax.text(x, 0.95 if cat==catlabels[0] else 0.72, cat, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=15, color=colorchan)
    rax.text(0.07,  -0.10, "MVA bins", horizontalalignment='center', verticalalignment='center', transform=rax.transAxes, fontsize=20, color='k')
    rax.set_xlabel('$\Delta$R(j,j)', fontsize=20)
    for sufix in ['png', 'pdf']: fig.savefig(outpath + outname + '.' + sufix)
    print('Saved to: ', outpath + outname + '.png')




if __name__ == "__main__":
    fname = outpath + outname + '.pkl.gz'
    if not os.path.exists(fname) or force:
        plt = plotter(path, prDic=processDic,  bkgList=bkglist, lumi=lumi, var=['counts', 'MVAscore', 'medianDRjj'])
        RebinVar(plt, 'MVAscore')
        RebinVar(plt, 'medianDRjj')
        counts   = plt.GetHistogram(var)
        systematics = [x.name for x in list(counts.identifiers('syst'))]
        process     = [x.name for x in list(counts.identifiers('process'))]
        print('Saving histograms to file: ', fname)
        CreateHistos(plt, systematics, process, channels, levels)
        DrawMasterHistogram(fname)
    else:
        DrawMasterHistogram(fname)
        #DrawMasterQCDunc(fname)
        DrawShapesMaser(fname, 'e')
        DrawShapesMaser(fname, 'm')
        DrawMasterUncWithStatBands(fname, syst='UE', outname='UE')
        DrawMasterUncWithStatBands(fname, syst='hdamp', outname='hdamp')
        DrawMasterUnc(fname, syst=['btagSF', 'elecSF', 'muonSF', 'prefire', 'JES'], outname='tt_experimental')

