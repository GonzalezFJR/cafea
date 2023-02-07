from config import *
import warnings
from datetime import datetime
from multiprocessing import Pool, Manager
warnings.filterwarnings("ignore", category=RuntimeWarning)

def Draw(plt, var, level, channel, outname, verbose=False):
  categories =  {'level':level, 'channel':channel}
  label = GetChLab(categories['channel']) + GetLevLab(categories['level']) 
  if channel == 'l': categories['channel'] = ['e', 'm']
  plt.SetCategories(categories)
  if not CheckHistoCategories(plt.hists[var], categories):
    if verbose: print(f'  > Skipping [{var}] cat = ', categories)
    return
  xtit = RebinVar(plt, var, level)
  plt.SetRegion(label)
  if verbose: print('  > Drawing: ', outname)
  plt.SetOutput(outname)
  #plt.SetLogY()
    
  aname = None
  if   var == 'l0pt': aname = 'lep0pt'
  elif var == 'l0eta': aname = 'lep0eta'
  plt.Stack(var, xtit=xtit, ytit=None, aname=aname, dosyst=True, verbose=verbose)

def DrawPar(L): 
  plt, var, lev, chan, outname, outdic = L
  if outdic != {}: 
    total = outdic['total']
    print("\r[{:<50}] {:.2f} %".format('#' * int(outdic['progress']/total*100/2), float(outdic['progress'])/total*100), end='')
  Draw(plt, var, lev, chan, outname)
  if outdic != {}: 
    outdic['progress'] += 1

def DrawAll(plt, outpath, vars, levels, channels, nSlots=4):
  doParallel = nSlots > 1
  plt.SetOutpath(outpath)
  inputs = []
  tot = len(vars)*len(levels)*len(channels)
  for c in channels:
    for l in levels:
      cat = {'level':l, 'channel':c}
      clab = GetChLab(c) + GetLevLab(l)
      for v in vars:
        outname = f'{c}_{l}_{v}'
        if l == 'incl' and v in ['j0pt', 'j0eta']: continue
        inputs.append([plt, v, l, c, outname])
        if not doParallel: 
          #Draw(plt, v, l, c, outname)
          pool = Pool(1)
          pool.map(DrawPar, [[plt, v, l, c, outname, {}]])
          pool.close()
          pool.join()
          progress = float(len(inputs))/tot
          print("\r[{:<50}] {:.2f} %".format('#' * int(float(len(inputs))/tot*100/2), float(len(inputs))/tot*100), end='')
  if doParallel: 
    if nSlots < 4: nSlots = 4
    if tot/nSlots > 40: print("WARNING: you are probably plotting to many plots for the amount of slots!! Total plots: %i, nSlots: %i. Increase the number of slots."%(tot,nSlots))
    print('Drawing %s plots with %i slots...'%(tot, nSlots))
    manager = Manager()
    outdict = manager.dict()
    outdict['progress'] = 0
    outdict['total'] = tot
    for i in range(len(inputs)): inputs[i].append(outdict)
    pool = Pool(nSlots)
    pool.map(DrawPar, inputs)
    pool.close()
    pool.join()

  print("\r[{:<50}] {:.2f} %".format('#' * 50, 100))


#################################################################################################
#################################################################################################

if __name__=="__main__":

  if not 'QCD.pkl.gz' in list(os.listdir(path)):
    print('WARNING: QCD file not found in input folder!!!!')
    exit()


  print(' >> Loading histograms! This might take a while...')
  plt = plotter(path, prDic=processDic, bkgList=bkglist, colors=colordic, lumi=lumi)
  plt.SetLumi(lumi, "pb$^{-1}$", "5.02 TeV")
  plt.SetRatio(True)
  plt.plotData = True
  plt.SetLegendLabels(diclegendlabels)

  outpath = baseweb+datatoday+'/ControlPlots/'
  print('[Control plots] Output = ', outpath)
  if not os.path.isdir(outpath): os.makedirs(outpath)
  plt.SetOutpath(outpath)

  # Analysis var
  variables_control = ['met', 'ht', 'mt', 'j0eta', 'j0pt', 'ept', 'eeta', 'mpt', 'meta']
  variables_MVA     = ['ht', 'j0pt', 'mjj', 'medianDRjj', 'mlb', 'medianDRuu', 'muu', 'dRlb'] 
  variables_extra   = ['st', 'j0eta', 'eeta', 'meta', 'mt', 'u0pt', 'u0eta', 'ptlb', 'sumallpt', 'minDRjj', 'ptuu', 'mjj', 'ptjj']
  mvaVar = "MVAscore"

  systematics = ['ISR', 'FSR', 'btagSF', 'elecSF', 'muonSF', 'JES', 'prefire']
  plt.SetSystematics(systematics)

  if not var is None:
    clab = ch if not isinstance(ch, list) and not len(list)>1 else 'l'
    outname = "custom_%s_%s_%s"%(var, clab, level)
    Draw(plt, var, level, ch, outname)
  else:
    ### Control plots
    #############################################
    levels = ['g4jets', '3j1b', '3j2b', '4j1b', '4j2b', 'g5j1b', 'g5j2b']
    channels = ['e', 'm']
    #DrawAll(plt, outpath, variables_control, levels, channels, nSlots=nSlots)

    ### B-tagging control plots
    #############################################
    outpath = baseweb+datatoday+'/B-tagging/'
    if not os.path.isdir(outpath): os.makedirs(outpath)
    levels = ['g4jets', 'g3jets']
    channels = ['e', 'm']
    varnjets = ['nbtags']
    #DrawAll(plt, outpath, varnjets, levels, channels, nSlots=nSlots)

    ### QCD plots
    #############################################
    levels = ['3j1b', '3j2b', '4j1b', '4j2b', 'g5j1b', 'g5j2b']
    channels = ['e', 'm']
    QCDoutpath = baseweb+datatoday+'/QCD/met/'
    print('[QCD met plots] Output = ', QCDoutpath)
    if not os.path.isdir(QCDoutpath): os.makedirs(QCDoutpath)
    variables_QCD = ['metnocut']
    #DrawAll(plt, QCDoutpath, variables_QCD, levels, channels, nSlots=nSlots)

    ### MVA plots
    #############################################
    levels = ['3j1b']
    channels = ['e', 'm', 'l']
    MVAoutpath = baseweb+datatoday+'/MVA/'
    print('[MVA plots] Output = ', MVAoutpath)
    if not os.path.isdir(MVAoutpath): os.makedirs(MVAoutpath)

    # MVA input variables
    outpath = MVAoutpath+'inputvars/'
    if not os.path.isdir(outpath): os.makedirs(outpath)
    DrawAll(plt, outpath, variables_MVA, levels, channels, nSlots=nSlots)
    exit()

    # MVA extra plots
    outpath = MVAoutpath+'extra/'
    if not os.path.isdir(outpath): os.makedirs(outpath)
    DrawAll(plt, outpath, variables_extra, levels, channels, nSlots=nSlots)

    # MVA score
    outpath = MVAoutpath+'score/'
    if not os.path.isdir(outpath): os.makedirs(outpath)
    DrawAll(plt, outpath, [mvaVar], levels, channels, nSlots=nSlots)
