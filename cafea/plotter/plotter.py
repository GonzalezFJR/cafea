from __future__ import print_function, division
from collections import defaultdict, OrderedDict
import gzip
import pickle
import json
import os
import uproot
import matplotlib.pyplot as plt
import numpy as np
import coffea
from coffea import hist, processor
from coffea.hist import plot, Hist
from cycler import cycler
from cafea.plotter.OutText import OutText
from cafea.plotter.Uncertainties import *

def StackOverUnderflow(v, overflow):
  if overflow=='all':
    return v
  elif overflow == 'over':
    return v[1:-2] + [sum(v[-2:])] 
  elif overflow == 'under':
    return [sum(v[0:2])] + v[2:]
  elif overflow == 'none':
    return [sum(v[0:2])] + v[2:-2] + [sum(v[-2:])]
  else:
    return [sum(v[0:2])] + v[2:-2] + [sum(v[-2:])]

def loadHistos(path, hname=None):
  if isinstance(path, str) and os.path.isdir(path):
    pathlist = []
    for f in os.listdir(path):
      if f.endswith('pkl.gz'): pathlist.append(path + '/' + f)
    path = pathlist
  if not isinstance(path, list): path = path.replace(' ', '').split() if ',' in path else [path]
  hists = {}
  for p in path:
    with gzip.open(p) as fin:
      hin = pickle.load(fin)
      for k in hin.keys():
        if k in hists: hists[k]+=hin[k]
        else:          hists[k]=hin[k]
  if hname is None: return hists
  return histis[hname]

def GetHisto(path, hname=None, categories={}, group=None, integrate=None, rebin=None):
  ''' Get a histogram from a pkl file (standard output from coffea analysis) '''
  hists = loadHistos(path)
  if hname is None:
    h = {}
    for hname in hists.keys(): h[hname] = GetHisto(path, hname)
    return h
  h = hists[hname]
  for cat in categories.keys():
    h = h.integrate(cat, categories[cat])
  if group is not None:
    if not isinstance(group, list) or not isinstance(group[0], list): 
      group = [group]
    for g in group:
      cat1, cat2, d = g
      h = h.group(hist.Cat(cat1, cat), hist.Cat(cat2, cat2), d)
  if integrate is not None:
    for i in integrate: h = h.integrate(i)
  if rebin is not None:
    axname, nbins = list(rebin.items())[0]
    h = Rebin(h, axname, nbins)
  return h

def CheckHistoCategories(hist, categories, checkValues=False):
  for c in categories:
    hasAxis = c in hist.axes()
    if hasAxis:
      cat = categories[c]
      if isinstance(cat, str) and ',' in cat: cat = cat.replace(' ', '').split(',')
      elif isinstance(cat, str): cat = [cat]
      for icat in cat:
        if not icat in [x.name for x in hist.identifiers(c)]: return False
    else: return False
  if checkValues:
    for c in categories:
      hist = hist.integrate(c, categories[c])
    values = hist.values()
    if values == {}: return False
  return True

def Rebin(h, axname, newbins, binN=None, includeLower=True, includeUpper=True):
  oldax = h.axis(axname)
  if binN is not None:
    oldbins = oldax.edges()
    bin0 = newbins
    indices = np.where(np.logical_and(oldbins>newbins, oldbins<binN))[0]
    if includeLower and (indices[0] > 0): 
      indices = np.concatenate( ([indices[0]-1], indices) )
    if includeUpper and (indices[-1] < (len(oldbins)-1)): 
      indices = np.append(indices, indices[-1]+1)
    newbins = oldbins[indices]
    newax = hist.Bin(axname, oldax.label, newbins)

  elif isinstance(newbins, int): # take n first bins...
    newedges = oldax.edges()[0::newbins]
    newax = hist.Bin(axname, oldax.label, newedges)

  else: # Give a new array with bins
    newax = hist.Bin(axname, oldax.label, newbins)

  return h.rebin(oldax, newax) 

def GetHistoMax(histo):
  ''' Returns the max value '''
  if isinstance(histo, list): return max([GetHistoMax(h) for h in histo])
  values = histo.values(overflow='all')
  values = values[list(values.keys())[0]]
  return max(values)

def GetClopperPearsonInterval(hnum, hden, overflow='all'):
  ''' Compute Clopper-Pearson interval from numerator and denominator histograms '''
  num = list(hnum.values(overflow='over')[()])#, dtype=int)
  den = list(hden.values(overflow='over')[()])#, dtype=int)
  if isinstance(num, list) and isinstance(num[0], np.ndarray):
    for i in range(len(num)):
      num[i] = np.array(StackOverUnderflow(list(num[i]),overflow), dtype=int)
      den[i] = np.array(StackOverUnderflow(list(den[i]),overflow), dtype=int)
    den = StackOverUnderflow(den, overflow)
    num = StackOverUnderflow(num, overflow)
  else: 
    num = np.array(StackOverUnderflow(num, overflow), dtype=int); den = np.array(StackOverUnderflow(den, overflow), dtype=int)
  num = np.array(num); den = np.array(den)
  down, up = hist.clopper_pearson_interval(num, den)
  ratio = np.array(num, dtype=float)/den
  return [ratio, down, up]

def GetEff(num, den):
  ''' Compute efficiency values from numerator and denominator histograms '''
  ratio, down, up =  GetClopperPearsonInterval(num, den)
  axis = num.axes()[0].name
  bins = num.axis(axis).edges()
  x    = num.axis(axis).centers()
  xlo  = bins[:-1]
  xhi  = bins[1:]
  return [[x, xlo-x, xhi-x],[ratio, down-ratio, up-ratio]]

def GetXYfromH1D(h, axis=None, mode='edges', errors=False, overflow=False, EFT_WCdict=None):
  ''' Get values and bin edges or centers from 1d histogram '''
  if EFT_WCdict is not None:
    WCs = np.array(h._wcnames)
    vals = np.zeros_like(WCs, dtype=float)
    for k in EFT_WCdict.keys():
      vals += np.where(WCs==k, EFT_WCdict[k], vals)
    h.set_wilson_coeff_from_array(vals)
  if isinstance(h, uproot.models.TH.Model_TH1D_v3): return GetXYfromTH1(h, mode, errors, overflow)
  if axis is None: axis = h.axes()[0]
  x = h.axis(axis).centers() if mode=='centers' else h.axis(axis).edges()
  y, ye = h.values(overflow=('over' if overflow else 'none'), sumw2=True)[()]
  #print('y = ', y)
  if overflow:
    ym1 = y[-1]; y = y[:-1]; y[-1] += ym1
    yem1 = ye[-1]; ye = ye[:-1]; ye[-1] = np.sqrt(ye[-1]*ye[-1] + yem1*yem1)
  if errors: return x, y, ye
  return x,y

def GetXYfromTH1(h, mode='edges', errors=False, overflow=False):
  _, x = h.to_numpy(flow=False)
  if mode in ['centers', 'centres', 'center', 'centre']: x = [ (x[i+1]+x[i])/2 for i in range(len(x)-1)]
  y  = h.values(flow=overflow)
  ye = h.errors(flow=overflow)
  if overflow:
    ym1 = y[-1]; y = y[1:-1]; y[-1] += ym1
    yem1 = ye[-1]; ye = ye[1:-1]; ye[-1] = np.sqrt(ye[-1]*ye[-1] + yem1*yem1)
  if errors: return x, y, ye
  return x,y

def GetXYfromH2D(h, axis1, axis2, mode='edges'):
  ''' Get values and bin edges or centers from 2d histogram, mode = edges, centers'''
  if mode=='edges':
    x = h.axis(axis1).edges();
    y = h.axis(axis2).edges();
  elif mode=='centers':
    x = h.axis(axis1).centers();
    y = h.axis(axis2).centers();
  z = h.values()[()]
  return x,y,z

def GetH1DfromXY(bins, values, yerr=None, ytit='Y', xtit='X', label=None, axisLabel=None):
  ''' Create a coffea-1d histogram from bins and values '''
  hname='hist'
  if label is None or axisLabel is None:
    h = hist.Hist(ytit, hist.Bin(hname, xtit, bins))
    centers = h.axis(hname).centers()
    h.fill(hist=centers, weight=values)#{hname:centers, 'weight':values})
  else:
    h = hist.Hist(ytit, hist.Bin(axisLabel, axisLabel), hist.Bin(hname, xtit, bins))
    centers = h.axis(hname).centers()
    h.fill(hist=centers, axisLabel=label, weight=values)#{hname:centers, 'weight':values})
  if yerr is not None: 
    origerr = h._sumw2[ (list(h._sumw2.keys())[0]) ]
    if len(origerr) == (len(yerr)+3): 
      yerr = np.insert(yerr, len(yerr), 0.0)
      yerr = np.insert(yerr, len(yerr), 0.0)
      yerr = np.insert(yerr, 0, 0.0)
    elif len(origerr) == (len(yerr)+2): 
      yerr = np.insert(yerr, len(yerr), 0.0)
      yerr = np.insert(yerr, 0, 0.0)
    elif len(origerr) == (len(yerr)+1):
      yerr = np.insert(yerr, 0, 0.0)
    h._sumw2[ (list(h._sumw2.keys())[0]) ] = yerr
  return h

def GetH2DfromXY(bins, values, ytit='Y', xtit=['X1', 'X2'], hname=['pt','eta']):
  ''' Create a coffea-2d histogram from bins and values '''
  h = hist.Hist(ytit, hist.Bin(hname[0], xtit[0], bins[0]), hist.Bin(hname[1], xtit[1], bins[1]))
  c0 = h.axis(hname[0]).centers()
  c1 = h.axis(hname[1]).centers()
  for i in range(len(c1)):
    row = values[i]
    x = c0
    y = np.array([c1[i]]*len(row))
    fill = {hname[0]:x, hname[1]:y, 'weight':row}
    h.fill(**fill)
  return h

def CheckArray(arr):
  if isinstance(arr, dict):
    keys = list(arr.keys())
    if len(keys) == 1:
      return arr[keys[0]]
    else:
      print("ERROR: array-like is dictionary with multiple keys" )
      return None
  elif isinstance(arr, list):
    return np.array(arr)
  elif isinstance(arr, coffea.hist.Hist) or isinstance(arr, coffea.hist.hist_tools.Hist) or hasattr(arr, "values"):
    return CheckArray(arr.values())
  return arr

def DrawUncBands(ax, hnom, up, down, ratioax=None, relative=False, alpha=None, hatch=None, fillcolor=None, color="gray"):
  nom = CheckArray(hnom)
  axis = hnom.axes()[0].name
  bins = hnom.axis(axis).edges()
  x = hnom.axis(axis).centers()
  Xlo = bins[:-1]
  Xhi = bins[1:]
  r2 = None
  for k in range(len(x)):
    yd = down[k]; yu = up[k]
    mm = x[k]; md = Xlo[k]; mu = Xhi[k]
    r1 = ax.fill_between([md, mm, mu], [yu, yu, yu], [yd, yd, yd], edgecolor=color, linewidth=0, facecolor=fillcolor if fillcolor is not None else 'none', alpha=alpha, hatch=hatch)
  if ratioax is not None:
    rup = up/nom
    rdo = down/nom
    for k in range(len(x)):
      yd = rdo[k]; yu = rup[k]
      mm = x[k]; md = Xlo[k]; mu = Xhi[k]
      r2 = ratioax.fill_between([md, mm, mu], 3*[yu], 3*[yd], edgecolor=color, linewidth=0, facecolor=fillcolor if fillcolor is not None else 'none', alpha=alpha, hatch=hatch)
  return r1, r2

def GetRatioAssymetricUncertainties(num, numDo, numUp, den, denDo, denUp):
  ''' Compute efficiencies from numerator and denominator counts histograms and uncertainties '''
  ratio = num/den
  uncUp = ratio*np.sqrt(numUp*numUp + denUp*denUp) 
  uncDo = ratio*np.sqrt(numDo*numDo + denDo*denDo) 
  return ratio, -uncDo, uncUp

def GetSFfromCountsHisto(hnumMC, hdenMC, hnumData, hdenData):
  ''' Compute scale factors from efficiency histograms for data and MC '''
  Xmc, Ymc = GetEff(hnumMC, hdenMC)
  Xda, Yda = GetEff(hnumData, hdenData)
  ratio, do, up = GetRatioAssymetricUncertainties(Yda[0], Yda[1], Yda[2], Ymc[0], Ymc[1], Ymc[2])
  return ratio, do, up

def PrintHisto(histo):
  for ax in [x.name for x in histo.sparse_axes()]:
    print("# Sparse axis [%s] -- Identifiers: "%ax)
    for idt in histo.identifiers(ax):
      print("   - %s"%idt)
  for ax in [x.name for x in histo.dense_axes()]:
    print("# Dense axis [%s] -- bins: "%(ax))
    print("   - ", histo.identifiers(ax))

'''
def GetHistoUnc(h, systList, systLabel='syst', nomlabel='norm', doRelative=True, includeStat=True):
  # Compute up and down arrays with relative or absolute errors 
  nom = 0
  up = ak.zeros(len(nom))
  do = ak.zeros(len(nom))
    # Quadratic sum
    up += np.power( np.where(err<0, 0, err), 2)
    do += np.power( np.where(err>0, 0, err), 2)
  up = np.sqrt(up)
  do = np.sqrt(do)
  if not doRelative:
    up = nom+up
    do = nom-do
  return do, up
 '''

def DrawEff(hnumMC, hdenMC, hnumData, hdenData, title='Efficiencies', xtit='|$\eta$|', doFill=None, mcColor='r', dataColor='k', outname='temp.png', verbose=1):
  ''' Draw histograms with scale factors from data and MC histograms with numerator and denominator '''
  Xmc, Ymc = GetEff(hnumMC, hdenMC)
  Xda, Yda = GetEff(hnumData, hdenData)
  ratio, do, up = GetRatioAssymetricUncertainties(Yda[0], Yda[1], Yda[2], Ymc[0], Ymc[1], Ymc[2])
  if verbose>2:
    print(' >> Values for histogram %s that will be saved as %s...'%(title, outname))
    print('Printing bins  histogram\n >> nom = ', Xmc[0], '\n >> do  = ', Xmc[1], '\n >> up  = ', Xmc[2])
    print('Printing MC    histogram\n >> nom = ', Ymc[0], '\n >> do  = ', Ymc[1], '\n >> up  = ', Ymc[2])
    print(' --- hnumMC = ', hnumMC.values()[()], ', sum = ', sum(hnumMC.values()[()]), ', hdenMC = ', hdenMC.values()[()], ', sum = ', sum(hdenMC.values()[()]))
    print('Printing Data  histogram\n >> nom = ', Yda[0], '\n >> do  = ', Yda[1], '\n >> up  = ', Yda[2])
    print(' --- hnumData = ', hnumData.values()[()], ', sum = ', sum(hnumData.values()[()]), ', hdenData = ', hdenData.values()[()], ', sum = ', sum(hdenData.values()[()]))
    print('Printing ratio histogram\n >> nom = ', ratio, '\n >> do  = ', do, '\n >> up  = ', up)
    print('')

  # Create the figures
  fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (2, 1)}, sharex=True)
  fig.subplots_adjust(hspace=.07)

  # Global options
  textParams = {'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 18, 'xtick.labelsize': 12, 'ytick.labelsize': 12}
  plt.rcParams.update(textParams)

  # Plot!
  ax.errorbar(Xmc[0], Ymc[0], yerr=np.abs(Ymc[1:]), xerr=np.abs(Xmc[1:]), ecolor=mcColor, color=mcColor, fmt='o', capsize=0, elinewidth=2, label='MC')
  ax.errorbar(Xda[0], Yda[0], yerr=np.abs(Yda[1:]), xerr=np.abs(Xda[1:]), ecolor=dataColor, color=dataColor, fmt='o', capsize=0, elinewidth=2, label='Data')
  rax.errorbar(Xmc[0], ratio, yerr=np.abs([do, up]), xerr=np.abs(Xmc[1:]), ecolor=dataColor, color=dataColor, fmt='o', capsize=0, elinewidth=2)

  if doFill is not None:
    if not isinstance(doFill, dict): doFill = {}
    if not 'alpha'     in doFill.keys(): doFill['alpha'    ] = 0.2
    if not 'hatch'     in doFill.keys(): doFill['hatch'    ] = '\/\/'
    if not 'facecolor' in doFill.keys(): doFill['facecolor'] = None
    nPoints = len(Xmc[0])
    for i in range(nPoints):
      mm = Xmc[0][i]; md = Xmc[1][i]+mm; mu = Xmc[2][i]+mm
      ax.fill_between([md, mm, mu], 3*[Ymc[0][i]+Ymc[1][i]], 3*[Ymc[0][i]+Ymc[2][i]], edgecolor=mcColor, facecolor=mcColor if doFill['facecolor'] is not None else 'none', alpha=doFill['alpha'], hatch=doFill['hatch'])
      rax.fill_between([md, mm, mu], 3*[ratio[i]+do[i]], 3*[ratio[i]+up[i]], edgecolor=dataColor, facecolor=dataColor if doFill['facecolor'] is not None else 'none', alpha=doFill['alpha'], hatch=doFill['hatch'])
      d = Xda[0][i]; dd = Xda[1][i]+d; du = Xda[2][i]+d
      ax.fill_between([dd, d, du], 3*[Yda[0][i]+Yda[1][i]], 3*[Yda[0][i]+Yda[2][i]], edgecolor=dataColor, facecolor=dataColor if doFill['facecolor'] is not None else 'none', alpha=doFill['alpha'], hatch=doFill['hatch'])

  #CMS  = plt.text(0., 1., r"$\bf{CMS}$ Preliminary", fontsize=16, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
  #lumi = plt.text(1., 1., r"%1.1f %s (%s)"%(self.lumi, self.lumiunit, self.sqrts), fontsize=20, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)

  # Titles
  ax.set_ylabel('Efficiency')
  ax.set_title(title)
  rax.set_ylabel('Scale factor')
  rax.set_xlabel(xtit)

  # Legend
  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles, labels)

  # Limits
  ax.set_xlim(Xmc[0][0]+Xmc[1][0], Xmc[0][-1]+Xmc[2][-1])
  ax.set_ylim(min(min(Ymc[0]+Ymc[1])*0.9, min(Yda[0]+Yda[1])*0.9), max(max(Ymc[0]+Ymc[2])*1.15, max(Yda[0]+Yda[2])*1.15))
  rax.set_xlim(Xmc[0][0]+Xmc[1][0], Xmc[0][-1]+Xmc[2][-1])
  rax.set_ylim(min(ratio+do)*0.9, max(ratio+up)*1.15) 
 
  print('New png created: ', outname)
  fig.savefig(outname)

def DrawComp(histos, colors='k', axis='', title='', labels=None, xtit=None, ytit=None, doFill=None, outname=None, verbose=False, EFT_WCdict=None):
  if not isinstance(histos, list): histos = [histos]
  if len(histos) <= 1:
    if EFT_WCdict != None:
      histos = histos*len(EFT_WCdict)
    else:
      # Nothing to compare...
      print('Warning: at least two histograms are needed! Nothing to compare...')
      return
  if isinstance(colors, str): colors = [colors]*len(histos)
  EFTdict = None if EFT_WCdict is None else EFT_WCdict[0]
  X, nom, unc = GetXYfromH1D(histos[0], mode='centers', errors=True, overflow=True, EFT_WCdict=EFTdict)
  bins, _ = GetXYfromH1D(histos[0], mode='bins')
  Xdo = X-bins[:-1]
  Xup = bins[1:]-X

  # Create the figures
  fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (2, 1)}, sharex=True)
  fig.subplots_adjust(hspace=.07)

  # Global options
  textParams = {'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 18, 'xtick.labelsize': 12, 'ytick.labelsize': 12}
  plt.rcParams.update(textParams)
  if doFill is not None:
    if not isinstance(doFill, dict): doFill = {}
    if not 'alpha'     in doFill.keys(): doFill['alpha'    ] = 0.2
    if not 'hatch'     in doFill.keys(): doFill['hatch'    ] = '\/\/'
    if not 'facecolor' in doFill.keys(): doFill['facecolor'] = None

  # Nominal
  #ax.errorbar(bins, nom, yerr=[unc, unc], xerr=np.abs(Xmc[1:]), ecolor=mcColor, color=mcColor, fmt='-', capsize=0, elinewidth=2, label='MC')
  ax.errorbar(X, nom, yerr=[unc, unc], xerr=[Xdo, Xup], ecolor=colors[0], color=colors[0], fmt='o', markersize=0, capsize=0, elinewidth=1, label=labels[0])
  if doFill is not None:
    for k in range(len(X)):
      mm = X[k]; md = mm-Xdo[k]; mu = mm+Xup[k]
      yd = nom[k]-unc[k]; yu = nom[k]+unc[k]
      ax.fill_between([md, mm, mu], 3*[yu], 3*[yd], edgecolor=colors[0], facecolor=colors[0] if doFill['facecolor'] is not None else 'none', alpha=doFill['alpha'], hatch=doFill['hatch'])

  ybins = (nom+unc);
  # Plot comparisons and rations
  for i in range(1, len(histos)):
    EFTdict = None if EFT_WCdict is None else EFT_WCdict[i]
    _, y, ye = GetXYfromH1D(histos[i], mode='centers', errors=True, overflow=True, EFT_WCdict=EFTdict)
    ax.errorbar(X, y, yerr=[ye, ye], xerr=[Xdo, Xup], ecolor=colors[i], color=colors[i], fmt='p', markersize=0, capsize=0, elinewidth=1, label=labels[i])
    ratio = y/nom
    ratio = np.where(np.isnan(ratio), 1, ratio)
    ratiounc = ratio*np.sqrt(np.power(unc/nom, 2) + np.power(ye/y, 2))
    rax.errorbar(X, ratio, yerr=np.abs([ratiounc, ratiounc]), xerr=[Xdo, Xup], ecolor=colors[i], color=colors[i], fmt='o', markersize=0, capsize=0)#, elinewidth=2)
    ybins = np.concatenate((ybins, (y+ye)));
  
    if doFill is not None:
      for k in range(len(X)):
        mm = X[k]; md = mm-Xdo[k]; mu = mm+Xup[k]
        yd = y[k]-ye[k]; yu = y[k]+ye[k]
        ax.fill_between([md, mm, mu], 3*[yu], 3*[yd], edgecolor=colors[i], facecolor=colors[i] if doFill['facecolor'] is not None else 'none', alpha=doFill['alpha'], hatch=doFill['hatch'])
        rax.fill_between([md, mm, mu], 3*[ratio[k]-ratiounc[k]], 3*[ratio[k]+ratiounc[k]], edgecolor=colors[i], facecolor=colors[i] if doFill['facecolor'] is not None else 'none', alpha=doFill['alpha'], hatch=doFill['hatch'])
        #d = Xda[0][k]; dd = Xda[1][k]+d; du = Xda[2][k]+d
        #ax.fill_between([md,mm, mu], 3*[Yda[0][k]+Yda[1][k]], 3*[Yda[0][k]+Yda[2][k]], edgecolor=dataColor, facecolor=dataColor if doFill['facecolor'] is not None else 'none', alpha=doFill['alpha'], hatch=doFill['hatch'])

  CMS  = plt.text(0., 1., r"$\bf{CMS}$ Preliminary", fontsize=16, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
  #lumi = plt.text(1., 1., r"%1.1f %s (%s)"%(self.lumi, self.lumiunit, self.sqrts), fontsize=20, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)

  # Titles
  rax.set_ylabel('Ratio')
  if ytit  is not None: ax.set_ylabel(ytit)
  if title is not None: ax.set_title(title)
  if xtit  is not None: 
    ax.set_xlabel('')
    rax.set_xlabel(xtit)
  else:
    rax.set_xlabel(ax.get_xlabel())

  # Legend
  handles, labels = ax.get_legend_handles_labels()
  ax.legend(handles, labels)

  # Limits
  ax.set_xlim(X[0]-Xdo[0], X[-1]+Xup[-1])
  rax.set_xlim(X[0]-Xdo[0], X[-1]+Xup[-1])
  ax.set_ylim(0, max(ybins)*1.2)
  #rax.set_ylim(min(ratio+do)*0.9, max(ratio+up)*1.15) 

  # output
  outname = (outname if outname is not None else 'temp.png')
  if not outname.endswith('.png'): outname += '.png'
  print('New png created: ', outname)
  fig.savefig(outname)

def DrawEff2D(h, xaxis, error=None, error2=None, xtit='', ytit='', tit='', outname='temp.png'):
  ''' Draw 2D histograms with scale factors from data and MC histograms with numerator and denominator '''
  fig, ax = plt.subplots(1, 1, figsize=(7,7))
  hist.plot2d(h, xaxis, ax, patch_opts={'cmap':'bwr'})
  values = h.values()[()];
  yaxis = h.axes()[0].name
  if yaxis == xaxis: yaxis = h.axes()[1].name
  ax.set_ylabel(ytit); ax.set_xlabel(xtit);
  ax.set_title(tit)
  xcenters = h.axis(xaxis).centers()
  ycenters = h.axis(yaxis).centers()
  if not error  is None: 
    if isinstance(error, Hist): error = error .values()[0]
  if not error2 is None: 
    if isinstance(error2, Hist): error2 = error2.values()[0]
  for i in range(len(values)):
    for j in range(len(values[i])):
      x = xcenters[i]
      y = ycenters[j]
      val = values[i][j]
      if not error is None and not error2 is None:
        ep = error [i][j]
        em = error2[i][j]
        plt.text(x, y, '%1.3f\n$^{+%1.3f}_{-%1.3f}$'%(val, ep, abs(em)), fontsize=12, horizontalalignment='center', verticalalignment='center')#, transform=ax.transAxes)
      elif not error is None:
        e = error[i][j]
        plt.text(x, y, '%1.3f\n$\pm$%1.3f'%(val, e), fontsize=16, horizontalalignment='center', verticalalignment='center')#, transform=ax.transAxes)
      else:
        plt.text(x, y, '%1.3f'%val, fontsize=16, horizontalalignment='center', verticalalignment='center')#, transform=ax.transAxes)
  print('New png created: ', outname)
  fig.savefig(outname)

def GetSystListForHisto(h, systAxisName="syst", normLabel='norm', stat=False):
  fullList = [x.name for x in list(h.identifiers(systAxisName))]
  systList = []
  for s in fullList:
    if s == normLabel: continue
    if not stat and s == 'stat': continue
    if s.lower().endswith('up') or s.lower().endswith('do'): s = s[:-2]
    elif s.lower().endswith('down'): s = s[:-4]
    if not s in systList: systList.append(s)
  return systList
  
def GetAsimov(histo, processes=None, prname='process', integrateAxis=True):
  if processes is None:
    processes = [x.name for x in histo.identifiers(prname)]
    if 'data' in processes: 
      processes.pop(processes.index('data'))
  var = [x.name for x in histo.dense_axes()][0]
  return histo.integrate(prname, processes)
  h = histo.group(prname, hist.Cat(prname, prname), {'asimov': processes})
  return h.integrate('process','asimov') if integrateAxis else h1d
  '''
  bins, values = GetXYfromH1D(h1d, axis=var, mode='centers', errors=False, overflow=False)
  print('values h1d = ', values)
  # copy a histogram and clean it, then fill
  datafill = []
  for b, d in zip(bins, np.array(values)): 
    datafill += [b]*d
  print('datafill = ', datafill)
  newh = h.copy(content=False)
  newh.fill(**{'weight' : np.ones_like(datafill), var : np.array(datafill), 'process':'asimov'})
  #print('Getting data from Asimov')
  return newh.integrate("process", "asimov")
  '''

def GetPseudodata(histo, processes=None, integrateAxis=True):
  ''' Only one sparse axes for processes, please '''
  prname = [x.name for x in histo.sparse_axes()][0]
  var = [x.name for x in histo.dense_axes()][0]
  if processes is None:
    processes = [x.name for x in histo.identifiers(prname)]
  h = histo.group(prname, hist.Cat(prname, prname), {'Pseudodata': processes})
  h1d = h.integrate('process', 'Pseudodata')
  bins, values = GetXYfromH1D(h1d, axis=var, mode='centers', errors=False, overflow=False)
  data = np.random.poisson(values)
  # copy a histogram and clean it, then fill
  datafill = []
  for b, d in zip(bins, data): datafill += [b]*d
  newh = h.copy(content=False)
  newh.fill(**{'weight' : np.ones_like(datafill), var : np.array(datafill), 'process':'Pseudodata'})
  #print('Getting pseudodata')
  return newh.integrate("process", "Pseudodata") if integrateAxis else newh

def GroupKeepOrder(histo, grouplists):
  # grouptuple = [[old_axis, new_axis, dict]]
  changing_axes = [x[0] for x in grouplists]
  axes = histo.sparse_axes()
  axes_names = [x.name for x in axes]
  for ax in axes_names[::-1]:
    if ax in changing_axes:
      for b1, b2, dic in grouplists:
        if b1 != ax: continue
        else: break
      histo = histo.group(hist.Cat(b1, b1), hist.Cat(b2, b2), dic)
    else:
      dic = {}
      for idt in histo.identifiers(ax):
        dic[idt] = idt
      histo = histo.group(hist.Cat(ax, ax), hist.Cat(ax, ax), dic)
  return histo


def GetFileNameFromPath(path):
  name = path if not '/' in path else path[path.rfind('/')+1:]
  if name.endswith('.pkl.gz'): name = name[:-7]
  return name







class plotter:
  def __init__(self, path, prDic={}, colors={}, bkgList=[], dataName='data', outpath='./temp/', output=None, lumi=59.7, sigList=[], verbose=1, var=None):
    self.verbose = verbose
    self.SetPath(path)
    self.SetProcessDic(prDic)
    self.SetBkgProcesses(bkgList)
    self.SetSignalProcesses(sigList)
    self.SetDataName(dataName)
    self.var = [var] if isinstance(var, str) else var
    self.listOfVars = []
    self.Load()
    self.SetOutpath(outpath)
    self.SetOutput(output)
    self.SetLumi(lumi)
    self.SetColors(colors)
    self.SetRegion()
    self.categories = {}
    self.multicategories = {}
    self.doLegend = True
    self.doRatio = True
    self.doStack = True
    self.doLogY = False
    self.invertStack = False
    self.plotData = True
    self.fill_opts = {'edgecolor': (0,0,0,0.3), 'alpha': 0.8}
    self.error_opts = {'label':'Stat. Unc.','hatch':'///','facecolor':'none','edgecolor':(0,0,0,.5),'linewidth': 0}
    self.textParams = {'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 18, 'xtick.labelsize': 12, 'ytick.labelsize': 12}
    self.data_err_opts = {'linestyle':'none', 'marker': '.', 'markersize': 10., 'color':'k', 'elinewidth': 1,}#'emarker': '_'
    self.SetRange()
    self.SetRatioRange()
    self.yRatioTit = 'Data / Pred.'
    self.extraBkg = None
    self.systLabel = 'syst'
    self.systNormLabel = 'norm'
    self.labels = []
    self.systList = None
    self.SetRebin()
    
  def SetRebin(self, var=None, rebin=None, bN=None, includeLower=True, includeUpper=True):
    self.rebin = rebin
    if var is None or rebin is None: return
    if bN is None:
      self.hists[var] = Rebin(self.hists[var], var, rebin) 
    if bN is not None:
      b0 = rebin;
      histo = self.hists[var]
      axname = [x.name for x in histo.dense_axes()]
      self.hists[var] = Rebin(self.hists[var], var, b0, bN, includeLower, includeUpper)

  def SetVerbose(verbose=1):
    ''' Set level of verbosity '''
    self.verbose = verbose

  def SetPath(self, path):
    ''' Set path to sample '''
    # String with paths to several pkl files
    if isinstance(path, str) and ',' in path: 
      path = path.replace(' ', '').split(',')
    # Path to a directory containing several pkl files
    if os.path.isdir(path):
      listpath = []
      for p in os.listdir(path):
        if p.endswith('.pkl.gz'): listpath.append(path+'/'+p)
      self.path = listpath
      return
    self.path = path

  def Load(self, path=''):
    ''' Get a dictionary histoname : histogram '''
    if path != '': self.SetPath(path)
    self.hists = {}
    self.other = {}
    listpath = self.path if isinstance(self.path, list) else [self.path]
    for path in listpath:
      name = GetFileNameFromPath(path)
      self.other[name] = {}
      with gzip.open(path) as fin:
        hin = pickle.load(fin)
        for k in hin.keys():
          if not isinstance(hin[k], coffea.hist.Hist):
            self.other[name][k] = hin[k] 
            continue
          if self.var is not None and k not in self.var: continue
          if not k in self.listOfVars: self.listOfVars.append(k)
          if k in self.hists: self.hists[k]+=hin[k]
          else:               self.hists[k]=hin[k]
    self.GroupProcesses()
    if self.verbose >= 3: print(self.hists)

  def GetListOfVars(self):
    ''' Get the full list of variables for the loaded histograms '''
    return self.listOfVars

  def SetProcessDic(self, prdic, sampleLabel='sample', processLabel='process'):
    ''' Set a dictionary process : samples '''
    self.prDic = OrderedDict()
    self.sampleLabel = sampleLabel
    self.processLabel = processLabel
    if len(prdic) == 0: return
    var = prdic[list(prdic.keys())[0]]
    if isinstance(var, str):
      for k in prdic:
        self.prDic[k] = (prdic[k].replace(' ', '').split(','))
    else:
      for k in groupDic:
        self.prDic[k] = (prdic[k])

  def GroupProcesses(self, prdic={}):
    ''' Move from grouping in samples to groping in processes '''
    if prdic != {}: self.SetProcessDic(prdic)
    for k in self.hists.keys(): 
      if len(self.hists[k].identifiers('sample')) == 0: continue
      if k == 'SumOfEFTweights': continue
      self.hists[k] = self.hists[k].group(hist.Cat(self.sampleLabel, self.sampleLabel), hist.Cat(self.processLabel, self.processLabel), self.prDic)

  def SetBkgProcesses(self, bkglist=[]):
    ''' Set the list of background processes '''
    self.bkglistOrig = bkglist.copy()
    self.bkglist = bkglist.copy()
    if isinstance(self.bkglist, str): 
      self.bkglist = self.bkglist.replace(' ', '').split(',')
    self.bkgdic = OrderedDict()
    for b in self.bkglist: self.bkgdic[b] = b

  def SetSignalProcesses(self, siglist=[]):
    ''' Set the list of signal processes '''
    self.signallist = siglist
    if isinstance(self.signallist, str): 
      self.signallist = self.signallist.replace(' ', '').split(',')
    self.signaldic = OrderedDict()
    for s in self.signallist: self.signaldic[b] = b

  def SetDataName(self, dataName='data'):
    ''' Set the name of the data process '''
    self.dataName = dataName
    if self.verbose >= 3: print("Data name: ", self.dataName )

  def SetHistoDic(self, histoDic={}):
    ''' Set dictionary with histoName : x-axis title '''
    self.histoDic = histoDic

  def SetOutpath(self, outpath='./temp/'):
    ''' Set output path '''
    self.outpath = outpath
    
  def SetOutput(self, output=None):
    ''' Set output name '''
    self.output = output

  def SetColors(self, colors={}):
    ''' Set a dictionary with a color for each process '''
    if isinstance(colors, str):
      colors = colors.replace(' ', '').split(',')
      self.SetColors(colors)
      return
    elif isinstance(colors, list):
      self.colors = {}
      for i in range(len(self.prDic)):
        key = list(self.prDic.keys())[i]
        if i < len(colors): self.colors[key] = colors[i]
        else              : self.colors[key] = '#000000'
      return
    elif isinstance(colors, dict):
      self.colors = colors
      for key in list(self.prDic.keys()):
        if not key in self.colors: self.colors[key] = 1

  def GetColors(self, processes=[]):
    ''' Get a list of colors for each process '''
    col = []
    for k in processes: 
      c = self.colors[k] if k in self.colors else 1
      col.append(c)
    return col

  def SetLumi(self, lumi=59.7, lumiunit='fb$^{-1}$', sqrts='13 TeV'):
    self.lumi = lumi
    self.lumiunit = lumiunit
    self.sqrts = sqrts
  
  def SetRange(self, xRange=None, yRange=None):
    self.xRange = None
    self.yRange = None

  def SetRatioRange(self, ymin=0.5, ymax=1.5):
    self.ratioRange = [ymin, ymax]

  def SetRegion(self, ref=None):
    self.region = ref

  def SetLabel(self, lab=None):
    self.retion = lab

  def SetYRatioTit(self, rat='Ratio'):
    self.yRatioTit = rat

  def SetCategories(self, dic):
    self.categories = dic

  def SetCategory(self, catname, values):
    self.categories[catname] = values

  def AddCategory(self, catname, catdic):
    self.multicategories[catname] = catdic

  def AddLabel(self, x, y, text, options={}):
    lab = {'x':x, 'y':y, 'text':text, 'options':options}
    self.labels.append(lab)

  def AddExtraBkgHist(self, h, add=False):
    if not isinstance(h, list): h = [h]
    self.extraBkg = h if self.extraBkg is None else (self.extraBkg + h)
    if add:
      self.AddExtraBkgToHisto()
    for ih in h:
      for pr in ih.identifiers('process'): 
        if not pr in self.bkglist:
          self.bkglist.append(pr.name)

  def ResetExtraBkg(self, resetBkg=True):
    self.extraBkg = None
    if resetBkg: self.bkglist = self.bkglistOrig.copy()

  def AddExtraBkgToHisto(self, h=None):
    if h is None: 
      for ex in self.extraBkg: 
        axes = ex.dense_axes()
        var = [x.name for x in axes][0]
        newh = self.hists[var].add(ex)
        self.hists[var] = newh
        self.ResetExtraBkg(False)
      return newh
    else:
      if not isinstance(self.extraBkg, list): return h
      for ex in self.extraBkg: 
        h += ex
    return h

  def SetMultiCategores(self, multidic={}):
    if multidic =={} and self.categories != {}:
      self.multicategories = {'Yields' : self.categories}
    elif multidic =={}:
      pass
    else:
      self.multicategories = multidic

  def GetHistogram(self, hname, process=None, categories=None, removeProcessAxis=True):
    ''' Returns a histogram with all categories contracted '''
    if categories == None: categories = self.categories
    h = self.hists[hname]
    if isinstance(process, str) and ',' in process: process = process.split(',')
    if isinstance(process, list): 
      prdic = {}
      for pr in process: prdic[pr] = pr
      h = h.group("process", hist.Cat("process", "process"), prdic)
    elif isinstance(process, str): 
      h = h[process]#.sum("process")
      if removeProcessAxis: h = h.sum("process")
    for cat in categories: 
      h = h.integrate(cat, categories[cat])
    #self.SetRebin(rebin)
    #if self.rebin is not None:
    #  h = Rebin(h, hname, self.rebin)
    return h

  def SetSystematics(self, syst=None):
    if isinstance(syst, str) and ',' in syst: syst = syst.replace(' ', '').split(',')
    elif isinstance(syst, str): syst = []
    self.systList = syst

  def doData(self, hname):
    ''' Check if data histogram exists '''
    if self.dataName.lower() in ['asimov', 'pseudodata']: return True
    else:
      return self.dataName in [str(x) for x in list(self.hists[hname].identifiers(self.processLabel))] and self.plotData

  def SetLegend(self, do=True):
    self.doLegend = do

  def SetRatio(self, do=True):
    self.doRatio = do

  def SetStack(self, do=True):
    self.doStack = do

  def SetInvertStack(self, do=True):
    self.invertStack = do

  def SetLogY(self, do=True):
    self.doLogY = do




  ######################################################################################
  ######## Uncertainties
  def GetUncertaintiesFromHist(self, systAxisName="syst", systNomName="norm", stat=False, syst=None, hist=None):
    ''' Get uncertainties object '''
    if isinstance(hist, str):
      h = self.GetHistogram(hname, self.bkglist)
    else:
      h = hist
    self.uncertainties = UncertHisto()
    if syst is None: 
      syst = GetSystListForHisto(h, systAxisName, systNomName, stat)
    #print("Uncetainty bands for these systematics: ", syst)
    for bkg in self.bkglist:
      hb = h.integrate(self.processLabel, bkg)
      hbnom = hb.integrate(systAxisName, systNomName)
      self.uncertainties.AddProcess(hbnom, bkg)
      for s in syst:
        systList = [x.name for x in list(h.identifiers(systAxisName))]
        up = None; do = None
        if s+'Up' in systList:
          up = hb.integrate(systAxisName, s+'Up')
          if up.values() == {}: up = None
        if s+'Down' in systList:
          do = hb.integrate(systAxisName, s+'Down')
          if do.values() == {}: do = None
        elif s+'Do' in systList:
          do = hb.integrate(systAxisName, s+'Do')
          if do.values() == {}: do = None
        if up is not None and do is not None:
          self.uncertainties.AddSyst(s, bkg, up, do)
        elif up is not None and do is None:
          self.uncertainties.AddSyst(s, bkg, up, down=None)
    return self.uncertainties.GetSystematic(self.bkglist, syst)



  ########################################################################
  ### Save combine
  def SaveCombine(self, var, channel, categories={}):
    import uproot3
    if categories == {}: categories = self.categories
    prlabel = self.processLabel
    systlabel = self.systLabel
    h = self.GetHistogram(var, categories=categories) 

    # Open the file
    out = self.outpath + channel + '.root'
    if os.path.isfile(out): os.system('mv %s %s.old'%(out, out))
    if not os.path.isdir(self.outpath): os.system('mkdir -p ' + self.outpath)
    fout = uproot3.create(out)

    # Get processes
    processes = [x.name for x in list(h.identifiers(prlabel))]
    processes_no_data = processes.copy()
    if 'data' in processes_no_data: processes_no_data.pop(processes_no_data.index('data'))
    
    # Nominal histograms
    hnorm = h.integrate(systlabel, self.systNormLabel)

    for pr in processes:
      hpr = hnorm.integrate(prlabel, pr)
      if pr.lower() in [self.dataName.lower(), 'data']: continue
      else: hpr.scale(self.lumi)
      fout[pr] = hist.export1d(hpr)

    if self.dataName.lower() == 'asimov': 
      hnorm.scale(self.lumi)
    hdata,_ = self.GetData(var, hnorm)
    fout['data_obs'] = hist.export1d(hdata)

    # Get systematics
    syst = GetSystListForHisto(h, systlabel, self.systNormLabel)
    for pr in processes:
      for s in syst:
        htempup = h.integrate(prlabel, pr).integrate(systlabel, s+'Up')#.integrate(prlabel, pr)
        htempdo = h.integrate(prlabel, pr).integrate(systlabel, s+'Down')#.integrate(prlabel, pr)
        if htempdo.values() == {}: 
          htempdo = h.integrate(prlabel, pr).integrate(systlabel, s+'Do')
        if htempup.values() == {} or htempdo.values() == {}: continue
        htempup.scale(self.lumi)
        htempdo.scale(self.lumi)
        fout["%s_%sUp"  %(pr, s)] = hist.export1d(htempup)
        fout["%s_%sDown"%(pr, s)] = hist.export1d(htempdo)
    print('Created file: ', out)
    fout.close()


  def GetData(self, hname, h=None, integrateAxis=True):
    if   self.dataName.lower() == 'pseudodata': 
      if h is None: print("ERROR: we need a histogram to create pseudodata!")
      hData = GetPseudodata(h, integrateAxis=integrateAxis)
      dataLabel = 'Pseudodata'
    elif self.dataName.lower() == 'asimov':
      if h is None: print("ERROR: we need a histogram to create an Asimov!")
      hData = GetAsimov(h, integrateAxis=integrateAxis)
      dataLabel = 'Asimov'
    else: 
      hData = self.GetHistogram(hname, self.dataName)
      dataLabel = 'Data'
    return hData, dataLabel












  def DrawComparison(self, var, process, selection, labels=[], color=[], lineStyle=[], scale=[], doRatio=True, xtit='', ytit=''):
    # Var can be list of N elemnts or a string
    # listOfProcessDics --> N elements of the form: process : {selection dic}, the comparisons are done w.r.t. the first one
    # labels, color, lineStyle, scale --> list of N elements

    # Deal with processes and other inputs
    self.doRatio = doRatio
 
    N1 = len(var)       if isinstance(var,       list) else 1
    N2 = len(process)   if isinstance(process,   list) else 1
    N3 = len(selection) if isinstance(selection, list) else 1
    N = max([N1, N2, N3])
    if N<=1:
      print('ERROR: var, process or selection must be a list with at least two elements')
      return
    if isinstance(var, str): var = [var]*N
    if isinstance(process, str): process = [process]*N
    if isinstance(selection, dict): selection = [seclection]*N
    if isinstance(color, str): color = [color]*N
    elif color == []: color = ['k']*N
    if isinstance(labels, str): labels = [labels]*N
    elif labels == []: labels = ['']*N
    if isinstance(lineStyle, str): lineStyle = [lineStyle]*N
    elif lineStyle == []: lineStyle = ['']*N
    if isinstance(scale, float): scale = [scale]*N
    elif scale == []: scale = [1.0]*N

    # Figure
    if self.doRatio:
      fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
      fig.subplots_adjust(hspace=.07)
    else:
      fig, ax = plt.subplots(1, 1, figsize=(7,7))#, gridspec_kw={"height_ratios": (3, 1)}, sharex=True)

    #selection[0]['process'] = process
    hmain = self.GetHistogram(var[0], process[0], selection[0])#, process[0],{})# selection[0])
    #print('var = ', var[0], ', process = ', process[0], ', selection = ', selection[0])
    #print(hmain)
    #PrintHisto(hmain)
    #print(hmain.values())
    hmain.scale(scale[0])
    hist.plot1d(hmain, ax=ax, clear=False, line_opts={'color':color[0]})
    histograms = [hmain]
    for i in range(1,N):
      h = self.GetHistogram(var[i], process[i], selection[i])
      hist.plot1d(h, ax=ax, clear=False, line_opts={'color':color[i]})
      hist.plotratio(h, hmain, clear=False, ax=rax, error_opts={'marker':'', 'linewidth':2, 'color':color[i], 'linestyle':'-'}, unc='num')
      histograms.append(h)

    ymax = GetHistoMax(histograms)
    ax.autoscale(axis='x', tight=True)
    ax.set_ylim(0, ymax*1.1)
    ax.set_xlabel(None)

    if self.doLegend:
      leg_anchor=(1., 1.)
      leg_loc='upper left'
      handles, blah = ax.get_legend_handles_labels()
      ax.legend(handles, labels)
    
    if self.doRatio:
      rax.set_ylabel(self.yRatioTit)
      rax.set_ylim(self.ratioRange[0], self.ratioRange[1])
    elif not self.doRatio:
      ax.set_xlabel(xtit)

    if self.doLogY:
      ax.set_yscale("log")
      ax.set_ylim(1,ax.get_ylim()[1]*5)        

    if not self.xRange is None: ax.set_xlim(xRange[0],xRange[1])
    if not self.yRange is None: ax.set_ylim(yRange[0],yRange[1])

    # Labels
    CMS  = plt.text(0., 1., r"$\bf{CMS}$ Preliminary", fontsize=16, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
    #lumi = plt.text(1., 1., r"%1.1f %s (%s)"%(self.lumi, self.lumiunit, self.sqrts), fontsize=20, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
    for lab in self.labels:
      plt.text(lab['x'], lab['y'], lab['text'], transform=ax.transAxes, **lab['options'])

    if not self.region is None:
      lab = plt.text(0.03, .98, self.region, fontsize=16, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
      if not self.doLogY: ax.set_ylim(0,ax.get_ylim()[1]*1.1)
    
    # Save
    os.system('mkdir -p %s'%self.outpath)
    if self.output is None: self.output = 'comp'
    fig.savefig(os.path.join(self.outpath, self.output+'.png'))
    print('New plot: ', os.path.join(self.outpath, self.output+'.png'))
    plt.close('all')
















  def Stack(self, hname={}, xtit='', ytit='', aname=None, dosyst=False):
    ''' prName can be a list of histograms or a dictionary 'histoName : xtit' '''
    if isinstance(hname, dict):
      for k in hname: self.Stack(k, hname[k], ytit)
      return
    if isinstance(hname, list):
      for k in hname: self.Stack(k, xtit, ytit)
      return
     
    density = False; binwnorm = None
    plt.rcParams.update(self.textParams)
    aname = hname if aname is None else aname

    if self.doData(hname) and self.doRatio:
      fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
      fig.subplots_adjust(hspace=.07)
    else:
      fig, ax = plt.subplots(1, 1, figsize=(7,7))#, gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
      fig.subplots_adjust(hspace=.07)

    fill_opts  = self.fill_opts
    error_opts = self.error_opts
    data_err_opts = self.data_err_opts
    if not self.doStack:
      error_opts = None
      fill_opts  = None

    h = self.GetHistogram(hname, self.bkglist)
    h.scale(self.lumi)
    if xtit=='': xtit = h.axis(aname).label
    self.AddExtraBkgToHisto(h)

    # Colors
    from cycler import cycler
    #print('bkg list = ', self.bkglist)
    colors = self.GetColors(self.bkglist)
    #if self.invertStack: 
    _n = len(h.identifiers('process'))-1
    colors = colors[_n::-1]
    ax.set_prop_cycle('color', colors)
    #if self.invertStack and type(h._axes[0])==hist.hist_tools.Cat:  h._axes[0]._sorted.reverse() 

    # Splitting into syst and nominal if syst exist
    drawSystBand = False
    if self.systLabel in [x.name for x in h.axes()] and dosyst:
      drawSystBand = True
      hsyst = h.copy()
      h = h.integrate(self.systLabel, self.systNormLabel)

    ybkg = h.integrate("process").values(overflow='all')
    if ybkg == {}: return #process not found
    ybkg = ybkg[list(ybkg.keys())[0]]
    ybkgmax = max(ybkg)
    hist.plot1d(h, overlay="process", ax=ax, clear=False, stack=self.doStack, order=self.bkglist[::-1], density=density, line_opts=None, fill_opts=fill_opts, error_opts=None if drawSystBand else self.error_opts, binwnorm=binwnorm)

    ydata = 0; ydatamax = 0
    if self.doData(hname):
      hData, dataLabel = self.GetData(hname, h)
      if self.systLabel in [x.name for x in hData.axes()]:
        hData = hData.integrate(self.systLabel, self.systNormLabel)
      hist.plot1d(hData, ax=ax, clear=False, error_opts=data_err_opts, binwnorm=binwnorm)
      ydata = hData.values(overflow='all')
      ydata = ydata[list(ydata.keys())[0]]
      ydatamax = max(ydata)

    ax.autoscale(axis='x', tight=True)
    ax.set_ylim(0, max(ybkgmax*1.1, ydatamax*1.1) )
    ax.set_xlabel(None)

    if self.doLegend:
      leg_anchor=(1., 1.)
      leg_loc='upper left'
      handles, labels = ax.get_legend_handles_labels()
      if self.doData(hname):
        handles = handles[-1:]+handles[:-1][::-1]
        labels = [dataLabel]+labels[:-1][::-1]
      ax.legend(handles, labels)#,bbox_to_anchor=leg_anchor,loc=leg_loc)
    
    if self.doData(hname) and self.doRatio:
      #hist.plotratio(hData, h.sum("process"), clear=False,ax=rax, error_opts=data_err_opts, denom_fill_opts={}, guide_opts={}, unc='num')
      hist.plotratio(hData, h.sum("process"), clear=False,ax=rax, error_opts=data_err_opts, denom_fill_opts= {'alpha':0, 'color':'none'} if drawSystBand else {}, guide_opts={}, unc='num')
      rax.set_ylabel(self.yRatioTit)
      rax.set_ylim(self.ratioRange[0], self.ratioRange[1])
      if xtit!='' and xtit is not None: rax.set_xlabel(xtit)
    elif not self.doRatio:
      ax.set_xlabel(xtit)

    if self.doLogY:
      ax.set_yscale("log")
      ax.set_ylim(1,ax.get_ylim()[1]*5)        

    if not self.xRange is None: ax.set_xlim(xRange[0],xRange[1])
    if not self.yRange is None: ax.set_ylim(yRange[0],yRange[1])

    # Draw uncertainty bands
    if drawSystBand:
      up, do = self.GetUncertaintiesFromHist(hist=hsyst, syst=self.systList)
      if self.doData(hname) and self.doRatio:
        r1, r2 = DrawUncBands(ax, h.sum('process'), up, do, ratioax=rax, hatch="\/\/", color="gray")
      else:
        r1, r2 = DrawUncBands(ax, h.sum('process'), up, do, hatch="\/\/", color="gray")

    # Labels
    CMS  = plt.text(0., 1., r"$\bf{CMS}$ Preliminary", fontsize=16, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
    lumi = plt.text(1., 1., r"%1.0f %s (%s)"%(self.lumi, self.lumiunit, self.sqrts), fontsize=20, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
    for lab in self.labels:
      plt.text(lab['x'], lab['y'], lab['text'], transform=ax.transAxes, **lab['options'])

    if not self.region is None:
      lab = plt.text(0.03, .98, self.region, fontsize=16, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
      ax.set_ylim(0,ax.get_ylim()[1]*1.1)
    
    # Save
    os.system('mkdir -p %s'%self.outpath)
    if self.output is None: self.output = hname
    fig.savefig(os.path.join(self.outpath, self.output+'.png'))
    fig.savefig(os.path.join(self.outpath, self.output+'.pdf'))
    print('New plot: ', os.path.join(self.outpath, self.output+'.png'))
    plt.close('all')
    #else: fig.savefig(os.path.join(self.outpath, hname+'_'+'_'.join(self.region.split())+'.png'))

  def GetYields(self, var='counts', cat=None, pr=None, doErr=False, syst=None):
    sumy = 0; sumerr = 0;
    dicyields = {}
    dicerrors = {}
    h = self.GetHistogram(var, self.bkglist, cat)
    systlabel = self.systLabel; norm = self.systNormLabel
    if systlabel in [x.name for x in h.axes()]:
      if syst is None: h = h.integrate(systlabel, norm)
      else: h = h.integrate(systlabel, syst)
    h.scale(self.lumi)
    for bkg in self.bkglist:
      y = h[bkg].integrate("process").values(overflow='all', sumw2=True)
      if y == {}: continue #process not found
      ye = np.sqrt(np.power(y[list(y.keys())[0]][1],2).sum())
      y  = y[list(y.keys())[0]][0].sum()
      sumy += y
      sumerr += ye*ye
      dicyields[bkg] = y
      dicerrors[bkg] = ye
    if self.doData(var):
      catdata = cat.copy(); 
      if 'syst' in catdata.keys(): catdata['syst'] = 'norm'
      hData = self.GetHistogram(var, self.dataName, catdata)
      ydata = hData.values(overflow='all', sumw2=True)
      nderr = np.sqrt(np.power(ydata[list(ydata.keys())[0]][1], 2).sum())
      ndata = ydata[list(ydata.keys())[0]][0].sum()
      dicyields[self.dataName] = ndata
      dicerrors[self.dataName] = nderr
    if pr is None:
      if not doErr: return dicyields
      return dicyields, dicerrors
    else:
      y = 0; ye = 0
      if isinstance(pr, str):
        if ',' in pr: pr = pr.replace(' ', '').split(',')
        else: pr = [pr]
      for p in pr:
        if p in dicyields.keys():
          y +=dicyields[p]
          ye+=dicerrors[p]*dicerrors[p]
      ye = np.sqrt(ye)
      if not doErr: return y
      return y, ye

  def GetYieldsSystUpDown(self, syst, process=None, var='counts', cat=None):
    if process is None: process = self.bkglist
    h = self.GetHistogram(var, self.bkglist, cat)
    systList = [x.name for x in h.identifiers(self.systLabel)]
    up = None; do = None
    if syst+'Up' in systList:
      up = self.GetYields(var, cat, process, syst=syst+'Up')
    if syst+'Down' in systList:
      do = self.GetYields(var, cat, process, syst=syst+'Down')
    elif syst+'Do' in systList:
      do = self.GetYields(var, cat, process, syst=syst+'Do')
    return up, do

  def PrintYields(self, var='counts', tform='%1.2f', save=False, multicategories={}, bkgprocess=None, signalprocess=None, doData=True, doTotBkg=True):
    if bkgprocess   !=None: self.SetBkgProcesses(bkgprocess)
    if signalprocess!=None: self.SetSignalProcesses(signalprocess)
    if multicategories != {} : 
      k0 = multicategories.keys()[0]
      if not isinstance(multicategories[k0], dict): # Not a multicategory, but single one
        self.SetMultiCategores({'Yields' : multicategories})
      else:
        self.SetMultiCategores(multicategories)
    else: 
      self.SetMultiCategores(multicategories)
    if self.multicategories == {}:
      print('[plotter.PrintYields] ERROR: no categories found!')
      exit()

    # Output name
    name = save if (save and save!='') else 'yields'

    # Create an OutText object: the output will be a tex file
    t = OutText(self.outpath, name, 'new', 'tex')
    t.bar()

    ncolumns = len(self.multicategories)
    t.SetTexAlign('l' + ' c'*ncolumns)
    dic = {}
    for k in self.multicategories.keys():
      self.SetCategories(self.multicategories[k])
      if k not in dic: continue
      dic[k] = self.GetYields(var)
    # header
    header = ''
    for label in dic: header += t.vsep() + ' ' + label
    t.line(header)
    t.sep()
    # backgrounds
    for bkg in self.bkglist:
      line = bkg
      for label in dic:
        line += t.vsep() + ' ' + (tform%(dic[label][bkg]))
      t.line(line) 
    if len(self.bkglist) > 0: t.sep()
    if doTotBkg and len(self.bkglist)>0:
      line = 'Total background'
      for label in dic:
        line += t.vsep() + ' ' + (tform%(sum([dic[label][bkg] for bkg in self.bkglist])))
      t.line(line) 
      t.sep()
    for sig in self.signallist:
      line = sig
      for label in dic:
        line += t.vsep() + ' ' + (tform%(dic[label][sig]))
      t.line(line) 
    if len(self.signallist) > 0:  t.sep()
    if doData:
      line = self.dataName
      for label in dic:
        line += t.vsep() + ' ' + tform%(dic[label][self.dataName]) 
      t.line(line)
      t.sep()
    t.write()


  '''
    # Get colors for the stack
    colors = self.GetColors(self.bkglist)
    ax.set_prop_cycle(cycler(color=colors))

    # Data
    dataOpts = {'linestyle':'none', 'marker':'.', 'markersize':10., 'color':'k', 'elinewidth':1}
    if self.dataName in [str(x) for x in list(self.hists[hname].identifiers(self.processLabel))]:
      plot.plot1d(self.hists[hname].sum('cut').sum('channel').sum('Zcat').sum('lepCat')[self.dataName],
        overlay=self.processLabel, ax=ax, clear=False, error_opts=dataOpts)

    # Background
    fillOpt = {'edgecolor': (0,0,0,0.3), 'alpha': 0.8}
    mcOpt   = {'label':'Stat. Unc.', 'hatch':'///', 'facecolor':'none', 'edgecolor':(0,0,0,.5), 'linewidth': 0}
    for bkg in self.bkgdic:
      fillOpti = {'edgecolor': (0,0,0,0.3), 'alpha': 0.8}
      fillOpti['color'] = self.colors[bkg]
      #h = self.hists[hname].sum('cut').sum('channel').sum('Zcat').sum('lepCat')[bkg] #.sum(self.processLabel)
      h = self.hists[hname]
      for cat in self.categories: h = h.integrate(cat, self.categories[cat])
      h = h[bkg]
      h.scale(self.lumi*1000)
      y = h.values(overflow='all')
      print(bkg, ' : ', y[list(y.keys())[0]].sum())
    h = self.hists[hname]
    for cat in self.categories: h = h.integrate(cat, self.categories[cat])
    plot.plot1d(h, ax=ax, clear=False, stack=True, fill_opts=fillOpti, overlay=self.processLabel )#, error_opts=mcOpt)
    hbkg = self.hists[hname]
    for cat in self.categories: hbkg = hbkg.integrate(cat, self.categories[cat])
    hbkg = hbkg.group(hist.Cat(self.processLabel,self.processLabel), hist.Cat(self.processLabel, self.processLabel), {'All bkg' : self.bkglist})
    plot.plot1d(hbkg, ax=ax, clear=False, overlay=self.processLabel)#, error_opts={'hatch':'///', 'facecolor':'none', 'edgecolor':(0,0,0,.5), 'linewidth': 0}, overlay=self.processLabel)

    #hbkg = self.hists[hname].group(hist.Cat(self.processLabel,self.processLabel), hist.Cat(self.processLabel, self.processLabel), self.bkgdic)
    #plot.plot1d(hbkg.sum('level').sum('channel'),
    #  overlay=self.processLabel, ax=ax, clear=False, stack=True, fill_opts=fillOpt, error_opts=mcOpt)

    # Signal
    #ax._get_lines.prop_cycler = ax._get_patches_for_fill.prop_cycler
    #args = {'linestyle':'--', 'linewidth': 5}
    #plot.plot1d(signal_hists[key].project('jet_selection','baggy').project('region','iszeroL'),
    #            ax=ax, overlay="process", clear=False, stack=False, line_opts=args)

    # Options
    ax.autoscale(axis='x', tight=True)
    if self.doLogY: ax.set_yscale('log')
    ax.set_ylim(.1, None)
    leg = ax.legend()
    region = plt.text(0., 1., u"☕ %s"%self.region, fontsize=20, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
    lumi = plt.text(1., 1., r"%1.1f %s (%s)"%(self.lumi, self.lumiunit, self.sqrts), fontsize=20, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
    os.system('mkdir -p %s'%self.outpath)
    fig.savefig(os.path.join(self.outpath, hname+'.png'))
    '''
