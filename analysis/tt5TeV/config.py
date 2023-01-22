from __future__ import print_function, division
from collections import defaultdict, OrderedDict
import gzip
import pickle
import json
import uproot3
import matplotlib.pyplot as plt
import numpy as np
np.seterr(divide='ignore', invalid='ignore', over='ignore')
from coffea import hist, processor 
from coffea.hist import plot
import os, sys
from cafea.plotter.OutText import OutText

from cafea.plotter.plotter import plotter, GetH1DfromXY
from cafea.plotter.plotter import *

import argparse
parser = argparse.ArgumentParser(description='You can customize your run')
parser.add_argument('--path',     '-p', default = 'histos/plots5TeV.pkl.gz', help = 'Path to pkl file')
parser.add_argument('--variable', '-v', default = None                 , help = 'Variable')
parser.add_argument('--channel',  '-c', default = 'em'                     , help = 'Channels')
parser.add_argument('--level',    '-l', default = 'incl'                   , help = 'Variable')
parser.add_argument('--output',   default = None                     , help = 'Name of the output png file')
parser.add_argument('--outpath',  '-o', default = None                     , help = 'Name of the output path')
parser.add_argument('--data',     '-d', action= 'store_true'             , help = 'Do data?')
parser.add_argument('--syst',     '-s', default= None             , help = 'Systematic choice')
parser.add_argument('--nSlots',   '-n', default= 4             , help = 'Number of slots for parallelization')
parser.add_argument('--verbose',   default= 0             , help = 'level of verbosity')
parser.add_argument('--force',  '-f', action= 'store_true'             , help = 'Force to overwrite')
args = parser.parse_args()

path  = args.path
var = args.variable
ch = args.channel
level = args.level
output = args.output
doData = args.data
outpatho = args.outpath
systch = args.syst
verbose = int(args.verbose)
nSlots = int(args.nSlots)
force = args.force
if outpatho is None: outpatho = 'temp/'
if not outpatho.endswith('/'): outpatho += '/'
#syst = 'norm'

pathQCD = path

from datetime import datetime
now = datetime.now()
datatoday = str(now.strftime('%d')) + str(now.strftime('%B')).lower()[:3] + str(now.strftime('%Y'))[2:]
baseweb = '/nfs/fanae/user/juanr/www/public/tt5TeV/ljets/'

# Convert string to list
if   isinstance(ch, str) and ',' in ch: ch = ch.replace(' ', '').split(',')
elif isinstance(ch, str): ch = [ch]
if   isinstance(level, str) and ',' in level: level = level.replace(' ', '').split(',')
lumi = 302; # pb
year = 2017

processDic = {
  'tt': 'ttPS',#, ttPS',
  'tW': 'tbarW, tW',
  'WJets': 'WJetsToLNu',#  'W0JetsToLNu, W1JetsToLNu, W2JetsToLNu, W3JetsToLNu',
  'QCD': 'QCD',
  'DY': 'DYJetsToLLMLL50, DYJetsToLLM10to50',
  'data' : 'SingleMuon, HighEGJet',
}

processDic_noQCD = processDic.copy()
processDic_noQCD.pop('QCD')

bkglist    = ['tt', 'tW', 'WJets', 'DY', 'QCD']
bkglist_noQCD = ['tt', 'tW', 'WJets', 'DY']
bkgnormunc = [0.05, 0.2, 0.2, 0.2, 0.2]

colordic ={
  'tt' : '#cc0000',
  'tW' : '#ffc207',
  'WJets': '#47ce33',
  'DY': '#3b78cb',
  'QCD' : '#aaaaaa',
}

colors = [colordic[k] for k in bkglist]

def GetChLab(channel):
  if isinstance(channel, list) and len(channel) > 1:
    channel = '$\ell$'
  elif isinstance(channel, list):
    channel = channel[0]
  if '_fake' in channel: 
    channel = 'non-iso ' + channel[0]
  channel = channel.replace('m', '$\mu$')
  return channel

def GetLevLab(lev):
  if   lev == 'incl'  : return ''
  elif lev == 'g2jets': return ', $\geq$2 jets'
  elif lev == 'g4jets': return ', $\geq$4 jets'
  elif lev == '0b'    : return ', $\geq$4 jets, 0b'
  elif lev == '1b'    : return ', $\geq$4 jets, 1b'
  elif lev == '2b'    : return ', $\geq$4 jets, 2b'
  elif lev == 'g5j1b' : return ', $\geq$4j, 1b'
  elif lev == 'g5j2b' : return ', $\geq$5j, 2b'
  return lev

def GetModSystHisto(path, fname, systname, var=None, prname='tt', samplab='sample', prlab='process', systlab='syst', systnormlab='norm'):
  h  = GetHisto(path+ fname +   '.pkl.gz', var, group=None)
  axes = [x.name for x in h.sparse_axes()]
  if not samplab in axes: return
  sampName = h.identifiers(samplab)[0]
  h = GroupKeepOrder(h, [[systlab, systlab, {systname:systnormlab}], [samplab, prlab, {prname:sampName}] ])
  return h

def GetModSystHistos(path, fname, systname, var=None):
  up = GetModSystHisto(path, fname+'Up',   systname+'Up', var)
  do = GetModSystHisto(path, fname+'Down', systname+'Down', var)
  return up, do


  #elif var in ['ht']:
  #  b0 = 2

def RebinVar(p, var, level=None):
  b0 = None; bN = None; binRebin=None
  xtit = None
  if var in ['minDRjj', 'minDRuu']:
    b0 = 0.4; bN = 2.0
  elif var =='medianDRjj':
    b0 = 1.0; bN = 3.5
  elif var =='MVAscore':
    b0 = 0.10; bN = 0.8; binRebin=2
  elif var == "njets" and 'level' != 'incl':
    b0 = 4; bN = 10
  elif var in ['st']:
    b0 = 120; bN = 600;
  elif var in ['sumallpt']:
    b0 = 0; bN = 200
    xtit = '$\sum_\mathrm{j,\ell}\,\mathrm{p}_{T}$ (GeV)'
  elif var in ['met','u0pt', 'ptuu', 'ptjj', 'metnocut']:
    b0 = 2;
  elif var in ['MVAscore']:
    b0 = 0.1; bN = 0.8
    binRebin = 2
  elif var in ['ht']:
    b0 = 4;
  if b0 is not None:
    p.SetRebin(var, b0, bN, includeLower=True, includeUpper=True, binRebin=binRebin)
  return xtit
