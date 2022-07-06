from __future__ import print_function, division
from collections import defaultdict, OrderedDict
import gzip
import pickle
import json
import uproot3
import matplotlib.pyplot as plt
import numpy as np
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
parser.add_argument('--level',    '-l', default = 'dilep'                   , help = 'Variable')
parser.add_argument('--output',   default = None                     , help = 'Name of the output png file')
parser.add_argument('--outpath',  '-o', default = None                     , help = 'Name of the output path')
parser.add_argument('--data',     '-d', action= 'store_true'             , help = 'Do data?')
args = parser.parse_args()

path  = args.path
var = args.variable
ch = args.channel
level = args.level
output = args.output
doData = args.data
outpatho = args.outpath
if outpatho is None: outpatho = 'temp/'
if not outpatho.endswith('/'): outpatho += '/'
syst = 'norm'

# Convert string to list
if   isinstance(ch, str) and ',' in ch: ch = ch.replace(' ', '').split(',')
elif isinstance(ch, str): ch = [ch]
lumi = 50; # pb
year = '2022'


processDic = {
  'tt': 'TTTo2L2Nu',
  'tW': 'tbarW, tW',
  'Nonprompt': 'WJetsToLNu, TTToSemiLep',
  'DY': 'DYJetsToLL_M50, DY_M10to50',
  'Diboson' : 'WW, WZ, ZZTo2L2Nu'#'WWTo2L2Nu, WZTo3LNu',#'WW, WZ, ZZTo2L2Nu',
}

bkglist    = ['tt', 'tW', 'Nonprompt', 'DY', 'Diboson']
bkglist = list(processDic.keys())
bkgnormunc = [0.05, 0.2, 0.2, 0.2, 0.3]

colordic ={
  'tt' : '#cc0000',
  'tW' : '#ffc207',
  'Nonprompt': '#47ce33',
  'DY': '#3b78cb',
  'Diboson' : '#fdffcb',
}

colors = [colordic[k] for k in colordic.keys()]

def GetChLab(channel):
  channel = channel.replace('m', '$\mu$')
  return channel

def GetLevLab(lev):
  if   lev == 'dilep'  : return ''
  elif lev == 'g2jets': return ', $\geq$2 jets'
  return ''

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

