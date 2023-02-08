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
parser.add_argument('--inputFile',  default=''             , help = 'Used for combine scripts')
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
inputFile = args.inputFile
force = args.force
if outpatho is None: outpatho = 'temp/'
if not outpatho.endswith('/'): outpatho += '/'
#syst = 'norm'

from datetime import datetime
now = datetime.now()
datatoday = str(now.strftime('%d')) + str(now.strftime('%B')).lower()[:3] + str(now.strftime('%Y'))[2:]

if   isinstance(ch, str) and ',' in ch: ch = ch.replace(' ', '').split(',')
elif isinstance(ch, str): ch = [ch]
if   isinstance(level, str) and ',' in level: level = level.replace(' ', '').split(',')
lumi = 302;
year = 2017

processDic = {
  'tt': 'tt',
  'tW': 'tbarW, tW',
  'WJets': 'WJetsToLNu',
  'DY': 'DYJetsToLLMLL50, DYJetsToLLM10to50',
  'data' : 'SingleMuon, HighEGJet',
}

diclegendlabels = {'None':'Data', 'tt':'$\\mathrm{t\\bar{t}}$', 'DY':'Drell-Yan', 'WJets':'W+jets', 'QCD':'QCD'}

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

def RebinVar(p, var, level=None):
  b0 = None; bN = None; binRebin=None
  xtit = None

  if var in ['minDRjj', 'minDRuu']:
    b0 = 0.4; bN = 2.0
  elif var =='medianDRjj':
    #b0 = 1.0; bN = 3.5
    b0 = 1.4; bN = 3.2
  elif var in ['medianDRuu']:
    b0 = 0.5; bN = 3.7
  elif var == "njets" and 'level' != 'incl':
    b0 = 4; bN = 10
  elif var in ['st']:
    b0 = 120; bN = 600;
  elif var in ['sumallpt']:
    b0 = 0; bN = 200
    xtit = '$\sum_\mathrm{j,\ell}\,\mathrm{p}_{T}$ (GeV)'
  elif var in ['met','u0pt', 'ptuu', 'ptjj']:
    b0 = 2;
  elif var in ['metnocut']:
    b0 = 4;
  elif var in ['MVAscore']:
    b0 = 0.2; bN = 0.8
    binRebin = 2
  elif var in ['ht']:
    b0 = 100; bN = 450
    binRebin = 2;
  elif var in ['j0pt']:
    b0 = 40; bN = 200
  elif var in ['mjj', 'muu']:
    b0 = 25; bN = 150
  elif var in ['mlb']:
    b0 = 25; bN = 200
  elif var in ['dRlb']:
    b0 = 0.5; bN = 2.9
  if b0 is not None:
    p.SetRebin(var, b0, bN, includeLower=True, includeUpper=True, binRebin=binRebin)
  return xtit
