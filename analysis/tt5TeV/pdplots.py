from config import *
from QCD import *
import pandas as pd
import os

outpath = path + 'tables/'
if not os.path.exists(outpath):
    os.makedirs(outpath)

var = 'counts'
processes = ['tt', 'tW', 'WJets', 'DY', 'QCD', 'data']
lev = ['3j1b', '3j2b', '4j1b', '4j2b', 'g5j1b', 'g5j2b']
ch = ['e', 'm']
syst = ['norm']


###############################################################################
def Yield(pr, lev, chan, syst='norm'):
    cat = {'level':lev, 'channel':chan, 'syst':syst}
    if pr != 'QCD': return plt.GetYields(cat=cat, pr=pr)
    else          : return qcd.GetYield(cat)

def CreateCSV(outpath='./', outname='yields'):
  plt = plotter(path, prDic=processDic, bkgList=bkglist, colors=colordic, lumi=lumi, var=var)
  qcd = QCD(path, prDic=processDic, bkglist=bkglist, lumi=lumi, var=var)

  dic = {'process':[], 'level':[], 'channel':[], 'syst':[], 'yield':[]}
  for pr in processes:
    for l in lev:
        for c in ch:
            for s in syst:
                dic['process'].append(pr)
                dic['level'].append(l)
                dic['channel'].append(c)
                dic['syst'].append(s)
                y = Yield(pr, l, c, s)
                dic['yield'].append(y)
                print('%s %s %s %s -- %s'%(pr, l, c, s, y))

  # Create a dataframe with the yields
  df = pd.DataFrame(dic)
  df.to_csv(outpath+outname+'.csv', index=False)
  return df

def GetYieldsTable(outpath='./', outname='yields'):
    if os.path.exists(outpath+outname+'.csv'):
        df = pd.read_csv(outpath+outname+'.csv')
        return df
    else:
        return CreateCSV(outpath, outname)

# Read the yields from the csv file
df = GetYieldsTable(outpath, 'yields')

# Create a new column with the total yield
dfe = df[df['channel'] == 'e']
dfm = df[df['channel'] == 'm']

import seaborn as sns
import matplotlib.pyplot as plt




def CreateYieldsTable(df, channel, syst='norm', outpath='./', outname='yields'):
    df = df[df['channel'] == channel]
    df = df[df['syst'] == syst]
    df = df[['process', 'level', 'yield']]
    df = df.pivot(index='process', columns='level', values='yield')

    # Add a row with the total yield
    df.loc['Total pred'] = df.sum() - df.loc['data']
    df = df.reindex(['tt', 'tW', 'WJets', 'DY', 'QCD', 'Total pred', 'data'])
    SaveTablePDF(df, outpath, outname)

def SaveTablePDF(df, outpath='./', outname='table'):
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
    df.to_latex(outpath+outname+'.tex', float_format="%.2f")
    # open pdf and modify
    with open(outpath+outname+'.tex', 'r') as file:
        pdf = file.read()
    pdf.replace('toprule', 'hline')
    pdf.replace('midrule', 'hline')
    pdf.replace('bottomrule', 'hline')
    pdf = r'\documentclass{article}' + '\n' + r'\usepackage{booktabs}' + '\n' + r'\begin{document}' + '\n' + pdf + '\n' + r'\end{document}'
    with open(outpath+outname+'.tex', 'w') as file:
        file.write(pdf)
        
    # quiet pdflatex
    os.system('pdflatex -interaction=nonstopmode -output-directory=%s %s.tex > '%(outpath, outpath+outname))
    print('Created %s.pdf'%(outpath+outname))
    if os.path.exists(outpath+outname+'.aux'): os.system('rm %s'%outpath+outname+'.aux')
    if os.path.exists(outpath+outname+'.log'): os.system('rm %s'%outpath+outname+'.log')


#CreateYieldsTable(df, 'e', outpath=outpath, outname='yields_e')
#CreateYieldsTable(df, 'm', outpath=outpath, outname='yields_m')


