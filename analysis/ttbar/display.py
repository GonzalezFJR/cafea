from config import *
import matplotlib.pyplot as plt

path    = 'histos/plots.pkl.gz'
process = 'tt'
var     = 'invmass'
level   = 'dilep'
ch      = 'em'

histo = loadHistos(path)[var]
histo = histo.integrate('sample')
histo = histo.integrate('channel', ch).integrate('level', level)
histo = histo.integrate('syst', 'norm').integrate('sign', 'OS')
PrintHisto(histo)
fig = plt.figure()
hist.plot1d(histo)
fig.savefig('temp.png')
