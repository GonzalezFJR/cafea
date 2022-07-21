from cafea.plotter.plotter import *
import matplotlib.pyplot as plt
path = 'DoubleMuon_5TeV.pkl.gz'

var = 'Z'
h = GetHisto(["DoubleMuon_dimuon.pkl.gz", "DoubleMuonLowMass_dimuon.pkl.gz"], var)

fig, ax = plt.subplots(1, 1, figsize=(7,7))
hist.plot1d(h, ax=ax, fill_opts=None)

if var == 'invmass':
  ax.set_yscale("log")
  ax.set_xscale("log")
  ax.set_ylim(1, 1e5)
  ax.set_xlim(0.2, 120)

elif var == 'Z':
  ax.set_xlim(80, 100)

fig.savefig(var+".png")
