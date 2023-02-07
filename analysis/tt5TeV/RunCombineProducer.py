import os
from config import *

if var is None: var = "medianDRjj"

outpath = path + "combineFiles/"
if not os.path.exists(outpath):
    os.makedirs(outpath)

pathcomb = "/nfs/fanae/user/juanr/CMSSW_10_2_13/src/combinescripts/tt5TeVljets/"+datatoday + "/" 
if not os.path.exists(pathcomb):
    os.makedirs(pathcomb)

levels = ['3j1b', '3j2b', '4j1b', '4j2b', 'g5j1b', 'g5j2b']
channels = ['e', 'm']
iteration = 0
total = len(channels)*len(levels)

for ch in channels:
    for level in levels:
        print("\r[{:<100}] {:.2f} % ".format('#' * int(float(iteration)/total*100), float(iteration)/total*100),end='')
        var = "medianDRjj" if level not in ['3j1b'] else "MVAscore"
        outname = "%s_%s_%s.root"%(var, ch, level)
        if not os.path.exists(f"{pathcomb+outname}"):
          command = "python analysis/tt5TeV/SaveRootfile.py -p %s -v %s -l %s -c %s --data"%(path, var, level, ch)
          if verbose >= 1: print("Running: %s"%(command))
          os.system(command)

          # Move the file to the combine folder
          mvcommand = f"cp {outpath+outname} {pathcomb+outname}"
          if verbose: print("Running: %s"%(mvcommand))
          os.system(mvcommand)

        # Create the datacard
        cardcommand = f"python analysis/tt5TeV/CreateDatacard.py -p {path} --inputFile {pathcomb+outname}"
        if verbose >= 1: print("Running: %s"%(cardcommand))
        os.system(cardcommand)
        if verbose >= 1: print("  ")
        iteration+= 1
print("\r[{:<100}] {:.2f} % ".format('#' * int(float(iteration)/total*100), float(iteration)/total*100))
print('Datacards created in ', pathcomb)
