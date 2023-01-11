import os
basepath = "histos5TeV/"
path = "10jan2023/"
var = "medianDRjj"
opath = "11jan2023/"

outpath = basepath + path + "combineFiles/"
if not os.path.exists(outpath):
    os.makedirs(outpath)

pathcomb = "/nfs/fanae/user/juanr/CMSSW_10_2_13/src/combinescripts/tt5TeVljets/"+opath
if not os.path.exists(pathcomb):
    os.makedirs(pathcomb)

levels = ['3j1b', '3j2b', '4j1b', '4j2b', 'g5j1b', 'g5j2b']
channels = ['e', 'm']

for ch in channels:
    for level in levels:
        var = "medianDRjj" if level not in ['3j1b'] else "MVAscore"
        outname = "%s_%s_%s.root"%(var, ch, level)
        if not os.path.exists(f"{pathcomb+outname}"):
          command = "python analysis/tt5TeV/saveCombineFile.py -p %s -v %s -l %s -c %s --data"%(basepath+path, var, level, ch)
          print("Running: %s"%(command))
          os.system(command)

          # Move the file to the combine folder
          mvcommand = f"cp {outpath+outname} {pathcomb+outname}"
          print("Running: %s"%(mvcommand))
          os.system(mvcommand)

        # Create the datacard
        cardcommand = f"python analysis/tt5TeV/CreateDatacard.py -p {pathcomb+outname}"
        print("Running: %s"%(cardcommand))
        os.system(cardcommand)
        print("  ")
