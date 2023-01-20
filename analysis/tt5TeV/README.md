## Analysis

The main analysis is in `tt5TeV.py`. To run it, use the `run.py` scrip. You need to have json files with the samples. For example, send a job to process tt sample with 

    python analysis/tt5TeV/run.py cafea/json/5TeV/poolphedex/TTPS.json -n 64 -j -o TTPS

To execute all the samples, you can use the script `run5TeV.sh`.

## QCD estimate

To estimate QCD, you need to have run over all the samples and have a folder with all the .pkl.gz files. The script `SaveQCD.py` takes those inputs and creates a `QCD.pkl.gz` file with the QCD estimate. Run the script as:

    python analysis/tt5TeV/SaveQCD.py -p histos5TeV/16jan2023/ -n 32

## Plotting and tables

There are several scripts to create plots and tables. You can find a description of some of them below.

 - ControlPlots.py: Produce control plots. You can modify the variables, channels and levels.
Example:

    python ControlPlots.py -p histos5TeV/16jan2023/ -n 16

 - PlotSystematics.py: Produce systematic plots, including comparisions. By default, it is done for ttbar only.
Example:

    python PlotSystematics.py -p histos5TeV/16jan2023/ 

## Datacards

You need to create rootfiles and datacards to produce fits and extract the cross section.
Rootfiles are created with `analysis/tt5TeV/saveCombineFile.py` and then, datacards are created with `analysis/tt5TeV/CreateDatacard.py`.
A script to create all the needed rootfiles and datacards in the analysis is executed as follows:

    python RunCombineProducer.py

You probably need to modify the inputs within the script.
