# CAFEA
Coffea analysis framework en Asturias

### Contents
- `analysis`:
   Subfolders with different analyses: creating histograms, applying selections...
   Also including analysis-dependent plotter scripts and/or jupyter files

- `cafea/cfg`:
  Configuration files (lists of samples, cross sections...)

- `cafea/data`:
  External inputs used in the analysis: scale factors, corrections...
  
- `cafea/json`:
   JSON files containing the lists of root files for each sample

- `cafea/analysis`:
  Auxiliar python modules to build an analysis: corrections, objects, selection

- `cafea/modules`:
  Scripts to manage files, create .json files, etc

- `cafea/plotter`:
  Tools to produce stack plots and other plots

### Set up the environment 

First, create a conda environment (this has to be done only once). First, edit the config file `environment.yml` and change the name of the environment that you would like to create. Then, execute:

    conda env create -f environment.yml

To activate the environmen, execute:

    conda activate my-conda-env

You have to install the cafea code as a python module using pip:

    pip install -e .

The `-e` option installs the project in editable mode (i.e. setuptools "develop mode"). If you wish to uninstall the package, you can do so by running `pip uninstall cafea`.

