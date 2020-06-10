# MergedHits
Identify and split merged hits in the CMS Pixel Tracker

# Analyzer + Notebook setup instruction
 Setup: (do not run any extra command from the moment you connect to fermilab/lxplus6, not lxplus-slc7)
	• cd /uscms_data/d3/hichemb/princeton/project1/ (go to your home directory).
	• source /cvmfs/cms.cern.ch/cmsset_default.csh (or sh for bash).
	• scram list (gives you list of CMSSW and check compatibility with your current architecture).
	• echo $SCRAM_ARCH (outputs: 'slc6_amd64_gcc476' for me, not compatible with the CMSSW environment I want to use (10_4_0), so I change architecture).
Instructions for CSH (default)	Instructions for BASH
• Setenv SCRAM_ARCH slc6_amd64_gcc700 	• SCRAM_ARCH=newarchictecture
	• export SCRAM_ARCH
	• echo $SCRAM_ARCH (should output: 'slc6_amd64_gcc700').
	• cmsrel CMSSW_10_4_0 (I chose this one).
	• cd CMSSW_10_4_0/src
	• cmsenv
	• (DO NOT MAKE ANY EXTRA DIRECTORY).
	• git clone https://github.com/nhaubrich/TrackingDstar 
	• scram b -j 8 (should get a few warnings).
	• cd TrackingDstar/LambaAnalyzer/test/

Running/Making root files
Run locally (fastest, good for testing):
	• cmsRun local_trkeffanalyzer_MC_GeneralTracks_cfg.py (runs on files from SingleNeutrino dataset I copied over) 

To run crab jobs:
	• voms-proxy-init -voms cms (use certificate, apply for grid certificate if you don't have one)
	• source /cvmfs/cms.cern.ch/crab3/crab.csh  OR .sh (depending on what you use) 
	• crab submit crab_MC_truth_cfg.py (can change RECO file used by editing crab_MC_truth_cfg.py)
	• (once you submit crab use 'crab status…(copy command from there)' to check on status.)

trkeffanalyzer_data_GeneralTracks_cfg.py  may be broken (missing input data ?))
cmsRun trkeffanalyzer_Data_GeneralTracks_cfg.py (outdated)
cmsRun trkeffanalyzer_MC_GeneralTracks_cfg.py (outdated)


ROOT Post-Processing
After running analyzer :
	• hadd output1.root *.root  (add root files if you have multiple files). (for crab jobs: hadd the files in your eos directory by simply including the path everytime before filename).
	• cp /eos/uscms/store/user/hichemb/RelValMinBias_13/crab_lambdaanalysis_relval11/190715_135818/0000/output1.root . To copy file from eos to local dir.
	• python makeLambdaHists.py  output1.root output2.root [remove duplicate lambdas (i.e. set their mass to -999) , this script can be merged with the initial python file).
	• python addBranchesForNNTraining.py output2.root (add pixel branches, outputs ofile2.root, only first lambda with mass>0  and keeping just important stuff) [UPDATE: now you can run this script as follow: python addBranchesForNNTraining.py output2.root output3.root rather than edit the name of the output in the file everytime].
	• python converRootToPandas.py ofile2.root (turns root file pixel info into pandas df, outputs pixelTrain.h5, can view panda file using python shell and command pandas.read_hdf("pixelTrain.h5")). [after the changes I made, run it as follow: python converRootToPandas.py output3.root output3.h5]


Jupyter Notebooks
	• Set up CERNBOX and SWAN.
	• Import Github URL: https://github.com/nhaubrich/MergedHits, now you have all the need script and files.
	• Go to SWAN bash terminal and run the following command 'pip install --user tables' .
	• Connect to your lxplus account and go to directory 'cd /eos/user/h/hboucham/' (where 'h' is your username first letter and 'hboucham' is your username, change accordingly).
	• Make a file called 'startup.sh' using your favorite text editor (Vim obviously) and include these 2 lines:
		○ #!/bin/bash
		○ export PYTHONPATH=$CERNBOX_HOME/.local/lib/python2.7/site-packages:$PYTHONPATH
	• When you start SWAN in the configuration setting copy the following in environment '$CERNBOX_HOME/startup.sh' and use Python 2 ('95a') [ very important !] and SLC7 (this option might be outdated).
	• Run 'do_merged.ipynb', if you did everything above it should run with no errors. If you get an error prompting you to import PyTables then you messed up somewhere.
	• Enjoy your plots.

