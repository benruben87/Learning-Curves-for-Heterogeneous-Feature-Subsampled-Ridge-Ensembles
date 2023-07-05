# Learning-Curves-for-Heterogeneous-Feature-Subsampled-Ridge-Ensembles

This repository houses code for the paper "Learning Curves for Heterogeneous Feature-Subsampled Ridge Ensembles" (arxiv, 2023) by Benjamin Ruben and Cengiz Pehlevan.

The folder "BlockRSCLibs" contains the custom mathematica libraries that were used for analytical simplification of the error for the equicorrelated data model.
The folder "EquiCorrDerivation" contains the mathematica files in which we performed the analyctical simplifications of the error for the equicorrelated data model, using the libraries in BlockRSCLibs

The folder "MainTextFigs" contains the following files:
"LinearRegressionExperiments.ipynb" contains the code for numerical calculations for figures 1 and 2.
"PLawCovTest_Analysis.ipynb" produces the plot for figure 1a, using output from "LinearRegressionExperiments.ipynb" and the "TheoryCurves.py" library.
"TripleGlobCorr_Analysis.ipynb" produces the plot for figure 1b, using output from "LinearRegressionExperiments.ipynb" and the "TheoryCurves.py" library.
"InterpThresh_GlobCorr_Analysis.ipynb" produces the plots for figure 2b, using output from "LinearRegressionExperiments.ipynb" and the "TheoryCurves.py" library.
"HetGlobTheoryCurves.ipynb" produces the plots for figure 3 using the "TheoryCurves.py" library.
"EquiCorrPhaseDiagrams.ipynb" produces the plots for figure 4 using the "TheoryCurves.py" library.

The folder "SupplementalFigs" contains files used for numerical experiments on CIFAR10 for figure S1
