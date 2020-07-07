# Multi-bunch-Tunes
Analytical model for eigenvalue analysis and time domain tracking of a rigid multi-bunch model

This repository contains two code files and 1 PDF.

1. eigen.py \n
Used for eigenvalue analysis of the obtained new transfer which is the sum of the transfer matrix without wakes and wake matrix.

2. time_domain.py \n
Used for turn by turn tracking of particle positions using the same matrix as the one used in eigen.py. \n
Prerequisites for turn by turn tracking: Harpy available at https://gitlab.cern.ch/jcoellod/harpy/-/blob/master/harmonic_analysis.py for harmonic analysis. \n

3. Detuning_on_Multi_Bunch.pdf \n
Mathematical explanation of the matrix used in the codes and its eigenvalue analysis.
