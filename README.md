# Multi-bunch-Tunes
Analytical model for eigenvalue analysis and time domain tracking of a rigid multi-bunch model

This repository contains two files:
1. eigen.py
Used for eigenvalue analysis of the obtained new transfer which is the sum of the transfer matrix without wakes and wake matrix.

2. time_domain.py
Used for turn by turn tracking of particle positions using the same matrix as the one used in eigen.py
Prerequisites for turn by turn tracking: Harpy available at https://gitlab.cern.ch/jcoellod/harpy/-/blob/master/harmonic_analysis.py for harmonic analysis
