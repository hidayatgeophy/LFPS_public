"""
LFPS - Low Frequency Passive Seismic Analysis Package
=====================================================

A comprehensive Python package for Low Frequency Passive Seismic (LFPS)
analysis as a Direct Hydrocarbon Indicator (DHI).

Based on: Saenger et al. (2009) - "A passive seismic survey over a gas field:
Analysis of low-frequency anomalies", Geophysics, 74(2), O29-O40.

Modules:
    - config: Default configuration parameters
    - io_utils: Data I/O (miniSEED, CSV coordinates)
    - preprocessing: Signal preprocessing (QC, filtering, windowing)
    - psd: Power Spectral Density computation
    - vhsr: Vertical-to-Horizontal Spectral Ratio
    - polarization: Polarization analysis (covariance matrix, eigenvalues)
    - mapping: Spatial interpolation and attribute mapping

Developed for: Pusat Survei Geologi, Indonesia
"""

__version__ = "1.0.0"
__author__ = "LFPS Analysis Tool"
