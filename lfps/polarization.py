"""
LFPS Polarization Analysis Module
====================================

Three-component polarization analysis for passive seismic data.
Computes polarization attributes from the covariance matrix eigenvalue
decomposition, including rectilinearity, dip angle, azimuth, and planarity.

These attributes help characterize wavefield properties above potential
hydrocarbon reservoirs, where body waves (P-waves) from reservoir-induced
microtremors show distinct polarization signatures.

Reference:
    Saenger et al. (2009), Geophysics, 74(2), O29-O40.
    Jurkevics, A. (1988). Polarization analysis of three-component
    array data. BSSA, 78(5), 1725-1743.
"""

import numpy as np
from scipy import signal as sig
import logging

logger = logging.getLogger(__name__)


def compute_covariance_matrix(z, n, e):
    """
    Compute the 3x3 covariance matrix from 3-component data.

    Parameters
    ----------
    z, n, e : np.ndarray
        Vertical, North-South, and East-West component data arrays.

    Returns
    -------
    np.ndarray
        3x3 covariance matrix. Component order: [E, N, Z]
        (geographic convention: x=E, y=N, z=Up)
    """
    # Stack components: rows = [E, N, Z]
    data_matrix = np.vstack([e, n, z])

    # Remove mean
    data_matrix = data_matrix - data_matrix.mean(axis=1, keepdims=True)

    # Covariance matrix
    cov = np.dot(data_matrix, data_matrix.T) / (data_matrix.shape[1] - 1)

    return cov


def eigenvalue_decomposition(cov_matrix):
    """
    Perform eigenvalue decomposition of the covariance matrix.

    Parameters
    ----------
    cov_matrix : np.ndarray
        3x3 covariance matrix.

    Returns
    -------
    eigenvalues : np.ndarray
        Eigenvalues sorted in descending order (λ1 ≥ λ2 ≥ λ3).
    eigenvectors : np.ndarray
        Corresponding eigenvectors as columns, sorted by eigenvalue.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort in descending order
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # Ensure eigenvalues are non-negative
    eigenvalues = np.maximum(eigenvalues, 0.0)

    return eigenvalues, eigenvectors


def extract_polarization_attributes(eigenvalues, eigenvectors):
    """
    Extract polarization attributes from eigenvalue decomposition.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Sorted eigenvalues (λ1 ≥ λ2 ≥ λ3).
    eigenvectors : np.ndarray
        Corresponding eigenvectors as columns.
        Component order: [E(x), N(y), Z(z)]

    Returns
    -------
    dict
        Polarization attributes:
        - rectilinearity: degree of linear polarization (0=circular, 1=linear)
        - planarity: degree of planar polarization
        - dip: dip angle of principal axis (degrees from horizontal)
        - azimuth: azimuth of principal axis (degrees from North, clockwise)
        - largest_eigenvalue: λ1 (total energy proxy)
        - eigenvalue_ratio_12: λ1/λ2 ratio
        - eigenvalue_ratio_13: λ1/λ3 ratio
    """
    l1, l2, l3 = eigenvalues
    v1 = eigenvectors[:, 0]  # Principal eigenvector [E, N, Z]

    # Rectilinearity (Flinn, 1965; Jurkevics, 1988)
    # R = 1 - (λ2 + λ3) / (2 * λ1)
    if l1 > 0:
        rectilinearity = 1.0 - (l2 + l3) / (2.0 * l1)
    else:
        rectilinearity = 0.0
    rectilinearity = np.clip(rectilinearity, 0.0, 1.0)

    # Planarity
    # F = 1 - 2*λ3 / (λ1 + λ2)
    if (l1 + l2) > 0:
        planarity = 1.0 - (2.0 * l3) / (l1 + l2)
    else:
        planarity = 0.0
    planarity = np.clip(planarity, 0.0, 1.0)

    # Dip angle (angle from horizontal plane)
    # v1 = [E, N, Z] = [vx, vy, vz]
    vx, vy, vz = v1
    horizontal_magnitude = np.sqrt(vx ** 2 + vy ** 2)

    if horizontal_magnitude > 0 or np.abs(vz) > 0:
        dip = np.degrees(np.arctan2(np.abs(vz), horizontal_magnitude))
    else:
        dip = 0.0

    # Azimuth (from North, clockwise)
    # atan2(E, N) gives angle from North
    azimuth = np.degrees(np.arctan2(vx, vy))
    if azimuth < 0:
        azimuth += 360.0

    # Eigenvalue ratios
    eigenvalue_ratio_12 = l1 / l2 if l2 > 0 else float("inf")
    eigenvalue_ratio_13 = l1 / l3 if l3 > 0 else float("inf")

    return {
        "rectilinearity": float(rectilinearity),
        "planarity": float(planarity),
        "dip": float(dip),
        "azimuth": float(azimuth),
        "largest_eigenvalue": float(l1),
        "eigenvalue_ratio_12": float(min(eigenvalue_ratio_12, 1e6)),
        "eigenvalue_ratio_13": float(min(eigenvalue_ratio_13, 1e6)),
    }


def polarization_analysis_windowed(trace_z, trace_n, trace_e, config):
    """
    Perform windowed polarization analysis on 3-component data.

    Computes polarization attributes for overlapping time windows,
    optionally filtered to specific frequency bands.

    Parameters
    ----------
    trace_z, trace_n, trace_e : obspy.Trace
        Three-component traces.
    config : dict
        Configuration with 'polarization' section.

    Returns
    -------
    dict
        Dictionary with keys for each frequency band, containing
        time-averaged polarization attributes.
    """
    pol_cfg = config.get("polarization", {})
    window_length = pol_cfg.get("window_length_sec", 5.0)
    overlap = pol_cfg.get("overlap_fraction", 0.5)
    freq_bands = pol_cfg.get("freq_bands", [(1.0, 6.0)])

    sr = trace_z.stats.sampling_rate
    results = {}

    for fmin, fmax in freq_bands:
        band_key = f"{fmin:.1f}-{fmax:.1f}Hz"
        logger.info(f"Polarization analysis for band {band_key}")

        # Bandpass filter for this frequency band
        z_filt = trace_z.copy()
        n_filt = trace_n.copy()
        e_filt = trace_e.copy()

        nyquist = sr / 2.0
        fmax_safe = min(fmax, nyquist * 0.9)

        for tr in [z_filt, n_filt, e_filt]:
            tr.detrend("demean")
            tr.filter("bandpass", freqmin=fmin, freqmax=fmax_safe,
                       corners=4, zerophase=True)

        # Window parameters
        nperseg = int(window_length * sr)
        step = int(nperseg * (1 - overlap))
        n_samples = min(len(z_filt.data), len(n_filt.data), len(e_filt.data))

        # Collect attributes per window
        window_attrs = []
        start = 0

        while start + nperseg <= n_samples:
            end = start + nperseg
            z_win = z_filt.data[start:end].astype(float)
            n_win = n_filt.data[start:end].astype(float)
            e_win = e_filt.data[start:end].astype(float)

            try:
                cov = compute_covariance_matrix(z_win, n_win, e_win)
                eigenvalues, eigenvectors = eigenvalue_decomposition(cov)
                attrs = extract_polarization_attributes(eigenvalues, eigenvectors)
                window_attrs.append(attrs)
            except Exception as ex:
                logger.debug(f"Window at {start/sr:.1f}s failed: {ex}")

            start += step

        if not window_attrs:
            logger.warning(f"No valid windows for band {band_key}")
            results[band_key] = None
            continue

        # Aggregate window results
        results[band_key] = _aggregate_polarization(window_attrs)

    return results


def _aggregate_polarization(window_attrs):
    """
    Aggregate polarization attributes across time windows.

    Parameters
    ----------
    window_attrs : list of dict
        List of polarization attribute dicts from each time window.

    Returns
    -------
    dict
        Aggregated attributes with mean, std, and median values.
    """
    keys = ["rectilinearity", "planarity", "dip", "largest_eigenvalue",
            "eigenvalue_ratio_12"]

    result = {"n_windows": len(window_attrs)}

    for key in keys:
        values = [w[key] for w in window_attrs if key in w and np.isfinite(w[key])]
        if values:
            result[f"{key}_mean"] = float(np.mean(values))
            result[f"{key}_std"] = float(np.std(values))
            result[f"{key}_median"] = float(np.median(values))
        else:
            result[f"{key}_mean"] = 0.0
            result[f"{key}_std"] = 0.0
            result[f"{key}_median"] = 0.0

    # Azimuth needs circular statistics
    azimuths = [w["azimuth"] for w in window_attrs if "azimuth" in w]
    if azimuths:
        az_rad = np.radians(azimuths)
        mean_sin = np.mean(np.sin(az_rad))
        mean_cos = np.mean(np.cos(az_rad))
        mean_az = np.degrees(np.arctan2(mean_sin, mean_cos))
        if mean_az < 0:
            mean_az += 360.0
        result["azimuth_mean"] = float(mean_az)

        # Circular standard deviation
        r = np.sqrt(mean_sin ** 2 + mean_cos ** 2)
        result["azimuth_std"] = float(np.degrees(np.sqrt(-2.0 * np.log(max(r, 1e-10)))))
    else:
        result["azimuth_mean"] = 0.0
        result["azimuth_std"] = 0.0

    # Raw window data for detailed analysis
    result["all_rectilinearity"] = [w["rectilinearity"] for w in window_attrs]
    result["all_dip"] = [w["dip"] for w in window_attrs]
    result["all_azimuth"] = [w["azimuth"] for w in window_attrs]
    result["all_eigenvalue"] = [w["largest_eigenvalue"] for w in window_attrs]

    return result
