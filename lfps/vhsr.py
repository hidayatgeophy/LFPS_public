"""
LFPS V/H Spectral Ratio (VHSR) Module
========================================

Computation of Vertical-to-Horizontal Spectral Ratio for hydrocarbon
microtremor detection. Unlike traditional H/V (Nakamura method for site
effects), LFPS uses V/H where elevated V/H at 1-6 Hz indicates
potential hydrocarbon accumulation.

Includes Konno-Ohmachi spectral smoothing.

Reference:
    Saenger et al. (2009), Geophysics, 74(2), O29-O40.
"""

import numpy as np
from scipy import signal
import logging

from .psd import compute_psd_welch

logger = logging.getLogger(__name__)


def compute_vhsr(trace_z, trace_n, trace_e, config):
    """
    Compute V/H Spectral Ratio for a 3-component station.

    Parameters
    ----------
    trace_z : obspy.Trace
        Vertical component trace.
    trace_n : obspy.Trace
        North-South component trace.
    trace_e : obspy.Trace
        East-West component trace.
    config : dict
        Configuration with 'psd' and 'vhsr' sections.

    Returns
    -------
    dict
        Dictionary with keys:
        - freqs: frequency array
        - vhsr: V/H spectral ratio array
        - vhsr_smooth: smoothed V/H ratio
        - peak_freq: peak frequency in target band
        - peak_amp: peak V/H amplitude
        - psd_z, psd_n, psd_e, psd_h: component PSDs
    """
    if trace_z is None:
        raise ValueError("Vertical (Z) component is required for VHSR")
    if trace_n is None and trace_e is None:
        raise ValueError("At least one horizontal component (N or E) is required")

    psd_cfg = config.get("psd", {})
    vhsr_cfg = config.get("vhsr", {})
    band_cfg = config.get("target_band", {})

    sr = trace_z.stats.sampling_rate
    window_length = psd_cfg.get("window_length_sec", 60.0)
    overlap = psd_cfg.get("overlap_fraction", 0.5)
    nfft_mult = psd_cfg.get("nfft_multiplier", 2)

    h_method = vhsr_cfg.get("horizontal_method", "geometric_mean")
    smooth_method = vhsr_cfg.get("smoothing_method", "konno_ohmachi")
    smooth_bw = vhsr_cfg.get("smoothing_bandwidth", 40)

    fmin = band_cfg.get("fmin", 1.0)
    fmax = band_cfg.get("fmax", 6.0)

    # Compute PSD for vertical component
    freqs, psd_z = compute_psd_welch(
        trace_z.data.astype(float), sr, window_length, overlap, nfft_mult
    )

    # Compute PSD for horizontal components
    psd_n = None
    psd_e = None

    if trace_n is not None:
        _, psd_n = compute_psd_welch(
            trace_n.data.astype(float), sr, window_length, overlap, nfft_mult
        )
    if trace_e is not None:
        _, psd_e = compute_psd_welch(
            trace_e.data.astype(float), sr, window_length, overlap, nfft_mult
        )

    # Combine horizontal components
    psd_h = combine_horizontal(psd_n, psd_e, method=h_method)

    # Compute V/H ratio safely
    # Use a small epsilon to prevent division by near-zero at spectral notches
    psd_h_safe = np.maximum(psd_h, 1e-15)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        vhsr = psd_z / psd_h_safe
        vhsr[~np.isfinite(vhsr)] = 0.0
        # Cap the maximum physical V/H ratio to prevent single-bin explosions
        # from dominating the average (e.g. 100 is already a massive anomaly)
        vhsr = np.clip(vhsr, 0, 500.0)

    # Smooth the V/H ratio
    if smooth_method == "konno_ohmachi":
        vhsr_smooth = konno_ohmachi_smoothing(freqs, vhsr, bandwidth=smooth_bw)
    elif smooth_method == "parzen":
        vhsr_smooth = parzen_smoothing(vhsr, window_len=smooth_bw)
    else:
        vhsr_smooth = vhsr.copy()

    # Extract peak in target band
    mask = (freqs >= fmin) & (freqs <= fmax)
    if np.any(mask):
        subset_freqs = freqs[mask]
        subset_vhsr = vhsr_smooth[mask]
        peak_idx = np.argmax(subset_vhsr)
        peak_freq = float(subset_freqs[peak_idx])
        peak_amp = float(subset_vhsr[peak_idx])
    else:
        peak_freq = 0.0
        peak_amp = 0.0

    # Compute average V/H in target band
    if np.any(mask):
        avg_vhsr = float(np.mean(vhsr_smooth[mask]))
    else:
        avg_vhsr = 0.0

    return {
        "freqs": freqs,
        "vhsr": vhsr,
        "vhsr_smooth": vhsr_smooth,
        "peak_freq": peak_freq,
        "peak_amp": peak_amp,
        "avg_vhsr": avg_vhsr,
        "psd_z": psd_z,
        "psd_n": psd_n,
        "psd_e": psd_e,
        "psd_h": psd_h,
    }


def combine_horizontal(psd_n, psd_e, method="geometric_mean"):
    """
    Combine horizontal component PSDs.

    Parameters
    ----------
    psd_n : np.ndarray or None
        PSD of North component.
    psd_e : np.ndarray or None
        PSD of East component.
    method : str
        Combination method:
        - 'geometric_mean': sqrt(N * E)
        - 'quadratic_mean': sqrt((N² + E²) / 2)
        - 'arithmetic_mean': (N + E) / 2
        - 'maximum': max(N, E)

    Returns
    -------
    np.ndarray
        Combined horizontal PSD.
    """
    if psd_n is not None and psd_e is not None:
        if method == "geometric_mean":
            return np.sqrt(psd_n * psd_e)
        elif method == "quadratic_mean":
            return np.sqrt((psd_n ** 2 + psd_e ** 2) / 2.0)
        elif method == "arithmetic_mean":
            return (psd_n + psd_e) / 2.0
        elif method == "maximum":
            return np.maximum(psd_n, psd_e)
        else:
            return np.sqrt(psd_n * psd_e)  # default to geometric mean
    elif psd_n is not None:
        return psd_n
    elif psd_e is not None:
        return psd_e
    else:
        raise ValueError("At least one horizontal component PSD is required")


def konno_ohmachi_smoothing(freqs, spectrum, bandwidth=40):
    """
    Apply Konno-Ohmachi smoothing to a spectrum.

    The Konno-Ohmachi smoothing window is defined as:
        W(f, fc) = [sin(b * log10(f/fc)) / (b * log10(f/fc))]^4

    where b is the bandwidth parameter (typically 40).

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array (Hz).
    spectrum : np.ndarray
        Spectrum to smooth.
    bandwidth : float
        Smoothing bandwidth parameter (higher = less smoothing).

    Returns
    -------
    np.ndarray
        Smoothed spectrum.

    Reference:
        Konno, K., & Ohmachi, T. (1998). Ground-motion characteristics
        estimated from spectral ratio between horizontal and vertical
        components of microtremor. BSSA, 88(1), 228-241.
    """
    smoothed = np.zeros_like(spectrum)
    n = len(freqs)

    for i in range(n):
        fc = freqs[i]
        if fc < 1e-10:
            smoothed[i] = spectrum[i]
            continue

        # Compute Konno-Ohmachi window
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = freqs / fc
            log_ratio = bandwidth * np.log10(ratio)

            # sinc-like function: [sin(x)/x]^4
            window = np.zeros_like(log_ratio)
            nonzero = np.abs(log_ratio) > 1e-10

            window[nonzero] = (np.sin(log_ratio[nonzero]) / log_ratio[nonzero]) ** 4
            window[~nonzero] = 1.0  # limit at center (l'Hôpital)

            # Set to zero for f <= 0
            window[freqs <= 0] = 0.0

        # Apply smoothing
        weight_sum = np.sum(window)
        if weight_sum > 0:
            smoothed[i] = np.sum(spectrum * window) / weight_sum
        else:
            smoothed[i] = spectrum[i]

    return smoothed


def parzen_smoothing(spectrum, window_len=11):
    """
    Apply Parzen window smoothing.

    Parameters
    ----------
    spectrum : np.ndarray
        Spectrum to smooth.
    window_len : int
        Smoothing window length (odd number).

    Returns
    -------
    np.ndarray
        Smoothed spectrum.
    """
    if window_len < 3:
        return spectrum.copy()

    if window_len % 2 == 0:
        window_len += 1

    window = signal.windows.parzen(window_len)
    window /= window.sum()

    # Pad edges
    s = np.pad(spectrum, (window_len // 2, window_len // 2), mode="edge")
    smoothed = np.convolve(s, window, mode="valid")

    return smoothed[: len(spectrum)]


def compute_vhsr_time_windows(trace_z, trace_n, trace_e, config,
                               short_window_sec=60.0, overlap=0.5):
    """
    Compute VHSR for individual time windows to assess temporal stability.

    Parameters
    ----------
    trace_z, trace_n, trace_e : obspy.Trace
        3-component traces.
    config : dict
        Configuration dictionary.
    short_window_sec : float
        Length of each time window in seconds.
    overlap : float
        Overlap fraction between windows.

    Returns
    -------
    dict
        Dictionary with:
        - freqs: frequency array
        - vhsr_windows: list of VHSR arrays per window
        - vhsr_mean: mean VHSR across windows
        - vhsr_std: std of VHSR across windows
        - n_windows: number of valid windows
    """
    sr = trace_z.stats.sampling_rate
    nperseg = int(short_window_sec * sr)
    step = int(nperseg * (1 - overlap))
    n_samples = min(len(trace_z.data), len(trace_n.data), len(trace_e.data))

    psd_cfg = config.get("psd", {})
    inner_window = min(psd_cfg.get("window_length_sec", 60.0), short_window_sec / 2)

    vhsr_windows = []
    freqs_out = None

    start = 0
    while start + nperseg <= n_samples:
        end = start + nperseg

        # Create sub-traces for this window
        sub_z = trace_z.copy()
        sub_z.data = trace_z.data[start:end].astype(float)

        sub_n = trace_n.copy()
        sub_n.data = trace_n.data[start:end].astype(float)

        sub_e = trace_e.copy()
        sub_e.data = trace_e.data[start:end].astype(float)

        try:
            # Use shorter internal window for PSD
            temp_config = config.copy()
            temp_config["psd"] = config.get("psd", {}).copy()
            temp_config["psd"]["window_length_sec"] = inner_window

            result = compute_vhsr(sub_z, sub_n, sub_e, temp_config)
            vhsr_windows.append(result["vhsr_smooth"])
            if freqs_out is None:
                freqs_out = result["freqs"]
        except Exception as e:
            logger.warning(f"VHSR window at {start/sr:.1f}s failed: {e}")

        start += step

    if not vhsr_windows:
        raise ValueError("No valid VHSR windows computed")

    vhsr_array = np.array(vhsr_windows)

    return {
        "freqs": freqs_out,
        "vhsr_windows": vhsr_windows,
        "vhsr_mean": np.mean(vhsr_array, axis=0),
        "vhsr_std": np.std(vhsr_array, axis=0),
        "n_windows": len(vhsr_windows),
    }
