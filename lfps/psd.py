"""
LFPS Power Spectral Density Module
=====================================

Computation of Power Spectral Density (PSD) for passive seismic data
using Welch's method. Includes PSD-IZ (integrated spectral energy above
noise floor) for LFPS anomaly characterization.

Reference:
    Saenger et al. (2009), Geophysics, 74(2), O29-O40.
"""

import numpy as np
from scipy import signal
import logging

logger = logging.getLogger(__name__)

# NumPy 2.0 compatibility: trapz → trapezoid
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def compute_psd_welch(data, sampling_rate, window_length_sec=60.0,
                      overlap_fraction=0.5, nfft_multiplier=2,
                      detrend="linear", window_type="hann"):
    """
    Compute Power Spectral Density using Welch's method.

    Parameters
    ----------
    data : np.ndarray
        1D time series data.
    sampling_rate : float
        Sampling rate in Hz.
    window_length_sec : float
        Window length in seconds for Welch's method.
    overlap_fraction : float
        Overlap fraction between windows (0.0 to 0.9).
    nfft_multiplier : int
        NFFT = nfft_multiplier * nperseg for zero-padding.
    detrend : str
        Detrend method: 'linear', 'constant', or False.
    window_type : str
        Window function: 'hann', 'hamming', 'blackman', etc.

    Returns
    -------
    freqs : np.ndarray
        Frequency array (Hz).
    psd : np.ndarray
        Power Spectral Density array (amplitude²/Hz).
    """
    nperseg = int(window_length_sec * sampling_rate)

    # Ensure nperseg doesn't exceed data length
    if nperseg > len(data):
        nperseg = len(data)
        logger.info(f"Window length adjusted to data length: {nperseg} samples")

    noverlap = int(nperseg * overlap_fraction)
    nfft = int(nperseg * nfft_multiplier)

    freqs, psd = signal.welch(
        data,
        fs=sampling_rate,
        window=window_type,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        scaling="density",
    )

    return freqs, psd


def compute_psd_multitaper(data, sampling_rate, nw=3.5, n_tapers=None):
    """
    Compute PSD using multitaper method for more robust spectral estimation.

    Parameters
    ----------
    data : np.ndarray
        1D time series data.
    sampling_rate : float
        Sampling rate in Hz.
    nw : float
        Time-bandwidth product (default: 3.5).
    n_tapers : int, optional
        Number of tapers (default: 2*nw - 1).

    Returns
    -------
    freqs : np.ndarray
        Frequency array (Hz).
    psd : np.ndarray
        Power Spectral Density array.
    """
    n = len(data)
    if n_tapers is None:
        n_tapers = int(2 * nw - 1)

    # Generate DPSS (Slepian) tapers
    tapers, concentrations = signal.windows.dpss(
        n, nw, Kmax=n_tapers, return_ratios=True
    )

    # Compute tapered FFT for each taper
    nfft = int(2 ** np.ceil(np.log2(n)))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sampling_rate)

    psd = np.zeros(len(freqs))
    for k in range(n_tapers):
        tapered_data = data * tapers[k]
        fft_vals = np.fft.rfft(tapered_data, n=nfft)
        psd += concentrations[k] * np.abs(fft_vals) ** 2

    # Normalize
    psd = psd / (sampling_rate * np.sum(concentrations))

    return freqs, psd


# ═══════════════════════════════════════════════════════════════════════
# PSD-IZ: Saenger et al. (2009) Methodology
# ═══════════════════════════════════════════════════════════════════════

def compute_noise_floor(freqs, psd, noise_fmin=1.0, noise_fmax=1.7):
    """
    Determine the noise floor level from the PSD spectrum.

    Following Saenger et al. (2009): the noise floor is the minimum
    PSD amplitude in a sub-band (typically 1-1.7 Hz). This lies
    between the ocean-wave peak and the anthropogenic noise peak.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array (Hz).
    psd : np.ndarray
        PSD values.
    noise_fmin : float
        Lower frequency for noise floor search (default: 1.0 Hz).
    noise_fmax : float
        Upper frequency for noise floor search (default: 1.7 Hz).

    Returns
    -------
    float
        Noise floor amplitude.
    """
    mask = (freqs >= noise_fmin) & (freqs <= noise_fmax)
    if not np.any(mask):
        # Fallback: use minimum of entire PSD
        return float(np.min(psd))
    return float(np.min(psd[mask]))


def compute_psd_iz(freqs, psd, noise_floor, fmin=1.0, fmax=4.0):
    """
    Compute PSD-IZ: integrated spectral energy ABOVE the noise floor
    within the target frequency band.

    This is the key DHI metric from Saenger et al. (2009).
    The shaded area in their Figure 3.

    PSD-IZ = ∫[fmin to fmax] max(PSD(f) - noise_floor, 0) df

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array (Hz).
    psd : np.ndarray
        PSD values.
    noise_floor : float
        Noise floor level (from compute_noise_floor).
    fmin : float
        Lower frequency of target band (Hz).
    fmax : float
        Upper frequency of target band (Hz).

    Returns
    -------
    float
        PSD-IZ value (integrated energy above noise floor).
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0

    # Subtract noise floor, clip negative values to zero
    psd_above_noise = np.maximum(psd[mask] - noise_floor, 0.0)

    # Trapezoidal integration
    energy = _trapz(psd_above_noise, freqs[mask])
    return float(energy)


def compute_station_psd(trace_z, trace_n, trace_e, config):
    """
    Compute PSD for all three components of a station.
    Implements PSD-IZ methodology from Saenger et al. (2009):
    - Welch averaging for robust spectral estimation
    - Noise floor subtraction before integration

    Parameters
    ----------
    trace_z, trace_n, trace_e : obspy.Trace or None
        Three-component traces.
    config : dict
        PSD configuration parameters.

    Returns
    -------
    dict
        Dictionary with keys: 'freqs', 'psd_z', 'psd_n', 'psd_e',
        'psd_h', 'band_energy_z', 'band_energy_h', 'spectral_ratio_band',
        'noise_floor_z', 'psd_iz'
    """
    psd_cfg = config.get("psd", {})
    band_cfg = config.get("target_band", {})

    window_length = psd_cfg.get("window_length_sec", 60.0)
    overlap = psd_cfg.get("overlap_fraction", 0.5)
    nfft_mult = psd_cfg.get("nfft_multiplier", 2)
    detrend = psd_cfg.get("detrend", "linear")

    fmin = band_cfg.get("fmin", 1.0)
    fmax = band_cfg.get("fmax", 6.0)

    # Noise floor detection sub-band (Saenger: 1-1.7 Hz)
    noise_fmin = psd_cfg.get("noise_floor_fmin", 1.0)
    noise_fmax = psd_cfg.get("noise_floor_fmax", 1.7)

    result = {
        "freqs": None, "psd_z": None, "psd_n": None, "psd_e": None,
        "psd_h": None, "noise_floor_z": None, "psd_iz": None,
        "band_energy_z": 0.0, "band_energy_h": 0.0,
        "spectral_ratio_band": 0.0,
    }

    # Get sampling rate from available trace
    ref_trace = trace_z or trace_n or trace_e
    if ref_trace is None:
        raise ValueError("No valid traces provided")
    sr = ref_trace.stats.sampling_rate

    # Compute PSD for each component
    if trace_z is not None:
        freqs, psd_z = compute_psd_welch(
            trace_z.data.astype(float), sr,
            window_length, overlap, nfft_mult, detrend
        )
        result["freqs"] = freqs
        result["psd_z"] = psd_z

    if trace_n is not None:
        freqs, psd_n = compute_psd_welch(
            trace_n.data.astype(float), sr,
            window_length, overlap, nfft_mult, detrend
        )
        result["freqs"] = freqs
        result["psd_n"] = psd_n

    if trace_e is not None:
        freqs, psd_e = compute_psd_welch(
            trace_e.data.astype(float), sr,
            window_length, overlap, nfft_mult, detrend
        )
        result["freqs"] = freqs
        result["psd_e"] = psd_e

    # Horizontal PSD (geometric mean of N and E)
    if result["psd_n"] is not None and result["psd_e"] is not None:
        result["psd_h"] = np.sqrt(result["psd_n"] * result["psd_e"])
    elif result["psd_n"] is not None:
        result["psd_h"] = result["psd_n"]
    elif result["psd_e"] is not None:
        result["psd_h"] = result["psd_e"]

    # ── PSD-IZ Calculation (Saenger et al. 2009) ──────────────────────
    # Vertical component
    if result["freqs"] is not None and result["psd_z"] is not None:
        noise_floor = compute_noise_floor(
            result["freqs"], result["psd_z"], noise_fmin, noise_fmax
        )
        result["noise_floor_z"] = noise_floor
        result["psd_iz"] = compute_psd_iz(
            result["freqs"], result["psd_z"], noise_floor, fmin, fmax
        )
        result["band_energy_z"] = result["psd_iz"]

    # Horizontal component
    if result["freqs"] is not None and result["psd_h"] is not None:
        noise_floor_h = compute_noise_floor(
            result["freqs"], result["psd_h"], noise_fmin, noise_fmax
        )
        result["band_energy_h"] = compute_psd_iz(
            result["freqs"], result["psd_h"], noise_floor_h, fmin, fmax
        )

    # Spectral ratio
    if result["band_energy_z"] > 0 and result["band_energy_h"] > 0:
        result["spectral_ratio_band"] = (
            result["band_energy_z"] / result["band_energy_h"]
        )

    return result


# ═══════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════

def extract_band_energy(freqs, psd, fmin, fmax):
    """
    Extract total integrated spectral energy within a frequency band
    (simple version, without noise floor subtraction).

    Parameters
    ----------
    freqs : np.ndarray
    psd : np.ndarray
    fmin, fmax : float

    Returns
    -------
    float
        Integrated energy (total area under PSD curve).
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    return float(_trapz(psd[mask], freqs[mask]))


def extract_peak_frequency(freqs, psd, fmin=0.5, fmax=10.0):
    """
    Find the peak frequency and amplitude in a given range.

    Parameters
    ----------
    freqs : np.ndarray
    psd : np.ndarray
    fmin, fmax : float

    Returns
    -------
    tuple
        (peak_frequency, peak_amplitude)
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0, 0.0

    subset_freqs = freqs[mask]
    subset_psd = psd[mask]

    peak_idx = np.argmax(subset_psd)
    return float(subset_freqs[peak_idx]), float(subset_psd[peak_idx])


def compute_normalized_psd(psd_station, psd_reference):
    """
    Compute normalized PSD relative to a reference (background) station.

    Parameters
    ----------
    psd_station : np.ndarray
    psd_reference : np.ndarray

    Returns
    -------
    np.ndarray
        Normalized PSD (ratio).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = psd_station / psd_reference
        normalized[~np.isfinite(normalized)] = 0.0
    return normalized
