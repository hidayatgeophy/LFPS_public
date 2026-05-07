"""
LFPS Configuration Module
==========================

Default parameters for all LFPS analysis stages.
All parameters are designed to be overridden via the Streamlit UI.
"""


DEFAULT_CONFIG = {

    # ── Bandpass Filter ───────────────────────────────────────────────
    "filter": {
        "freqmin": 0.5,         # Minimum frequency (Hz)
        "freqmax": 15.0,        # Maximum frequency (Hz)
        "corners": 4,           # Filter order (number of poles)
        "zerophase": True,      # Zero-phase (acausal) filtering
    },

    # ── PSD Computation ───────────────────────────────────────────────
    "psd": {
        "window_length_sec": 60.0,   # Window length in seconds
        "overlap_fraction": 0.5,     # Overlap fraction (0.0 - 0.9)
        "nfft_multiplier": 2,        # NFFT = nfft_multiplier * nperseg
        "detrend": "linear",         # Detrend method: 'linear', 'constant'
        "noise_floor_fmin": 1.0,     # Noise floor search band min (Hz)
        "noise_floor_fmax": 1.7,     # Noise floor search band max (Hz)
    },

    # ── Target Frequency Band ─────────────────────────────────────────
    "target_band": {
        "fmin": 1.0,            # Target band minimum (Hz)
        "fmax": 6.0,            # Target band maximum (Hz)
    },

    # ── VHSR (V/H Spectral Ratio) ────────────────────────────────────
    "vhsr": {
        "horizontal_method": "geometric_mean",  # 'geometric_mean', 'quadratic_mean', 'arithmetic_mean'
        "smoothing_method": "konno_ohmachi",     # 'konno_ohmachi', 'parzen', 'none'
        "smoothing_bandwidth": 40,               # Konno-Ohmachi bandwidth parameter
    },

    # ── Polarization Analysis ─────────────────────────────────────────
    "polarization": {
        "window_length_sec": 5.0,    # Short window for polarization (seconds)
        "overlap_fraction": 0.5,     # Overlap fraction
        "freq_bands": [              # Frequency bands to analyze
            (1.0, 2.0),
            (2.0, 4.0),
            (4.0, 6.0),
            (1.0, 6.0),
        ],
    },

    # ── Spatial Mapping ───────────────────────────────────────────────
    "mapping": {
        "method": "idw",             # 'idw', 'rbf', 'linear', 'cubic'
        "grid_resolution": 100,      # Number of grid points per axis
        "idw_power": 2.0,            # IDW power parameter
        "rbf_function": "multiquadric",  # RBF kernel function
        "colormap": "RdYlGn_r",     # Matplotlib colormap for HC anomaly
    },

    # ── Preprocessing ─────────────────────────────────────────────────
    "preprocessing": {
        "detrend": True,             # Apply detrending
        "demean": True,              # Remove mean
        "taper": True,               # Apply taper
        "taper_fraction": 0.05,      # Taper fraction at each end
    },
}


# Component identification patterns (last character of channel code)
COMPONENT_MAP = {
    "Z": ["Z", "3"],       # Vertical
    "N": ["N", "1", "Y"],  # North-South / Component 1
    "E": ["E", "2", "X"],  # East-West / Component 2
}

# Geobit extension-based component mapping
GEOBIT_EXT_MAP = {
    ".che": "E",    # East
    ".chn": "N",    # North
    ".chz": "Z",    # Vertical
}

# Supported file extensions
SUPPORTED_FORMATS = [
    ".mseed", ".miniseed", ".seed", ".sac", ".segy", ".sgy",
    # Geobit native extensions
    ".che", ".chn", ".chz",
]
