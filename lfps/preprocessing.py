"""
LFPS Preprocessing Module
===========================

Signal preprocessing for passive seismic data including:
- Quality control (spike detection, gap detection)
- Detrending and demeaning
- Tapering
- Bandpass filtering
- Time window segmentation
"""

import numpy as np
import logging
from copy import deepcopy
from obspy import Stream

logger = logging.getLogger(__name__)


def preprocess_stream(stream, config, copy=True):
    """
    Apply full preprocessing pipeline to a 3-component stream.

    Parameters
    ----------
    stream : obspy.Stream
        Raw seismic stream (3 components).
    config : dict
        Configuration dictionary with 'preprocessing' and 'filter' keys.
    copy : bool
        If True, work on a copy of the stream.

    Returns
    -------
    obspy.Stream
        Preprocessed stream.
    dict
        QC report for each trace.
    """
    if copy:
        stream = stream.copy()

    prep_cfg = config.get("preprocessing", {})
    filt_cfg = config.get("filter", {})
    qc_reports = {}

    for tr in stream:
        trace_id = f"{tr.stats.station}.{tr.stats.channel}"

        # Quality control
        qc = quality_control(tr)
        qc_reports[trace_id] = qc

        if not qc["passed"]:
            logger.warning(f"QC failed for {trace_id}: {qc['issues']}")
            continue

        # Detrend
        if prep_cfg.get("detrend", True):
            tr.detrend("linear")
            tr.detrend("demean")

        # Demean (additional)
        if prep_cfg.get("demean", True):
            tr.detrend("demean")

        # Taper
        if prep_cfg.get("taper", True):
            taper_frac = prep_cfg.get("taper_fraction", 0.05)
            tr.taper(max_percentage=taper_frac, type="cosine")

        # Bandpass filter
        freqmin = filt_cfg.get("freqmin", 0.5)
        freqmax = filt_cfg.get("freqmax", 15.0)
        corners = filt_cfg.get("corners", 4)
        zerophase = filt_cfg.get("zerophase", True)

        # Ensure freqmax doesn't exceed Nyquist
        nyquist = tr.stats.sampling_rate / 2.0
        if freqmax >= nyquist:
            freqmax = nyquist * 0.9
            logger.info(f"Adjusted freqmax to {freqmax:.1f} Hz (Nyquist: {nyquist} Hz)")

        tr.filter(
            "bandpass",
            freqmin=freqmin,
            freqmax=freqmax,
            corners=corners,
            zerophase=zerophase,
        )

    return stream, qc_reports


def quality_control(trace):
    """
    Perform quality control checks on a single trace.

    Parameters
    ----------
    trace : obspy.Trace
        Single seismic trace.

    Returns
    -------
    dict
        QC report with 'passed' flag and 'issues' list.
    """
    data = trace.data.astype(float)
    issues = []

    # Check for all-zero data
    if np.all(data == 0):
        issues.append("All-zero data")

    # Check for NaN/Inf values
    if np.any(np.isnan(data)):
        issues.append(f"Contains {np.sum(np.isnan(data))} NaN values")

    if np.any(np.isinf(data)):
        issues.append(f"Contains {np.sum(np.isinf(data))} Inf values")

    # Check for spikes (values > 10 * std)
    if len(data) > 0 and np.std(data) > 0:
        std_val = np.std(data)
        mean_val = np.mean(data)
        spike_threshold = 10 * std_val
        n_spikes = np.sum(np.abs(data - mean_val) > spike_threshold)
        spike_ratio = n_spikes / len(data)
        if spike_ratio > 0.01:  # More than 1% spikes
            issues.append(f"High spike ratio: {spike_ratio:.2%}")
    else:
        issues.append("Zero or constant amplitude")

    # Check minimum data length (at least 10 seconds)
    duration = trace.stats.npts / trace.stats.sampling_rate
    if duration < 10.0:
        issues.append(f"Very short duration: {duration:.1f}s")

    # Check for data gaps (large jumps)
    if len(data) > 1:
        diff = np.abs(np.diff(data))
        if np.std(data) > 0:
            gap_threshold = 50 * np.std(data)
            n_gaps = np.sum(diff > gap_threshold)
            if n_gaps > 5:
                issues.append(f"Detected {n_gaps} possible data gaps")

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "stats": {
            "mean": float(np.nanmean(data)),
            "std": float(np.nanstd(data)),
            "min": float(np.nanmin(data)) if len(data) > 0 else 0,
            "max": float(np.nanmax(data)) if len(data) > 0 else 0,
            "duration_sec": duration,
            "sampling_rate": trace.stats.sampling_rate,
            "npts": trace.stats.npts,
        },
    }


def remove_spikes(trace, threshold_factor=10, window_size=5):
    """
    Remove spikes from trace data using median filter approach.

    Parameters
    ----------
    trace : obspy.Trace
        Input trace (modified in-place).
    threshold_factor : float
        Spike detection threshold as multiple of local std.
    window_size : int
        Window size for local statistics.

    Returns
    -------
    int
        Number of spikes removed.
    """
    data = trace.data.astype(float)
    n_spikes = 0

    # Compute running median and std
    half_win = window_size // 2
    for i in range(len(data)):
        start = max(0, i - half_win)
        end = min(len(data), i + half_win + 1)
        local_data = np.concatenate([data[start:i], data[i+1:end]])

        if len(local_data) == 0:
            continue

        local_median = np.median(local_data)
        local_std = np.std(local_data)

        if local_std > 0 and np.abs(data[i] - local_median) > threshold_factor * local_std:
            data[i] = local_median
            n_spikes += 1

    trace.data = data
    return n_spikes


def segment_trace(trace, window_length_sec, overlap_fraction=0.5):
    """
    Segment a trace into overlapping time windows.

    Parameters
    ----------
    trace : obspy.Trace
        Input trace.
    window_length_sec : float
        Window length in seconds.
    overlap_fraction : float
        Fraction of overlap between consecutive windows (0.0 to 0.9).

    Returns
    -------
    list of np.ndarray
        List of data segments.
    list of float
        Center times of each segment (in seconds from trace start).
    """
    sr = trace.stats.sampling_rate
    nperseg = int(window_length_sec * sr)
    step = int(nperseg * (1.0 - overlap_fraction))

    if step < 1:
        step = 1

    data = trace.data.astype(float)
    n_samples = len(data)

    segments = []
    center_times = []

    start = 0
    while start + nperseg <= n_samples:
        segment = data[start: start + nperseg]
        center_sample = start + nperseg // 2
        center_time = center_sample / sr

        segments.append(segment)
        center_times.append(center_time)

        start += step

    return segments, center_times


def synchronize_streams(station_data_list):
    """
    Synchronize multiple station streams to common time window.

    Parameters
    ----------
    station_data_list : list of dict
        List of station info dicts (from match_stations)

    Returns
    -------
    tuple
        (common_starttime, common_endtime, valid_stations)
    """
    starttimes = []
    endtimes = []

    for sdata in station_data_list:
        if sdata["starttime"] and sdata["endtime"]:
            starttimes.append(sdata["starttime"])
            endtimes.append(sdata["endtime"])

    if not starttimes:
        raise ValueError("No valid station data for synchronization")

    common_start = max(starttimes)
    common_end = min(endtimes)

    if common_start >= common_end:
        raise ValueError(
            f"No overlapping time window found.\n"
            f"Latest start: {common_start}\n"
            f"Earliest end: {common_end}"
        )

    duration = common_end - common_start
    logger.info(f"Common time window: {common_start} to {common_end} ({duration:.1f}s)")

    return common_start, common_end


def trim_to_common_window(stream, starttime, endtime):
    """
    Trim all traces in stream to common time window.

    Parameters
    ----------
    stream : obspy.Stream
    starttime : obspy.UTCDateTime
    endtime : obspy.UTCDateTime

    Returns
    -------
    obspy.Stream
        Trimmed stream.
    """
    stream = stream.copy()
    stream.trim(starttime=starttime, endtime=endtime)
    return stream
