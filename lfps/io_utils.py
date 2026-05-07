"""
LFPS I/O Utilities
===================

Functions for loading miniSEED data and CSV station coordinates.
Handles matching between station IDs in filenames and coordinate files.
"""

import os
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import read, Stream, UTCDateTime

from .config import COMPONENT_MAP, SUPPORTED_FORMATS, GEOBIT_EXT_MAP

logger = logging.getLogger(__name__)


def load_mseed_directory(directory_path, file_pattern="*", progress_callback=None):
    """
    Load all miniSEED files from a directory.

    Parameters
    ----------
    directory_path : str
        Path to directory containing miniSEED files.
    file_pattern : str
        Glob pattern to filter files (default: '*' for all supported formats).
    progress_callback : callable, optional
        Function(current, total, filename) called for progress updates.

    Returns
    -------
    dict
        Dictionary of {station_id: obspy.Stream} where station_id is derived
        from the filename (without extension).
    """
    directory = Path(directory_path)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    # Collect all supported files
    all_files = []
    for ext in SUPPORTED_FORMATS:
        all_files.extend(directory.glob(f"{file_pattern}{ext}"))
        all_files.extend(directory.glob(f"{file_pattern}{ext.upper()}"))

    # Remove duplicates and sort
    all_files = sorted(set(all_files))

    if not all_files:
        raise FileNotFoundError(
            f"No supported seismic files found in: {directory_path}\n"
            f"Supported formats: {SUPPORTED_FORMATS}"
        )

    station_streams = {}
    errors = []

    for i, filepath in enumerate(all_files):
        station_id = filepath.stem  # Filename without extension
        try:
            st = read(str(filepath))
            station_streams[station_id] = st
            logger.info(f"Loaded {station_id}: {len(st)} traces")
        except Exception as e:
            errors.append((station_id, str(e)))
            logger.warning(f"Failed to load {station_id}: {e}")

        if progress_callback:
            progress_callback(i + 1, len(all_files), station_id)

    if errors:
        logger.warning(f"Failed to load {len(errors)} file(s)")

    return station_streams, errors


# ═══════════════════════════════════════════════════════════════════════
# GEOBIT HIERARCHICAL DIRECTORY SUPPORT
# Structure: Root > Day > Hour > 10min > [210 mseed files]
# Example:   E:\10Oktober\011024\011024_00\011024_0000\*.mseed
# ═══════════════════════════════════════════════════════════════════════

def scan_directory_tree(root_path):
    """
    Recursively scan a directory tree and discover all leaf directories
    containing miniSEED files (e.g., Geobit 10-minute recording folders).

    A 'leaf directory' is defined as a directory that contains mseed files
    and has no subdirectories that also contain mseed files.

    Parameters
    ----------
    root_path : str
        Root directory to scan (e.g., 'E:\\10Oktober').

    Returns
    -------
    list of dict
        List of discovered leaf directories, each with:
        - 'path': absolute path to the leaf directory
        - 'relative_path': path relative to root
        - 'n_files': number of mseed files found
        - 'folder_name': name of the leaf folder
        - 'depth': directory depth from root
    """
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root_path}")

    leaf_dirs = []

    for dirpath, dirnames, filenames in os.walk(str(root)):
        # Count mseed files in this directory
        mseed_files = [
            f for f in filenames
            if any(f.lower().endswith(ext) for ext in SUPPORTED_FORMATS)
        ]

        if mseed_files:
            rel_path = os.path.relpath(dirpath, str(root))
            depth = len(Path(rel_path).parts) if rel_path != "." else 0

            leaf_dirs.append({
                "path": os.path.abspath(dirpath),
                "relative_path": rel_path,
                "n_files": len(mseed_files),
                "folder_name": os.path.basename(dirpath),
                "depth": depth,
            })

    # Sort by path for chronological order
    leaf_dirs.sort(key=lambda x: x["path"])

    logger.info(f"Found {len(leaf_dirs)} directories with mseed files under {root_path}")
    return leaf_dirs


def load_leaf_directory(leaf_path, progress_callback=None):
    """
    Load all miniSEED files from a single leaf directory (one 10-min segment).
    Groups files by station ID extracted from the filename or mseed header.

    Each file typically represents one component (Z, N, or E) of one station.
    Files are grouped into 3-component Streams per station.

    Parameters
    ----------
    leaf_path : str
        Path to leaf directory containing mseed files.
    progress_callback : callable, optional
        Function(current, total, filename) for progress updates.

    Returns
    -------
    dict
        Dictionary of {station_id: obspy.Stream} where each Stream
        contains the 3-component traces for that station.
    list
        List of (filename, error_message) for failed files.
    """
    directory = Path(leaf_path)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {leaf_path}")

    # Collect mseed files
    all_files = []
    for ext in SUPPORTED_FORMATS:
        all_files.extend(directory.glob(f"*{ext}"))
        all_files.extend(directory.glob(f"*{ext.upper()}"))
    all_files = sorted(set(all_files))

    if not all_files:
        raise FileNotFoundError(f"No mseed files in: {leaf_path}")

    # Load all traces and group by station
    station_streams = {}
    errors = []

    for i, filepath in enumerate(all_files):
        try:
            st = read(str(filepath))
            # Determine component from file extension (Geobit: .ChE, .ChN, .ChZ)
            file_ext = filepath.suffix.lower()
            geobit_comp = GEOBIT_EXT_MAP.get(file_ext)  # 'E', 'N', or 'Z'

            for tr in st:
                # Extract station ID from filename
                sta_id = _extract_station_id(tr, filepath)

                # If Geobit format, set the channel code from extension
                # so component identification works correctly
                if geobit_comp:
                    if not tr.stats.channel or tr.stats.channel.strip() == "":
                        tr.stats.channel = f"Ch{geobit_comp}"
                    elif tr.stats.channel[-1].upper() not in ("Z", "N", "E",
                                                               "1", "2", "3"):
                        tr.stats.channel = f"Ch{geobit_comp}"

                if sta_id not in station_streams:
                    station_streams[sta_id] = Stream()
                station_streams[sta_id] += tr

        except Exception as e:
            errors.append((filepath.name, str(e)))
            logger.warning(f"Failed to load {filepath.name}: {e}")

        if progress_callback:
            progress_callback(i + 1, len(all_files), filepath.name)

    logger.info(
        f"Loaded {len(all_files)} files → {len(station_streams)} stations "
        f"from {leaf_path}"
    )
    return station_streams, errors


def _extract_station_id(trace, filepath):
    """
    Extract station ID from a trace, trying multiple strategies:
    1. Geobit filename pattern: SO01_SO_241001_000000.ChE → "SO01"
    2. ObsPy header (trace.stats.station)
    3. Generic filename parsing

    Parameters
    ----------
    trace : obspy.Trace
    filepath : Path

    Returns
    -------
    str
        Station identifier.
    """
    fname = filepath.stem  # e.g. "SO01_SO_241001_000000"
    file_ext = filepath.suffix.lower()

    # Strategy 1: Geobit naming pattern
    # Format: STATION_NETWORK_YYMMDD_HHMMSS.ChX
    # Example: SO01_SO_241001_000000.ChE
    if file_ext in GEOBIT_EXT_MAP:
        parts = fname.split("_")
        if len(parts) >= 1 and parts[0]:
            return parts[0]  # "SO01"

    # Strategy 2: Use ObsPy station header if available
    sta = trace.stats.station.strip() if trace.stats.station else ""
    if sta and sta not in ("", "None", "UNKNOWN"):
        return sta

    # Strategy 3: Generic filename parsing
    parts = fname.replace(".", "_").replace("-", "_").split("_")
    if len(parts) >= 2:
        # Try to find the station part (not a channel code)
        channel_codes = {"BHZ", "BHN", "BHE", "HHZ", "HHN", "HHE",
                         "EHZ", "EHN", "EHE", "SHZ", "SHN", "SHE",
                         "HNZ", "HNN", "HNE", "HGZ", "HGN", "HGE",
                         "DPZ", "DPN", "DPE", "Z", "N", "E"}
        station_parts = [p for p in parts if p.upper() not in channel_codes]
        if station_parts:
            return station_parts[0]

    # Fallback: use full filename stem
    return fname


def load_multiple_segments(leaf_paths, merge=True, progress_callback=None):
    """
    Load and optionally merge multiple 10-minute segments into
    continuous station streams. This is critical for LFPS analysis
    which benefits from longer time series for low-frequency resolution.

    Parameters
    ----------
    leaf_paths : list of str
        List of leaf directory paths to load (chronological order).
    merge : bool
        If True, merge traces from different segments into continuous
        streams per station using ObsPy's Stream.merge().
    progress_callback : callable, optional
        Function(current, total, segment_name) for progress.

    Returns
    -------
    dict
        Dictionary of {station_id: obspy.Stream} with merged traces.
    list
        All errors from all segments.
    int
        Total duration in minutes (number of segments × 10).
    """
    all_station_streams = {}
    all_errors = []

    for i, leaf_path in enumerate(leaf_paths):
        seg_name = Path(leaf_path).name
        if progress_callback:
            progress_callback(i + 1, len(leaf_paths), seg_name)

        try:
            seg_streams, seg_errors = load_leaf_directory(leaf_path)
            all_errors.extend(seg_errors)

            for sta_id, stream in seg_streams.items():
                if sta_id not in all_station_streams:
                    all_station_streams[sta_id] = Stream()
                all_station_streams[sta_id] += stream

        except Exception as e:
            all_errors.append((seg_name, str(e)))
            logger.warning(f"Failed to load segment {seg_name}: {e}")

    # Merge traces per station
    if merge:
        for sta_id in all_station_streams:
            try:
                all_station_streams[sta_id].merge(
                    method=1,             # use latest sample in overlap
                    fill_value="interpolate",  # fill gaps by interpolation
                    interpolation_samples=0,
                )
            except Exception as e:
                logger.warning(f"Merge failed for {sta_id}: {e}")

    total_minutes = len(leaf_paths) * 10
    logger.info(
        f"Loaded {len(leaf_paths)} segments ({total_minutes} min) → "
        f"{len(all_station_streams)} stations"
    )

    return all_station_streams, all_errors, total_minutes


def build_directory_tree_info(leaf_dirs):
    """
    Build a hierarchical summary of the directory structure for display.

    Parameters
    ----------
    leaf_dirs : list of dict
        Output from scan_directory_tree().

    Returns
    -------
    pd.DataFrame
        Summary table with folder hierarchy and file counts.
    dict
        Nested dictionary representing the tree structure.
    """
    rows = []
    tree = {}

    for ld in leaf_dirs:
        parts = Path(ld["relative_path"]).parts
        # Build nested tree
        node = tree
        for part in parts:
            if part not in node:
                node[part] = {}
            node = node[part]

        rows.append({
            "Folder": ld["folder_name"],
            "Full Path": ld["relative_path"],
            "Files": ld["n_files"],
            "Stations (est.)": ld["n_files"] // 3,
            "Depth": ld["depth"],
        })

    return pd.DataFrame(rows), tree


def load_coordinates(csv_path, station_col=None, lat_col=None, lon_col=None,
                     elev_col=None, encoding="utf-8"):
    """
    Load station coordinates from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to CSV file containing station coordinates.
    station_col : str, optional
        Column name for station ID. Auto-detected if None.
    lat_col : str, optional
        Column name for latitude. Auto-detected if None.
    lon_col : str, optional
        Column name for longitude. Auto-detected if None.
    elev_col : str, optional
        Column name for elevation. Auto-detected if None.
    encoding : str
        File encoding (default: 'utf-8').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: station_id, latitude, longitude, [elevation]
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Coordinate file not found: {csv_path}")

    # Try different encodings and separators
    df = None
    last_error = None
    for enc in [encoding, "utf-8", "latin-1", "cp1252", "utf-8-sig"]:
        for sep in [",", ";", "\t", " "]:
            try:
                _df = pd.read_csv(str(csv_path), sep=sep, encoding=enc,
                                  engine="python")
                if len(_df.columns) >= 3:
                    df = _df
                    break
            except Exception as e:
                last_error = e
                continue
        if df is not None:
            break

    if df is None:
        raise ValueError(
            f"Could not parse CSV file: {csv_path}\n"
            f"Last error: {last_error}\n"
            f"Pastikan file CSV memiliki minimal 3 kolom "
            f"(station, lon/longitude, lat/latitude) dengan separator "
            f"koma, titik koma, tab, atau spasi."
        )

    logger.info(f"CSV loaded: {len(df)} rows, columns: {list(df.columns)}")

    # Auto-detect column names if not provided
    cols_lower = {c: c.lower().strip() for c in df.columns}

    if station_col is None:
        station_col = _find_column(cols_lower, ["station", "sta", "id", "name", "station_id", "sta_id", "stasiun", "kode"])
    if lat_col is None:
        lat_col = _find_column(cols_lower, ["lat", "latitude", "lintang", "y"])
    if lon_col is None:
        lon_col = _find_column(cols_lower, ["lon", "long", "longitude", "bujur", "x"])
    if elev_col is None:
        elev_col = _find_column(cols_lower, ["elev", "elevation", "alt", "altitude", "z", "ketinggian"], required=False)

    # Build standardized DataFrame
    result = pd.DataFrame()
    result["station_id"] = df[station_col].astype(str).str.strip()
    result["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
    result["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")

    if elev_col and elev_col in df.columns:
        result["elevation"] = pd.to_numeric(df[elev_col], errors="coerce")
    else:
        result["elevation"] = 0.0

    # Drop rows with invalid coordinates
    valid_mask = result["latitude"].notna() & result["longitude"].notna()
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        logger.warning(f"Dropped {n_invalid} rows with invalid coordinates")
    result = result[valid_mask].reset_index(drop=True)

    logger.info(f"Loaded {len(result)} stations with valid coordinates")
    return result


def _find_column(cols_lower, candidates, required=True):
    """Find column name matching any candidate pattern."""
    for col, col_low in cols_lower.items():
        for candidate in candidates:
            if candidate == col_low or candidate in col_low:
                return col

    if required:
        raise ValueError(
            f"Could not auto-detect column. Looked for: {candidates}\n"
            f"Available columns: {list(cols_lower.keys())}"
        )
    return None


def match_stations(station_streams, coordinates_df):
    """
    Match station streams with coordinate data.

    Parameters
    ----------
    station_streams : dict
        Dictionary of {station_id: Stream}
    coordinates_df : pd.DataFrame
        DataFrame with 'station_id' column

    Returns
    -------
    matched : list of dict
        List of matched station info dicts
    unmatched_streams : list of str
        Station IDs with data but no coordinates
    unmatched_coords : list of str
        Station IDs with coordinates but no data
    """
    stream_ids = set(station_streams.keys())
    coord_ids = set(coordinates_df["station_id"].values)

    matched_ids = stream_ids & coord_ids
    unmatched_streams = sorted(stream_ids - coord_ids)
    unmatched_coords = sorted(coord_ids - stream_ids)

    matched = []
    for sid in sorted(matched_ids):
        row = coordinates_df[coordinates_df["station_id"] == sid].iloc[0]
        stream = station_streams[sid]
        components = identify_components(stream)

        matched.append({
            "station_id": sid,
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "elevation": row.get("elevation", 0.0),
            "stream": stream,
            "components": components,
            "n_traces": len(stream),
            "sampling_rate": stream[0].stats.sampling_rate if len(stream) > 0 else None,
            "starttime": min(tr.stats.starttime for tr in stream) if len(stream) > 0 else None,
            "endtime": max(tr.stats.endtime for tr in stream) if len(stream) > 0 else None,
        })

    return matched, unmatched_streams, unmatched_coords


def identify_components(stream):
    """
    Identify Z, N, E components from a Stream.

    Parameters
    ----------
    stream : obspy.Stream
        Stream with 3-component traces

    Returns
    -------
    dict
        Dictionary with keys 'Z', 'N', 'E' mapping to trace indices.
        Missing components are set to None.
    """
    components = {"Z": None, "N": None, "E": None}

    for i, tr in enumerate(stream):
        channel = tr.stats.channel
        if not channel:
            # Fallback: assume order Z, N, E
            comp_order = ["Z", "N", "E"]
            if i < len(comp_order):
                components[comp_order[i]] = i
            continue

        last_char = channel[-1].upper()
        for comp, patterns in COMPONENT_MAP.items():
            if last_char in patterns:
                components[comp] = i
                break

    return components


def get_three_components(stream, components):
    """
    Extract Z, N, E traces from a stream using component mapping.

    Parameters
    ----------
    stream : obspy.Stream
    components : dict
        Component mapping from identify_components()

    Returns
    -------
    tuple
        (trace_z, trace_n, trace_e) - None for missing components
    """
    trace_z = stream[components["Z"]] if components["Z"] is not None else None
    trace_n = stream[components["N"]] if components["N"] is not None else None
    trace_e = stream[components["E"]] if components["E"] is not None else None

    return trace_z, trace_n, trace_e


def get_station_summary(station_streams):
    """
    Generate a summary table of all loaded stations.

    Returns
    -------
    pd.DataFrame
        Summary with station info, sampling rate, duration, etc.
    """
    rows = []
    for sid, stream in station_streams.items():
        components = identify_components(stream)
        has_z = components["Z"] is not None
        has_n = components["N"] is not None
        has_e = components["E"] is not None

        sr = stream[0].stats.sampling_rate if len(stream) > 0 else None
        npts = stream[0].stats.npts if len(stream) > 0 else 0
        duration = npts / sr if sr else 0

        rows.append({
            "Station": sid,
            "Traces": len(stream),
            "Has Z": "✅" if has_z else "❌",
            "Has N": "✅" if has_n else "❌",
            "Has E": "✅" if has_e else "❌",
            "Sampling Rate (Hz)": sr,
            "Duration (s)": round(duration, 1),
            "Samples": npts,
            "Channels": ", ".join(tr.stats.channel for tr in stream),
        })

    return pd.DataFrame(rows)
