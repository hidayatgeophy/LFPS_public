"""
LFPS Analyzer - Low Frequency Passive Seismic Analysis
Streamlit Application for Direct Hydrocarbon Indicator Analysis

Reference: Saenger et al. (2009), Geophysics, 74(2), O29-O40
Developed for: Pusat Survei Geologi, Indonesia
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lfps.config import DEFAULT_CONFIG
from lfps.io_utils import (load_mseed_directory, load_coordinates,
                           match_stations, identify_components,
                           get_three_components, get_station_summary,
                           scan_directory_tree, load_leaf_directory,
                           load_multiple_segments, build_directory_tree_info)
from lfps.preprocessing import preprocess_stream, trim_to_common_window, synchronize_streams
from lfps.psd import compute_station_psd, extract_band_energy, extract_peak_frequency
from lfps.vhsr import compute_vhsr
from lfps.polarization import polarization_analysis_windowed
from lfps.mapping import (create_interpolation_grid, interpolate_attribute,
                          plot_attribute_map, plot_composite_map, normalize_attribute)

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(page_title="LFPS Analyzer", page_icon="🌍", layout="wide")

# ── Helper Functions ─────────────────────────────────────────────────
def _extract_attribute_values(matched, attr_name, cfg, state):
    """Extract attribute values for all matched stations."""
    values, valid_lats, valid_lons, valid_labels = [], [], [], []
    tkey = (f"{cfg['target_band']['fmin']:.1f}-"
            f"{cfg['target_band']['fmax']:.1f}Hz")

    for sdata in matched:
        sid = sdata["station_id"]
        val = None

        if attr_name == "PSD Band Energy (Z)" and state["psd_results"]:
            r = state["psd_results"].get(sid)
            val = r.get("band_energy_z") if r else None
        elif attr_name == "PSD Band Energy (H)" and state["psd_results"]:
            r = state["psd_results"].get(sid)
            val = r.get("band_energy_h") if r else None
        elif attr_name == "VHSR Peak Amplitude" and state["vhsr_results"]:
            r = state["vhsr_results"].get(sid)
            val = r.get("peak_amp") if r else None
        elif attr_name == "VHSR Peak Frequency" and state["vhsr_results"]:
            r = state["vhsr_results"].get(sid)
            val = r.get("peak_freq") if r else None
        elif attr_name == "VHSR Avg (band)" and state["vhsr_results"]:
            r = state["vhsr_results"].get(sid)
            val = r.get("avg_vhsr") if r else None
        elif state["pol_results"]:
            pr = state["pol_results"].get(sid, {})
            bd = pr.get(tkey) if pr else None
            if bd:
                pol_map = {
                    "Rectilinearity": "rectilinearity_mean",
                    "Dip": "dip_mean",
                    "Planarity": "planarity_mean",
                    "Eigenvalue Ratio": "eigenvalue_ratio_12_mean",
                }
                val = bd.get(pol_map.get(attr_name, ""), None)

        if val is not None and np.isfinite(val):
            values.append(val)
            valid_lats.append(sdata["latitude"])
            valid_lons.append(sdata["longitude"])
            valid_labels.append(sid)

    return values, valid_lats, valid_lons, valid_labels

def _show_export(labels, lats, lons, values, attr_name):
    """Show download button for results."""
    export_df = pd.DataFrame({
        "station_id": labels,
        "latitude": lats,
        "longitude": lons,
        "value": values
    })
    csv_data = export_df.to_csv(index=False)
    fname = f"lfps_{attr_name.lower().replace(' ', '_')}.csv"
    st.download_button("⬇️ Download Results CSV", csv_data, fname, "text/csv")


# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {font-size:2.2rem; font-weight:700; color:#1E88E5;
                  text-align:center; padding:0.5rem 0;}
    .sub-header {font-size:1rem; color:#666; text-align:center; margin-bottom:1.5rem;}
    .metric-card {background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                  padding:1rem; border-radius:10px; color:white; text-align:center;}
    .stTabs [data-baseweb="tab-list"] {gap:8px;}
    .stTabs [data-baseweb="tab"] {padding:8px 20px; font-weight:600;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌍 LFPS Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Low Frequency Passive Seismic — Direct Hydrocarbon Indicator</div>',
            unsafe_allow_html=True)

# ── Session State Init ───────────────────────────────────────────────
for key in ["raw_streams", "coords_df", "matched", "config",
            "preprocessed", "psd_results", "vhsr_results", "pol_results",
            "leaf_dirs", "scan_done"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "config" not in st.session_state or st.session_state["config"] is None:
    import copy
    st.session_state["config"] = copy.deepcopy(DEFAULT_CONFIG)

cfg = st.session_state["config"]

# ── Tabs ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📂 Data Loading", "🔧 Preprocessing", "📊 PSD Analysis",
    "📈 VHSR Analysis", "🔄 Polarization", "🗺️ Mapping"
])

def _sanitize_path(p):
    """Clean user-entered path: strip quotes, normalize separators."""
    if not p:
        return p
    # Remove surrounding quotes (users often paste with quotes)
    p = p.strip().strip('"').strip("'").strip()
    # Normalize path separators
    p = os.path.normpath(p)
    return p

# =====================================================================
# TAB 1: DATA LOADING
# =====================================================================
with tab1:
    st.header("📂 Data Loading")

    # ── Loading Mode Selection ────────────────────────────────────────
    load_mode = st.radio(
        "📦 Loading Mode",
        ["🔬 Geobit Recursive (Recommended)", "📁 Simple Directory"],
        horizontal=True,
        help="Geobit mode scans folder hierarchy (Bulan > Hari > Jam > 10menit)"
    )

    csv_path = _sanitize_path(st.text_input("📄 Coordinates CSV Path",
                             placeholder=r"E:\Coordinates\stations.csv"))

    # ── MODE A: Geobit Recursive ──────────────────────────────────────
    if "Geobit" in load_mode:
        st.subheader("🔍 Scan Geobit Directory Tree")
        st.caption(
            "Masukkan folder induk (misal `E:\\10Oktober`). Program akan "
            "otomatis menelusuri semua subfolder dan menemukan folder-folder "
            "berisi file mseed (folder 10-menit)."
        )
        root_dir = _sanitize_path(st.text_input("📁 Root Directory",
                                 placeholder=r"E:\10Oktober"))

        if st.button("🔍 Scan Directory", use_container_width=True):
            if root_dir:
                with st.spinner("Scanning directory tree..."):
                    try:
                        leaf_dirs = scan_directory_tree(root_dir)
                        st.session_state["leaf_dirs"] = leaf_dirs
                        st.session_state["scan_done"] = True
                        st.success(
                            f"✅ Found **{len(leaf_dirs)}** folders with mseed files"
                        )
                    except Exception as e:
                        st.error(f"❌ Scan error: {e}")

        # Show discovered folders and allow selection
        if st.session_state.get("leaf_dirs"):
            leaf_dirs = st.session_state["leaf_dirs"]
            tree_df, _ = build_directory_tree_info(leaf_dirs)
            with st.expander(f"📂 Directory Tree ({len(leaf_dirs)} folders)", expanded=True):
                st.dataframe(tree_df, use_container_width=True, height=250)

            # Selection mode
            sel_mode = st.radio(
                "Pilih segment:",
                ["Single folder (1 × 10 menit)",
                 "Multiple folders (merge jadi time series panjang)",
                 "All folders (gabung semua)"],
                horizontal=True
            )

            selected_paths = []
            if "Single" in sel_mode:
                folder_names = [ld["relative_path"] for ld in leaf_dirs]
                sel = st.selectbox("Pilih folder", folder_names)
                if sel:
                    selected_paths = [
                        ld["path"] for ld in leaf_dirs
                        if ld["relative_path"] == sel
                    ]
            elif "Multiple" in sel_mode:
                folder_names = [ld["relative_path"] for ld in leaf_dirs]
                sel_list = st.multiselect(
                    "Pilih folder-folder (urutan kronologis)",
                    folder_names, default=folder_names[:6]
                )
                selected_paths = [
                    ld["path"] for ld in leaf_dirs
                    if ld["relative_path"] in sel_list
                ]
            else:  # All
                selected_paths = [ld["path"] for ld in leaf_dirs]

            merge_opt = st.checkbox(
                "🔗 Merge segments (gabungkan menjadi trace kontinu per stasiun)",
                value=True,
                help="Penting untuk resolusi frekuensi rendah yang lebih baik"
            )

            n_seg = len(selected_paths)
            st.info(f"📊 Selected: **{n_seg}** segment(s) "
                    f"≈ **{n_seg * 10} menit** data")

            if st.button("🚀 Load Selected Segments", type="primary",
                         use_container_width=True):
                if not selected_paths:
                    st.error("Pilih minimal 1 folder.")
                elif not csv_path:
                    st.error("Masukkan path CSV koordinat.")
                else:
                    # Load mseed segments
                    with st.spinner(
                        f"Loading {len(selected_paths)} segment(s)..."
                    ):
                        try:
                            progress = st.progress(0, text="Loading...")
                            def geobit_prog(cur, total, name):
                                progress.progress(
                                    cur / total,
                                    text=f"Segment {cur}/{total}: {name}"
                                )
                            streams, errors, total_min = load_multiple_segments(
                                selected_paths, merge=merge_opt,
                                progress_callback=geobit_prog
                            )
                            st.session_state["raw_streams"] = streams
                            if errors:
                                st.warning(
                                    f"⚠️ {len(errors)} file(s) failed"
                                )
                                with st.expander("Show errors"):
                                    for fname, err in errors[:20]:
                                        st.text(f"{fname}: {err}")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")

                    # Load coordinates
                    with st.spinner("Loading coordinates..."):
                        try:
                            coords = load_coordinates(csv_path)
                            st.session_state["coords_df"] = coords
                        except Exception as e:
                            st.error(f"❌ CSV error: {e}")

                    # Match stations
                    if (st.session_state["raw_streams"]
                            and st.session_state["coords_df"] is not None):
                        matched, unm_s, unm_c = match_stations(
                            st.session_state["raw_streams"],
                            st.session_state["coords_df"]
                        )
                        st.session_state["matched"] = matched
                        st.success(
                            f"✅ Loaded **{len(matched)}** matched stations "
                            f"| {total_min} menit data"
                        )
                        if unm_s:
                            st.warning(
                                f"Stasiun tanpa koordinat: "
                                f"{', '.join(unm_s[:10])}"
                            )
                        if unm_c:
                            st.info(
                                f"Koordinat tanpa data: "
                                f"{', '.join(unm_c[:10])}"
                            )

    # ── MODE B: Simple Directory ──────────────────────────────────────
    else:
        mseed_dir = _sanitize_path(st.text_input("📁 MiniSEED Directory Path",
                                  placeholder=r"C:\Data\LFPS\mseed"))
        if st.button("🚀 Load Data", type="primary", use_container_width=True):
            if not mseed_dir or not csv_path:
                st.error("Provide both directory and CSV paths.")
            else:
                with st.spinner("Loading miniSEED files..."):
                    try:
                        progress = st.progress(0, text="Loading...")
                        def update_prog(cur, total, name):
                            progress.progress(cur/total, text=f"Loading {name}...")
                        streams, errors = load_mseed_directory(
                            mseed_dir, progress_callback=update_prog)
                        st.session_state["raw_streams"] = streams
                        if errors:
                            st.warning(f"⚠️ {len(errors)} file(s) failed")
                    except Exception as e:
                        st.error(f"❌ Error loading data: {e}")

                with st.spinner("Loading coordinates..."):
                    try:
                        coords = load_coordinates(csv_path)
                        st.session_state["coords_df"] = coords
                    except Exception as e:
                        st.error(f"❌ Error loading coordinates: {e}")

                if (st.session_state["raw_streams"]
                        and st.session_state["coords_df"] is not None):
                    matched, unm_s, unm_c = match_stations(
                        st.session_state["raw_streams"],
                        st.session_state["coords_df"])
                    st.session_state["matched"] = matched
                    st.success(f"✅ Loaded {len(matched)} matched stations")
                    if unm_s:
                        st.warning(
                            f"Stations without coordinates: "
                            f"{', '.join(unm_s[:10])}"
                        )

    # ── Display loaded data (shared for both modes) ───────────────────
    if st.session_state["raw_streams"]:
        st.subheader("📋 Station Summary")
        summary_df = get_station_summary(st.session_state["raw_streams"])
        st.dataframe(summary_df, use_container_width=True, height=300)

        if st.session_state["coords_df"] is not None:
            st.subheader("🗺️ Station Locations")
            cdf = st.session_state["coords_df"]
            fig = px.scatter_mapbox(cdf, lat="latitude", lon="longitude",
                                    hover_name="station_id", zoom=10,
                                    mapbox_style="open-street-map", height=400)
            fig.update_traces(marker=dict(size=10, color="#E53935"))
            st.plotly_chart(fig, use_container_width=True)

        # Waveform preview (downsampled for display)
        st.subheader("📉 Waveform Preview")
        sids = sorted(st.session_state["raw_streams"].keys())
        sel_sta = st.selectbox("Select station", sids, key="wf_sta")
        if sel_sta:
            stream = st.session_state["raw_streams"][sel_sta]
            fig = make_subplots(rows=len(stream), cols=1, shared_xaxes=True,
                                vertical_spacing=0.05)
            colors = ["#1E88E5", "#43A047", "#E53935"]
            MAX_POINTS = 10000  # max points per trace for display
            for i, tr in enumerate(stream):
                npts = tr.stats.npts
                t = np.arange(npts) / tr.stats.sampling_rate
                data = tr.data
                # Downsample if too many points
                if npts > MAX_POINTS:
                    step = npts // MAX_POINTS
                    t = t[::step]
                    data = data[::step]
                fig.add_trace(go.Scattergl(x=t, y=data, name=tr.stats.channel,
                              line=dict(width=0.5, color=colors[i % 3])),
                              row=i+1, col=1)
                fig.update_yaxes(title_text=tr.stats.channel, row=i+1, col=1)
            fig.update_xaxes(title_text="Time (s)", row=len(stream), col=1)
            fig.update_layout(height=400, title=f"Station: {sel_sta}",
                              showlegend=False,
                              margin=dict(l=60,r=20,t=40,b=40))
            st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# TAB 2: PREPROCESSING
# =====================================================================
with tab2:
    st.header("🔧 Preprocessing")
    if not st.session_state["matched"]:
        st.info("⬅️ Load data first in the Data Loading tab.")
    else:
        st.subheader("Filter Parameters")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            cfg["filter"]["freqmin"] = st.number_input("Min Freq (Hz)", 0.1, 20.0,
                                                        cfg["filter"]["freqmin"], 0.1)
        with c2:
            cfg["filter"]["freqmax"] = st.number_input("Max Freq (Hz)", 1.0, 50.0,
                                                        cfg["filter"]["freqmax"], 0.5)
        with c3:
            cfg["filter"]["corners"] = st.number_input("Filter Order", 1, 10,
                                                        cfg["filter"]["corners"])
        with c4:
            cfg["filter"]["zerophase"] = st.checkbox("Zero-phase", cfg["filter"]["zerophase"])

        st.subheader("Target Frequency Band (Hz)")
        tc1, tc2 = st.columns(2)
        with tc1:
            cfg["target_band"]["fmin"] = st.number_input("Target Fmin", 0.1, 10.0,
                                                          cfg["target_band"]["fmin"], 0.1)
        with tc2:
            cfg["target_band"]["fmax"] = st.number_input("Target Fmax", 1.0, 20.0,
                                                          cfg["target_band"]["fmax"], 0.5)

        if st.button("⚙️ Run Preprocessing", type="primary", use_container_width=True):
            matched = st.session_state["matched"]
            preprocessed = {}
            progress = st.progress(0, text="Preprocessing...")
            qc_all = {}
            for i, sdata in enumerate(matched):
                sid = sdata["station_id"]
                progress.progress((i+1)/len(matched), text=f"Processing {sid}...")
                try:
                    pp_stream, qc = preprocess_stream(sdata["stream"], cfg)
                    preprocessed[sid] = {"stream": pp_stream,
                                         "components": identify_components(pp_stream)}
                    qc_all[sid] = qc
                except Exception as e:
                    st.warning(f"⚠️ {sid}: {e}")
            st.session_state["preprocessed"] = preprocessed
            st.success(f"✅ Preprocessed {len(preprocessed)} stations")

            # Show QC summary
            qc_rows = []
            for sid, qc in qc_all.items():
                for trid, report in qc.items():
                    qc_rows.append({"Station": sid, "Trace": trid,
                                    "Passed": "✅" if report["passed"] else "❌",
                                    "Issues": "; ".join(report["issues"]) if report["issues"] else "—"})
            if qc_rows:
                with st.expander("📋 QC Report"):
                    st.dataframe(pd.DataFrame(qc_rows), use_container_width=True)

# =====================================================================
# TAB 3: PSD ANALYSIS
# =====================================================================
with tab3:
    st.header("📊 PSD Analysis")
    if not st.session_state["preprocessed"]:
        st.info("⬅️ Run preprocessing first.")
    else:
        st.subheader("PSD Parameters")
        p1, p2 = st.columns(2)
        with p1:
            cfg["psd"]["window_length_sec"] = st.number_input(
                "Window Length (s)", 5.0, 300.0,
                cfg["psd"]["window_length_sec"], 5.0)
        with p2:
            cfg["psd"]["overlap_fraction"] = st.slider(
                "Overlap", 0.0, 0.9, cfg["psd"]["overlap_fraction"], 0.1)

        st.subheader("🔧 Noise Floor Detection (Saenger et al.)")
        st.caption(
            "Noise floor = PSD minimum dalam sub-band ini. "
            "PSD-IZ = area DI ATAS noise floor dalam target band."
        )
        n1, n2 = st.columns(2)
        with n1:
            cfg["psd"]["noise_floor_fmin"] = st.number_input(
                "Noise Floor Fmin (Hz)", 0.5, 5.0,
                cfg["psd"].get("noise_floor_fmin", 1.0), 0.1)
        with n2:
            cfg["psd"]["noise_floor_fmax"] = st.number_input(
                "Noise Floor Fmax (Hz)", 1.0, 5.0,
                cfg["psd"].get("noise_floor_fmax", 1.7), 0.1)

        if st.button("📊 Compute PSD", type="primary", use_container_width=True):
            preprocessed = st.session_state["preprocessed"]
            psd_results = {}
            progress = st.progress(0)
            for i, (sid, pdata) in enumerate(preprocessed.items()):
                progress.progress((i+1)/len(preprocessed), text=f"PSD: {sid}")
                try:
                    stream = pdata["stream"]
                    comps = pdata["components"]
                    tr_z = stream[comps["Z"]] if comps["Z"] is not None else None
                    tr_n = stream[comps["N"]] if comps["N"] is not None else None
                    tr_e = stream[comps["E"]] if comps["E"] is not None else None
                    psd_results[sid] = compute_station_psd(tr_z, tr_n, tr_e, cfg)
                except Exception as e:
                    st.warning(f"⚠️ {sid}: {e}")
            st.session_state["psd_results"] = psd_results
            st.success(f"✅ PSD computed for {len(psd_results)} stations")

        if st.session_state["psd_results"]:
            psd_results = st.session_state["psd_results"]
            sids = sorted(psd_results.keys())

            # PSD-IZ curves (linear scale, like Saenger Figure 3)
            st.subheader("📉 PSD-IZ Curves (Saenger Method)")
            sel_stas = st.multiselect("Select stations to plot", sids,
                                       default=sids[:3], key="psd_sel")
            if sel_stas:
                fig = go.Figure()
                fmin_t = cfg["target_band"]["fmin"]
                fmax_t = cfg["target_band"]["fmax"]

                for sid in sel_stas:
                    r = psd_results[sid]
                    if r["freqs"] is not None and r["psd_z"] is not None:
                        freqs = r["freqs"]
                        psd_z = r["psd_z"]
                        nf = r.get("noise_floor_z", 0)

                        # Show PSD curve (linear scale, like paper)
                        freq_mask = freqs <= cfg["filter"]["freqmax"]
                        fig.add_trace(go.Scatter(
                            x=freqs[freq_mask], y=psd_z[freq_mask],
                            name=f"{sid} (PSD)", mode="lines"))

                        # Show noise floor line for first station
                        if sid == sel_stas[0] and nf > 0:
                            fig.add_hline(
                                y=nf, line_dash="dash", line_color="gray",
                                annotation_text=f"Noise Floor = {nf:.2e}")

                        # Shaded area (PSD-IZ) for first station
                        if sid == sel_stas[0]:
                            band_mask = (freqs >= fmin_t) & (freqs <= fmax_t)
                            if np.any(band_mask):
                                psd_above = np.maximum(psd_z[band_mask] - nf, 0)
                                fig.add_trace(go.Scatter(
                                    x=freqs[band_mask],
                                    y=psd_z[band_mask],
                                    fill="tonexty" if nf > 0 else None,
                                    mode="lines", line=dict(width=0),
                                    showlegend=False))
                                if nf > 0:
                                    fig.add_trace(go.Scatter(
                                        x=freqs[band_mask],
                                        y=np.full(band_mask.sum(), nf),
                                        fill=None, mode="lines",
                                        line=dict(width=0),
                                        showlegend=False))

                # Target band highlight
                fig.add_vrect(x0=fmin_t, x1=fmax_t,
                              fillcolor="red", opacity=0.08,
                              annotation_text="Target Band")
                fig.update_layout(
                    title="PSD-IZ: Vertical Component (Linear Scale)",
                    xaxis_title="Frequency (Hz)",
                    yaxis_title="PSD (arbitrary units)",
                    height=500,
                    xaxis=dict(range=[0.5, cfg["filter"]["freqmax"]]))
                st.plotly_chart(fig, use_container_width=True)

            # PSD-IZ Summary Table with QC
            st.subheader("⚡ PSD-IZ Summary (Saenger Method)")

            # QC thresholds
            VH_RATIO_MAX = 10.0  # V/H ratio > this → suspect sensor
            PSD_IZ_H_MIN = 1e-6  # PSD-IZ(H) < this → dead horizontal

            energy_rows = []
            flagged_stations = []
            for sid in sids:
                r = psd_results[sid]
                psd_iz_v = r.get("psd_iz", 0) or 0
                psd_iz_h = r.get("band_energy_h", 0) or 0
                vh_ratio = r.get("spectral_ratio_band", 0) or 0
                nf = r.get("noise_floor_z", 0) or 0

                # QC checks
                issues = []
                if psd_iz_h < PSD_IZ_H_MIN:
                    issues.append("🔴 Horizontal sensor mati/rusak")
                if vh_ratio > VH_RATIO_MAX:
                    issues.append(f"🟠 V/H={vh_ratio:.1f} (>10, abnormal)")
                if psd_iz_v == 0:
                    issues.append("🔴 Tidak ada energi vertikal")

                if issues:
                    flagged_stations.append(sid)

                energy_rows.append({
                    "Station": sid,
                    "PSD-IZ (V)": f"{psd_iz_v:.4f}",
                    "PSD-IZ (H)": f"{psd_iz_h:.4f}",
                    "Noise Floor": f"{nf:.2e}",
                    "V/H Ratio": f"{vh_ratio:.3f}" if vh_ratio < 1e6 else "∞",
                    "QC": "✅" if not issues else " | ".join(issues),
                })

            edf = pd.DataFrame(energy_rows)
            st.dataframe(edf, use_container_width=True)

            # QC Warning & Manual Exclude
            if flagged_stations:
                st.warning(
                    f"⚠️ **{len(flagged_stations)} stasiun** terdeteksi anomali: "
                    f"{', '.join(flagged_stations)}\n\n"
                    f"**Penyebab umum:** sensor horizontal mati/rusak, kabel putus."
                )

            st.markdown("#### 🚫 Manajemen Stasiun Outlier")
            st.caption("Pilih stasiun yang ingin diabaikan pada proses VHSR, Polarisasi, dan Mapping. Stasiun anomali otomatis terpilih, namun Anda bisa mengubahnya.")
            
            excluded_stas = st.multiselect(
                "Eksklusi Stasiun:",
                options=sids,
                default=flagged_stations,
                help="Stasiun yang masuk daftar ini tidak akan diikutkan di analisis selanjutnya."
            )

            # Update session state flags
            for s in sids:
                if s in st.session_state["psd_results"]:
                    st.session_state["psd_results"][s]["excluded"] = (s in excluded_stas)
                    
            if excluded_stas:
                st.info(f"✅ {len(excluded_stas)} stasiun akan dilewati.")


# =====================================================================
# TAB 4: VHSR ANALYSIS
# =====================================================================
with tab4:
    st.header("📈 VHSR Analysis (V/H Spectral Ratio)")
    if not st.session_state["preprocessed"]:
        st.info("⬅️ Run preprocessing first.")
    else:
        st.subheader("VHSR Parameters")
        v1, v2, v3 = st.columns(3)
        with v1:
            cfg["vhsr"]["horizontal_method"] = st.selectbox(
                "H Combination", ["geometric_mean", "quadratic_mean", "arithmetic_mean"],
                index=0)
        with v2:
            cfg["vhsr"]["smoothing_method"] = st.selectbox(
                "Smoothing", ["konno_ohmachi", "parzen", "none"], index=0)
        with v3:
            cfg["vhsr"]["smoothing_bandwidth"] = st.number_input(
                "Bandwidth", 10, 100, cfg["vhsr"]["smoothing_bandwidth"], 5)

        if st.button("📈 Compute VHSR", type="primary", use_container_width=True):
            preprocessed = st.session_state["preprocessed"]
            vhsr_results = {}
            progress = st.progress(0)
            for i, (sid, pdata) in enumerate(preprocessed.items()):
                progress.progress((i+1)/len(preprocessed), text=f"VHSR: {sid}")
                
                # Check if excluded in PSD step
                psd_res = st.session_state.get("psd_results") or {}
                if psd_res.get(sid, {}).get("excluded", False):
                    continue

                try:
                    stream = pdata["stream"]
                    comps = pdata["components"]
                    tr_z = stream[comps["Z"]] if comps["Z"] is not None else None
                    tr_n = stream[comps["N"]] if comps["N"] is not None else None
                    tr_e = stream[comps["E"]] if comps["E"] is not None else None
                    vhsr_results[sid] = compute_vhsr(tr_z, tr_n, tr_e, cfg)
                except Exception as e:
                    st.warning(f"⚠️ {sid}: {e}")
            st.session_state["vhsr_results"] = vhsr_results
            st.success(f"✅ VHSR computed for {len(vhsr_results)} stations")

        if st.session_state["vhsr_results"]:
            vhsr_results = st.session_state["vhsr_results"]
            sids = sorted(vhsr_results.keys())

            st.subheader("📉 V/H Spectral Ratio Curves")
            sel = st.multiselect("Select stations", sids, default=sids[:5], key="vhsr_sel")
            if sel:
                fig = go.Figure()
                for sid in sel:
                    r = vhsr_results[sid]
                    fig.add_trace(go.Scatter(x=r["freqs"], y=r["vhsr_smooth"],
                                             name=sid, mode="lines"))
                fmin, fmax = cfg["target_band"]["fmin"], cfg["target_band"]["fmax"]
                fig.add_vrect(x0=fmin, x1=fmax, fillcolor="red", opacity=0.1,
                              annotation_text="Target Band")
                fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                              annotation_text="V/H = 1")
                fig.update_layout(title="V/H Spectral Ratio (Smoothed)",
                                  xaxis_title="Frequency (Hz)",
                                  yaxis_title="V/H Ratio", height=450,
                                  xaxis=dict(range=[0, cfg["filter"]["freqmax"]]))
                st.plotly_chart(fig, use_container_width=True)

            # Summary table
            st.subheader("📋 VHSR Summary")
            rows = []
            for sid in sids:
                r = vhsr_results[sid]
                rows.append({"Station": sid, "Peak Freq (Hz)": round(r["peak_freq"], 2),
                              "Peak V/H": round(r["peak_amp"], 3),
                              "Avg V/H (band)": round(r["avg_vhsr"], 3)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

# =====================================================================
# TAB 5: POLARIZATION
# =====================================================================
with tab5:
    st.header("🔄 Polarization Analysis")
    if not st.session_state["preprocessed"]:
        st.info("⬅️ Run preprocessing first.")
    else:
        st.subheader("Polarization Parameters")
        pc1, pc2 = st.columns(2)
        with pc1:
            cfg["polarization"]["window_length_sec"] = st.number_input(
                "Window (s)", 1.0, 60.0, cfg["polarization"]["window_length_sec"], 1.0)
        with pc2:
            cfg["polarization"]["overlap_fraction"] = st.slider(
                "Overlap", 0.0, 0.9, cfg["polarization"]["overlap_fraction"], 0.1, key="pol_ov")

        if st.button("🔄 Compute Polarization", type="primary", use_container_width=True):
            # Ensure current target band is always computed
            t_fmin = cfg["target_band"]["fmin"]
            t_fmax = cfg["target_band"]["fmax"]
            t_band = (t_fmin, t_fmax)
            
            if "freq_bands" not in cfg["polarization"]:
                cfg["polarization"]["freq_bands"] = []
            if t_band not in cfg["polarization"]["freq_bands"]:
                cfg["polarization"]["freq_bands"].insert(0, t_band)

            preprocessed = st.session_state["preprocessed"]
            pol_results = {}
            progress = st.progress(0)
            for i, (sid, pdata) in enumerate(preprocessed.items()):
                progress.progress((i+1)/len(preprocessed), text=f"Polarization: {sid}")
                
                # Check if excluded in PSD step
                psd_res = st.session_state.get("psd_results") or {}
                if psd_res.get(sid, {}).get("excluded", False):
                    continue

                try:
                    stream = pdata["stream"]
                    comps = pdata["components"]
                    tr_z = stream[comps["Z"]] if comps["Z"] is not None else None
                    tr_n = stream[comps["N"]] if comps["N"] is not None else None
                    tr_e = stream[comps["E"]] if comps["E"] is not None else None
                    if tr_z and tr_n and tr_e:
                        pol_results[sid] = polarization_analysis_windowed(
                            tr_z, tr_n, tr_e, cfg)
                except Exception as e:
                    st.warning(f"⚠️ {sid}: {e}")
            st.session_state["pol_results"] = pol_results
            st.success(f"✅ Polarization computed for {len(pol_results)} stations")

        if st.session_state["pol_results"]:
            pol_results = st.session_state["pol_results"]
            # Build summary for the full target band
            target_key = f"{cfg['target_band']['fmin']:.1f}-{cfg['target_band']['fmax']:.1f}Hz"
            rows = []
            for sid in sorted(pol_results.keys()):
                bands = pol_results[sid]
                for bkey, bdata in bands.items():
                    if bdata is None:
                        continue
                    rows.append({
                        "Station": sid, "Band": bkey,
                        "Rectilinearity": round(bdata["rectilinearity_mean"], 3),
                        "Dip (°)": round(bdata["dip_mean"], 1),
                        "Azimuth (°)": round(bdata["azimuth_mean"], 1),
                        "Eigenvalue": f"{bdata['largest_eigenvalue_mean']:.2e}",
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, height=400)

# =====================================================================
# TAB 6: MAPPING
# =====================================================================
with tab6:
    st.header("🗺️ Attribute Mapping")
    matched = st.session_state["matched"]
    if not matched:
        st.info("⬅️ Load and process data first.")
    else:
        # ── Interpolation Settings ────────────────────────────────────
        st.subheader("⚙️ Interpolation Settings")
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            cfg["mapping"]["method"] = st.selectbox(
                "Method", ["idw", "rbf", "linear", "cubic"], index=0)
        with mc2:
            cfg["mapping"]["grid_resolution"] = st.number_input(
                "Grid Resolution", 20, 500,
                cfg["mapping"]["grid_resolution"], 10)
        with mc3:
            cfg["mapping"]["idw_power"] = st.number_input(
                "IDW Power", 0.5, 5.0, cfg["mapping"]["idw_power"], 0.5)

        cmap_choice = st.selectbox("Colormap",
            ["RdYlGn_r", "hot", "inferno", "viridis", "magma",
             "coolwarm", "jet", "Spectral_r"])

        # Collect station coords
        lats = np.array([s["latitude"] for s in matched])
        lons = np.array([s["longitude"] for s in matched])
        sids_matched = [s["station_id"] for s in matched]

        # ── Mapping Mode ──────────────────────────────────────────────
        map_mode = st.radio(
            "🗺️ Mapping Mode",
            ["📊 Individual Attribute Map",
             "🎯 Composite LFPS Score (Weighted DHI)"],
            horizontal=True
        )

        # ==============================================================
        # MODE A: Individual Attribute Map
        # ==============================================================
        if "Individual" in map_mode:
            # Determine available attributes
            available = []
            if st.session_state["psd_results"]:
                available += ["PSD Band Energy (Z)", "PSD Band Energy (H)"]
            if st.session_state["vhsr_results"]:
                available += ["VHSR Peak Amplitude", "VHSR Peak Frequency",
                              "VHSR Avg (band)"]
            if st.session_state["pol_results"]:
                available += ["Rectilinearity", "Dip", "Planarity",
                              "Eigenvalue Ratio"]

            if not available:
                st.info("Run PSD, VHSR, or Polarization analysis first.")
            else:
                sel_attr = st.selectbox("Select Attribute to Map", available)

                if st.button("🗺️ Generate Map", type="primary",
                             use_container_width=True):
                    values, valid_lats, valid_lons, valid_labels = \
                        _extract_attribute_values(
                            matched, sel_attr, cfg, st.session_state)

                    if len(values) < 3:
                        st.error("Need at least 3 stations with valid values.")
                    else:
                        values = np.array(values)
                        valid_lats = np.array(valid_lats)
                        valid_lons = np.array(valid_lons)

                        with st.spinner("Interpolating..."):
                            grid_lon2, grid_lat2 = create_interpolation_grid(
                                valid_lats, valid_lons,
                                cfg["mapping"]["grid_resolution"])
                            grid_vals = interpolate_attribute(
                                valid_lons, valid_lats, values,
                                grid_lon2, grid_lat2, cfg)

                        fig = plot_attribute_map(
                            grid_lon2, grid_lat2, grid_vals,
                            valid_lons, valid_lats, values,
                            title=f"LFPS: {sel_attr}",
                            colorbar_label=sel_attr,
                            cmap=cmap_choice,
                            station_labels=valid_labels)
                        st.pyplot(fig, use_container_width=True)

                        # Export
                        _show_export(valid_labels, valid_lats, valid_lons,
                                     values, sel_attr)

        # ==============================================================
        # MODE B: Composite LFPS Score (Weighted DHI)
        # ==============================================================
        else:
            st.subheader("🎯 Composite LFPS Score")
            st.caption(
                "Gabungkan beberapa atribut menjadi satu skor DHI (0–1) "
                "dengan bobot yang dapat diatur. Setiap atribut dinormalisasi "
                "menggunakan min-max scaling sebelum pembobotan."
            )

            # Check available data
            has_psd = st.session_state["psd_results"] is not None
            has_vhsr = st.session_state["vhsr_results"] is not None
            has_pol = st.session_state["pol_results"] is not None

            if not (has_psd or has_vhsr or has_pol):
                st.warning("⚠️ Jalankan minimal satu analisis "
                           "(PSD/VHSR/Polarization) terlebih dahulu.")
            else:
                st.markdown("#### 📐 Weight Configuration")
                st.caption(
                    "Atur bobot (0 = tidak digunakan). "
                    "Bobot akan dinormalisasi otomatis agar total = 1.0"
                )

                # Weight sliders for each attribute
                w_col1, w_col2 = st.columns(2)
                weights = {}

                with w_col1:
                    st.markdown("**Spectral Attributes**")
                    if has_psd:
                        weights["psd_energy_z"] = st.slider(
                            "PSD Band Energy (Z)", 0.0, 1.0, 0.2, 0.05,
                            help="Energi spektral vertikal di target band")
                    if has_vhsr:
                        weights["vhsr_peak_amp"] = st.slider(
                            "VHSR Peak Amplitude", 0.0, 1.0, 0.25, 0.05,
                            help="Amplitudo puncak V/H ratio")
                        weights["vhsr_avg"] = st.slider(
                            "VHSR Avg (band)", 0.0, 1.0, 0.15, 0.05,
                            help="Rata-rata V/H di target band")

                with w_col2:
                    st.markdown("**Polarization Attributes**")
                    if has_pol:
                        weights["rectilinearity"] = st.slider(
                            "Rectilinearity", 0.0, 1.0, 0.15, 0.05,
                            help="Tinggi → gelombang linear (P-wave)")
                        weights["dip"] = st.slider(
                            "Dip Angle", 0.0, 1.0, 0.15, 0.05,
                            help="Tinggi → gerak partikel vertikal")
                        weights["planarity_inv"] = st.slider(
                            "Planarity (inverse)", 0.0, 1.0, 0.05, 0.05,
                            help="Rendah planarity → bukan surface wave")
                        weights["eigenvalue_ratio"] = st.slider(
                            "Eigenvalue Ratio", 0.0, 1.0, 0.05, 0.05,
                            help="Tinggi → satu arah dominan")

                # Show normalized weights
                total_w = sum(weights.values())
                if total_w > 0:
                    norm_weights = {k: v/total_w for k, v in weights.items()}
                    wdf = pd.DataFrame([
                        {"Attribute": k.replace("_", " ").title(),
                         "Raw Weight": f"{v:.2f}",
                         "Normalized": f"{norm_weights[k]:.1%}"}
                        for k, v in weights.items() if v > 0
                    ])
                    st.dataframe(wdf, use_container_width=True, hide_index=True)

                if st.button("🎯 Generate Composite DHI Map", type="primary",
                             use_container_width=True):
                    if total_w == 0:
                        st.error("Set at least one weight > 0")
                    else:
                        tkey = (f"{cfg['target_band']['fmin']:.1f}-"
                                f"{cfg['target_band']['fmax']:.1f}Hz")
                        norm_weights = {k: v/total_w
                                        for k, v in weights.items()}

                        # Collect all attribute values per station
                        composite_data = {}
                        for sdata in matched:
                            sid = sdata["station_id"]
                            attrs = {}
                            valid = True

                            # PSD
                            if "psd_energy_z" in norm_weights \
                                    and norm_weights["psd_energy_z"] > 0:
                                r = (st.session_state["psd_results"] or {}) \
                                    .get(sid)
                                if r and r.get("band_energy_z") is not None:
                                    attrs["psd_energy_z"] = r["band_energy_z"]
                                else:
                                    valid = False

                            # VHSR
                            if "vhsr_peak_amp" in norm_weights \
                                    and norm_weights["vhsr_peak_amp"] > 0:
                                r = (st.session_state["vhsr_results"] or {}) \
                                    .get(sid)
                                if r and r.get("peak_amp") is not None:
                                    attrs["vhsr_peak_amp"] = r["peak_amp"]
                                else:
                                    valid = False

                            if "vhsr_avg" in norm_weights \
                                    and norm_weights["vhsr_avg"] > 0:
                                r = (st.session_state["vhsr_results"] or {}) \
                                    .get(sid)
                                if r and r.get("avg_vhsr") is not None:
                                    attrs["vhsr_avg"] = r["avg_vhsr"]
                                else:
                                    valid = False

                            # Polarization
                            if has_pol:
                                pr = (st.session_state["pol_results"] or {}) \
                                    .get(sid, {})
                                bd = pr.get(tkey) if pr else None

                                for pol_key in ["rectilinearity", "dip",
                                                "planarity_inv",
                                                "eigenvalue_ratio"]:
                                    if pol_key in norm_weights \
                                            and norm_weights[pol_key] > 0:
                                        if bd:
                                            src_key = {
                                                "rectilinearity":
                                                    "rectilinearity_mean",
                                                "dip": "dip_mean",
                                                "planarity_inv":
                                                    "planarity_mean",
                                                "eigenvalue_ratio":
                                                    "eigenvalue_ratio_12_mean",
                                            }[pol_key]
                                            val = bd.get(src_key)
                                            if val is not None:
                                                attrs[pol_key] = val
                                            else:
                                                valid = False
                                        else:
                                            valid = False

                            if valid and attrs:
                                composite_data[sid] = attrs

                        if len(composite_data) < 3:
                            st.error(
                                f"Only {len(composite_data)} stations have "
                                f"all required attributes. Need ≥ 3."
                            )
                        else:
                            # Normalize each attribute (min-max → 0 to 1)
                            attr_keys = list(
                                next(iter(composite_data.values())).keys()
                            )
                            attr_arrays = {
                                k: np.array([
                                    composite_data[s][k]
                                    for s in composite_data
                                ])
                                for k in attr_keys
                            }

                            normalized = {}
                            for k, arr in attr_arrays.items():
                                normed = normalize_attribute(arr, "minmax")
                                # Invert planarity (low = good for DHI)
                                if k == "planarity_inv":
                                    normed = 1.0 - normed
                                normalized[k] = normed

                            # Compute weighted composite score
                            scores = np.zeros(len(composite_data))
                            for k in attr_keys:
                                w = norm_weights.get(k, 0)
                                scores += w * normalized[k]

                            # Get coords for valid stations
                            valid_sids = list(composite_data.keys())
                            comp_lats, comp_lons = [], []
                            for sdata in matched:
                                if sdata["station_id"] in valid_sids:
                                    comp_lats.append(sdata["latitude"])
                                    comp_lons.append(sdata["longitude"])
                            comp_lats = np.array(comp_lats)
                            comp_lons = np.array(comp_lons)

                            # Interpolate composite score
                            with st.spinner("Interpolating composite..."):
                                grid_lon2, grid_lat2 = \
                                    create_interpolation_grid(
                                        comp_lats, comp_lons,
                                        cfg["mapping"]["grid_resolution"])
                                grid_composite = interpolate_attribute(
                                    comp_lons, comp_lats, scores,
                                    grid_lon2, grid_lat2, cfg)

                            # Plot composite map
                            fig = plot_attribute_map(
                                grid_lon2, grid_lat2, grid_composite,
                                comp_lons, comp_lats, scores,
                                title="LFPS Composite DHI Score",
                                colorbar_label="DHI Score (0=Low, 1=High)",
                                cmap=cmap_choice,
                                station_labels=valid_sids,
                                vmin=0, vmax=1)
                            st.pyplot(fig, use_container_width=True)

                            # Show individual + composite maps side by side
                            with st.expander("📊 Individual Attribute Maps"):
                                attr_grids = {}
                                for k in attr_keys:
                                    vals = normalized[k]
                                    g = interpolate_attribute(
                                        comp_lons, comp_lats, vals,
                                        grid_lon2, grid_lat2, cfg)
                                    label = k.replace("_", " ").title()
                                    if k == "planarity_inv":
                                        label = "1 - Planarity"
                                    attr_grids[label] = g

                                fig2 = plot_composite_map(
                                    grid_lon2, grid_lat2, attr_grids,
                                    comp_lons, comp_lats,
                                    list(attr_grids.keys()),
                                    station_labels=valid_sids)
                                st.pyplot(fig2, use_container_width=True)

                            # Score table
                            st.subheader("📋 Station Scores")
                            score_rows = []
                            for i, sid in enumerate(valid_sids):
                                row = {"Station": sid,
                                       "DHI Score": round(scores[i], 3)}
                                for k in attr_keys:
                                    label = k.replace("_", " ").title()
                                    row[f"{label} (norm)"] = round(
                                        normalized[k][i], 3)
                                score_rows.append(row)
                            score_df = pd.DataFrame(score_rows).sort_values(
                                "DHI Score", ascending=False)
                            st.dataframe(score_df, use_container_width=True)

                            # Export
                            _show_export(valid_sids, comp_lats, comp_lons,
                                         scores, "composite_dhi")


# ── Footer ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
    "LFPS Analyzer v1.0 | Based on Saenger et al. (2009) | "
    "Hidayat</div>",
    unsafe_allow_html=True)
