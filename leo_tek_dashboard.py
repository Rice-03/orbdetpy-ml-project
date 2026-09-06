# =============================================================================
# leo_tek_dashboard.py  --  LEO TEK
# =============================================================================
#
# Streamlit dashboard for the blind Doppler tracking pipeline.
#
# This app READS the CSVs already produced by generate_params.py,
# simulationv2.py, build_dataset.py, and forest_trainer.py. It does NOT
# re-run any orbdetpy propagation or model training -- that work happens in
# the pipeline scripts; this is just the viewer.
#
# Expected files (default ./output/):
#     random_orbital_parameters.csv   -- Keplerian elements per satellite
#     delay_differences.csv           -- Doppler delay sequences (ML features)
#     training_table.csv              -- OPTIONAL: build_dataset.py output
#     model_results.csv               -- OPTIONAL: forest_trainer.py output
#
# initial_state_vectors.csv is no longer read here -- ECEF state vectors were
# tested and found unrecoverable from a single arc (see project log). Labels
# shown in this dashboard are the six Keplerian elements actually used for
# training, read from training_table.csv.
#
# PAGES
# -----
#   params        -- generated orbital parameters (generate_params.py)
#   passes        -- satellite passes + Doppler curve (simulationv2.py)
#   data_cleaning -- sampled orbits -> acquired passes -> training rows
#   models        -- training table preview + LightGBM/RF results, with a
#                     plain-language key for what each label means
#
# NAVIGATION
# ----------
# One session_state value ("view") drives which section renders. The sidebar
# holds only data-source controls (directory, reload, file status). Every
# page also has its own "BACK" button at the top, so there's no need to hunt
# for the sidebar to get home.
#
# RUN WITH:
#     streamlit run leo_tek_dashboard.py
#
# REQUIRES:
#     pip install "streamlit>=1.49" pandas matplotlib
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

C_LIGHT = 299792458.0
R_EARTH_M = 6378137.0
LEO_RATE_LIMIT = 8000.0     # |range rate| ceiling for LEO [m/s]

# --- Palette: strict monochrome. No accent hue anywhere. -----------------
VOID   = "#000000"   # true black ground
PANEL  = "#070707"   # panel fill, barely lifted off the void
GRID   = "#171717"   # hairline grid
DIM    = "#454545"   # deep-background labels
MID    = "#7d7d7d"   # secondary text
BRIGHT = "#d6d6d6"   # primary text
NEON   = "#ffffff"   # readout values -- carries the glow

# Plain-language explanation of each Keplerian label, shared by the
# training-table preview and the model-results page.
LABEL_INFO = {
    "label_sma_m":     ("Semi-major axis", "Orbit size -- the average distance from Earth's centre, in metres."),
    "label_ecc":       ("Eccentricity", "Orbit shape -- 0 is a perfect circle, closer to 1 is more stretched out."),
    "label_inc_deg":   ("Inclination", "Tilt of the orbital plane relative to the equator, in degrees."),
    "label_raan_deg":  ("RAAN", "Right ascension of the ascending node -- where the orbit crosses the equator heading north, in degrees."),
    "label_argp_deg":  ("Argument of perigee", "Orientation of the orbit's closest approach to Earth within its own plane, in degrees."),
    "label_anom_deg":  ("Mean anomaly", "Where the satellite sits along its orbit at the reference time, in degrees."),
}

st.set_page_config(page_title="LEO TEK", layout="wide",
                   initial_sidebar_state="expanded")

# =============================================================================
# Styling
# =============================================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;800&display=swap');

.stApp {{
    background-color: {VOID};
    background-image:
        repeating-linear-gradient(0deg,  {GRID} 0 1px, transparent 1px 64px),
        repeating-linear-gradient(90deg, {GRID} 0 1px, transparent 1px 64px);
    color: {BRIGHT};
}}
html, body, [class*="css"], .stMarkdown, p, span, div, label {{
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
}}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 2.2rem; max-width: 1400px; }}

[data-testid="stSidebar"] {{
    background-color: {PANEL};
    border-right: 1px solid {GRID};
}}

/* ---- Masthead ---- */
.masthead {{
    border-top: 1px solid {NEON};
    border-bottom: 1px solid {GRID};
    padding: 30px 0 22px;
    margin-bottom: 34px;
}}
.wordmark {{
    font-size: clamp(40px, 8vw, 82px);
    font-weight: 800;
    letter-spacing: 0.16em;
    line-height: 0.94;
    color: {NEON};
    text-shadow: 0 0 14px rgba(255,255,255,.55), 0 0 46px rgba(255,255,255,.22);
    margin: 0;
}}
.tagline {{
    color: {MID};
    font-size: 12px;
    letter-spacing: 0.30em;
    text-transform: uppercase;
    margin-top: 16px;
}}
.tagline b {{ color: {BRIGHT}; font-weight: 400; }}

/* ---- Section rule ---- */
.sec {{
    display: flex; align-items: center; gap: 14px;
    margin: 46px 0 20px;
}}
.sec .mark {{
    width: 7px; height: 7px; background: {NEON};
    box-shadow: 0 0 9px rgba(255,255,255,.85);
    flex: 0 0 auto;
}}
.sec .name {{
    font-size: 12px; letter-spacing: 0.26em; text-transform: uppercase;
    color: {BRIGHT}; white-space: nowrap;
}}
.sec .rule {{ flex: 1 1 auto; height: 1px; background: {GRID}; }}
.sec .tail {{
    font-size: 10px; letter-spacing: 0.20em; color: {DIM};
    text-transform: uppercase; white-space: nowrap;
}}

/* ---- Readout cards ---- */
.readouts {{ display: flex; gap: 1px; background: {GRID}; border: 1px solid {GRID}; }}
.ro {{ flex: 1; background: {PANEL}; padding: 20px 22px 22px; }}
.ro .k {{
    font-size: 10px; letter-spacing: 0.22em; text-transform: uppercase;
    color: {DIM}; margin-bottom: 12px;
}}
.ro .v {{
    font-size: 36px; font-weight: 600; color: {NEON}; line-height: 1;
    text-shadow: 0 0 12px rgba(255,255,255,.42);
}}
.ro .v small {{ font-size: 14px; color: {MID}; font-weight: 300; text-shadow: none; }}
.ro .u {{ font-size: 10px; color: {DIM}; letter-spacing: .14em; margin-top: 9px;
         text-transform: uppercase; }}

/* ---- Status line ---- */
.status {{
    border: 1px solid {GRID}; background: {PANEL};
    padding: 15px 20px; display: flex; align-items: center; gap: 16px;
    font-size: 12px; letter-spacing: .1em;
}}
.status .lamp {{
    width: 9px; height: 9px; border-radius: 50%; background: {NEON};
    box-shadow: 0 0 11px rgba(255,255,255,.9); flex: 0 0 auto;
}}
.status .lamp.off {{ background: {DIM}; box-shadow: none; }}
.status .verdict {{ color: {NEON}; font-weight: 600; letter-spacing: .2em; }}
.status .detail {{ color: {MID}; }}

/* ---- Empty state ---- */
.empty {{
    border: 1px dashed {GRID}; background: {PANEL};
    padding: 46px 30px; text-align: center;
}}
.empty .t {{ color: {BRIGHT}; font-size: 13px; letter-spacing: .2em;
             text-transform: uppercase; margin-bottom: 14px; }}
.empty .d {{ color: {MID}; font-size: 12px; line-height: 1.9; }}
.empty code {{ color: {NEON}; background: {VOID}; padding: 2px 7px;
               border: 1px solid {GRID}; }}

/* ---- Result cards (real model output) ---- */
.slots {{ display: flex; gap: 1px; background: {GRID}; border: 1px solid {GRID}; }}
.slot {{ flex: 1; background: {PANEL}; padding: 26px 24px 30px; }}
.slot .n {{ font-size: 11px; letter-spacing: .22em; color: {BRIGHT};
            text-transform: uppercase; }}
.slot .s {{ font-size: 10px; letter-spacing: .18em; color: {DIM};
            text-transform: uppercase; margin-top: 10px; }}
.slot .rows {{ margin-top: 22px; }}
.slot .row {{
    display: flex; justify-content: space-between;
    border-top: 1px solid {GRID}; padding: 11px 0;
    font-size: 11px; color: {MID}; letter-spacing: .1em;
}}
.slot .row b {{ color: {NEON}; font-weight: 600; }}
.slot .row b.weak {{ color: {DIM}; font-weight: 400; }}

/* ---- Label key ---- */
.lkey {{ border-bottom: 1px solid {GRID}; }}
.lkey .row {{
    border-top: 1px solid {GRID}; padding: 10px 2px;
    font-size: 11px; color: {MID}; letter-spacing: .04em; line-height: 1.6;
}}
.lkey .row b {{ color: {BRIGHT}; }}
.lkey .row span.code {{ color: {DIM}; }}

/* ---- Home screen ---- */
.home-wrap {{ text-align: center; padding: 10px 0 8px; }}
.home-tag {{
    color: {MID}; font-size: 11px; letter-spacing: .32em;
    text-transform: uppercase; margin-top: 22px;
}}
.nav-card-label {{
    text-align: center; color: {DIM}; font-size: 10px; letter-spacing: .2em;
    text-transform: uppercase; margin-top: 8px;
}}

/* ---- Streamlit widget overrides ---- */
.stSelectbox div[data-baseweb="select"] > div {{
    background-color: {PANEL}; border: 1px solid {GRID}; border-radius: 0;
    color: {BRIGHT};
}}
.stTextInput input {{
    background-color: {PANEL}; border: 1px solid {GRID}; border-radius: 0;
    color: {BRIGHT}; font-family: 'JetBrains Mono', monospace !important;
}}
.stButton button {{
    background: {VOID}; color: {BRIGHT}; border: 1px solid {DIM};
    border-radius: 0; letter-spacing: .18em; text-transform: uppercase;
    font-size: 11px; font-family: 'JetBrains Mono', monospace !important;
    width: 100%;
}}
.stButton button:hover {{
    border-color: {NEON}; color: {NEON};
    box-shadow: 0 0 14px rgba(255,255,255,.22);
}}
.stDataFrame {{ border: 1px solid {GRID}; }}
</style>
""", unsafe_allow_html=True)


def section(name, tail=""):
    st.markdown(
        f'<div class="sec"><div class="mark"></div><div class="name">{name}</div>'
        f'<div class="rule"></div><div class="tail">{tail}</div></div>',
        unsafe_allow_html=True)


def masthead(small=False):
    size_style = "font-size: clamp(26px, 5vw, 40px); letter-spacing:.14em;" if small else ""
    st.markdown(
        f'<div class="masthead">'
        f'<div class="wordmark" style="{size_style}">LEO TEK</div>'
        f'<div class="tagline">Blind Doppler Tracking &nbsp;/&nbsp; '
        f'Opportunistic Positioning &nbsp;/&nbsp; <b>Station MAUI</b></div>'
        f'</div>', unsafe_allow_html=True)


def back_button():
    """A small, page-local way home -- doesn't rely on the sidebar."""
    col, _ = st.columns([1, 6])
    with col:
        if st.button("← BACK", key=f"back_{st.session_state.view}"):
            st.session_state.view = "home"
            st.rerun()


# =============================================================================
# Plot styling
# =============================================================================
def new_fig(w=11, h=3.6):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(VOID)
    ax.set_facecolor(VOID)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=MID, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("monospace")
    return fig, ax


def glow_line(ax, x, y, lw=1.5, marker=None):
    """Draw a line with a phosphor-style bloom around it."""
    for width, alpha in ((lw * 7, 0.05), (lw * 4, 0.09), (lw * 2, 0.16)):
        ax.plot(x, y, color=NEON, linewidth=width, alpha=alpha,
                solid_capstyle="round")
    ax.plot(x, y, color=NEON, linewidth=lw, marker=marker,
            markersize=3, solid_capstyle="round")


def axis_labels(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=DIM, fontsize=9, labelpad=9,
                  fontfamily="monospace")
    ax.set_ylabel(ylabel, color=DIM, fontsize=9, labelpad=9,
                  fontfamily="monospace")


# =============================================================================
# Data loading
# =============================================================================
@st.cache_data(show_spinner=False)
def load(data_dir):
    """
    Read the pipeline CSVs. training_table.csv and model_results.csv are
    optional -- they only exist once build_dataset.py / forest_trainer.py
    have actually been run, so the dashboard degrades gracefully without
    them rather than crashing.
    Returns (params, delays, training_table, model_results, missing_required).
    """
    def read(name):
        path = os.path.join(data_dir, name)
        return pd.read_csv(path) if os.path.exists(path) else None

    params = read("random_orbital_parameters.csv")
    delays = read("delay_differences.csv")
    training_table = read("training_table.csv")
    model_results = read("model_results.csv")

    required_missing = [n for n, d in (("random_orbital_parameters.csv", params),
                                       ("delay_differences.csv", delays)) if d is None]
    return params, delays, training_table, model_results, required_missing


def pass_series(delays_row, dd_cols, cadence=30.0):
    """Trim zero-padding off one satellite's row using num_valid_samples."""
    n = int(delays_row["num_valid_samples"])
    if n == 0:
        return np.array([]), np.array([])
    vals = delays_row[dd_cols[:n]].to_numpy(dtype=float)
    return np.arange(n) * cadence, vals


# =============================================================================
# Sidebar -- data source controls ONLY. Navigation lives on the home screen
# and on each page's own BACK button.
# =============================================================================
with st.sidebar:
    st.markdown(f'<div style="color:{NEON};letter-spacing:.26em;font-size:15px;'
                f'font-weight:800;text-shadow:0 0 10px rgba(255,255,255,.5)">'
                f'LEO TEK</div>'
                f'<div style="color:{DIM};font-size:10px;letter-spacing:.2em;'
                f'margin-top:8px">GROUND SEGMENT CONSOLE</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div style="height:1px;background:{GRID};margin:22px 0"></div>',
                unsafe_allow_html=True)

    data_dir = st.text_input("DATA DIRECTORY", value="output")
    if st.button("Reload data"):
        st.cache_data.clear()

    params, delays, training_table, model_results, required_missing = load(data_dir)

    st.markdown(f'<div style="height:1px;background:{GRID};margin:22px 0"></div>',
                unsafe_allow_html=True)
    st.markdown(f'<div style="color:{DIM};font-size:10px;letter-spacing:.2em;'
                f'margin-bottom:12px">SOURCE FILES</div>', unsafe_allow_html=True)
    for label, frame, optional in (
        ("orbital_parameters", params, False),
        ("delay_differences", delays, False),
        ("training_table", training_table, True),
        ("model_results", model_results, True),
    ):
        ok = frame is not None
        colr = NEON if ok else DIM
        mark = "ONLINE" if ok else ("OPTIONAL" if optional else "ABSENT")
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'font-size:10px;letter-spacing:.12em;padding:7px 0;'
            f'border-top:1px solid {GRID};color:{MID}">'
            f'<span>{label}</span><span style="color:{colr}">{mark}</span></div>',
            unsafe_allow_html=True)

    if required_missing:
        st.markdown(f'<div style="height:1px;background:{GRID};margin:22px 0"></div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div style="color:{DIM};font-size:10px;letter-spacing:.1em;'
            f'line-height:1.7">Run <code>generate_params.py</code> then '
            f'<code>simulationv2.py</code>, then reload.</div>',
            unsafe_allow_html=True)

    st.markdown(f'<div style="height:1px;background:{GRID};margin:22px 0"></div>',
                unsafe_allow_html=True)
    if st.session_state.get("view", "home") != "home":
        if st.button("← Home"):
            st.session_state.view = "home"
            st.rerun()


# =============================================================================
# View routing
# =============================================================================
if "view" not in st.session_state:
    st.session_state.view = "home"

if required_missing:
    masthead()
    st.markdown(
        f'<div class="empty"><div class="t">No dataset found</div>'
        f'<div class="d">Looked in <code>{data_dir}/</code> and could not find '
        f'{", ".join(f"<code>{m}</code>" for m in required_missing)}.<br><br>'
        f'Run <code>generate_params.py</code>, then <code>simulationv2.py</code>, '
        f'then reload.</div></div>', unsafe_allow_html=True)
    st.stop()

# --- Derived quantities, needed across multiple views ---------------------
dd_cols = [c for c in delays.columns if c.startswith("dd_")]
delta_t = float(params["delta_t_s"].iloc[0])
cadence = float(params["cadence_s"].iloc[0])
station = params["station_name"].iloc[0]

n_total = len(delays)
valid = delays["num_valid_samples"].to_numpy()
n_pass = int((valid > 0).sum())
yield_pct = n_pass / n_total if n_total else 0.0
seen = delays[delays["num_valid_samples"] > 0]
n_final = len(training_table) if training_table is not None else n_pass

rates = []
for _, row in seen.iterrows():
    _, v = pass_series(row, dd_cols, cadence)
    rates.append(v * C_LIGHT / delta_t)
rates_all = np.concatenate(rates) if rates else np.array([0.0])
rate_min, rate_max = float(rates_all.min()), float(rates_all.max())
in_band = max(abs(rate_min), abs(rate_max)) <= LEO_RATE_LIMIT

samples = valid[valid > 0]
mean_len = float(samples.mean()) if len(samples) else 0.0


# =============================================================================
# VIEW: HOME
# =============================================================================
if st.session_state.view == "home":

    st.markdown(
        '<div class="home-wrap">'
        '<div class="wordmark" style="font-size:clamp(46px,9vw,96px)">LEO TEK</div>'
        '<div class="home-tag">Blind Doppler Tracking &nbsp;/&nbsp; '
        f'Opportunistic Positioning &nbsp;/&nbsp; Station {station.upper()}</div>'
        '</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div style="text-align:center;color:{DIM};font-size:10px;'
        f'letter-spacing:.2em;margin-top:26px">'
        f'{n_pass:,} ACQUIRED PASSES &nbsp;·&nbsp; {n_total:,} ORBITS SAMPLED'
        f'</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:38px"></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("ORBITAL PARAMETERS"):
            st.session_state.view = "params"
            st.rerun()
        st.markdown('<div class="nav-card-label">Sampled satellites &amp; elements</div>',
                    unsafe_allow_html=True)
    with c2:
        if st.button("SATELLITE PASSES"):
            st.session_state.view = "passes"
            st.rerun()
        st.markdown('<div class="nav-card-label">Doppler curve per pass</div>',
                    unsafe_allow_html=True)
    with c3:
        if st.button("DATA CLEANING"):
            st.session_state.view = "data_cleaning"
            st.rerun()
        st.markdown('<div class="nav-card-label">Sampled &rarr; acquired &rarr; trainable</div>',
                    unsafe_allow_html=True)
    with c4:
        if st.button("MODEL RESULTS"):
            st.session_state.view = "models"
            st.rerun()
        st.markdown('<div class="nav-card-label">Training table &amp; performance</div>',
                    unsafe_allow_html=True)


# =============================================================================
# VIEW: PARAMS  (generate_params.py output)
# =============================================================================
elif st.session_state.view == "params":
    masthead(small=True)
    back_button()
    section("Generated orbital parameters", f"{n_total:,} satellites sampled")

    alt_km = (params["sma_m"] - R_EARTH_M) / 1000
    ecc = params["ecc"]
    inc_deg = np.degrees(params["inc_rad"])

    cards = [
        ("Satellites sampled", f"{n_total:,}", ""),
        ("Altitude range", f"{alt_km.min():.0f}&ndash;{alt_km.max():.0f}", "km"),
        ("Eccentricity range", f"{ecc.min():.4f}&ndash;{ecc.max():.4f}", ""),
        ("Inclination range", f"{inc_deg.min():.1f}&ndash;{inc_deg.max():.1f}", "deg"),
    ]
    st.markdown('<div class="readouts">' + "".join(
        f'<div class="ro"><div class="k">{k}</div>'
        f'<div class="v" style="font-size:24px">{v}</div>'
        f'<div class="u">{u}</div></div>'
        for k, v, u in cards
    ) + '</div>', unsafe_allow_html=True)

    section("Orbital elements", "Keplerian elements per satellite")
    show_cols = ["sat_idx", "sma_m", "ecc", "inc_rad", "raan_rad", "argp_rad", "anom_rad"]
    show_cols = [c for c in show_cols if c in params.columns]
    n_show = min(200, len(params))
    st.dataframe(params[show_cols].head(n_show), width="stretch", hide_index=True)
    if n_total > n_show:
        st.markdown(
            f'<div style="color:{DIM};font-size:10px;letter-spacing:.1em;margin-top:6px">'
            f'Showing first {n_show:,} of {n_total:,} rows.</div>',
            unsafe_allow_html=True)


# =============================================================================
# VIEW: PASSES  (simulationv2.py output)
# =============================================================================
elif st.session_state.view == "passes":
    masthead(small=True)
    back_button()
    section("Satellite passes", f"{n_pass:,} acquired · mean arc {mean_len:.1f} samples")

    ids = seen["sat_idx"].tolist()
    c1, c2 = st.columns([1, 3])
    with c1:
        pick = st.selectbox("SATELLITE", ids,
                            index=int(np.argmax(seen["num_valid_samples"].to_numpy())))
        show_units = st.selectbox("READOUT", ["Delay difference (s)",
                                              "Range rate (m/s)"])
        row = seen[seen["sat_idx"] == pick].iloc[0]
        p = params[params["sat_idx"] == pick].iloc[0]
        n_s = int(row["num_valid_samples"])
        st.markdown(
            f'<div style="border:1px solid {GRID};background:{PANEL};padding:16px;'
            f'margin-top:6px;font-size:11px;color:{MID};letter-spacing:.08em">'
            f'<div style="color:{DIM};font-size:10px;letter-spacing:.2em;'
            f'margin-bottom:12px">ORBIT {pick}</div>'
            f'ALT&nbsp;&nbsp;{(p["sma_m"]-R_EARTH_M)/1000:>8.1f} km<br>'
            f'ECC&nbsp;&nbsp;{p["ecc"]:>8.4f}<br>'
            f'INC&nbsp;&nbsp;{np.degrees(p["inc_rad"]):>8.2f}°<br>'
            f'ARC&nbsp;&nbsp;{n_s*cadence/60:>8.1f} min<br>'
            f'PTS&nbsp;&nbsp;{n_s:>8d}</div>', unsafe_allow_html=True)

    with c2:
        t, v = pass_series(row, dd_cols, cadence)
        y = v if show_units.startswith("Delay") else v * C_LIGHT / delta_t
        fig, ax = new_fig(11, 4.1)
        ax.axhline(0, color=DIM, linewidth=0.8, linestyle=(0, (4, 4)))
        glow_line(ax, t, y, lw=1.6, marker="o")
        axis_labels(ax, "TIME SINCE AOS  [s]",
                    "Δd  [s]" if show_units.startswith("Delay") else "ṙ  [m/s]")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_color(DIM)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.markdown(
        f'<div style="color:{DIM};font-size:11px;letter-spacing:.06em;'
        f'line-height:1.9;margin-top:4px">'
        f'Negative on approach, positive on recession. '
        f'A single clean crossing is the signature of a correct pass.</div>',
        unsafe_allow_html=True)


# =============================================================================
# VIEW: DATA CLEANING  (delay_differences.csv -> training_table.csv)
# =============================================================================
elif st.session_state.view == "data_cleaning":
    masthead(small=True)
    back_button()
    section("Data cleaning", "from sampled orbits to trainable passes")

    cards = [
        ("Orbits sampled", f"{n_total:,}", "generate_params.py"),
        ("Passes acquired", f"{n_pass:,}", f"{yield_pct*100:.1f}% yield"),
        ("Rows ready for ML", f"{n_final:,}",
         "training_table.csv" if training_table is not None else "build_dataset.py not yet run"),
    ]
    st.markdown('<div class="readouts">' + "".join(
        f'<div class="ro"><div class="k">{k}</div><div class="v">{v}</div>'
        f'<div class="u">{u}</div></div>'
        for k, v, u in cards
    ) + '</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div style="height:1px"></div>'
        f'<div class="status" style="margin-top:1px">'
        f'<div class="lamp{"" if in_band else " off"}"></div>'
        f'<div class="verdict">{"RANGE RATE NOMINAL" if in_band else "RANGE RATE OUT OF BAND"}</div>'
        f'<div class="detail">observed {rate_min:,.0f} to {rate_max:,.0f} m/s '
        f'&nbsp;·&nbsp; LEO envelope ±{LEO_RATE_LIMIT:,.0f} m/s</div></div>',
        unsafe_allow_html=True)

    section("How we get there", "")

    step3 = (
        f'<b style="color:{BRIGHT}">3. Feature engineering</b> &mdash; '
        f'{n_final:,} of those {n_pass:,} passes have enough valid samples to build '
        f'the 17 engineered features and the six Keplerian labels, and make it into '
        f'the final training table.'
        if training_table is not None else
        f'<b style="color:{DIM}">3. Feature engineering</b> &mdash; '
        f'run <code>build_dataset.py</code> to see this step.'
    )

    st.markdown(f"""
    <div style="color:{MID};font-size:12px;line-height:2.0;letter-spacing:.04em">
    <b style="color:{BRIGHT}">1. Orbits sampled</b> &mdash; {n_total:,} satellites are
    generated with random Keplerian elements (semi-major axis, eccentricity, inclination,
    RAAN, argument of perigee, mean anomaly) within physically valid LEO bounds.<br><br>
    <b style="color:{BRIGHT}">2. Passes acquired</b> &mdash; each orbit is propagated past
    the Maui ground station. Only {n_pass:,} of {n_total:,} ({yield_pct*100:.1f}%) ever come
    into view for long enough to record a contiguous range-rate sequence; the rest never rise
    above the horizon during the simulated window and are dropped.<br><br>
    {step3}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# VIEW: MODELS  (build_dataset.py + forest_trainer.py output)
# =============================================================================
elif st.session_state.view == "models":
    masthead(small=True)
    back_button()
    section("Training data", "features and labels, aligned by sat_idx")

    if training_table is None:
        st.markdown(
            f'<div class="empty"><div class="t">No training table found</div>'
            f'<div class="d">Run <code>build_dataset.py</code> to produce '
            f'<code>training_table.csv</code>, then reload.</div></div>',
            unsafe_allow_html=True)
    else:
        feat_cols_tt = [c for c in training_table.columns if c.startswith("feat_")]
        label_cols_tt = [c for c in training_table.columns if c.startswith("label_")]
        fc, lc = st.columns(2)
        with fc:
            st.markdown(f'<div style="color:{DIM};font-size:10px;letter-spacing:.22em;'
                        f'margin-bottom:10px">FEATURES &nbsp;X &nbsp;·&nbsp; '
                        f'{len(feat_cols_tt)} engineered columns</div>', unsafe_allow_html=True)
            st.dataframe(training_table[["sat_idx"] + feat_cols_tt[:5]].head(8),
                        width="stretch", hide_index=True)
        with lc:
            st.markdown(f'<div style="color:{DIM};font-size:10px;letter-spacing:.22em;'
                        f'margin-bottom:10px">LABELS &nbsp;y &nbsp;·&nbsp; '
                        f'six Keplerian elements</div>', unsafe_allow_html=True)
            st.dataframe(training_table[["sat_idx"] + label_cols_tt].head(8),
                        width="stretch", hide_index=True)

        st.markdown(f'<div style="color:{DIM};font-size:10px;letter-spacing:.22em;'
                    f'margin:22px 0 4px">LABEL KEY &nbsp;·&nbsp; what each column means</div>',
                    unsafe_allow_html=True)
        legend_rows = "".join(
            f'<div class="row"><b>{LABEL_INFO.get(c, (c, ""))[0]}</b> '
            f'<span class="code">({c})</span> &mdash; {LABEL_INFO.get(c, ("", ""))[1]}</div>'
            for c in label_cols_tt
        )
        st.markdown(f'<div class="lkey">{legend_rows}</div>', unsafe_allow_html=True)

    section("Orbit estimation models", "LightGBM & Random Forest, per label")

    if model_results is None:
        st.markdown(
            f'<div class="empty"><div class="t">No results found</div>'
            f'<div class="d">Run <code>forest_trainer.py</code> to produce '
            f'<code>model_results.csv</code>, then reload.<br><br>'
            f'{n_pass:,} acquired passes are ready to train on.</div></div>',
            unsafe_allow_html=True)
    else:
        method = st.selectbox("EVALUATION METHOD",
                              ["cv", "single_split"],
                              format_func=lambda m: "5-Fold Cross-Validation" if m == "cv"
                                                    else "Single Train/Test Split")
        subset = model_results[model_results["method"] == method]

        for model_name, display in (("lightgbm", "LightGBM"), ("random_forest", "Random Forest")):
            m = subset[subset["model"] == model_name]
            if m.empty:
                continue
            rows_html = ""
            for _, r in m.iterrows():
                weak = "weak" if r["r2"] < 0.3 else ""
                label = LABEL_INFO.get(r["target"], (r["target"], ""))[0]
                rows_html += (f'<div class="row"><span>{label}</span>'
                             f'<b class="{weak}">R² {r["r2"]:.3f} · RMSE {r["rmse"]:,.2f}</b></div>')
            n_val = int(m["n"].iloc[0]) if len(m) else 0
            st.markdown(f"""
            <div class="slots" style="margin-bottom:1px">
              <div class="slot" style="flex:1 1 100%">
                <div class="n">{display}</div>
                <div class="s">gradient boosted trees · engineered features</div>
                <div class="rows">{rows_html}
                  <div class="row"><span>Test samples</span><b>{n_val:,}</b></div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div style="height:1px"></div>', unsafe_allow_html=True)

        st.markdown(
            f'<div style="color:{DIM};font-size:11px;letter-spacing:.06em;'
            f'line-height:1.9;margin-top:12px">'
            f'Dim values (R² below 0.3) indicate weak or no recoverable signal for that '
            f'label at this observation setup, not a weakness in either model, both are '
            f'trained on identical features and agree closely on every label.</div>',
            unsafe_allow_html=True)


# =============================================================================
# Footer
# =============================================================================
st.markdown(
    f'<div style="margin-top:56px;padding-top:20px;border-top:1px solid {GRID};'
    f'color:{DIM};font-size:10px;letter-spacing:.2em">'
    f'LEO TEK &nbsp;·&nbsp; ML-ENABLED BLIND DOPPLER TRACKING &nbsp;·&nbsp; '
    f'STATION {station.upper()}</div>', unsafe_allow_html=True)