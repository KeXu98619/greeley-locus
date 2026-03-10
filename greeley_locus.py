"""
greeley_locus.py
Streamlit app – City of Greeley Transportation Dashboard

Data inputs (same folder as this script):
  - dow_tod_segments_speed&sample.parquet       (speed + attributes, no geometry)
  - segments_geometry.geojson                   (unique segment geometries)
  - trip_summary_with_expansion_area.parquet    (OD trip summary by area & hour)
  - region_geometry.geojson                     (area polygon geometries)
  - aadt_by_segment_dow_tod.parquet             (AADT by segment, DOW, time-of-day)
    Expected columns: newSegmentId, dow, timeSetName, aadt
  - logo.png                                    (organisation logo)

Run:  streamlit run greeley_locus.py
Requires: streamlit>=1.40, pydeck, plotly, geopandas, pandas, numpy, pyarrow
"""

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import pydeck as pdk
import plotly.graph_objects as go
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Greeley Transportation Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Password gate ──────────────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    _, pwd_col, _ = st.columns([1, 2, 1])
    with pwd_col:
        logo_path_login = Path("logo.png")
        if logo_path_login.exists():
            st.image(str(logo_path_login), width=180)
        st.markdown("## City of Greeley Dashboard")
        pwd_input = st.text_input("Password", type="password", key="pwd_field")
        if st.button("Login", use_container_width=True):
            if pwd_input == st.secrets.get("APP_PASSWORD", ""):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password — please try again.")
    st.stop()

# ── File paths ─────────────────────────────────────────────────────────────────
PARQUET_PATH      = "dow_tod_segments_speed&sample.parquet"
GEOM_PATH         = "segments_geometry.geojson"
TRIP_PARQUET_PATH = "trip_summary_with_expansion_area.parquet"
REGION_GEOM_PATH  = "region_geometry.geojson"
AADT_PARQUET_PATH = "aadt_by_segment_dow_tod.parquet"
LOGO_PATH         = "logo.png"

# ── Geofencing area exclusions ─────────────────────────────────────────────────
# Add/remove area numbers to exclude them from the Geofencing Trips map.
# Set to [] to restore all areas.
EXCLUDED_AREAS = [3]

# ── Map / data constants ───────────────────────────────────────────────────────
VIEW_STATE   = pdk.ViewState(latitude=40.420376, longitude=-104.693693, zoom=14, pitch=0)
MAP_STYLE    = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
DOW_OPTIONS  = ["Mon-Thur", "Fri", "Sat", "Sun"]
HOUR_LABELS  = [f"{h}:00–{h+1}:00" for h in range(24)]
TOD_AVG_ALL  = "All (avg across hours)"
TOD_SUM_ALL  = "All (sum across hours)"


# ── Geometry helpers ───────────────────────────────────────────────────────────
def tod_sort_key(label: str) -> int:
    try:
        return int(label.split(":")[0])
    except Exception:
        return -1


def get_bin_hours(label: str) -> int:
    """Return number of hours spanned by a TOD label like '10:00-16:00'."""
    import re
    m = re.match(r"^(\d+):00\s*[-–]\s*(\d+):00$", str(label).strip())
    if not m:
        return 1
    s, e = int(m.group(1)), int(m.group(2))
    if e <= s:
        e += 24
    return max(e - s, 1)


def geom_to_path(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "LineString":
        return [[round(x, 6), round(y, 6)] for x, y in geom.coords]
    if geom.geom_type == "MultiLineString":
        coords = []
        for part in geom.geoms:
            coords.extend([[round(x, 6), round(y, 6)] for x, y in part.coords])
        return coords
    return None


def geom_to_polygon(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return [[round(x, 6), round(y, 6)] for x, y in geom.exterior.coords]
    if geom.geom_type == "MultiPolygon":
        largest = max(geom.geoms, key=lambda p: p.area)
        return [[round(x, 6), round(y, 6)] for x, y in largest.exterior.coords]
    return None


# ── Color helpers ──────────────────────────────────────────────────────────────
def speed_to_color(series: pd.Series) -> list:
    """Red (slow) → Yellow → Green (fast). Gray for NaN."""
    valid = series.dropna()
    vmin = float(valid.min()) if len(valid) else 0.0
    vmax = float(valid.max()) if len(valid) else 1.0
    rng  = (vmax - vmin) if (vmax - vmin) > 0 else 1.0
    out  = []
    for v in series:
        if pd.isna(v):
            out.append([160, 160, 160, 120])
        else:
            t = float((v - vmin) / rng)
            r = int(255 * (1.0 - t))
            g = int(220 * t + 35)
            out.append([r, g, 0, 220])
    return out


def ratio_to_color(series: pd.Series) -> list:
    """Grey at ratio≈1, orange/red above 1 (Scenario A faster), blue below 1 (A slower).
    NaN = fully transparent (hidden)."""
    out = []
    for v in series:
        if pd.isna(v):
            out.append([0, 0, 0, 0])  # fully transparent – not shown
        elif v >= 1.0:
            t = min((v - 1.0) / 0.5, 1.0)
            r = int(180 + 60 * t)   # 180→240
            g = int(180 - 120 * t)  # 180→60
            b = int(180 - 160 * t)  # 180→20
            out.append([r, g, b, 230])
        else:
            t = min((1.0 - v) / 0.5, 1.0)
            r = int(180 - 160 * t)  # 180→20
            g = int(180 - 110 * t)  # 180→70
            b = int(180 + 60 * t)   # 180→240
            out.append([r, g, b, 230])
    return out


def blue_gradient_color(series: pd.Series, alpha: int = 210) -> list:
    """Light blue (low) → Dark blue (high). Gray for NaN."""
    valid = series.dropna()
    if len(valid) == 0:
        return [[160, 160, 160, 120]] * len(series)
    vmin = float(valid.min())
    vmax = float(valid.max())
    rng  = (vmax - vmin) if (vmax - vmin) > 0 else 1.0
    out  = []
    for v in series:
        if pd.isna(v):
            out.append([160, 160, 160, 120])
        else:
            t = float((v - vmin) / rng)
            r = int(220 - 200 * t)
            g = int(235 - 185 * t)
            b = 255
            out.append([r, g, b, alpha])
    return out


# ── Colorbar figures ───────────────────────────────────────────────────────────
def make_speed_colorbar(vmin: float, vmax: float) -> go.Figure:
    """Horizontal gradient colorbar for speed: red → yellow → green."""
    if pd.isna(vmin) or pd.isna(vmax) or vmin >= vmax:
        return None
    n = 100
    vals = [vmin + (vmax - vmin) * i / (n - 1) for i in range(n)]
    colorscale = [
        [0.0, "rgb(255,35,0)"],
        [0.5, "rgb(200,145,0)"],
        [1.0, "rgb(35,220,59)"],
    ]
    fig = go.Figure(go.Heatmap(
        z=[vals], x=vals, y=[""],
        colorscale=colorscale,
        showscale=False,
        zmin=vmin, zmax=vmax,
        hoverinfo="none",
    ))
    fig.update_layout(
        height=62,
        margin=dict(t=24, b=22, l=10, r=10),
        title=dict(
            text=(
                f"Avg Speed (mph)  |  "
                f"Slow ◀  {vmin:.0f} ──── {vmax:.0f}  ▶ Fast  "
                f"|  Gray = No data"
            ),
            font_size=11, x=0.5,
        ),
        xaxis=dict(showgrid=False, zeroline=False, tickformat=".0f"),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_volume_colorbar(vmin: float, vmax: float) -> go.Figure:
    """Horizontal gradient colorbar for Volume: light blue → dark blue."""
    if pd.isna(vmin) or pd.isna(vmax) or vmin >= vmax:
        return None
    n = 100
    vals = [vmin + (vmax - vmin) * i / (n - 1) for i in range(n)]
    colorscale = [
        [0.0, "rgb(220,235,255)"],
        [0.5, "rgb(100,160,255)"],
        [1.0, "rgb(0,63,181)"],
    ]
    fig = go.Figure(go.Heatmap(
        z=[vals], x=vals, y=[""],
        colorscale=colorscale,
        showscale=False,
        zmin=vmin, zmax=vmax,
        hoverinfo="none",
    ))
    fig.update_layout(
        height=62,
        margin=dict(t=24, b=22, l=10, r=10),
        title=dict(
            text=(
                f"Volume  |  "
                f"Low ◀  {int(vmin):,} ──── {int(vmax):,}  ▶ High  "
                f"|  Gray = No data"
            ),
            font_size=11, x=0.5,
        ),
        xaxis=dict(showgrid=False, zeroline=False, tickformat=",d"),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Pydeck layer builders ──────────────────────────────────────────────────────
def make_layer(df: pd.DataFrame) -> pdk.Layer:
    return pdk.Layer(
        "PathLayer",
        data=df,
        id="segments",
        get_path="path",
        get_color="color",
        get_width=10,
        width_min_pixels=2,
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 0, 255],
    )


def make_polygon_layer(df: pd.DataFrame) -> pdk.Layer:
    return pdk.Layer(
        "PolygonLayer",
        data=df,
        id="areas",
        get_polygon="polygon",
        get_fill_color="color",
        get_line_color=[80, 80, 80, 200],
        line_width_min_pixels=1,
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 220, 0, 120],
        stroked=True,
    )


def bar_chart(
    x,
    y: pd.Series,
    title: str,
    color: str = "#2196F3",
    ymax: float = None,
    yaxis_title: str = "Avg Speed (mph)",
    xaxis_title: str = "Time of Day",
    bar_name: str = "Systemwide",
) -> go.Figure:
    fig = go.Figure(
        go.Bar(x=x, y=y, marker_color=color, opacity=0.85, name=bar_name)
    )
    fig.update_layout(
        title=dict(text=title, font_size=13),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis_tickangle=-45,
        yaxis=dict(range=[0, ymax] if ymax else None),
        height=310,
        margin=dict(t=40, b=70, l=50, r=10),
        xaxis=dict(gridcolor="#e0e0e0"),
        yaxis_gridcolor="#e0e0e0",
    )
    return fig


# ── Data loading (cached) ──────────────────────────────────────────────────────
@st.cache_data
def load_speed() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_PATH)
    df["averageSpeed"] = df["averageSpeed"].where(df["averageSpeed"] > 0, other=np.nan)
    return df


@st.cache_data
def load_geometry() -> pd.DataFrame:
    gdf = gpd.read_file(GEOM_PATH).to_crs(epsg=4326)
    gdf["path"] = gdf.geometry.apply(geom_to_path)
    keep = [c for c in ["newSegmentId", "segmentId", "path"] if c in gdf.columns]
    return gdf[keep].dropna(subset=["path"]).reset_index(drop=True)


@st.cache_data
def load_segment_attrs() -> pd.DataFrame:
    return (
        pd.read_parquet(PARQUET_PATH, columns=["newSegmentId", "street", "frc"])
        .drop_duplicates("newSegmentId")
        .reset_index(drop=True)
    )


@st.cache_data
def load_trips() -> pd.DataFrame:
    return pd.read_parquet(TRIP_PARQUET_PATH)


@st.cache_data
def load_area_geometry() -> pd.DataFrame:
    gdf = gpd.read_file(REGION_GEOM_PATH).to_crs(epsg=4326)
    gdf["name"] = gdf["name"].astype(int)
    gdf = gdf[gdf["name"] <= 29].copy()
    gdf["polygon"] = gdf.geometry.apply(geom_to_polygon)
    return gdf[["name", "polygon"]].dropna(subset=["polygon"]).reset_index(drop=True)


@st.cache_data
def load_aadt() -> pd.DataFrame:
    """
    Expected columns: newSegmentId, dow, timeSetName, aadt
    Column name normalisation handles common casing variants automatically.
    """
    df = pd.read_parquet(AADT_PARQUET_PATH)
    # Normalise column names
    col_map = {}
    lcols = {c.lower(): c for c in df.columns}
    for src, dst in [("newsegmentid", "newSegmentId"), ("timesetname", "timeSetName"), ("aadt", "aadt")]:
        if src in lcols and lcols[src] != dst:
            col_map[lcols[src]] = dst
    if col_map:
        df = df.rename(columns=col_map)
    if "aadt" in df.columns:
        df["aadt"] = pd.to_numeric(df["aadt"], errors="coerce")
        df["aadt"] = df["aadt"].where(df["aadt"] >= 0, other=np.nan)
    return df


@st.cache_data
def load_volume_hourly() -> pd.DataFrame:
    """Expand multi-hour TOD bins (e.g. '0:00-7:00') into individual hourly rows.
    Volume is distributed evenly across the hours in each bin."""
    import re
    df = load_aadt()
    pattern = re.compile(r"^(\d+):00\s*[-–]\s*(\d+):00$")

    def get_hourly_labels(label):
        m = pattern.match(str(label).strip())
        if not m:
            return [label]
        s, e = int(m.group(1)), int(m.group(2))
        if e <= s:
            e += 24
        return [f"{h}:00-{h+1}:00" for h in range(s, e)]

    df = df.copy()
    df["_labels"] = df["timeSetName"].apply(get_hourly_labels)
    df["_n"] = df["_labels"].apply(len)
    df["aadt"] = df["aadt"] / df["_n"]
    df_exp = df.explode("_labels").copy()
    df_exp["timeSetName"] = df_exp["_labels"]
    df_exp = df_exp.drop(columns=["_labels", "_n"])
    df_exp = df_exp.rename(columns={"aadt": "volume"})
    return df_exp.reset_index(drop=True)


ALL_VOL_TOD = [f"{h}:00-{h+1}:00" for h in range(24)]


# ── Load all data ──────────────────────────────────────────────────────────────
try:
    df_speed       = load_speed()
    df_geom        = load_geometry()
    df_seg_attr    = load_segment_attrs()
    df_trips       = load_trips()
    df_areas       = load_area_geometry()
    df_aadt        = load_aadt()
    df_vol_hourly  = load_volume_hourly()
except FileNotFoundError as e:
    st.error(
        f"Data file not found: {e}\n"
        "Make sure all data files are in the same folder as this script."
    )
    st.stop()

# Apply geofencing exclusions
if EXCLUDED_AREAS:
    df_areas = df_areas[~df_areas["name"].isin(EXCLUDED_AREAS)].reset_index(drop=True)

ALL_TOD = sorted(df_speed["timeSetName"].unique(), key=tod_sort_key)

if "timeSetName" in df_aadt.columns:
    ALL_AADT_TOD = sorted(df_aadt["timeSetName"].unique(), key=tod_sort_key)
else:
    ALL_AADT_TOD = ALL_TOD  # fallback


# ── Tooltip format helpers ─────────────────────────────────────────────────────
def fmt_speed(v):
    return f"{int(round(v))} mph" if pd.notna(v) else "No data"

def fmt_frc(v):
    return str(int(v)) if pd.notna(v) else "None"

def fmt_str(v):
    return str(v) if (v is not None and pd.notna(v)) else "None"

def fmt_trips(v):
    return f"{int(round(v))} trips/day" if pd.notna(v) else "No Data"

def fmt_aadt(v):
    return f"{int(round(v)):,}" if pd.notna(v) else "No Data"


# ── Header: title (left) + logo (right) ───────────────────────────────────────
hdr_title, hdr_logo = st.columns([8, 1])
with hdr_title:
    st.title("City of Greeley Dashboard")
with hdr_logo:
    logo_path = Path(LOGO_PATH)
    if logo_path.exists():
        st.image(str(logo_path), width=110)

# ── Session state: persistent selections (survive filter changes) ──────────────
_init_state = {
    "spd_sel_id": None, "spd_sel_street": None,
    "ratio_sel_id": None, "ratio_sel_street": None,
    "trip_sel_area": None,
    "vol_sel_id": None, "vol_sel_street": None,
    # Key counters — increment to force pydeck component recreation (clears selection)
    "spd_key_ctr": 0, "ratio_key_ctr": 0,
    "trip_key_ctr": 0, "vol_key_ctr": 0,
}
for _k, _v in _init_state.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Speed", "Trips by parking zones", "Traffic Volumes"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – ROAD PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:

    col_metric, col_filters = st.columns([1, 4])
    with col_metric:
        metric = st.radio("**Metric**", ["Average Speed", "Speed Ratio"])

    with col_filters:
        if metric == "Average Speed":
            fa, fb, _ = st.columns([1, 2, 1])
            dow = fa.selectbox("Day of Week", DOW_OPTIONS)
            tod = fb.selectbox("Time of Day (for map)", [TOD_AVG_ALL] + ALL_TOD)
        else:
            fa, fb, fc, fd = st.columns(4)
            dow_a = fa.selectbox("DOW – Scenario A", DOW_OPTIONS, key="dow_a",
                                 index=DOW_OPTIONS.index("Mon-Thur"))
            _tod_a_opts = [TOD_AVG_ALL] + ALL_TOD
            tod_a = fb.selectbox("TOD – Scenario A", _tod_a_opts, key="tod_a",
                                 index=_tod_a_opts.index("8:00-9:00") if "8:00-9:00" in _tod_a_opts else 0)
            dow_b = fc.selectbox("DOW – Scenario B", DOW_OPTIONS, key="dow_b",
                                 index=DOW_OPTIONS.index("Sat"))
            _tod_b_opts = [TOD_AVG_ALL] + ALL_TOD
            tod_b = fd.selectbox("TOD – Scenario B", _tod_b_opts, key="tod_b",
                                 index=_tod_b_opts.index("8:00-9:00") if "8:00-9:00" in _tod_b_opts else 0)
            st.caption("Speed Ratio = Avg Speed (Scenario A) ÷ Avg Speed (Scenario B)")

    st.divider()
    map_col, chart_col = st.columns([3, 2])

    # ── AVERAGE SPEED MODE ────────────────────────────────────────────────────
    if metric == "Average Speed":
        base = df_speed[df_speed["dow"] == dow].copy()

        if tod == TOD_AVG_ALL:
            seg_speed = (
                base.groupby("newSegmentId")["averageSpeed"]
                .mean().rename("displaySpeed").reset_index()
            )
        else:
            seg_speed = (
                base[base["timeSetName"] == tod]
                .groupby("newSegmentId")["averageSpeed"]
                .mean().rename("displaySpeed").reset_index()
            )

        map_df = (
            df_geom
            .merge(df_seg_attr, on="newSegmentId", how="left")
            .merge(seg_speed,   on="newSegmentId", how="left")
        )
        map_df["color"]        = speed_to_color(map_df["displaySpeed"])
        map_df["speed_label"]  = map_df["displaySpeed"].apply(fmt_speed)
        map_df["frc_label"]    = map_df["frc"].apply(fmt_frc)
        map_df["street_label"] = map_df["street"].apply(fmt_str)

        sys_avg = (
            base.groupby("timeSetName")["averageSpeed"]
            .mean().reindex(ALL_TOD).reset_index()
            .rename(columns={"averageSpeed": "avg_speed"})
        )

        valid_speeds = map_df["displaySpeed"].dropna()
        spd_vmin = float(valid_speeds.min()) if len(valid_speeds) else 0.0
        spd_vmax = float(valid_speeds.max()) if len(valid_speeds) else 1.0

        with map_col:
            st.subheader("Avg Speed")
            cb_fig = make_speed_colorbar(spd_vmin, spd_vmax)
            if cb_fig:
                st.plotly_chart(cb_fig, use_container_width=True,
                                config={"displayModeBar": False})
            layer = make_layer(
                map_df[["newSegmentId", "path", "color",
                         "speed_label", "street_label", "frc_label"]]
            )
            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=VIEW_STATE,
                map_style=MAP_STYLE,
                tooltip={
                    "text": (
                        "Segment ID: {newSegmentId}\n"
                        "Street: {street_label}\n"
                        "FRC: {frc_label}\n"
                        "Avg Speed: {speed_label}"
                    )
                },
            )
            event = st.pydeck_chart(
                deck,
                on_select="rerun",
                selection_mode="single-object",
                use_container_width=True,
                height=470,
                key=f"map_avg_speed_{st.session_state.spd_key_ctr}",
            )

        with chart_col:
            try:
                objs = event.selection.objects.get("segments", [])
                if objs:
                    st.session_state.spd_sel_id     = objs[0].get("newSegmentId")
                    st.session_state.spd_sel_street = objs[0].get("street_label")
            except Exception:
                pass
            selected_id     = st.session_state.spd_sel_id
            selected_street = st.session_state.spd_sel_street

            # ── Selected segment info at top of chart column ───────────────
            if selected_id:
                _c_info, _c_clr = st.columns([3, 1])
                _c_info.markdown(
                    f"**Selected:** `{selected_id}`  \n"
                    f"**Street:** {selected_street or 'None'}"
                )
                if _c_clr.button("✕ Clear", key="clr_spd"):
                    st.session_state.spd_sel_id     = None
                    st.session_state.spd_sel_street = None
                    st.session_state.spd_key_ctr    += 1
                    st.rerun()

            fig_sys = bar_chart(
                x=sys_avg["timeSetName"],
                y=sys_avg["avg_speed"],
                title="Systemwide Avg Speed",
                color="#2196F3",
            )
            if selected_id:
                seg_sys = (
                    base[base["newSegmentId"] == selected_id]
                    .groupby("timeSetName")["averageSpeed"].mean()
                    .reindex(ALL_TOD).reset_index()
                    .rename(columns={"averageSpeed": "avg_speed"})
                )
                fig_sys.add_trace(
                    go.Scatter(
                        x=seg_sys["timeSetName"],
                        y=seg_sys["avg_speed"],
                        mode="lines+markers",
                        name="Selected segment",
                        line=dict(color="#FF9800", width=2),
                        marker=dict(size=6),
                    )
                )
                fig_sys.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
            st.plotly_chart(fig_sys, use_container_width=True)

            if selected_id:
                seg_tod = (
                    base[base["newSegmentId"] == selected_id]
                    .groupby("timeSetName")["averageSpeed"].mean()
                    .reindex(ALL_TOD).reset_index()
                    .rename(columns={"timeSetName": "Time of Day",
                                     "averageSpeed": "Avg Speed (mph)"})
                )
                seg_tod["Avg Speed (mph)"] = seg_tod["Avg Speed (mph)"].apply(
                    lambda v: int(round(v)) if pd.notna(v) else None
                )
                st.markdown("**Selected Segment — Avg Speed by Time of Day**")
                st.dataframe(
                    seg_tod,
                    use_container_width=True,
                    hide_index=True,
                    height=min(420, 38 + len(seg_tod) * 35),
                )
            else:
                st.info("Click a segment on the map to see its time-of-day speed profile.")

    # ── SPEED RATIO MODE ──────────────────────────────────────────────────────
    else:
        # Scenario A
        if tod_a == TOD_AVG_ALL:
            speed_a = (
                df_speed[df_speed["dow"] == dow_a]
                .groupby("newSegmentId")["averageSpeed"].mean().rename("speed_a")
            )
        else:
            speed_a = (
                df_speed[(df_speed["dow"] == dow_a) & (df_speed["timeSetName"] == tod_a)]
                .groupby("newSegmentId")["averageSpeed"].mean().rename("speed_a")
            )

        # Scenario B
        if tod_b == TOD_AVG_ALL:
            speed_b = (
                df_speed[df_speed["dow"] == dow_b]
                .groupby("newSegmentId")["averageSpeed"].mean().rename("speed_b")
            )
        else:
            speed_b = (
                df_speed[(df_speed["dow"] == dow_b) & (df_speed["timeSetName"] == tod_b)]
                .groupby("newSegmentId")["averageSpeed"].mean().rename("speed_b")
            )

        ratio_df = pd.concat([speed_a, speed_b], axis=1).reset_index()
        ratio_df["ratio"] = ratio_df["speed_a"] / ratio_df["speed_b"]

        map_df = (
            df_geom
            .merge(df_seg_attr, on="newSegmentId", how="left")
            .merge(ratio_df,    on="newSegmentId", how="left")
        )
        map_df["color"]        = ratio_to_color(map_df["ratio"])
        map_df["ratio_label"]  = map_df["ratio"].apply(
            lambda v: f"{v:.3f}" if pd.notna(v) else "No data"
        )
        map_df["spd_a_label"]  = map_df["speed_a"].apply(fmt_speed)
        map_df["spd_b_label"]  = map_df["speed_b"].apply(fmt_speed)
        map_df["frc_label"]    = map_df["frc"].apply(fmt_frc)
        map_df["street_label"] = map_df["street"].apply(fmt_str)

        tod_a_lbl = "All hrs" if tod_a == TOD_AVG_ALL else tod_a
        tod_b_lbl = "All hrs" if tod_b == TOD_AVG_ALL else tod_b

        with map_col:
            st.subheader("Speed Ratio — Scenario A ÷ Scenario B")
            st.markdown(
                "<span style='color:#1446F0'>&#9632;</span> Scenario A slower &nbsp;&nbsp;"
                "<span style='color:#B4B4B4'>&#9632;</span> Similar &nbsp;&nbsp;"
                "<span style='color:#F03C14'>&#9632;</span> Scenario A faster",
                unsafe_allow_html=True,
            )
            layer = make_layer(
                map_df[[
                    "newSegmentId", "path", "color",
                    "street_label", "frc_label",
                    "ratio_label", "spd_a_label", "spd_b_label",
                ]]
            )
            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=VIEW_STATE,
                map_style=MAP_STYLE,
                tooltip={
                    "text": (
                        "Segment ID: {newSegmentId}\n"
                        "Street: {street_label}\n"
                        "FRC: {frc_label}\n"
                        "Ratio (Scenario A / Scenario B): {ratio_label}\n"
                        "Speed Scenario A: {spd_a_label}\n"
                        "Speed Scenario B: {spd_b_label}"
                    )
                },
            )
            ratio_event = st.pydeck_chart(
                deck,
                on_select="rerun",
                selection_mode="single-object",
                use_container_width=True,
                height=520,
                key=f"map_ratio_{st.session_state.ratio_key_ctr}",
            )

        with chart_col:
            try:
                r_objs = ratio_event.selection.objects.get("segments", [])
                if r_objs:
                    st.session_state.ratio_sel_id     = r_objs[0].get("newSegmentId")
                    st.session_state.ratio_sel_street = r_objs[0].get("street_label")
            except Exception:
                pass
            ratio_sel_id     = st.session_state.ratio_sel_id
            ratio_sel_street = st.session_state.ratio_sel_street

            if ratio_sel_id:
                # ── Selected segment info at top of chart column ───────────
                _ri, _rclr = st.columns([3, 1])
                _ri.markdown(
                    f"**Selected:** `{ratio_sel_id}`  \n"
                    f"**Street:** {ratio_sel_street or 'None'}"
                )
                if _rclr.button("✕ Clear", key="clr_ratio"):
                    st.session_state.ratio_sel_id     = None
                    st.session_state.ratio_sel_street = None
                    st.session_state.ratio_key_ctr    += 1
                    st.rerun()

                if tod_a == TOD_AVG_ALL:
                    seg_spd_a = df_speed[
                        (df_speed["dow"] == dow_a) &
                        (df_speed["newSegmentId"] == ratio_sel_id)
                    ]["averageSpeed"].mean()
                else:
                    seg_spd_a = df_speed[
                        (df_speed["dow"] == dow_a) &
                        (df_speed["timeSetName"] == tod_a) &
                        (df_speed["newSegmentId"] == ratio_sel_id)
                    ]["averageSpeed"].mean()

                if tod_b == TOD_AVG_ALL:
                    seg_spd_b = df_speed[
                        (df_speed["dow"] == dow_b) &
                        (df_speed["newSegmentId"] == ratio_sel_id)
                    ]["averageSpeed"].mean()
                else:
                    seg_spd_b = df_speed[
                        (df_speed["dow"] == dow_b) &
                        (df_speed["timeSetName"] == tod_b) &
                        (df_speed["newSegmentId"] == ratio_sel_id)
                    ]["averageSpeed"].mean()

                m1, m2 = st.columns(2)
                m1.metric(f"Avg Speed Scenario A ({dow_a} | {tod_a_lbl})", fmt_speed(seg_spd_a))
                m2.metric(f"Avg Speed Scenario B ({dow_b} | {tod_b_lbl})", fmt_speed(seg_spd_b))

                # Comparison chart: segment speed Scenario A vs Scenario B across all TODs
                seg_line_a = (
                    df_speed[
                        (df_speed["dow"] == dow_a) &
                        (df_speed["newSegmentId"] == ratio_sel_id)
                    ]
                    .groupby("timeSetName")["averageSpeed"].mean()
                    .reindex(ALL_TOD).reset_index()
                    .rename(columns={"averageSpeed": "avg_speed"})
                )
                seg_line_b = (
                    df_speed[
                        (df_speed["dow"] == dow_b) &
                        (df_speed["newSegmentId"] == ratio_sel_id)
                    ]
                    .groupby("timeSetName")["averageSpeed"].mean()
                    .reindex(ALL_TOD).reset_index()
                    .rename(columns={"averageSpeed": "avg_speed"})
                )
                # Per-bar colors: full color for selected TOD, dimmed for others
                _ac = [
                    "#E53935" if (tod_a == TOD_AVG_ALL or t == tod_a)
                    else "rgba(229,57,53,0.28)"
                    for t in seg_line_a["timeSetName"]
                ]
                _bc = [
                    "#1E88E5" if (tod_b == TOD_AVG_ALL or t == tod_b)
                    else "rgba(30,136,229,0.28)"
                    for t in seg_line_b["timeSetName"]
                ]
                fig_seg_cmp = go.Figure()
                fig_seg_cmp.add_trace(go.Bar(
                    x=seg_line_a["timeSetName"], y=seg_line_a["avg_speed"],
                    name=f"Scenario A: {dow_a} | {tod_a_lbl}", marker_color=_ac,
                ))
                fig_seg_cmp.add_trace(go.Bar(
                    x=seg_line_b["timeSetName"], y=seg_line_b["avg_speed"],
                    name=f"Scenario B: {dow_b} | {tod_b_lbl}", marker_color=_bc,
                ))
                fig_seg_cmp.update_layout(
                    title=dict(text="Selected Segment — Scenario A vs Scenario B by TOD", font_size=13),
                    barmode="group",
                    xaxis_title="Time of Day",
                    yaxis_title="Avg Speed (mph)",
                    xaxis_tickangle=-45,
                    height=310,
                    margin=dict(t=40, b=70, l=50, r=10),
                    xaxis_gridcolor="#e0e0e0",
                    yaxis_gridcolor="#e0e0e0",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_seg_cmp, use_container_width=True)

            else:
                # ── No segment selected: show systemwide summary ───────────
                valid = ratio_df["ratio"].dropna()
                c1, c2, c3 = st.columns(3)
                c1.metric("Median Ratio", f"{valid.median():.3f}" if len(valid) else "–")
                c2.metric("Mean Ratio",   f"{valid.mean():.3f}"   if len(valid) else "–")
                c3.metric("# Segments",   f"{len(valid):,}")

                n_slower  = int((valid < 0.95).sum())
                n_faster  = int((valid > 1.05).sum())
                n_similar = len(valid) - n_slower - n_faster
                st.markdown(
                    f"| Scenario A slower by >5% | Similar (±5%) | Scenario A faster by >5% |\n"
                    f"|---|---|---|\n"
                    f"| **{n_slower}** segs | **{n_similar}** segs | **{n_faster}** segs |"
                )

                st.markdown("---")
                st.markdown("**Systemwide Avg Speed Comparison**")
                st.caption("Click a segment on the map to see its speed comparison.")

                sys_a = (
                    df_speed[df_speed["dow"] == dow_a]
                    .groupby("timeSetName")["averageSpeed"].mean()
                    .reindex(ALL_TOD).reset_index()
                    .rename(columns={"averageSpeed": "avg_speed"})
                )
                sys_b = (
                    df_speed[df_speed["dow"] == dow_b]
                    .groupby("timeSetName")["averageSpeed"].mean()
                    .reindex(ALL_TOD).reset_index()
                    .rename(columns={"averageSpeed": "avg_speed"})
                )
                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Bar(
                    x=sys_a["timeSetName"], y=sys_a["avg_speed"],
                    name=f"Scenario A: {dow_a}", marker_color="#E53935", opacity=0.8,
                ))
                fig_cmp.add_trace(go.Bar(
                    x=sys_b["timeSetName"], y=sys_b["avg_speed"],
                    name=f"Scenario B: {dow_b}", marker_color="#1E88E5", opacity=0.8,
                ))
                fig_cmp.update_layout(
                    barmode="group",
                    xaxis_title="Time of Day",
                    yaxis_title="Avg Speed (mph)",
                    xaxis_tickangle=-45,
                    height=310,
                    margin=dict(t=10, b=70, l=50, r=10),
                    xaxis_gridcolor="#e0e0e0",
                    yaxis_gridcolor="#e0e0e0",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_cmp, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – GEOFENCING TRIPS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:

    t2f1, t2f2, _ = st.columns([1, 2, 3])
    with t2f1:
        trip_dow = st.selectbox("Day of Week", DOW_OPTIONS, key="trip_dow")
    with t2f2:
        trip_hour = st.selectbox(
            "Hour of Day (for map)",
            options=[None] + list(range(24)),
            format_func=lambda h: "All (sum across hours)" if h is None else HOUR_LABELS[h],
            key="trip_hour",
        )

    st.divider()
    map_col2, chart_col2 = st.columns([3, 2])

    # Aggregate trips for map
    if trip_hour is None:
        trip_map_agg = (
            df_trips[df_trips["dow"] == trip_dow]
            .groupby("destination_region")["daily_trip_wt"]
            .sum().rename("daily_trips").reset_index()
        )
        hour_label_str = "All Hours"
    else:
        trip_map_agg = (
            df_trips[
                (df_trips["dow"] == trip_dow) &
                (df_trips["start_hour"] == trip_hour)
            ]
            .groupby("destination_region")["daily_trip_wt"]
            .sum().rename("daily_trips").reset_index()
        )
        hour_label_str = HOUR_LABELS[trip_hour]

    map_areas = df_areas.merge(
        trip_map_agg,
        left_on="name", right_on="destination_region", how="left",
    )
    map_areas["color"]      = blue_gradient_color(map_areas["daily_trips"])
    map_areas["area_label"] = map_areas["name"].apply(lambda n: str(n))
    map_areas["trip_label"] = map_areas["daily_trips"].apply(fmt_trips)

    # Systemwide hourly aggregate for chart
    trip_hourly_all = (
        df_trips[df_trips["dow"] == trip_dow]
        .groupby("start_hour")["daily_trip_wt"]
        .sum()
        .reindex(range(24), fill_value=0)
        .reset_index()
        .rename(columns={"daily_trip_wt": "daily_trips"})
    )
    total_sys_daily = trip_hourly_all["daily_trips"].sum()

    with map_col2:
        st.subheader("Daily Trips Ending by Area")
        st.markdown(
            "<span style='color:#DCF0FF'>&#9632;</span> Fewer trips &nbsp;&nbsp;"
            "<span style='color:#003FB5'>&#9632;</span> More trips &nbsp;&nbsp;"
            "<span style='color:#A0A0A0'>&#9632;</span> No data",
            unsafe_allow_html=True,
        )
        poly_layer = make_polygon_layer(
            map_areas[["name", "polygon", "color", "area_label", "trip_label"]]
        )
        deck2 = pdk.Deck(
            layers=[poly_layer],
            initial_view_state=VIEW_STATE,
            map_style=MAP_STYLE,
            tooltip={"text": "Area: {area_label}\nDaily Trips: {trip_label}"},
        )
        event2 = st.pydeck_chart(
            deck2,
            on_select="rerun",
            selection_mode="single-object",
            use_container_width=True,
            height=520,
            key=f"map_geofence_{st.session_state.trip_key_ctr}",
        )

    with chart_col2:
        # Persist selection across filter changes
        try:
            objs2 = event2.selection.objects.get("areas", [])
            if objs2:
                st.session_state.trip_sel_area = int(objs2[0].get("name"))
        except Exception:
            pass
        selected_area = st.session_state.trip_sel_area

        # ── Summary / selected info at top ────────────────────────────────
        sel_daily_total = None
        trip_hourly_sel = None
        y_sel = None
        if selected_area is not None:
            trip_hourly_sel = (
                df_trips[
                    (df_trips["dow"] == trip_dow) &
                    (df_trips["destination_region"] == selected_area)
                ]
                .groupby("start_hour")["daily_trip_wt"]
                .sum()
                .reindex(range(24), fill_value=0)
                .reset_index()
                .rename(columns={"daily_trip_wt": "daily_trips"})
            )
            sel_daily_total = trip_hourly_sel["daily_trips"].sum()

            # Volume number first, then ID + Clear
            st.metric("Selected Area Daily Trips", f"{int(round(sel_daily_total)):,}")
            _ai, _aclr = st.columns([3, 1])
            _ai.markdown(f"**Selected: Area {selected_area}**")
            if _aclr.button("✕ Clear", key="clr_trip"):
                st.session_state.trip_sel_area = None
                st.session_state.trip_key_ctr  += 1
                st.rerun()
        else:
            st.metric("Total Daily Trips (All Areas)", f"{int(round(total_sys_daily)):,}")

        # Chart scale toggle — lives next to the chart, not in the filter bar
        trip_scale = st.radio(
            "Chart scale",
            ["Daily Trips", "% of Total Daily Trips"],
            key="trip_scale",
            horizontal=True,
            label_visibility="collapsed",
        )

        # Scale bars
        if trip_scale == "% of Total Daily Trips":
            y_sys = (trip_hourly_all["daily_trips"] / total_sys_daily * 100).fillna(0) \
                    if total_sys_daily > 0 else trip_hourly_all["daily_trips"] * 0
            yaxis_t2 = "% of Daily Trips"
        else:
            y_sys = trip_hourly_all["daily_trips"]
            yaxis_t2 = "Daily Trips"

        if selected_area is not None and sel_daily_total is not None:
            if trip_scale == "% of Total Daily Trips":
                y_sel = (
                    trip_hourly_sel["daily_trips"] / sel_daily_total * 100
                ).fillna(0) if sel_daily_total > 0 else trip_hourly_sel["daily_trips"] * 0
            else:
                y_sel = trip_hourly_sel["daily_trips"]

        fig_all = go.Figure(go.Bar(
            x=HOUR_LABELS, y=y_sys,
            marker_color="#2196F3", opacity=0.85, name="All Areas",
        ))

        if y_sel is not None:
            fig_all.add_trace(
                go.Scatter(
                    x=HOUR_LABELS, y=y_sel,
                    mode="lines+markers",
                    name=f"Area {selected_area}",
                    line=dict(color="#FF9800", width=2),
                    marker=dict(size=6),
                )
            )
            fig_all.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

        fig_all.update_layout(
            title=dict(text="All Areas — Daily Trips by Hour", font_size=13),
            xaxis_title="Hour of Day",
            yaxis_title=yaxis_t2,
            xaxis_tickangle=-45,
            height=310,
            margin=dict(t=40, b=70, l=50, r=10),
            xaxis=dict(gridcolor="#e0e0e0"),
            yaxis_gridcolor="#e0e0e0",
        )
        st.plotly_chart(fig_all, use_container_width=True)

        if selected_area is not None and y_sel is not None:
            # Hourly profile chart for selected area
            fig_sel_area = go.Figure(go.Bar(
                x=HOUR_LABELS, y=y_sel,
                marker_color="#FF9800", opacity=0.85, name=f"Area {selected_area}",
            ))
            fig_sel_area.update_layout(
                title=dict(
                    text=f"Area {selected_area} — Daily Trips by Hour",
                    font_size=13,
                ),
                xaxis_title="Hour of Day",
                yaxis_title=yaxis_t2,
                xaxis_tickangle=-45,
                height=280,
                margin=dict(t=40, b=70, l=50, r=10),
                xaxis=dict(gridcolor="#e0e0e0"),
                yaxis_gridcolor="#e0e0e0",
            )
            st.plotly_chart(fig_sel_area, use_container_width=True)
        else:
            st.info("Click an area on the map to see its hourly trip profile.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – TRAFFIC VOLUMES
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:

    v3f1, v3f2, _ = st.columns([1, 2, 3])
    with v3f1:
        vol_dow = st.selectbox("Day of Week", DOW_OPTIONS, key="vol_dow")
    with v3f2:
        vol_tod = st.selectbox(
            "Time of Day (as shown on map)",
            [TOD_SUM_ALL] + ALL_AADT_TOD,
            key="vol_tod",
        )

    st.divider()
    map_col3, chart_col3 = st.columns([3, 2])

    # ── Build map data from original AADT dataframe ────────────────────────────
    vol_base = df_aadt[df_aadt["dow"] == vol_dow].copy()

    if vol_tod == TOD_SUM_ALL:
        seg_vol = (
            vol_base.groupby("newSegmentId")["aadt"]
            .sum().rename("displayVol").reset_index()
        )
    else:
        seg_vol = (
            vol_base[vol_base["timeSetName"] == vol_tod]
            .groupby("newSegmentId")["aadt"]
            .sum().rename("displayVol").reset_index()
        )

    vol_map_df = (
        df_geom
        .merge(df_seg_attr, on="newSegmentId", how="left")
        .merge(seg_vol,     on="newSegmentId", how="left")
    )
    vol_map_df["color"]        = blue_gradient_color(vol_map_df["displayVol"], alpha=240)
    vol_map_df["vol_label"]    = vol_map_df["displayVol"].apply(fmt_aadt)
    vol_map_df["frc_label"]    = vol_map_df["frc"].apply(fmt_frc)
    vol_map_df["street_label"] = vol_map_df["street"].apply(fmt_str)

    valid_vol = vol_map_df["displayVol"].dropna()
    vol_vmin = float(valid_vol.min()) if len(valid_vol) else 0.0
    vol_vmax = float(valid_vol.max()) if len(valid_vol) else 1.0

    with map_col3:
        st.subheader("Daily Volumes")
        cb_vol = make_volume_colorbar(vol_vmin, vol_vmax)
        if cb_vol:
            st.plotly_chart(cb_vol, use_container_width=True,
                            config={"displayModeBar": False})

        vol_layer = make_layer(
            vol_map_df[[
                "newSegmentId", "path", "color",
                "vol_label", "street_label", "frc_label",
            ]]
        )
        vol_deck = pdk.Deck(
            layers=[vol_layer],
            initial_view_state=VIEW_STATE,
            map_style=MAP_STYLE,
            tooltip={
                "text": (
                    "Segment ID: {newSegmentId}\n"
                    "Street: {street_label}\n"
                    "FRC: {frc_label}\n"
                    "Volume: {vol_label}"
                )
            },
        )
        vol_event = st.pydeck_chart(
            vol_deck,
            on_select="rerun",
            selection_mode="single-object",
            use_container_width=True,
            height=470,
            key=f"map_vol_{st.session_state.vol_key_ctr}",
        )

    with chart_col3:
        # Persist selection across filter changes
        try:
            v_objs = vol_event.selection.objects.get("segments", [])
            if v_objs:
                st.session_state.vol_sel_id     = v_objs[0].get("newSegmentId")
                st.session_state.vol_sel_street = v_objs[0].get("street_label")
        except Exception:
            pass
        vol_sel_id     = st.session_state.vol_sel_id
        vol_sel_street = st.session_state.vol_sel_street

        if vol_sel_id:
            # ── Selected segment: aggregate by original TOD bins ──────────
            seg_vol_by_tod = (
                vol_base[vol_base["newSegmentId"] == vol_sel_id]
                .groupby("timeSetName")["aadt"]
                .sum()
                .reindex(ALL_AADT_TOD).reset_index()
                .rename(columns={"aadt": "seg_vol"})
            )

            if vol_tod == TOD_SUM_ALL:
                sel_display_vol = seg_vol_by_tod["seg_vol"].sum()
            else:
                _row = seg_vol_by_tod[seg_vol_by_tod["timeSetName"] == vol_tod]
                sel_display_vol = _row["seg_vol"].values[0] if len(_row) else np.nan

            # ── Volume metric first, then segment ID + Clear ──────────────
            tod_suffix = vol_tod if vol_tod != TOD_SUM_ALL else "Daily Total"
            st.metric(f"{tod_suffix} Volume", fmt_aadt(sel_display_vol))
            _vi3, _vclr3 = st.columns([3, 1])
            _vi3.markdown(
                f"**Selected:** `{vol_sel_id}` — **{vol_sel_street or 'None'}**"
            )
            if _vclr3.button("✕ Clear", key="clr_vol"):
                st.session_state.vol_sel_id     = None
                st.session_state.vol_sel_street = None
                st.session_state.vol_key_ctr    += 1
                st.rerun()

            # ── Bar chart: avg hourly for multi-hour bins; 0:xx last ──────
            seg_vol_bar = seg_vol_by_tod.copy()
            seg_vol_bar["n_hours"]      = seg_vol_bar["timeSetName"].apply(get_bin_hours)
            seg_vol_bar["seg_vol_norm"] = seg_vol_bar["seg_vol"] / seg_vol_bar["n_hours"]
            # Sort: bins starting ≥7 first (ascending), then bins starting <7 at end
            seg_vol_bar["_sort"] = seg_vol_bar["timeSetName"].apply(
                lambda lbl: (int(lbl.split(":")[0]) + 24)
                if int(lbl.split(":")[0]) < 7 else int(lbl.split(":")[0])
            )
            seg_vol_bar = seg_vol_bar.sort_values("_sort")

            fig_vol_seg = go.Figure(go.Bar(
                x=seg_vol_bar["timeSetName"],
                y=seg_vol_bar["seg_vol_norm"],
                marker_color="#2196F3", opacity=0.85, name="Avg Hourly Volume",
            ))
            fig_vol_seg.update_layout(
                title=dict(text="Selected Segment — Avg Hourly Volume by TOD", font_size=13),
                xaxis_title="Time of Day",
                yaxis_title="Avg Hourly Volume",
                xaxis_tickangle=-45,
                height=310,
                margin=dict(t=40, b=70, l=50, r=10),
                xaxis=dict(gridcolor="#e0e0e0", categoryorder="array",
                           categoryarray=list(seg_vol_bar["timeSetName"])),
                yaxis_gridcolor="#e0e0e0",
            )
            st.plotly_chart(fig_vol_seg, use_container_width=True)

            # ── Datatable: actual totals per TOD bin ──────────────────────
            seg_vol_table = seg_vol_by_tod.rename(
                columns={"timeSetName": "Time of Day", "seg_vol": "Volume"}
            ).copy()
            seg_vol_table["Volume"] = seg_vol_table["Volume"].apply(
                lambda v: int(round(v)) if pd.notna(v) else None
            )
            st.markdown("**Selected Segment — Volume by Time of Day**")
            st.dataframe(
                seg_vol_table,
                use_container_width=True,
                hide_index=True,
                height=min(420, 38 + len(seg_vol_table) * 35),
            )
        else:
            st.info("Click a segment on the map to see its volume profile.")
