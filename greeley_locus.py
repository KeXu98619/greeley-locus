"""
greeley_road_performance.py
Streamlit app – City of Greeley Transportation Dashboard

Data inputs (same folder as this script):
  - dow_tod_segments_speed&sample.parquet          (speed + attributes, no geometry)
  - segments_geometry.geojson                      (unique segment geometries)
  - trip_summary_with_expansion_area.parquet       (OD trip summary by area & hour)
  - region_geometry.geojson                        (area polygon geometries)

Run:  streamlit run greeley_road_performance.py
Requires: streamlit>=1.40, pydeck, plotly, geopandas, pandas, numpy, pyarrow
"""

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import pydeck as pdk
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Greeley Transportation Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",   # sidebar unused; keep it out of the way
)

# ── File paths ────────────────────────────────────────────────────────────────
PARQUET_PATH      = "dow_tod_segments_speed&sample.parquet"
GEOM_PATH         = "segments_geometry.geojson"
TRIP_PARQUET_PATH = "trip_summary_with_expansion_area.parquet"
REGION_GEOM_PATH  = "region_geometry.geojson"

# ── Map defaults ──────────────────────────────────────────────────────────────
VIEW_STATE  = pdk.ViewState(latitude=40.420376, longitude=-104.693693, zoom=14, pitch=0)
MAP_STYLE   = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
DOW_OPTIONS = ["Mon-Thur", "Fri", "Sat", "Sun"]
HOUR_LABELS = [f"{h}:00–{h+1}:00" for h in range(24)]


# ── Helpers ───────────────────────────────────────────────────────────────────
def tod_sort_key(label: str) -> int:
    """Sort time-of-day labels by start hour: '7:00-8:00' → 7, '0:00-7:00' → 0."""
    try:
        return int(label.split(":")[0])
    except Exception:
        return -1


def geom_to_path(geom):
    """Convert a Shapely LineString geometry to pydeck-compatible [[lon, lat], ...] list."""
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
    """Convert a Shapely Polygon geometry to pydeck-compatible [[lon, lat], ...] list."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return [[round(x, 6), round(y, 6)] for x, y in geom.exterior.coords]
    if geom.geom_type == "MultiPolygon":
        largest = max(geom.geoms, key=lambda p: p.area)
        return [[round(x, 6), round(y, 6)] for x, y in largest.exterior.coords]
    return None


def speed_to_color(series: pd.Series) -> list:
    """Red (slow) → Yellow → Green (fast). Gray for NaN."""
    vmin = series.min()
    vmax = series.max()
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
    """Blue = A slower (ratio < 1), Red = A faster (ratio > 1). Gray = no data."""
    out = []
    for v in series:
        if pd.isna(v):
            out.append([160, 160, 160, 120])
        elif v > 1.0:
            intensity = min((v - 1.0) / 0.5, 1.0)
            out.append([int(50 + 205 * intensity), 50, 50, 230])
        else:
            intensity = min((1.0 - v) / 0.5, 1.0)
            out.append([50, 50, int(50 + 205 * intensity), 230])
    return out


def trips_to_color(series: pd.Series) -> list:
    """Light blue (few trips) → Dark blue (many trips). Gray for NaN (No Data)."""
    valid = series.dropna()
    if len(valid) == 0:
        return [[160, 160, 160, 120]] * len(series)
    vmin = valid.min()
    vmax = valid.max()
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
            out.append([r, g, b, 210])
    return out


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


# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_speed() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_PATH)
    df["averageSpeed"] = df["averageSpeed"].where(df["averageSpeed"] > 0, other=np.nan)
    return df


@st.cache_data
def load_geometry() -> pd.DataFrame:
    """Returns a plain DataFrame with 'path' column for pydeck."""
    gdf = gpd.read_file(GEOM_PATH).to_crs(epsg=4326)
    gdf["path"] = gdf.geometry.apply(geom_to_path)
    keep = [c for c in ["newSegmentId", "segmentId", "path"] if c in gdf.columns]
    return gdf[keep].dropna(subset=["path"]).reset_index(drop=True)


@st.cache_data
def load_segment_attrs() -> pd.DataFrame:
    """Unique segment-level attributes (street name, FRC) from the parquet."""
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
    """Returns a plain DataFrame with 'polygon' column for pydeck, areas 1–29 only."""
    gdf = gpd.read_file(REGION_GEOM_PATH).to_crs(epsg=4326)
    gdf["name"] = gdf["name"].astype(int)
    gdf = gdf[gdf["name"] <= 29].copy()
    gdf["polygon"] = gdf.geometry.apply(geom_to_polygon)
    return gdf[["name", "polygon"]].dropna(subset=["polygon"]).reset_index(drop=True)


# ── Load all data ─────────────────────────────────────────────────────────────
try:
    df_speed    = load_speed()
    df_geom     = load_geometry()
    df_seg_attr = load_segment_attrs()
    df_trips    = load_trips()
    df_areas    = load_area_geometry()
except FileNotFoundError as e:
    st.error(
        f"Data file not found: {e}\n"
        "Make sure all data files are in the same folder as this script."
    )
    st.stop()

ALL_TOD = sorted(df_speed["timeSetName"].unique(), key=tod_sort_key)


# ── Shared tooltip helpers ────────────────────────────────────────────────────
def fmt_speed(v):
    return f"{int(round(v))} mph" if pd.notna(v) else "No data"

def fmt_frc(v):
    return str(int(v)) if pd.notna(v) else "None"

def fmt_str(v):
    return str(v) if (v is not None and pd.notna(v)) else "None"

def fmt_trips(v):
    return f"{int(round(v))} trips/day" if pd.notna(v) else "No Data"


# ── Tabs ──────────────────────────────────────────────────────────────────────
st.title("City of Greeley – Transportation Dashboard")
tab1, tab2 = st.tabs(["Road Performance", "Geofencing Trips"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – ROAD PERFORMANCE
# All filters live inside this tab so Tab 2 is completely self-contained.
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Inline filter row ─────────────────────────────────────────────────────
    col_metric, col_filters = st.columns([1, 4])
    with col_metric:
        metric = st.radio("**Metric**", ["Average Speed", "Speed Ratio"])

    with col_filters:
        if metric == "Average Speed":
            fa, fb, _ = st.columns([1, 2, 1])
            dow = fa.selectbox("Day of Week", DOW_OPTIONS)
            tod = fb.selectbox("Time of Day (map)", ["All (avg across hours)"] + ALL_TOD)
        else:
            fa, fb, fc, fd = st.columns(4)
            dow_a = fa.selectbox("DOW – A", DOW_OPTIONS, key="dow_a",
                                  index=DOW_OPTIONS.index("Mon-Thur"))
            tod_a = fb.selectbox("TOD – A", ALL_TOD, key="tod_a",
                                  index=ALL_TOD.index("8:00-9:00") if "8:00-9:00" in ALL_TOD else 0)
            dow_b = fc.selectbox("DOW – B", DOW_OPTIONS, key="dow_b",
                                  index=DOW_OPTIONS.index("Sat"))
            tod_b = fd.selectbox("TOD – B", ALL_TOD, key="tod_b",
                                  index=ALL_TOD.index("8:00-9:00") if "8:00-9:00" in ALL_TOD else 0)
            st.caption("Ratio = Avg Speed (A) ÷ Avg Speed (B)")

    st.divider()
    map_col, chart_col = st.columns([3, 2])

    # ── AVERAGE SPEED MODE ────────────────────────────────────────────────────
    if metric == "Average Speed":

        base = df_speed[df_speed["dow"] == dow].copy()

        if tod == "All (avg across hours)":
            seg_speed = (
                base.groupby("newSegmentId")["averageSpeed"]
                .mean().rename("displaySpeed").reset_index()
            )
            map_title = f"Avg Speed — {dow}  |  All Hours"
        else:
            seg_speed = (
                base[base["timeSetName"] == tod]
                .groupby("newSegmentId")["averageSpeed"]
                .mean().rename("displaySpeed").reset_index()
            )
            map_title = f"Avg Speed — {dow}  |  {tod}"

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

        with map_col:
            st.subheader(map_title)
            st.markdown(
                "<span style='color:#FF2200'>&#9632;</span> Slow &nbsp;&nbsp;"
                "<span style='color:#FFB300'>&#9632;</span> Moderate &nbsp;&nbsp;"
                "<span style='color:#23DC3B'>&#9632;</span> Fast &nbsp;&nbsp;"
                "<span style='color:#A0A0A0'>&#9632;</span> No data",
                unsafe_allow_html=True,
            )
            layer = make_layer(
                map_df[[
                    "newSegmentId", "path", "color",
                    "speed_label", "street_label", "frc_label",
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
                        "Avg Speed: {speed_label}"
                    )
                },
            )
            event = st.pydeck_chart(
                deck,
                on_select="rerun",
                selection_mode="single-object",
                use_container_width=True,
                height=520,
            )

        with chart_col:
            selected_id     = None
            selected_street = None
            try:
                objs = event.selection.objects.get("segments", [])
                if objs:
                    selected_id     = objs[0].get("newSegmentId")
                    selected_street = objs[0].get("street_label")
            except Exception:
                pass

            fig_sys = bar_chart(
                x=sys_avg["timeSetName"],
                y=sys_avg["avg_speed"],
                title=f"Systemwide Avg Speed — {dow}",
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
                st.markdown(
                    f"**Selected:** `{selected_id}`  \n"
                    f"**Street:** {selected_street or 'None'}"
                )
                seg_tod = (
                    base[base["newSegmentId"] == selected_id]
                    .groupby("timeSetName")["averageSpeed"].mean()
                    .reindex(ALL_TOD).reset_index()
                    .rename(columns={"averageSpeed": "avg_speed"})
                )
                fig_seg = bar_chart(
                    x=seg_tod["timeSetName"],
                    y=seg_tod["avg_speed"],
                    title="Selected Segment — Avg Speed by Time of Day",
                    color="#FF9800",
                )
                st.plotly_chart(fig_seg, use_container_width=True)
            else:
                st.info("Click a segment on the map to see its time-of-day speed profile.")

    # ── SPEED RATIO MODE ──────────────────────────────────────────────────────
    else:
        speed_a = (
            df_speed[(df_speed["dow"] == dow_a) & (df_speed["timeSetName"] == tod_a)]
            .groupby("newSegmentId")["averageSpeed"].mean().rename("speed_a")
        )
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
        map_df["ratio_label"]  = map_df["ratio"].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "No data")
        map_df["spd_a_label"]  = map_df["speed_a"].apply(fmt_speed)
        map_df["spd_b_label"]  = map_df["speed_b"].apply(fmt_speed)
        map_df["frc_label"]    = map_df["frc"].apply(fmt_frc)
        map_df["street_label"] = map_df["street"].apply(fmt_str)

        with map_col:
            st.subheader(f"Speed Ratio — ({dow_a} / {tod_a})  ÷  ({dow_b} / {tod_b})")
            st.markdown(
                "<span style='color:#3232FF'>&#9632;</span> A slower (ratio &lt; 1) &nbsp;&nbsp;"
                "<span style='color:#A0A0A0'>&#9632;</span> No data &nbsp;&nbsp;"
                "<span style='color:#FF3232'>&#9632;</span> A faster (ratio &gt; 1)",
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
                        "Ratio (A/B): {ratio_label}\n"
                        f"Speed A ({dow_a} {tod_a}): " + "{spd_a_label}\n"
                        f"Speed B ({dow_b} {tod_b}): " + "{spd_b_label}"
                    )
                },
            )
            st.pydeck_chart(deck, use_container_width=True, height=520)

        with chart_col:
            valid = ratio_df["ratio"].dropna()

            c1, c2, c3 = st.columns(3)
            c1.metric("Median Ratio", f"{valid.median():.3f}" if len(valid) else "–")
            c2.metric("Mean Ratio",   f"{valid.mean():.3f}"   if len(valid) else "–")
            c3.metric("# Segments",   f"{len(valid):,}")

            n_slower  = int((valid < 0.95).sum())
            n_faster  = int((valid > 1.05).sum())
            n_similar = len(valid) - n_slower - n_faster
            st.markdown(
                f"| A slower by >5% | Similar (±5%) | A faster by >5% |\n"
                f"|---|---|---|\n"
                f"| **{n_slower}** segs | **{n_similar}** segs | **{n_faster}** segs |"
            )

            st.markdown("---")
            st.markdown("**Systemwide Avg Speed Comparison**")

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
                name=f"A: {dow_a}", marker_color="#E53935", opacity=0.8,
            ))
            fig_cmp.add_trace(go.Bar(
                x=sys_b["timeSetName"], y=sys_b["avg_speed"],
                name=f"B: {dow_b}", marker_color="#1E88E5", opacity=0.8,
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
# TAB 2 – Geofencing Trips
# Completely self-contained; no sidebar controls involved.
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:

    # ── Inline filters ────────────────────────────────────────────────────────
    t2f1, t2f2, _ = st.columns([1, 2, 2])
    with t2f1:
        trip_dow = st.selectbox("Day of Week", DOW_OPTIONS, key="trip_dow")
    with t2f2:
        # None = "All (sum across hours)" sentinel
        trip_hour = st.selectbox(
            "Hour of Day (map)",
            options=[None] + list(range(24)),
            format_func=lambda h: "All (sum across hours)" if h is None else HOUR_LABELS[h],
            key="trip_hour",
        )

    st.divider()
    map_col2, chart_col2 = st.columns([3, 2])

    # ── Aggregate trips for map ────────────────────────────────────────────────
    if trip_hour is None:
        # Sum across all 24 hours
        trip_map_agg = (
            df_trips[df_trips["dow"] == trip_dow]
            .groupby("destination_region")["daily_trip_wt"]
            .sum()
            .rename("daily_trips")
            .reset_index()
        )
        hour_label_str = "All Hours"
    else:
        trip_map_agg = (
            df_trips[
                (df_trips["dow"] == trip_dow) &
                (df_trips["start_hour"] == trip_hour)
            ]
            .groupby("destination_region")["daily_trip_wt"]
            .sum()
            .rename("daily_trips")
            .reset_index()
        )
        hour_label_str = HOUR_LABELS[trip_hour]

    # Left-join keeps all 29 areas (missing ones → NaN → "No Data")
    map_areas = df_areas.merge(
        trip_map_agg,
        left_on="name",
        right_on="destination_region",
        how="left",
    )
    map_areas["color"]      = trips_to_color(map_areas["daily_trips"])
    map_areas["area_label"] = map_areas["name"].apply(lambda n: str(n))
    map_areas["trip_label"] = map_areas["daily_trips"].apply(fmt_trips)

    # ── Aggregate trips for bar chart: all hours, selected dow ────────────────
    trip_hourly_all = (
        df_trips[df_trips["dow"] == trip_dow]
        .groupby("start_hour")["daily_trip_wt"]
        .sum()
        .reindex(range(24), fill_value=0)
        .reset_index()
        .rename(columns={"daily_trip_wt": "daily_trips"})
    )

    # ── MAP ───────────────────────────────────────────────────────────────────
    with map_col2:
        st.subheader(f"Daily Trips Ending by Area — {trip_dow}  |  {hour_label_str}")
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
        )

    # ── CHARTS ────────────────────────────────────────────────────────────────
    with chart_col2:
        # Parse map click
        selected_area = None
        try:
            objs2 = event2.selection.objects.get("areas", [])
            if objs2:
                selected_area = int(objs2[0].get("name"))
        except Exception:
            pass

        # All-areas hourly bar (always shown)
        fig_all = bar_chart(
            x=HOUR_LABELS,
            y=trip_hourly_all["daily_trips"],
            title=f"All Areas — Daily Trips by Hour ({trip_dow})",
            color="#2196F3",
            yaxis_title="Daily Trips",
            xaxis_title="Hour of Day",
            bar_name="All Areas",
        )

        # If an area is selected, overlay its hourly profile as a line
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
            fig_all.add_trace(
                go.Scatter(
                    x=HOUR_LABELS,
                    y=trip_hourly_sel["daily_trips"],
                    mode="lines+markers",
                    name=f"Area {selected_area}",
                    line=dict(color="#FF9800", width=2),
                    marker=dict(size=6),
                )
            )
            fig_all.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

        st.plotly_chart(fig_all, use_container_width=True)

        # Selected area detail bar chart
        if selected_area is not None:
            st.markdown(f"**Selected:** Area {selected_area}")
            fig_sel = bar_chart(
                x=HOUR_LABELS,
                y=trip_hourly_sel["daily_trips"],
                title=f"Area {selected_area} — Daily Trips by Hour ({trip_dow})",
                color="#FF9800",
                yaxis_title="Daily Trips",
                xaxis_title="Hour of Day",
                bar_name=f"Area {selected_area}",
            )
            st.plotly_chart(fig_sel, use_container_width=True)
        else:
            st.info("Click an area on the map to see its hourly trip profile.")
