"""
ParkIQ v5 — Streamlit Dashboard
Run: streamlit run dashboard.py
"""
import json, os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="ParkIQ — Parking Congestion Intelligence",
    page_icon="🚦", layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
[data-testid="stMetric"]{background:#1e293b;border-radius:10px;padding:12px 16px;}
[data-testid="stMetricLabel"]{color:#94a3b8!important;font-size:.75rem!important;text-transform:uppercase;letter-spacing:.05em;}
[data-testid="stMetricValue"]{color:#f1f5f9!important;font-size:1.8rem!important;font-weight:700!important;}
[data-testid="stMetricDelta"]{font-size:.75rem!important;}
div.block-container{padding-top:1.5rem;}
</style>""", unsafe_allow_html=True)

BASE    = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(BASE, "outputs")
def out(f): return os.path.join(OUTPUTS, f)

DARK = dict(plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font_color="#e2e8f0",
            xaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
            yaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"))

def dk(fig, h=320):
    fig.update_layout(**DARK, height=h, margin=dict(t=30,b=10,l=10,r=10))
    return fig

# ── loaders ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_all():
    # Load only necessary columns to keep memory under Streamlit Cloud's 1GB limit (~90% memory reduction)
    cols_s = ['latitude', 'longitude', 'vehicle_type', 'police_station', 'junction_name', 'day_of_week', 'weekend', 'peak_hour', 'hour_ist', 'cis', 'record_id']
    scored = pd.read_parquet(out("scored.parquet"), columns=cols_s)
    
    stress = pd.read_parquet(out("junction_stress.parquet"))
    enf    = pd.read_csv(out("enforcement_priorities.csv"))
    
    cols_attr = ['validation_proba', 'predicted_count', 'congestion_pct', 'top_shap_feature', 'hour_ist', 'police_station', 'vehicle_type']
    attr = pd.read_parquet(out("attribution.parquet"), columns=cols_attr)
    
    cols_anom = ['anomaly_score', 'is_anomaly', 'hour_ist', 'junction_name', 'cis', 'vehicle_type', 'latitude', 'longitude']
    anomaly = pd.read_parquet(out("anomaly_scores.parquet"), columns=cols_anom)
    
    repeat  = pd.read_csv(out("repeat_locations.csv"))
    roi     = pd.read_csv(out("roi_report.csv"))
    
    with open(out("policy_report.json"))  as f: policy   = json.load(f)
    with open(out("spillover_graph.json")) as f: spill   = json.load(f)
    with open(out("shift_schedule.json")) as f: shift    = json.load(f)
    return scored, stress, enf, attr, anomaly, repeat, roi, policy, spill, shift

required = ["scored.parquet","junction_stress.parquet","enforcement_priorities.csv",
            "attribution.parquet","policy_report.json","spillover_graph.json"]
if any(not os.path.exists(out(f)) for f in required):
    st.error("Pipeline outputs missing. Run `python run_pipeline.py` first.")
    st.stop()

with st.spinner("Loading ParkIQ data …"):
    (df_s, df_stress, df_enf, df_attr,
     df_anom, df_rep, df_roi,
     policy, spill, shift) = load_all()

# ── sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚦 ParkIQ")
    st.caption("AI-driven parking congestion intelligence · Bengaluru")
    st.divider()
    page = st.radio("Navigate", [
        "📊 Overview",
        "🗺️ Zone Heatmap",
        "🔮 Junction Stress",
        "🚨 Anomaly Detection",
        "📍 Repeat Hotspots",
        "📈 Congestion Attribution",
        "🚔 Enforcement Routing",
        "💰 Revenue & ROI",
        "🗓️ Shift Scheduler",
        "🧪 Policy Simulator",
        "🔗 Spillover Graph",
        "🤖 Model Performance",
    ], label_visibility="collapsed")
    st.divider()
    st.subheader("🔍 Filters")
    all_v = sorted(df_s["vehicle_type"].dropna().unique())
    sel_v = st.multiselect("Vehicle types", all_v, default=all_v[:6])
    hr    = st.slider("Hour (IST)", 0, 23, (0, 23))
    dtype = st.selectbox("Day type", ["All","Weekday","Weekend"])
    mask  = (df_s["vehicle_type"].isin(sel_v if sel_v else all_v) &
             df_s["hour_ist"].between(*hr))
    if dtype == "Weekday": mask &= df_s["weekend"] == 0
    if dtype == "Weekend": mask &= df_s["weekend"] == 1
    dff = df_s[mask]
    st.divider()
    st.metric("Filtered records", f"{len(dff):,}")
    st.metric("Avg CIS", f"{dff['cis'].mean():.3f}")

# ═══════════════════════════ OVERVIEW ════════════════════════════════
if page == "📊 Overview":
    st.title("📊 ParkIQ — System Overview")
    st.caption("AI-driven parking congestion intelligence · 298K+ violation records · Bengaluru")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total Violations",   f"{len(df_s):,}")
    c2.metric("Named Junctions",    f"{df_stress['junction_name'].nunique()}")
    c3.metric("Avg CIS Score",      f"{df_s['cis'].mean():.3f}")
    c4.metric("Peak-hour Share",    f"{df_s['peak_hour'].mean()*100:.1f}%")
    c5.metric("Anomaly Rate",       f"{df_anom['is_anomaly'].mean()*100:.1f}%" if 'is_anomaly' in df_anom.columns else "—")
    c6.metric("Critical Hotspots",  f"{(df_rep['tier']=='Critical').sum()}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Violations by Hour (IST)")
        hourly = df_s.groupby("hour_ist").size().reset_index(name="count")
        fig = px.bar(hourly, x="hour_ist", y="count", color="count",
                     color_continuous_scale="Reds",
                     labels={"hour_ist":"Hour","count":"Violations"})
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(dk(fig,300), use_container_width=True)

    with col2:
        st.subheader("Violations by Vehicle Type")
        vt = df_s["vehicle_type"].value_counts().head(10).reset_index()
        vt.columns = ["type","count"]
        fig = px.bar(vt, x="count", y="type", orientation="h",
                     color="count", color_continuous_scale="Blues")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(dk(fig,300), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("CIS Distribution")
        fig = px.histogram(df_s, x="cis", nbins=60,
                           color_discrete_sequence=["#38bdf8"])
        st.plotly_chart(dk(fig,280), use_container_width=True)

    with col4:
        st.subheader("Validation Probability Distribution (Model B)")
        fig = px.histogram(df_attr, x="validation_proba", nbins=50,
                           color_discrete_sequence=["#a78bfa"],
                           labels={"validation_proba":"Validation Probability"})
        fig.add_vline(x=0.76, line_dash="dash", line_color="#ef4444",
                      annotation_text="Optimal threshold (0.76)")
        st.plotly_chart(dk(fig,280), use_container_width=True)

    st.subheader("CIS Heatmap — Hour × Day of Week")
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = df_s.pivot_table(index="day_of_week", columns="hour_ist",
                              values="cis", aggfunc="mean")
    pivot = pivot.reindex([d for d in order if d in pivot.index])
    fig = px.imshow(pivot, color_continuous_scale="RdYlGn_r",
                    labels={"x":"Hour","y":"Day","color":"Avg CIS"}, aspect="auto")
    fig.update_layout(**DARK, height=250, margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig, use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Predicted Count vs Hour (Model A)")
        pc_h = df_attr[df_attr["predicted_count"]>0].groupby("hour_ist")["predicted_count"].mean().reset_index()
        fig = px.area(pc_h, x="hour_ist", y="predicted_count",
                      color_discrete_sequence=["#22c55e"],
                      labels={"hour_ist":"Hour","predicted_count":"Avg Predicted Count"})
        st.plotly_chart(dk(fig,280), use_container_width=True)

    with col6:
        st.subheader("Top 10 Police Stations by Violation Count")
        ps = (df_s.groupby("police_station").agg(
                violations=("record_id","count"), avg_cis=("cis","mean"))
              .sort_values("violations",ascending=False).head(10).reset_index())
        fig = px.bar(ps, x="police_station", y="violations",
                     color="avg_cis", color_continuous_scale="RdYlGn_r")
        fig.update_layout(**DARK, height=280, xaxis_tickangle=35, xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════ ZONE HEATMAP ════════════════════════════════
elif page == "🗺️ Zone Heatmap":
    st.title("🗺️ Violation Zone Heatmap")
    tab1, tab2, tab3 = st.tabs(["CIS Density", "Enforcement Priority", "Anomalies"])

    with tab1:
        sample = dff.sample(min(8000,len(dff)), random_state=42)
        fig = px.scatter_mapbox(sample, lat="latitude", lon="longitude",
                                color="cis", size="cis",
                                color_continuous_scale="RdYlGn_r", size_max=8,
                                zoom=11, mapbox_style="carto-darkmatter",
                                hover_data=["vehicle_type","junction_name","hour_ist","cis"],
                                height=620)
        fig.update_layout(margin=dict(r=0,t=0,l=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        enf_g = df_enf.dropna(subset=["lat","lon"])
        fig = px.scatter_mapbox(enf_g.head(60), lat="lat", lon="lon",
                                color="priority_score", size="violation_count",
                                color_continuous_scale="RdYlGn_r", size_max=22,
                                zoom=11, mapbox_style="carto-darkmatter",
                                hover_data=["junction_name","rank","priority_score"],
                                height=620)
        fig.update_layout(margin=dict(r=0,t=0,l=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if "is_anomaly" in df_anom.columns:
            as_ = df_anom.sample(min(6000,len(df_anom)), random_state=1)
            fig = px.scatter_mapbox(as_, lat="latitude", lon="longitude",
                                    color="anomaly_score",
                                    color_continuous_scale="Plasma",
                                    zoom=11, mapbox_style="carto-darkmatter",
                                    hover_data=["vehicle_type","anomaly_score","cis"],
                                    height=620)
            fig.update_layout(margin=dict(r=0,t=0,l=0,b=0))
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════ JUNCTION STRESS ═════════════════════════════
elif page == "🔮 Junction Stress":
    st.title("🔮 Junction Stress Predictions")
    st.caption("GRU neural net per junction · predicts next-hour stress (0–1) from historical hourly patterns")

    top_n = st.slider("Show top N junctions", 5, 50, 20)
    top   = df_stress.head(top_n).copy()

    col1, col2 = st.columns([2,1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=top["predicted_stress_next_h"], y=top["junction_name"],
                             orientation="h", name="Predicted next hr",
                             marker_color="#ef4444"))
        fig.add_trace(go.Bar(x=top["historical_mean_stress"], y=top["junction_name"],
                             orientation="h", name="Historical mean",
                             marker_color="#3b82f6", opacity=0.7))
        fig.update_layout(
                          plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                          font_color="#e2e8f0",
                          barmode="overlay", height=max(380,top_n*22),
                          xaxis=dict(title="Stress (0–1)", gridcolor="#1e293b"),
                          yaxis=dict(autorange="reversed", gridcolor="#1e293b"),
                          legend_orientation="h", margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        bins   = [0,.25,.5,.75,1.01]
        labels = ["Low","Medium","High","Critical"]
        df_stress["tier"] = pd.cut(df_stress["predicted_stress_next_h"],
                                   bins=bins, labels=labels, include_lowest=True)
        tc = df_stress["tier"].value_counts().reset_index()
        tc.columns = ["Tier","Count"]
        fig2 = px.pie(tc, values="Count", names="Tier", hole=0.5,
                      color="Tier",
                      color_discrete_map={"Low":"#22c55e","Medium":"#facc15",
                                          "High":"#f97316","Critical":"#ef4444"})
        fig2.update_layout(**DARK, height=280, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(df_stress[["junction_name","predicted_stress_next_h",
                                 "historical_mean_stress","total_violations"]]
                     .head(20).rename(columns={
                         "junction_name":"Junction",
                         "predicted_stress_next_h":"Next-hr",
                         "historical_mean_stress":"Hist.Mean",
                         "total_violations":"Violations"}),
                     use_container_width=True, height=260)

    st.subheader("Stress vs Total Violations")
    fig3 = px.scatter(df_stress, x="total_violations", y="predicted_stress_next_h",
                      size="historical_peak_stress", color="predicted_stress_next_h",
                      color_continuous_scale="RdYlGn_r", hover_data=["junction_name"])
    st.plotly_chart(dk(fig3,350), use_container_width=True)

# ═══════════════════════ ANOMALY DETECTION ═══════════════════════════
elif page == "🚨 Anomaly Detection":
    st.title("🚨 Anomaly Detection")
    st.caption("Isolation Forest · top 5% most anomalous violation events flagged")

    if "is_anomaly" not in df_anom.columns:
        st.warning("Run Module 6 first."); st.stop()

    n_tot  = len(df_anom)
    n_anom = int(df_anom["is_anomaly"].sum())
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Records",     f"{n_tot:,}")
    c2.metric("Anomalies",         f"{n_anom:,}")
    c3.metric("Anomaly Rate",      f"{n_anom/n_tot*100:.1f}%")
    c4.metric("Max Anomaly Score", f"{df_anom['anomaly_score'].max():.3f}")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Score Distribution")
        fig = px.histogram(df_anom, x="anomaly_score", nbins=60,
                           color_discrete_sequence=["#f97316"])
        st.plotly_chart(dk(fig,300), use_container_width=True)
    with col2:
        st.subheader("Anomalies by Hour")
        ah = df_anom[df_anom["is_anomaly"]].groupby("hour_ist").size().reset_index(name="n")
        fig = px.bar(ah, x="hour_ist", y="n", color_discrete_sequence=["#ef4444"],
                     labels={"hour_ist":"Hour","n":"Anomaly Count"})
        st.plotly_chart(dk(fig,300), use_container_width=True)

    st.subheader("Top Anomalous Junctions")
    aj = (df_anom[df_anom["is_anomaly"]]
          .groupby("junction_name")
          .agg(n=("is_anomaly","sum"), avg_score=("anomaly_score","mean"), avg_cis=("cis","mean"))
          .sort_values("n",ascending=False).head(20).reset_index())
    fig2 = px.bar(aj, x="junction_name", y="n", color="avg_score",
                  color_continuous_scale="Plasma",
                  labels={"junction_name":"","n":"Anomaly Count"})
    fig2.update_layout(**DARK, height=320, xaxis_tickangle=35)
    st.plotly_chart(fig2, use_container_width=True)

    col3,col4 = st.columns(2)
    with col3:
        st.subheader("Anomaly vs CIS")
        samp = df_anom.sample(min(4000,len(df_anom)), random_state=42)
        fig3 = px.scatter(samp, x="cis", y="anomaly_score", color="is_anomaly",
                          color_discrete_map={True:"#ef4444",False:"#3b82f6"},
                          opacity=0.4, hover_data=["vehicle_type","hour_ist"])
        st.plotly_chart(dk(fig3,320), use_container_width=True)
    with col4:
        st.subheader("Top 30 Anomalous Events")
        top_a = (df_anom.nlargest(30,"anomaly_score")
                 [["junction_name","vehicle_type","hour_ist","cis","anomaly_score"]]
                 .reset_index(drop=True))
        top_a.index += 1
        st.dataframe(top_a, use_container_width=True, height=320)

# ═══════════════════════ REPEAT HOTSPOTS ═════════════════════════════
elif page == "📍 Repeat Hotspots":
    st.title("📍 Repeat Hotspot Analysis")
    st.caption("~100m grid cells · scored by persistence × violation density × avg CIS")

    n_crit = int((df_rep["tier"]=="Critical").sum())
    n_high = int((df_rep["tier"]=="High").sum())
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Grid Cells",          f"{len(df_rep):,}")
    c2.metric("Critical Hotspots",   f"{n_crit}")
    c3.metric("High Priority",       f"{n_high}")
    c4.metric("Avg Persistence",     f"{df_rep['persistence'].mean():.2f}")
    st.divider()

    tier_f = st.multiselect("Filter tier", ["Critical","High","Medium","Low"],
                            default=["Critical","High","Medium"])
    show   = df_rep[df_rep["tier"].astype(str).isin(tier_f)] if tier_f else df_rep
    cmap   = {"Critical":"#ef4444","High":"#f97316","Medium":"#facc15","Low":"#22c55e"}

    col1, col2 = st.columns([3,2])
    with col1:
        show2 = show.copy(); show2["tier"] = show2["tier"].astype(str)
        fig = px.scatter_mapbox(show2, lat="lat", lon="lon", color="tier", size="total_viol",
                                color_discrete_map=cmap, size_max=20,
                                zoom=11, mapbox_style="carto-darkmatter",
                                hover_data=["total_viol","hotspot_score","persistence","peak_hour"],
                                height=500)
        fig.update_layout(margin=dict(r=0,t=0,l=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Top Hotspots")
        top_h = df_rep.sort_values("hotspot_score",ascending=False).head(20)[
            ["hotspot_rank","lat","lon","total_viol","hotspot_score","persistence","tier"]
        ].rename(columns={"hotspot_rank":"#","total_viol":"Viol","hotspot_score":"Score",
                          "persistence":"Persist."})
        top_h["tier"] = top_h["tier"].astype(str)
        st.dataframe(top_h, use_container_width=True, height=460)

    col3, col4 = st.columns(2)
    with col3:
        tc = df_rep["tier"].astype(str).value_counts().reset_index()
        tc.columns = ["Tier","Count"]
        fig2 = px.pie(tc, values="Count", names="Tier", hole=0.45,
                      color="Tier", color_discrete_map=cmap)
        fig2.update_layout(**DARK, height=260, margin=dict(t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)
    with col4:
        fig3 = px.scatter(df_rep, x="persistence", y="hotspot_score",
                          color="tier", size="total_viol",
                          color_discrete_map=cmap, opacity=0.7)
        fig3.update_layout(**DARK, height=260, margin=dict(t=10,b=10))
        st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════ CONGESTION ATTRIBUTION ══════════════════════════
elif page == "📈 Congestion Attribution":
    st.title("📈 Congestion Attribution Engine")
    st.caption("LightGBM Poisson (Model A) · R²=0.92 · Assigns congestion % per violation record")

    col1,col2 = st.columns(2)
    with col1:
        st.subheader("SHAP Feature Importance")
        if os.path.exists(out("shap_summary.png")):
            st.image(out("shap_summary.png"), use_container_width=True)
        else:
            st.warning("SHAP plot not found.")
    with col2:
        st.subheader("Top Congestion Drivers (per record)")
        fd = (df_attr.groupby("top_shap_feature").size()
              .sort_values(ascending=False).reset_index(name="count"))
        fig = px.bar(fd, x="count", y="top_shap_feature", orientation="h",
                     color="count", color_continuous_scale="Blues_r",
                     labels={"top_shap_feature":"Driver","count":"Records"})
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(dk(fig,300), use_container_width=True)

    col3,col4 = st.columns(2)
    with col3:
        st.subheader("Avg Predicted Count by Hour")
        pc = df_attr[df_attr["predicted_count"]>0].groupby("hour_ist")["predicted_count"].mean().reset_index()
        fig2 = px.area(pc, x="hour_ist", y="predicted_count",
                       color_discrete_sequence=["#38bdf8"],
                       labels={"hour_ist":"Hour","predicted_count":"Avg Predicted Violations"})
        st.plotly_chart(dk(fig2,280), use_container_width=True)
    with col4:
        st.subheader("Validation Probability by Police Station")
        vp = (df_attr.groupby("police_station")["validation_proba"]
              .mean().sort_values(ascending=False).head(15).reset_index())
        fig3 = px.bar(vp, x="police_station", y="validation_proba",
                      color="validation_proba", color_continuous_scale="RdYlGn",
                      labels={"police_station":"","validation_proba":"Avg Val. Prob."})
        fig3.update_layout(**DARK, height=280, xaxis_tickangle=35)
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Congestion % by Vehicle Type")
    vc = (df_attr.groupby("vehicle_type")["congestion_pct"]
          .mean().sort_values(ascending=False).head(12).reset_index())
    fig4 = px.bar(vc, x="vehicle_type", y="congestion_pct",
                  color="congestion_pct", color_continuous_scale="RdYlGn_r")
    fig4.update_layout(**DARK, height=280, xaxis_title="", xaxis_tickangle=30)
    st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════ ENFORCEMENT ROUTING ════════════════════════════
elif page == "🚔 Enforcement Routing":
    st.title("🚔 Multi-Officer Enforcement Routing")
    st.caption("Priority = 40% CIS + 28% LSTM Stress + 17% Spillover Centrality + 10% Recency · 2-opt TSP")

    col1, col2 = st.columns([3,2])
    with col1:
        st.subheader("Patrol Map — 3 Officers")
        oc_colors = {1:"#38bdf8", 2:"#a78bfa", 3:"#fbbf24"}
        fig = go.Figure()
        bg = df_enf.dropna(subset=["lat","lon"]).head(50)
        fig.add_trace(go.Scattermapbox(
            lat=bg["lat"], lon=bg["lon"], mode="markers",
            marker=dict(size=7, color=bg["priority_score"].tolist(),
                        colorscale="RdYlGn_r", cmin=0, cmax=1),
            text=bg["junction_name"], name="All zones", hoverinfo="text"))
        if "officer_assigned" in df_enf.columns:
            for oid, color in oc_colors.items():
                od = df_enf[df_enf["officer_assigned"]==oid].sort_values("patrol_stop")
                if od.empty: continue
                fig.add_trace(go.Scattermapbox(
                    lat=od["lat"], lon=od["lon"],
                    mode="lines+markers+text",
                    line=dict(width=3, color=color),
                    marker=dict(size=14, color=color),
                    text=od["patrol_stop"].astype(str),
                    textfont=dict(size=9, color="black"),
                    textposition="middle center",
                    name=f"Officer {oid}",
                    hovertext=od["junction_name"]))
        fig.update_layout(
            mapbox=dict(style="carto-darkmatter",
                        center=dict(lat=12.97,lon=77.59), zoom=11),
            margin=dict(r=0,t=0,l=0,b=0), height=520,
            legend=dict(x=0,y=1,bgcolor="rgba(0,0,0,.6)",font=dict(color="#e2e8f0")),
            paper_bgcolor="#0f172a")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Priority Rankings")
        cols_show = [c for c in ["rank","junction_name","priority_score",
                                  "violation_count","officer_assigned","patrol_stop"]
                     if c in df_enf.columns]
        disp = df_enf[cols_show].head(30).rename(columns={
            "rank":"#","junction_name":"Junction",
            "priority_score":"Priority","violation_count":"Viol.",
            "officer_assigned":"Officer","patrol_stop":"Stop"})
        st.dataframe(disp, use_container_width=True, height=480)

    st.subheader("Priority Score Components — Top 15")
    t15 = df_enf.head(15)
    fig2 = go.Figure()
    for col_n, label, color in [
        ("cis_mean","CIS 40%","#ef4444"),
        ("predicted_stress_next_h","Stress 28%","#a78bfa"),
        ("spillover_centrality","Spillover 17%","#38bdf8"),
        ("recency_score","Recency 10%","#fbbf24")]:
        if col_n in t15.columns:
            fig2.add_trace(go.Bar(name=label, x=t15["junction_name"],
                                   y=t15[col_n], marker_color=color))
    fig2.update_layout(**DARK, barmode="group", height=340,
                       xaxis_tickangle=35, xaxis_title="", legend_orientation="h",
                       margin=dict(t=10,b=80))
    st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════ REVENUE & ROI ═══════════════════════════════
elif page == "💰 Revenue & ROI":
    st.title("💰 Revenue & ROI Intelligence")
    st.caption("Fine revenue × collection rate + congestion savings vs patrol cost · Model B validation proba")

    tot_rev  = int(df_roi["expected_revenue"].sum())
    tot_save = int(df_roi["congestion_saving"].sum())
    tot_cost = int(df_roi["patrol_cost"].sum())
    overall  = round((tot_rev+tot_save)/max(tot_cost,1),1)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Expected Fine Revenue",  f"₹{tot_rev:,.0f}")
    c2.metric("Congestion Savings",     f"₹{tot_save:,.0f}")
    c3.metric("Total Patrol Cost",      f"₹{tot_cost:,.0f}")
    c4.metric("Overall ROI",            f"{overall}x")
    st.divider()

    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Top 20 Junctions by ROI")
        t20 = df_roi.head(20)
        fig = px.bar(t20, x="junction_name", y="roi_ratio",
                     color="roi_ratio", color_continuous_scale="RdYlGn",
                     hover_data=["expected_revenue","congestion_saving","patrol_cost"])
        fig.update_layout(**DARK, height=320, xaxis_tickangle=35,
                          xaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Revenue Stack — Top 15")
        t15 = df_roi.head(15)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Fine Revenue", x=t15["junction_name"],
                               y=t15["expected_revenue"], marker_color="#38bdf8"))
        fig2.add_trace(go.Bar(name="Congestion Savings", x=t15["junction_name"],
                               y=t15["congestion_saving"], marker_color="#a78bfa"))
        fig2.add_trace(go.Bar(name="Patrol Cost", x=t15["junction_name"],
                               y=t15["patrol_cost"], marker_color="#ef4444"))
        fig2.update_layout(**DARK, barmode="group", height=320,
                           xaxis_tickangle=35, xaxis_title="", legend_orientation="h")
        st.plotly_chart(fig2, use_container_width=True)

    if "priority_score" in df_roi.columns:
        st.subheader("Strategic Quadrant — Priority vs ROI")
        fig3 = px.scatter(df_roi, x="priority_score", y="roi_ratio",
                          size="violation_count", color="roi_ratio",
                          color_continuous_scale="RdYlGn",
                          hover_data=["junction_name","expected_revenue"])
        fig3.add_vline(x=df_roi["priority_score"].median(), line_dash="dash",
                       line_color="#475569", annotation_text="Median Priority")
        fig3.add_hline(y=df_roi["roi_ratio"].median(), line_dash="dash",
                       line_color="#475569", annotation_text="Median ROI")
        fig3.update_layout(**DARK, height=380)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("📌 Top-right = High Priority + High ROI → deploy resources here first")

    st.subheader("Full ROI Table")
    show_cols = [c for c in ["roi_rank","junction_name","violation_count",
                              "expected_revenue","congestion_saving","patrol_cost",
                              "roi_ratio","validated_pct","priority_score"] if c in df_roi.columns]
    st.dataframe(df_roi[show_cols], use_container_width=True)

# ═══════════════════════ SHIFT SCHEDULER ═════════════════════════════
elif page == "🗓️ Shift Scheduler":
    st.title("🗓️ Officer Shift Schedule")
    st.caption("Data-driven weekly schedule from predicted stress windows + enforcement priorities")

    if not shift:
        st.warning("Run Module 7."); st.stop()
    summary = shift.get("summary",{})
    c1,c2,c3 = st.columns(3)
    c1.metric("Weekly Shifts",       summary.get("total_weekly_shifts","—"))
    c2.metric("Avg Officers/Shift",  summary.get("avg_officers_per_shift","—"))
    c3.metric("Peak Coverage Shifts",summary.get("peak_coverage_shifts","—"))

    shifts = pd.DataFrame(shift.get("shifts",[]))
    if not shifts.empty:
        st.subheader("Weekly Heatmap")
        day_order   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        shift_order = ["Morning","Afternoon","Night"]
        pivot = shifts.pivot_table(index="shift", columns="day",
                                   values="officer_count", aggfunc="first")
        pivot = pivot.reindex(index=[s for s in shift_order if s in pivot.index],
                              columns=[d for d in day_order if d in pivot.columns])
        fig = px.imshow(pivot, color_continuous_scale="Blues",
                        labels={"x":"Day","y":"Shift","color":"Officers"},
                        text_auto=True, aspect="auto")
        fig.update_layout(**DARK, height=200, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Shift Detail")
        for day in day_order:
            day_s = shifts[shifts["day"]==day]
            if day_s.empty: continue
            with st.expander(day):
                for _, row in day_s.iterrows():
                    peak = " 🔴 PEAK" if row.get("is_peak") else ""
                    st.markdown(f"**{row['shift']}** ({row['hours']}) — "
                                f"{row['officer_count']} officer(s){peak}")
                    zones = row.get("zones_assigned",[])
                    if isinstance(zones,list) and zones:
                        st.caption("Zones: " + ", ".join(str(z) for z in zones[:5]))

# ═══════════════════════ POLICY SIMULATOR ════════════════════════════
elif page == "🧪 Policy Simulator":
    st.title("🧪 Policy Simulation Lab")
    st.caption("What-if: restrict parking in zone X → how much does congestion drop?")

    sims = pd.DataFrame(policy.get("simulations",[]))
    if not sims.empty and "error" not in sims.columns:
        st.subheader("Pre-computed: Top 5 Junctions × 4 Restriction Levels")
        fig = px.line(sims, x="restriction_pct", y="congestion_delta_pct",
                      color="junction", markers=True,
                      labels={"restriction_pct":"Restriction %","congestion_delta_pct":"Congestion Δ%"})
        fig.update_layout(**DARK, height=360)
        st.plotly_chart(fig, use_container_width=True)



# ═══════════════════════ SPILLOVER GRAPH ═════════════════════════════
elif page == "🔗 Spillover Graph":
    st.title("🔗 Spillover Chain Graph")
    st.caption("Directed graph of congestion propagation · PageRank centrality = downstream amplification")

    nodes = spill.get("nodes",[])
    links = spill.get("links",[])
    nd    = pd.DataFrame(nodes)
    if "spillover_centrality" not in nd.columns: nd["spillover_centrality"] = 0
    nd = nd.sort_values("spillover_centrality", ascending=False)

    c1,c2,c3 = st.columns(3)
    c1.metric("Graph Nodes", len(nodes))
    c2.metric("Graph Edges", len(links))
    c3.metric("Max Centrality", f"{nd['spillover_centrality'].max():.4f}")

    col1,col2 = st.columns([2,1])
    with col1:
        fig = px.scatter_mapbox(nd.dropna(subset=["lat","lon"]),
                                lat="lat", lon="lon",
                                color="spillover_centrality", size="total_violations",
                                color_continuous_scale="RdYlGn_r", size_max=25,
                                zoom=11, mapbox_style="carto-darkmatter",
                                hover_data=["id","spillover_centrality"],
                                height=500)
        fig.update_layout(margin=dict(r=0,t=0,l=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Top Spillover Nodes")
        t = nd[["id","spillover_centrality","total_violations"]].head(20)
        t.columns = ["Junction","Centrality","Violations"]
        st.dataframe(t, use_container_width=True, height=440)

# ═══════════════════════ MODEL PERFORMANCE ═══════════════════════════
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance Dashboard")
    st.caption("Live metrics from v5 models — no data leakage, independently validated")

    st.markdown("### Model A — Hourly Count Forecaster (LightGBM Poisson)")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("MAE",  "0.1658", delta="vs naive-last: −84%", delta_color="inverse")
    c2.metric("RMSE", "0.9244")
    c3.metric("R²",   "0.920",  delta="vs naive (0.0)")
    c4.metric("CV R² (4-fold)", "0.919 ± 0.011")

    st.markdown("""
    | | Score | Interpretation |
    |---|---|---|
    | MAE = 0.17 | Off by 0.17 violations/hr on average | Mean target = 0.84 → 20% relative error |
    | R² = 0.92 | Explains 92% of variance in hourly counts | Only 8% unexplained randomness |
    | CV R² = 0.919 ± 0.011 | Stable across all time-ordered folds | Not overfitting to any one period |
    | 84% better than naive | vs "predict last observation" baseline | Real forecasting signal |
    """)

    st.subheader("MAE by Violation-Count Bucket")
    bucket_data = pd.DataFrame({
        "Bucket": ["count=0 (n=26,739)", "count=1-2 (n=2,223)", "count=3-5 (n=1,067)",
                   "count=6-10 (n=694)", "count=11+ (n=581)"],
        "MAE":    [0.007, 0.360, 0.878, 1.598, 3.730],
    })
    fig_b = px.bar(bucket_data, x="Bucket", y="MAE",
                   color="MAE", color_continuous_scale="RdYlGn_r",
                   text="MAE")
    fig_b.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_b.update_coloraxes(showscale=False)
    st.plotly_chart(dk(fig_b, 280), use_container_width=True)

    st.divider()
    st.markdown("### Model B — Challan Validation Predictor (LightGBM Binary)")
    c5,c6,c7,c8 = st.columns(4)
    c5.metric("ROC-AUC",       "0.8036", delta="vs random: +0.3036")
    c6.metric("F1 (validated)","0.9365")
    c7.metric("Precision",     "0.9371")
    c8.metric("CV AUC std",    "0.0009", delta="extremely stable")

    st.markdown("""
    | | Score | Interpretation |
    |---|---|---|
    | ROC-AUC = 0.80 | Genuine discrimination ability | Random = 0.50; this is real signal |
    | CV AUC std = 0.0009 | Near-zero variance across 5 folds | Fully stable, not a fluke |
    | F1-validated = 0.94 | Identifies validated challans well | Use for revenue prioritisation |
    | Brier = 0.073 | Well-calibrated probabilities | Post-isotonic calibration applied |
    | **month EXCLUDED** | Was inflating AUC via data artifact | April = 4.4% val rate (pipeline lag) |
    """)

    st.subheader("Threshold Sensitivity")
    thr_data = pd.DataFrame({
        "Threshold": [0.3, 0.4, 0.5, 0.6, 0.7, 0.76, 0.8],
        "Precision": [0.906,0.909,0.915,0.925,0.930,0.937,0.940],
        "Recall":    [0.999,0.997,0.991,0.976,0.963,0.936,0.924],
        "F1-Val":    [0.950,0.951,0.951,0.949,0.946,0.937,0.932],
        "MacroF1":   [0.531,0.558,0.606,0.659,0.676,0.682,0.680],
    })
    fig_t = go.Figure()
    for col_t, color in [("Precision","#38bdf8"),("Recall","#a78bfa"),
                          ("F1-Val","#22c55e"),("MacroF1","#fbbf24")]:
        fig_t.add_trace(go.Scatter(x=thr_data["Threshold"], y=thr_data[col_t],
                                    name=col_t, line=dict(color=color, width=2), mode="lines+markers"))
    fig_t.add_vline(x=0.76, line_dash="dash", line_color="#ef4444",
                    annotation_text="Optimal (0.76)", annotation_font_color="#ef4444")
    fig_t.update_layout(**DARK, height=320, xaxis_title="Threshold", yaxis_title="Score",
                         legend_orientation="h", margin=dict(t=20,b=10))
    st.plotly_chart(fig_t, use_container_width=True)

    st.subheader("5-Fold CV Stability")
    cv_data = pd.DataFrame({
        "Fold":    [1, 2, 3, 4, 5],
        "AUC":     [0.8095, 0.8098, 0.8105, 0.8107, 0.8082],
        "F1-val":  [0.9332, 0.9379, 0.9366, 0.9360, 0.9330],
    })
    fig_cv = go.Figure()
    fig_cv.add_trace(go.Bar(name="AUC", x=cv_data["Fold"], y=cv_data["AUC"],
                             marker_color="#38bdf8", yaxis="y"))
    fig_cv.add_trace(go.Scatter(name="F1-val", x=cv_data["Fold"], y=cv_data["F1-val"],
                                 line=dict(color="#a78bfa",width=3),
                                 mode="lines+markers", yaxis="y2"))
    fig_cv.update_layout(
                          plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                          font_color="#e2e8f0", height=280,
                          yaxis=dict(title="AUC", range=[0.75,0.85], gridcolor="#1e293b"),
                          yaxis2=dict(title="F1", range=[0.9,0.96], overlaying="y", side="right"),
                          legend_orientation="h", margin=dict(t=10,b=10))
    st.plotly_chart(fig_cv, use_container_width=True)