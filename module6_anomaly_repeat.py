"""
ParkIQ — Module 6: Anomaly Detection + Repeat Hotspot Analysis
- Isolation Forest detects anomalous spikes in violation density
- Spatial clustering (grid-based) finds persistent repeat hotspots
- Location-level repeat offence rate scoring
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from config import SCORED_PARQUET, ANOMALY_PARQUET, REPEAT_CSV


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Isolation Forest on temporal + spatial + CIS features."""
    print("  Running Isolation Forest anomaly detection …")
    features = ["cis", "vehicle_weight", "prox_score", "time_mult",
                "hour_ist", "weekday_num", "near_junction", "near_crossing"]
    feat_cols = [c for c in features if c in df.columns]

    X = df[feat_cols].fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = IsolationForest(n_estimators=200, contamination=0.05,
                          random_state=42, n_jobs=-1)
    df = df.copy()
    df["anomaly_score"]  = -clf.fit(Xs).score_samples(Xs)   # higher = more anomalous
    df["is_anomaly"]     = clf.predict(Xs) == -1             # True = anomalous
    df["anomaly_score"]  = df["anomaly_score"].round(4)

    n_anom = df["is_anomaly"].sum()
    print(f"  Detected {n_anom:,} anomalous records ({n_anom/len(df)*100:.1f}%)")
    return df


def repeat_hotspot_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Grid the city into ~100m cells; find cells with persistent violations
    across multiple time windows → true hotspots, not one-off events.
    """
    print("  Analysing repeat hotspots …")
    df = df.copy()
    # ~100m grid cells (approx 0.001° lat/lon at Bengaluru latitude)
    GRID = 0.001
    df["grid_lat"] = (df["latitude"]  / GRID).round(0) * GRID
    df["grid_lon"] = (df["longitude"] / GRID).round(0) * GRID
    df["grid_id"]  = df["grid_lat"].astype(str) + "_" + df["grid_lon"].astype(str)

    df["created_datetime"] = pd.to_datetime(df["created_datetime"], utc=True, errors="coerce")
    df["week"] = df["created_datetime"].dt.isocalendar().week.fillna(0).astype(int)
    df["year"] = df["created_datetime"].dt.year

    agg = (df.groupby("grid_id")
             .agg(
                 lat          = ("grid_lat",   "first"),
                 lon          = ("grid_lon",   "first"),
                 total_viol   = ("record_id",  "count"),
                 avg_cis      = ("cis",        "mean"),
                 max_cis      = ("cis",        "max"),
                 weeks_active = ("week",       "nunique"),
                 months_active= ("month",      "nunique"),
                 peak_hour    = ("hour_ist",   lambda x: int(x.mode()[0]) if len(x) else 0),
                 top_vehicle  = ("vehicle_type", lambda x: str(x.mode()[0]) if len(x) else ""),
                 near_junction= ("near_junction","max"),
                 police_station=("police_station", lambda x: str(x.mode()[0]) if len(x) else ""),
             )
             .reset_index())

    # Persistence score: how consistently does this cell produce violations?
    # weeks_active / max_possible_weeks
    total_weeks = df["week"].nunique()
    agg["persistence"] = (agg["weeks_active"] / max(total_weeks, 1)).clip(0, 1).round(4)

    # Hotspot score = normalised(total_viol) * 0.4 + avg_cis_norm * 0.3 + persistence * 0.3
    def norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    agg["hotspot_score"] = (
        0.40 * norm(agg["total_viol"]) +
        0.30 * norm(agg["avg_cis"]) +
        0.30 * agg["persistence"]
    ).round(4)

    agg = agg.sort_values("hotspot_score", ascending=False).reset_index(drop=True)
    agg["hotspot_rank"] = agg.index + 1

    # Tier classification — percentile-based so tiers are always meaningful
    # regardless of the absolute score range in the dataset.
    # Top 5% = Critical, next 15% = High, next 30% = Medium, rest = Low
    q95 = agg["hotspot_score"].quantile(0.95)
    q80 = agg["hotspot_score"].quantile(0.80)
    q50 = agg["hotspot_score"].quantile(0.50)
    agg["tier"] = "Low"
    agg.loc[agg["hotspot_score"] >= q50, "tier"] = "Medium"
    agg.loc[agg["hotspot_score"] >= q80, "tier"] = "High"
    agg.loc[agg["hotspot_score"] >= q95, "tier"] = "Critical"
    agg["tier"] = pd.Categorical(agg["tier"],
                                 categories=["Low","Medium","High","Critical"],
                                 ordered=True)

    print(f"  {len(agg)} grid cells analysed")
    print(f"  Critical hotspots: {(agg['tier']=='Critical').sum()}")
    print(f"  High hotspots:     {(agg['tier']=='High').sum()}")
    return agg


def run():
    print("[Module 6] Anomaly Detection + Repeat Hotspot Analysis …")
    df = pd.read_parquet(SCORED_PARQUET)

    # Anomaly detection
    df = detect_anomalies(df)
    df.to_parquet(ANOMALY_PARQUET, index=False)
    print(f"  Saved → {ANOMALY_PARQUET}")

    # Repeat hotspot analysis
    repeat_df = repeat_hotspot_analysis(df)
    repeat_df.to_csv(REPEAT_CSV, index=False)
    print(f"  Saved → {REPEAT_CSV}")

    print("  Top 5 Critical Hotspots:")
    top = repeat_df[repeat_df["tier"] == "Critical"].head(5)
    for _, r in top.iterrows():
        print(f"    Rank {int(r.hotspot_rank):3d}  lat={r.lat:.4f} lon={r.lon:.4f}  "
              f"violations={int(r.total_viol):4d}  persistence={r.persistence:.2f}  "
              f"score={r.hotspot_score:.3f}")

    return df, repeat_df


if __name__ == "__main__":
    run()