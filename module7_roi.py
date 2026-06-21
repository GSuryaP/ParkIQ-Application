"""
ParkIQ — Module 7: Revenue & ROI Intelligence Engine
"""
import json
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from config import (SCORED_PARQUET, ENFORCEMENT_CSV, JUNCTION_STRESS,
                    ROI_CSV, SHIFT_JSON, FINE_SCHEDULE)

CONGESTION_COST_PER_VH = 120
AVG_VEHICLES_AFFECTED  = 15
AVG_DELAY_HOURS        = 0.08
OFFICER_COST_PER_HR    = 250


def estimate_fine(vtype_list):
    best = FINE_SCHEDULE["DEFAULT"]
    for vt in vtype_list:
        key = str(vt).upper().strip()
        for fine_key, fine_val in FINE_SCHEDULE.items():
            if fine_key in key:
                best = max(best, fine_val)
    return best


def compute_junction_roi(df: pd.DataFrame, enf_df: pd.DataFrame) -> pd.DataFrame:
    print("  Computing per-junction ROI …")
    df = df.copy()
    # violation_type is already a list/array in parquet
    df["vtype_list"] = df["violation_type"].apply(
        lambda v: list(v) if hasattr(v, '__iter__') and not isinstance(v, str) else [str(v)]
    )
    df["fine_amount"] = df["vtype_list"].apply(estimate_fine)

    collection_rate = float(df["validated"].mean()) if "validated" in df.columns else 0.65
    collection_rate = max(0.3, min(collection_rate, 0.9))

    agg = (df[df["junction_name"] != "No Junction"]
           .groupby("junction_name")
           .agg(
               violation_count  = ("record_id",    "count"),
               avg_cis          = ("cis",          "mean"),
               total_fine_pot   = ("fine_amount",  "sum"),
               lat              = ("latitude",     "mean"),
               lon              = ("longitude",    "mean"),
               validated_pct    = ("validated",    "mean"),
           ).reset_index())

    agg["expected_revenue"] = (agg["total_fine_pot"] * collection_rate).round(0).astype(int)
    agg["congestion_saving"] = (
        agg["violation_count"] * AVG_VEHICLES_AFFECTED * AVG_DELAY_HOURS * CONGESTION_COST_PER_VH
    ).round(0).astype(int)
    agg["patrol_visits"] = np.ceil(agg["violation_count"] / 50).astype(int)
    agg["patrol_cost"]   = (agg["patrol_visits"] * 1.5 * OFFICER_COST_PER_HR).round(0).astype(int)
    agg["total_benefit"] = agg["expected_revenue"] + agg["congestion_saving"]
    agg["roi_ratio"]     = (agg["total_benefit"] / (agg["patrol_cost"] + 1)).round(2)
    agg["roi_pct"]       = ((agg["roi_ratio"] - 1) * 100).round(1)

    if "junction_name" in enf_df.columns:
        agg = agg.merge(enf_df[["junction_name","rank","priority_score"]],
                        on="junction_name", how="left")

    agg = agg.sort_values("roi_ratio", ascending=False).reset_index(drop=True)
    agg["roi_rank"] = agg.index + 1
    print(f"  Top ROI junction: {agg.iloc[0]['junction_name']} (ROI {agg.iloc[0]['roi_ratio']:.1f}x)")
    return agg


def build_shift_schedule(enf_df: pd.DataFrame) -> dict:
    print("  Building officer shift schedule …")
    top_zones = enf_df.head(20)["junction_name"].tolist()

    days   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    shifts_def = {
        "Morning"  : ("06:00–14:00", False),
        "Afternoon": ("14:00–22:00", True),
        "Night"    : ("22:00–06:00", False),
    }

    schedule = {"shifts": [], "summary": {}}
    officer_map = {"Morning": 3, "Afternoon": 4, "Night": 2}
    zone_map    = {"Morning": top_zones[:5], "Afternoon": top_zones[:8], "Night": top_zones[:3]}

    for day in days:
        is_weekend = day in ["Saturday","Sunday"]
        for shift_name, (hours, is_peak) in shifts_def.items():
            oc = officer_map[shift_name]
            if is_weekend and shift_name == "Night": oc = 1
            schedule["shifts"].append({
                "day"           : day,
                "shift"         : shift_name,
                "hours"         : hours,
                "officer_count" : oc,
                "zones_assigned": zone_map[shift_name],
                "is_peak"       : is_peak and not is_weekend,
            })

    schedule["summary"] = {
        "total_weekly_shifts"    : len(schedule["shifts"]),
        "avg_officers_per_shift" : round(np.mean([s["officer_count"] for s in schedule["shifts"]]), 1),
        "peak_coverage_shifts"   : sum(1 for s in schedule["shifts"] if s["is_peak"]),
    }
    return schedule


def run():
    print("[Module 7] Revenue & ROI Intelligence …")
    df        = pd.read_parquet(SCORED_PARQUET)
    enf_df    = pd.read_csv(ENFORCEMENT_CSV)

    roi_df = compute_junction_roi(df, enf_df)
    roi_df.to_csv(ROI_CSV, index=False)
    print(f"  Saved → {ROI_CSV}")

    schedule = build_shift_schedule(enf_df)
    with open(SHIFT_JSON, "w") as f:
        json.dump(schedule, f, indent=2)
    print(f"  Saved → {SHIFT_JSON}")

    total_rev  = roi_df["expected_revenue"].sum()
    total_save = roi_df["congestion_saving"].sum()
    total_cost = roi_df["patrol_cost"].sum()
    print(f"\n  ── Revenue Summary ──")
    print(f"  Expected fine revenue  : ₹{total_rev:,.0f}")
    print(f"  Congestion savings     : ₹{total_save:,.0f}")
    print(f"  Total patrol cost      : ₹{total_cost:,.0f}")
    print(f"  Overall ROI            : {(total_rev+total_save)/max(total_cost,1):.1f}x")
    return roi_df, schedule

if __name__ == "__main__":
    run()
