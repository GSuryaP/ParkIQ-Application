"""
ParkIQ — Module 0: Data Ingestion & Preprocessing
Dataset is pre-cleaned; we parse list fields, derive features, add vehicle weights.
"""
import ast
import pandas as pd
import numpy as np
from config import RAW_CSV, SCORED_PARQUET, VEHICLE_WEIGHTS, DEFAULT_VEHICLE_WEIGHT, PEAK_HOURS


def parse_list_field(val):
    if pd.isna(val):
        return []
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return [str(val)]


def map_vehicle_weight(vtype: str) -> float:
    return VEHICLE_WEIGHTS.get(str(vtype).upper().strip(), DEFAULT_VEHICLE_WEIGHT)


def proximity_score(near_junction: int, near_crossing: int, main_road: int) -> float:
    if near_junction and near_crossing:
        return 1.0
    if near_junction:
        return 0.8
    if near_crossing:
        return 0.6
    if main_road:
        return 0.5
    return 0.3


def time_multiplier(hour_ist: int, weekend: int) -> float:
    if hour_ist in PEAK_HOURS:
        base = 1.5
    elif 6 <= hour_ist < 8 or 20 <= hour_ist < 22:
        base = 1.2
    elif hour_ist < 6 or hour_ist >= 22:
        base = 0.6
    else:
        base = 1.0
    return base * (0.85 if weekend else 1.0)


def run():
    print("[Module 0] Loading dataset …")
    df = pd.read_csv(RAW_CSV, low_memory=False)
    print(f"  Rows: {len(df):,}  Columns: {len(df.columns)}")

    # Parse list fields
    df["violation_type"] = df["violation_type"].apply(parse_list_field)
    df["offence_code"]   = df["offence_code"].apply(parse_list_field)

    # Fill nulls
    df["junction_name"] = df["junction_name"].fillna("No Junction")
    df["vehicle_type"]  = df["vehicle_type"].fillna("UNKNOWN").str.upper().str.strip()

    # Vehicle weight
    df["vehicle_weight"] = df["vehicle_type"].map(map_vehicle_weight)

    # Violation flags
    def has_crossing(vt): return int(any("CROSSING" in str(v).upper() for v in vt))
    def has_main_road(vt): return int(any("MAIN ROAD" in str(v).upper() for v in vt))
    def is_near_junction(jn): return 0 if str(jn).strip().lower() in ("no junction", "", "nan") else 1

    df["near_junction"] = df["junction_name"].apply(is_near_junction)
    df["near_crossing"] = df["violation_type"].apply(has_crossing)
    df["main_road"]     = df["violation_type"].apply(has_main_road)

    # Proximity & time scores
    df["prox_score"] = df.apply(
        lambda r: proximity_score(r["near_junction"], r["near_crossing"], r["main_road"]), axis=1
    )
    df["time_mult"] = df.apply(
        lambda r: time_multiplier(r["hour_ist"], r["weekend"]), axis=1
    )

    # CIS = proximity × vehicle_weight × time_of_day
    raw_cis = df["prox_score"] * df["vehicle_weight"] * df["time_mult"]
    cis_max = raw_cis.max()
    df["cis"] = (raw_cis / cis_max * 10).round(4)

    # Validated flag
    df["validated"] = df["data_sent_to_scita"].astype(str).str.upper().isin(["TRUE", "1"])

    # Record ID
    df = df.reset_index(drop=True)
    df["record_id"] = df.index

    print(f"  CIS → min={df['cis'].min():.3f}  mean={df['cis'].mean():.3f}  max={df['cis'].max():.3f}")
    df.to_parquet(SCORED_PARQUET, index=False)
    print(f"  Saved → {SCORED_PARQUET}")
    return df


if __name__ == "__main__":
    run()
