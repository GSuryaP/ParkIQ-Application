"""ParkIQ — Shared Configuration"""
import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RAW_CSV              = os.path.join(DATA_DIR, "cleaned_dataset.csv")
SCORED_PARQUET       = os.path.join(OUTPUT_DIR, "scored.parquet")
JUNCTION_STRESS      = os.path.join(OUTPUT_DIR, "junction_stress.parquet")
SPILLOVER_JSON       = os.path.join(OUTPUT_DIR, "spillover_graph.json")
ENFORCEMENT_CSV      = os.path.join(OUTPUT_DIR, "enforcement_priorities.csv")
ATTRIBUTION_PARQUET  = os.path.join(OUTPUT_DIR, "attribution.parquet")
SHAP_PNG             = os.path.join(OUTPUT_DIR, "shap_summary.png")
POLICY_REPORT        = os.path.join(OUTPUT_DIR, "policy_report.json")
ANOMALY_PARQUET      = os.path.join(OUTPUT_DIR, "anomaly_scores.parquet")
REPEAT_CSV           = os.path.join(OUTPUT_DIR, "repeat_locations.csv")
ROI_CSV              = os.path.join(OUTPUT_DIR, "roi_report.csv")
SHIFT_JSON           = os.path.join(OUTPUT_DIR, "shift_schedule.json")

VEHICLE_WEIGHTS = {
    "TANKER": 3.0, "BUS": 2.5, "TRUCK": 2.5, "LGV": 2.0,
    "HGV": 2.0, "MINI TRUCK": 2.0, "MAXI-CAB": 1.8,
    "PASSENGER AUTO": 1.5, "GOODS AUTO": 1.5, "AUTO": 1.5,
    "CAR": 1.0, "MOPED": 0.6, "SCOOTER": 0.5, "MOTOR CYCLE": 0.5, "CYCLE": 0.3,
}
DEFAULT_VEHICLE_WEIGHT = 1.0
PEAK_HOURS = [8, 9, 10, 17, 18, 19, 20]
JUNCTION_RADIUS_M  = 200
SPILLOVER_RADIUS_M = 600

# Fine amounts per violation type (INR)
FINE_SCHEDULE = {
    "NO PARKING": 500,
    "WRONG PARKING": 500,
    "PARKING NEAR ROAD CROSSING": 1000,
    "PARKING IN A MAIN ROAD": 750,
    "PARKING NEAR JUNCTION": 1000,
    "PARKING ON FOOTPATH": 500,
    "DEFAULT": 500,
}
