"""
ParkIQ — Module 4: Attribution Engine (v5.1)

MODEL A — Hourly Count Forecaster (LightGBM Poisson)
  Uses FULL dataset including April — count patterns are real regardless
  of whether challans were later validated.

MODEL B — Challan Validation Predictor (LightGBM Binary)
  TRAINING  : November–March only (April labels untrustworthy — pipeline lag
               means April challans show validated=False simply because they
               hadn't been reviewed yet at data-extract time, not because
               they were actually rejected)
  INFERENCE : Full dataset including April — model generates
               validation_proba for every record so officers can still
               prioritise April challans by predicted confidence
  This preserves data integrity: April violations are NOT discarded,
  only their labels are excluded from supervised training.
"""

import os, warnings
warnings.filterwarnings("ignore")
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score, brier_score_loss,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.isotonic import IsotonicRegression
import shap

from config import SCORED_PARQUET, ATTRIBUTION_PARQUET, SHAP_PNG

MIN_JUNCTION_RECORDS = 500
OPTUNA_TRIALS_A      = 10
OPTUNA_TRIALS_B      = 10


# ════════════════════════════════════════════════════════════════════════
# MODEL A — HOURLY COUNT FORECASTER
# ════════════════════════════════════════════════════════════════════════

def build_hourly_ts(df: pd.DataFrame):
    """
    Uses FULL df including April.
    April violations are real events — their counts should inform the model.
    """
    print("  [A] Building hourly time-series (full dataset incl. April) …")

    named = df[df["junction_name"] != "No Junction"].copy()
    named["hour_ist"] = named["hour_ist"].fillna(0).astype(int)

    jcounts = named.groupby("junction_name").size()
    rich    = jcounts[jcounts >= MIN_JUNCTION_RECORDS].index
    named   = named[named["junction_name"].isin(rich)]
    print(f"    Kept {len(rich)} / {jcounts.shape[0]} junctions "
          f"(≥{MIN_JUNCTION_RECORDS} records)")

    agg = (named.groupby(["junction_name","date_dt","hour_ist"])
           .agg(count      = ("record_id",     "count"),
                heavy_pct  = ("vehicle_weight", lambda x: (x>=2.0).mean()),
                avg_weight = ("vehicle_weight", "mean"),
                lat        = ("latitude",       "first"),
                lon        = ("longitude",      "first"),
                weekday    = ("weekday_num",    "first"),
                month      = ("month",          "first"))
           .reset_index().rename(columns={"date_dt":"date"}))

    agg["weekend"]          = (agg["weekday"] >= 5).astype(int)
    agg["peak_hour"]        = agg["hour_ist"].isin([8,9,10,17,18,19,20]).astype(int)
    agg["day_of_month"]     = agg["date"].dt.day
    agg["days_since_start"] = (agg["date"] - agg["date"].min()).dt.days
    agg["hour_sin"]         = np.sin(2 * np.pi * agg["hour_ist"] / 24)
    agg["hour_cos"]         = np.cos(2 * np.pi * agg["hour_ist"] / 24)
    agg = agg.sort_values(["junction_name","date","hour_ist"]).reset_index(drop=True)

    all_j = agg["junction_name"].unique()
    all_d = pd.date_range(agg["date"].min(), agg["date"].max(), freq="D")
    idx   = pd.MultiIndex.from_product([all_j, all_d, list(range(24))],
                                        names=["junction_name","date","hour_ist"])
    full  = agg.set_index(["junction_name","date","hour_ist"]).reindex(idx).reset_index()
    full["count"] = full["count"].fillna(0)
    meta = ["heavy_pct","avg_weight","lat","lon","weekday","month","weekend",
            "peak_hour","day_of_month","days_since_start","hour_sin","hour_cos"]
    full[meta] = full.groupby("junction_name")[meta].ffill().bfill()

    grp = full.groupby("junction_name")["count"]
    for lag, name in [(1,"lag1"),(2,"lag2"),(24,"lag24"),(48,"lag48"),(168,"lag168")]:
        full[name] = grp.shift(lag).fillna(0)
    for win, name in [(3,"roll3"),(6,"roll6"),(24,"roll24"),(48,"roll48")]:
        full[name] = grp.shift(1).transform(
            lambda x: x.rolling(win, min_periods=1).mean()).fillna(0)
    full["roll_std6"]    = grp.shift(1).transform(
        lambda x: x.rolling(6, min_periods=2).std()).fillna(0)
    full["ewm6"]         = grp.shift(1).transform(
        lambda x: x.ewm(span=6, min_periods=1).mean()).fillna(0)
    full["lag168_roll3"] = grp.shift(168).transform(
        lambda x: x.rolling(3*168, min_periods=1).mean()).fillna(0)

    le = LabelEncoder()
    full["junction_id"] = le.fit_transform(full["junction_name"])
    full["count_rank"]  = full.groupby("date")["count"].rank(pct=True)

    has_signal = (full["count"]>0)|(full["lag1"]>0)|(full["lag24"]>0)|(full["roll24"]>0)
    full = full[has_signal].reset_index(drop=True)
    print(f"    Rows: {len(full):,}   Junctions: {full['junction_name'].nunique()}")
    return full, le, rich


FEAT_A_BASE = [
    "hour_ist","hour_sin","hour_cos","weekday","month","weekend","peak_hour",
    "day_of_month","days_since_start","junction_id","lat","lon","count_rank",
    "heavy_pct","avg_weight",
    "lag1","lag2","lag24","lag48","lag168",
    "roll3","roll6","roll24","roll48","roll_std6","ewm6","lag168_roll3",
]


def add_target_encoding_A(ts, feat_cols, cutoff_date):
    train     = ts[ts["date"] <= cutoff_date].copy()
    jh_mean   = train.groupby(["junction_name","hour_ist"])["count"].mean().rename("junc_hour_mean")
    jdow_mean = train.groupby(["junction_name","weekday"])["count"].mean().rename("junc_dow_mean")
    ts = ts.merge(jh_mean.reset_index(),   on=["junction_name","hour_ist"], how="left")
    ts = ts.merge(jdow_mean.reset_index(), on=["junction_name","weekday"],  how="left")
    ts["junc_hour_mean"] = ts["junc_hour_mean"].fillna(ts["roll24"])
    ts["junc_dow_mean"]  = ts["junc_dow_mean"].fillna(ts["roll24"])
    return ts, feat_cols + ["junc_hour_mean","junc_dow_mean"]


def tune_model_a(X_tr, y_tr, X_te, y_te):
    print(f"    Tuning ({OPTUNA_TRIALS_A} trials) …")
    def objective(trial):
        p = dict(objective="poisson", metric="mse", verbose=-1, n_jobs=-1,
                 num_leaves       = trial.suggest_int("nl", 31, 127),
                 min_data_in_leaf = trial.suggest_int("mdl", 10, 60),
                 learning_rate    = trial.suggest_float("lr", 0.02, 0.12, log=True),
                 feature_fraction = trial.suggest_float("ff", 0.6, 1.0),
                 bagging_fraction = trial.suggest_float("bf", 0.6, 1.0),
                 bagging_freq=5,
                 lambda_l1 = trial.suggest_float("l1", 0, 1.0),
                 lambda_l2 = trial.suggest_float("l2", 0, 5.0),
                 max_depth = trial.suggest_int("md", 4, 8))
        cb = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)]
        m  = lgb.train(p, lgb.Dataset(X_tr, y_tr), num_boost_round=400,
                       valid_sets=[lgb.Dataset(X_te, y_te)], callbacks=cb)
        return mean_absolute_error(y_te, m.predict(X_te).clip(0))
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=OPTUNA_TRIALS_A, show_progress_bar=False)
    return study.best_params


def train_model_a(ts, feat_cols):
    print("\n  [Model A] LightGBM Poisson count forecaster …")
    cutoff = ts["date"].quantile(0.80)
    train  = ts[ts["date"] <= cutoff]
    test   = ts[ts["date"] >  cutoff]
    print(f"    Train: {len(train):,}  ({train['date'].min().date()} → {train['date'].max().date()})")
    print(f"    Test : {len(test):,}   ({test['date'].min().date()} → {test['date'].max().date()})")

    avail  = [f for f in feat_cols if f in ts.columns]
    X_tr, y_tr = train[avail].fillna(0).values, train["count"].values
    X_te, y_te = test[avail].fillna(0).values,  test["count"].values

    best_p = tune_model_a(X_tr, y_tr, X_te, y_te)
    best_p.update(dict(objective="poisson", metric="mse", verbose=-1,
                       n_jobs=-1, bagging_freq=5))
    best_p["learning_rate"]    = best_p.pop("lr",  best_p.get("learning_rate", 0.04))
    best_p["feature_fraction"] = best_p.pop("ff",  best_p.get("feature_fraction", 0.8))
    best_p["bagging_fraction"] = best_p.pop("bf",  best_p.get("bagging_fraction", 0.8))
    best_p["lambda_l1"]        = best_p.pop("l1",  best_p.get("lambda_l1", 0.1))
    best_p["lambda_l2"]        = best_p.pop("l2",  best_p.get("lambda_l2", 1.0))
    best_p["num_leaves"]       = best_p.pop("nl",  best_p.get("num_leaves", 63))
    best_p["min_data_in_leaf"] = best_p.pop("mdl", best_p.get("min_data_in_leaf", 20))
    best_p["max_depth"]        = best_p.pop("md",  best_p.get("max_depth", 6))

    cb    = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
    model = lgb.train(best_p, lgb.Dataset(X_tr, y_tr), num_boost_round=400,
                      valid_sets=[lgb.Dataset(X_te, y_te)], callbacks=cb)
    preds = model.predict(X_te).clip(0)

    mae  = mean_absolute_error(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    r2   = r2_score(y_te, preds)
    bl_last = mean_absolute_error(y_te, test["lag1"].values)
    bl_24h  = mean_absolute_error(y_te, test["roll24"].values)

    print(f"\n    ── Model A Results ──────────────────────────────────")
    print(f"    MAE  = {mae:.4f}   (naive-last={bl_last:.4f}  naive-24h={bl_24h:.4f})")
    print(f"    RMSE = {rmse:.4f}")
    print(f"    R²   = {r2:.4f}")
    print(f"    Improvement vs naive-last : {(bl_last-mae)/bl_last*100:.1f}%")
    print(f"    Best iteration: {model.best_iteration}")
    return model, avail, test, preds, y_te


# ════════════════════════════════════════════════════════════════════════
# MODEL B — CHALLAN VALIDATION PREDICTOR
# ════════════════════════════════════════════════════════════════════════

FEAT_B_CAT = ["police_station","vehicle_type","offence_code","center_code","junction_name"]
FEAT_B_NUM = ["latitude","longitude","hour_ist","weekday_num","weekend"]
# month EXCLUDED — would let model learn pipeline timing, not real signal
# April EXCLUDED from TRAINING only — labels untrustworthy (not-yet-processed)
# April INCLUDED at INFERENCE — predictions still generated for all records


def add_station_hour_te(data, train_mask):
    """Station×hour target encoding — fitted on train rows only."""
    global_mean = data.loc[train_mask, "validated"].mean()
    smooth_k    = 20
    grp = data.loc[train_mask].groupby(["police_station","hour_ist"])["validated"]
    te  = (grp.sum() + smooth_k * global_mean) / (grp.count() + smooth_k)
    te  = te.rename("station_hour_te")
    data = data.merge(te.reset_index(), on=["police_station","hour_ist"], how="left")
    data["station_hour_te"] = data["station_hour_te"].fillna(global_mean)
    return data


def add_violation_richness(df):
    def count_types(v):
        if v is None or (not isinstance(v, (list, np.ndarray)) and pd.isna(v)):
            return 1
        return len(list(v)) if hasattr(v, '__iter__') and not isinstance(v, str) else 1
    df["violation_richness"] = df["violation_type"].apply(count_types)
    return df


def tune_model_b(X_tr, y_tr, X_te, y_te, spw):
    print(f"    Tuning ({OPTUNA_TRIALS_B} trials) …")
    def objective(trial):
        p = dict(objective="binary", metric="auc", verbose=-1, n_jobs=-1,
                 scale_pos_weight=spw, bagging_freq=5,
                 num_leaves       = trial.suggest_int("nl", 15, 63),
                 min_data_in_leaf = trial.suggest_int("mdl", 30, 100),
                 learning_rate    = trial.suggest_float("lr", 0.01, 0.1, log=True),
                 feature_fraction = trial.suggest_float("ff", 0.6, 1.0),
                 bagging_fraction = trial.suggest_float("bf", 0.6, 1.0),
                 lambda_l1 = trial.suggest_float("l1", 0, 2.0),
                 lambda_l2 = trial.suggest_float("l2", 0, 5.0))
        cb = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)]
        m  = lgb.train(p, lgb.Dataset(X_tr, y_tr), num_boost_round=400,
                       valid_sets=[lgb.Dataset(X_te, y_te)], callbacks=cb)
        return -roc_auc_score(y_te, m.predict(X_te))
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=OPTUNA_TRIALS_B, show_progress_bar=False)
    return study.best_params


def find_best_threshold(y_te, proba):
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.2, 0.8, 0.01):
        f = f1_score(y_te, (proba>=t).astype(int), average="macro", zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return round(best_t, 2), round(best_f1, 4)


def train_model_b(df):
    print("\n  [Model B] LightGBM challan validation predictor …")
    print("    Training  : November–March only (April labels = pipeline artifact)")
    print("    Inference : Full dataset including April (all records get predictions)")

    data = df.copy()
    data = add_violation_richness(data)

    # ── TRAINING SET: exclude April (untrustworthy labels) ───────────
    # April challans show validated=False only because they hadn't been
    # reviewed at extraction time — NOT because they were truly rejected.
    # We train on months where the label is ground truth.
    data_train = data[data["month"] != 4].copy()
    # data (full) is used for inference in attach_to_records()

    print(f"    Training set  : {len(data_train):,} records (Nov–Mar)")
    print(f"    Inference set : {len(data):,} records (full, incl. {(data['month']==4).sum():,} April)")

    # ── Encode categoricals on training set ──────────────────────────
    encoders = {}
    for col in FEAT_B_CAT:
        le = LabelEncoder()
        data_train[f"{col}_enc"] = le.fit_transform(
            data_train[col].astype(str).fillna("__NA__"))
        encoders[col] = le

    # ── Split THEN target-encode (no leakage) ────────────────────────
    y_train = data_train["validated"].astype(int).values
    idx     = np.arange(len(data_train))
    tr_idx, te_idx = train_test_split(idx, test_size=0.20,
                                      random_state=42, stratify=y_train)
    train_mask = pd.Series(False, index=data_train.index)
    train_mask.iloc[tr_idx] = True

    data_train = add_station_hour_te(data_train, train_mask)

    feat_cols = ([f"{c}_enc" for c in FEAT_B_CAT] +
                 FEAT_B_NUM + ["station_hour_te","violation_richness"])
    feat_cols = [f for f in feat_cols if f in data_train.columns]

    X = data_train[feat_cols].fillna(0).values
    y = data_train["validated"].astype(int).values
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    pos_rate = y_tr.mean()
    spw      = (y_tr==0).sum() / max((y_tr==1).sum(), 1)
    print(f"    Validation rate (train): {pos_rate*100:.1f}%   scale_pos_weight: {spw:.3f}")

    best_p = tune_model_b(X_tr, y_tr, X_te, y_te, spw)
    best_p.update(dict(objective="binary", metric="auc", verbose=-1,
                       n_jobs=-1, bagging_freq=5, scale_pos_weight=spw))
    best_p["learning_rate"]    = best_p.pop("lr",  0.03)
    best_p["feature_fraction"] = best_p.pop("ff",  0.8)
    best_p["bagging_fraction"] = best_p.pop("bf",  0.8)
    best_p["lambda_l1"]        = best_p.pop("l1",  0.5)
    best_p["lambda_l2"]        = best_p.pop("l2",  2.0)
    best_p["num_leaves"]       = best_p.pop("nl",  31)
    best_p["min_data_in_leaf"] = best_p.pop("mdl", 50)

    cb  = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
    clf = lgb.train(best_p, lgb.Dataset(X_tr, y_tr), num_boost_round=400,
                    valid_sets=[lgb.Dataset(X_te, y_te)], callbacks=cb)

    proba = clf.predict(X_te)
    ir    = IsotonicRegression(out_of_bounds="clip")
    ir.fit(proba, y_te)
    proba_cal = ir.transform(proba)

    best_t, _ = find_best_threshold(y_te, proba_cal)
    preds     = (proba_cal >= best_t).astype(int)

    auc   = roc_auc_score(y_te, proba_cal)
    prauc = average_precision_score(y_te, proba_cal)
    f1    = f1_score(y_te, preds)
    prec  = precision_score(y_te, preds, zero_division=0)
    rec   = recall_score(y_te, preds, zero_division=0)
    brier = brier_score_loss(y_te, proba_cal)
    f1_nv = f1_score(y_te, preds, pos_label=0)

    print(f"\n    ── Model B Results ──────────────────────────────────")
    print(f"    ROC-AUC        = {auc:.4f}   (random=0.5000)")
    print(f"    PR-AUC         = {prauc:.4f}")
    print(f"    Brier score    = {brier:.4f}  (0=perfect)")
    print(f"    Best threshold = {best_t}")
    print(f"    F1 (validated) = {f1:.4f}")
    print(f"    F1 (not-valid) = {f1_nv:.4f}")
    print(f"    Precision      = {prec:.4f}   Recall = {rec:.4f}")
    print(f"    Best iteration : {clf.best_iteration}")

    # Return data_train so attach_to_records can reuse the TE mapping
    return clf, feat_cols, encoders, X_te, y_te, proba_cal, best_t, ir, data_train


# ════════════════════════════════════════════════════════════════════════
# SHAP
# ════════════════════════════════════════════════════════════════════════

def save_shap(model_a, feat_a, X_a, model_b, feat_b, X_b):
    print("  Computing SHAP (both models) …")
    os.makedirs("/tmp/mplconfig", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor("#0f172a")

    for ax, model, feats, Xs, title in [
        (axes[0], model_a, feat_a, X_a,
         "Model A — Hourly Count Forecaster\n(Poisson LightGBM · full dataset incl. April)"),
        (axes[1], model_b, feat_b, X_b,
         "Model B — Validation Predictor\n(Binary LightGBM · trained Nov–Mar · infers on all)"),
    ]:
        ax.set_facecolor("#0f172a")
        exp = shap.TreeExplainer(model)
        sv  = np.abs(exp.shap_values(Xs))
        if isinstance(sv, list): sv = sv[1]
        mean_abs = sv.mean(axis=0)
        order    = np.argsort(mean_abs)
        labels   = [feats[i] for i in order]
        values   = mean_abs[order]
        colors   = plt.cm.RdYlGn_r(np.linspace(0.15, 0.9, len(values)))
        bars     = ax.barh(labels, values, color=colors, edgecolor="#1e3a5f", linewidth=0.4)
        for bar, v in zip(bars, values):
            ax.text(v + values.max()*0.01, bar.get_y()+bar.get_height()/2,
                    f"{v:.3f}", va="center", fontsize=7.5, color="#e2e8f0")
        ax.set_title(title, color="#e2e8f0", fontsize=9.5, pad=8)
        ax.set_xlabel("Mean |SHAP value|", color="#94a3b8", fontsize=9)
        ax.tick_params(colors="#cbd5e1", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#334155")

    plt.tight_layout(pad=2)
    plt.savefig(SHAP_PNG, dpi=140, bbox_inches="tight", facecolor="#0f172a")
    plt.close("all")
    print(f"  SHAP chart saved → {SHAP_PNG}")


# ════════════════════════════════════════════════════════════════════════
# ATTACH PREDICTIONS TO ALL RECORDS (including April)
# ════════════════════════════════════════════════════════════════════════

def attach_to_records(df, ts, model_a, feat_a,
                      model_b, feat_b, encoders, ir, best_t, data_train):
    print("\n  Attaching predictions to ALL records (incl. April) …")
    out = df.copy()
    out = add_violation_richness(out)

    # ── Model B inference on FULL dataset ────────────────────────────
    # Encode using encoders fitted on Nov–Mar training set
    # Unknown categories (e.g. new junctions in April) fall back to first class
    out_b = out.copy()
    for col in FEAT_B_CAT:
        le    = encoders[col]
        known = set(le.classes_)
        out_b[f"{col}_enc"] = le.transform(
            out_b[col].astype(str).fillna("__NA__").map(
                lambda v, k=known, c=le.classes_: v if v in k else c[0]))

    # Recompute station×hour TE from training data (Nov–Mar) and apply to all
    gm   = data_train["validated"].mean()
    sm_k = 20
    grp  = data_train.groupby(["police_station","hour_ist"])["validated"]
    te   = (grp.sum() + sm_k * gm) / (grp.count() + sm_k)
    te   = te.rename("station_hour_te").reset_index()
    out_b = out_b.merge(te, on=["police_station","hour_ist"], how="left")
    out_b["station_hour_te"] = out_b["station_hour_te"].fillna(gm)

    X_b               = out_b[[f for f in feat_b if f in out_b.columns]].fillna(0).values
    raw_proba         = model_b.predict(X_b)
    out["validation_proba"] = ir.transform(raw_proba).round(4)

    # ── Model A inference ─────────────────────────────────────────────
    avail = [f for f in feat_a if f in ts.columns]
    ts2   = ts.copy()
    ts2["predicted_count"] = model_a.predict(ts2[avail].fillna(0).values).clip(0).round(1)
    ts2["date_str"]        = ts2["date"].astype(str)
    out["date_str"]        = out["date_dt"].astype(str)

    out = out.merge(
        ts2[["junction_name","date_str","hour_ist","predicted_count","count"]],
        on=["junction_name","date_str","hour_ist"], how="left")

    max_c                  = out["predicted_count"].max()
    out["predicted_count"] = out["predicted_count"].fillna(0)
    out["congestion_pct"]  = (out["predicted_count"] / max(max_c, 1) * 100).round(2)

    # Per-record top driver (rule-based from available columns)
    out["top_shap_feature"] = "time_of_day"
    out.loc[(out["peak_hour"]==1) & (out["near_junction"]==1),
            "top_shap_feature"] = "peak_hour + junction"
    out.loc[(out["peak_hour"]==1) & (out["near_junction"]==0),
            "top_shap_feature"] = "peak_hour"
    out.loc[(out["vehicle_weight"]>=2.0) & (out["near_junction"]==1),
            "top_shap_feature"] = "heavy_vehicle + junction"
    out.loc[(out["vehicle_weight"]>=2.0) & (out["near_junction"]==0),
            "top_shap_feature"] = "heavy_vehicle"
    out.loc[(out["near_junction"]==1) & (out["peak_hour"]==0),
            "top_shap_feature"] = "junction_proximity"
    out.loc[(out["weekend"]==1),
            "top_shap_feature"] = "weekend_pattern"
    out["top_shap_value"] = out["cis"].round(4)

    return out


# ════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════

def run():
    print("[Module 4] Attribution Engine v5.1 …")
    print("  Model A: full dataset (April counts are real violations)")
    print("  Model B: train on Nov–Mar, infer on all (April labels untrustworthy)\n")

    df = pd.read_parquet(SCORED_PARQUET)
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], utc=True, errors="coerce")
    df["date_dt"]     = pd.to_datetime(df["created_datetime"].dt.date.astype(str), errors="coerce")
    df["hour_ist"]    = df["hour_ist"].fillna(0).astype(int)
    df["weekday_num"] = df["weekday_num"].fillna(0).astype(int)
    df["month"]       = df["month"].fillna(1).astype(int)
    df = df[df["date_dt"].notna()].reset_index(drop=True)

    # Model A — uses full df including April
    ts, le_j, rich = build_hourly_ts(df)
    cutoff          = ts["date"].quantile(0.80)
    ts, feat_a      = add_target_encoding_A(ts, FEAT_A_BASE[:], cutoff)
    model_a, feat_a, test_ts, preds_a, y_a = train_model_a(ts, feat_a)

    # Model B — trains on Nov–Mar, returns data_train for TE reuse
    (model_b, feat_b, encoders,
     X_te_b, y_te_b, proba_b,
     best_t, ir, data_train) = train_model_b(df)

    # SHAP
    avail_a = [f for f in feat_a if f in test_ts.columns]
    Xa = test_ts[avail_a].fillna(0).sample(min(2000, len(test_ts)), random_state=42).values
    Xb = X_te_b[:min(2000, len(X_te_b))]
    save_shap(model_a, feat_a, Xa, model_b, feat_b, Xb)

    # Attach to ALL records (full df, including April)
    df_out = attach_to_records(df, ts, model_a, feat_a,
                               model_b, feat_b, encoders, ir, best_t, data_train)
    df_out.to_parquet(ATTRIBUTION_PARQUET, index=False)
    print(f"\n  Saved → {ATTRIBUTION_PARQUET}  ({len(df_out):,} records)")
    return df_out, model_a, model_b


if __name__ == "__main__":
    run()