"""
Coral Reef Bleaching Prediction App
Hugging Face Spaces / Local Gradio App
Loads 4 models from models/ folder

Model types:
  LR  — sklearn Pipeline (StandardScaler + PolynomialFeatures + LogisticRegression)
          NO separate scaler file. Pass raw DataFrame directly.
  RF  — RandomForestClassifier       — no scaler
  XGB — XGBClassifier                — no scaler
  SVM — SVC with svm_scaler.pkl      — needs_scaling=True in metadata

Temperatures in KELVIN to match training data.
297K ≈ 24°C | 300K ≈ 27°C | 303K ≈ 30°C | 306K ≈ 33°C
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import sklearn
sklearn.set_config(enable_metadata_routing=False)

import gradio as gr
import joblib, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline as _SkPipeline

# ── Load models ────────────────────────────────────────────────────────────────
# LR has NO separate scaler file — it is a Pipeline with preprocessing inside.
# SVM is the only model with svm_scaler.pkl (needs_scaling=True in metadata).
MODEL_PREFIXES = {
    "Logistic Regression": "lr",
    "Random Forest"       : "rf",
    "XGBoost"             : "xgb",
    "SVM (RBF Kernel)"    : "svm",
}

MODELS   = {}
SCALERS  = {}
FEATURES = {}
METADATA = {}

for name, prefix in MODEL_PREFIXES.items():
    mpath = f"models/{prefix}_model.pkl"
    fpath = f"models/{prefix}_features.pkl"
    spath = f"models/{prefix}_scaler.pkl"
    jpath = f"models/{prefix}_metadata.json"

    if not os.path.exists(mpath):
        print(f"WARNING: {mpath} not found — {name} skipped")
        continue

    MODELS[name]   = joblib.load(mpath)
    FEATURES[name] = joblib.load(fpath) if os.path.exists(fpath) else []
    # Only load scaler if the file actually exists (LR will always be None here)
    SCALERS[name]  = joblib.load(spath) if os.path.exists(spath) else None
    METADATA[name] = json.load(open(jpath)) if os.path.exists(jpath) else {}

    is_pipe  = isinstance(MODELS[name], _SkPipeline)
    needs_sc = METADATA[name].get("needs_scaling", False)
    print(f"Loaded: {name:<25} pipeline={is_pipe}  needs_scaling={needs_sc}  scaler={'yes' if SCALERS[name] else 'no'}")

# Feature list: metadata 'features' key takes priority over the .pkl file
FEAT_DICT = {
    name: meta.get("features", FEATURES.get(name, []))
    for name, meta in METADATA.items()
}

COLORS = {
    "Logistic Regression": "#E74C3C",
    "Random Forest"       : "#2ECC71",
    "XGBoost"             : "#F39C12",
    "SVM (RBF Kernel)"    : "#3498DB",
}

ALL_FEAT_COLS = [
    "ClimSST", "Temperature_Mean", "Temperature_Minimum", "Temperature_Maximum",
    "SSTA", "SSTA_DHW", "TSA", "TSA_DHW", "TSA_DHW_Frequency",
    "Windspeed", "SSTA_Frequency", "SSTA_Frequency_Standard_Deviation",
    "Turbidity_ct", "Turbidity", "Cyclone_Frequency",
    "Distance", "Depth", "Latitude_Degrees", "Longitude_Degrees", "Date_Year",
]

# ── Sample reef sites — Kelvin temperatures, DHW within training range 0–2.97 ──
SAMPLE_SITES = {
    "Great Barrier Reef 2016 (Bleach)":
        [299.5,302.0,298.5,303.5,2.1,2.5,2.5,2.5,0.30,4.2,0.40,0.20,0.04,0.04,0.10,15.0, 5.0,-18.3,147.2,2016],
    "Great Barrier Reef 2020 (Mild)":
        [299.5,300.5,297.8,302.0,0.8,1.2,1.0,1.2,0.12,5.1,0.20,0.10,0.05,0.05,0.10,15.0, 5.0,-18.3,147.2,2020],
    "Maldives Healthy (2019)":
        [299.8,300.2,298.5,301.5,0.3,0.5,0.4,0.5,0.05,6.0,0.10,0.05,0.03,0.03,0.05, 8.0, 6.0,  4.2, 73.5,2019],
    "Maldives Bleached (2016)":
        [299.8,302.5,298.8,303.8,2.8,2.5,3.1,2.5,0.40,3.5,0.50,0.25,0.03,0.03,0.05, 8.0, 6.0,  4.2, 73.5,2016],
    "Red Sea Resilient":
        [298.5,299.0,297.0,301.0,0.1,0.2,0.2,0.2,0.02,7.0,0.05,0.03,0.08,0.08,0.02,20.0, 8.0, 22.5, 37.8,2018],
    "Caribbean Warm Season (2015)":
        [299.3,301.5,297.8,302.8,1.7,2.2,2.0,2.2,0.25,5.5,0.35,0.15,0.06,0.06,0.08,10.0, 4.0, 17.5,-66.0,2015],
    "Sri Lanka Coast (2016)":
        [299.6,301.0,298.0,302.5,1.2,1.8,1.5,1.8,0.18,4.8,0.25,0.12,0.07,0.07,0.06, 5.0, 3.0,  6.9, 81.5,2016],
    "Deep Protected Reef (Low Risk)":
        [297.5,298.0,296.5,299.5,-0.2,0.1,-0.1,0.1,0.01,8.0,0.02,0.01,0.12,0.12,0.01,30.0,25.0,-22.0,114.0,2017],
    "Turbid Coastal Reef":
        [299.0,301.2,297.5,302.5,1.9,2.5,2.2,2.5,0.28,3.2,0.38,0.18,0.35,0.35,0.09, 2.0, 3.0, -8.5,115.2,2016],
    "El Nino Hotspot (2016)":
        [300.5,302.8,299.0,303.8,2.5,2.9,2.8,2.9,0.45,2.8,0.55,0.28,0.02,0.02,0.12,12.0, 4.0,-16.0,145.5,2016],
}


# ── Universal predict helper ───────────────────────────────────────────────────
def predict_with_model(name, model, X_raw_df):
    """
    Routes correctly for every model type:

      LR  (Pipeline) — X is passed as a DataFrame directly.
                        The Pipeline's own StandardScaler + PolyFeatures handle
                        all preprocessing internally. SCALERS[name] is None.

      RF / XGB       — no scaler, convert to numpy array and predict.

      SVM            — SCALERS[name] is svm_scaler, needs_scaling=True in metadata.
                        Scale first, then predict.
    """
    feats    = FEAT_DICT[name]
    scaler   = SCALERS[name]                               # None for LR, RF, XGB
    needs_sc = METADATA[name].get("needs_scaling", False)  # True only for SVM

    # Align to training columns — keep as DataFrame so Pipeline gets named cols
    cols = [f for f in feats if f in X_raw_df.columns]
    X    = X_raw_df[cols].copy()

    if isinstance(model, _SkPipeline):
        # LR: Pipeline does its own scaling — pass raw DataFrame, do NOT touch X
        pass

    elif needs_sc and scaler is not None:
        # SVM: apply the saved scaler before predicting
        n_exp = scaler.n_features_in_
        if X.shape[1] != n_exp:
            X_full = pd.DataFrame(0.0, index=X.index, columns=feats)
            for c in cols:
                X_full[c] = X[c].values
            X = scaler.transform(X_full.values)
        else:
            X = scaler.transform(X.values)

    else:
        # RF / XGB: no scaling needed
        X = X.values

    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[:, 1][0])
    return pred, prob


# ── Prediction functions ───────────────────────────────────────────────────────
def run_prediction(ClimSST, Temperature_Mean, Temperature_Minimum, Temperature_Maximum,
                   SSTA, SSTA_DHW, TSA, TSA_DHW, TSA_DHW_Frequency, Windspeed,
                   SSTA_Frequency, SSTA_Frequency_Standard_Deviation,
                   Turbidity_ct, Turbidity, Cyclone_Frequency,
                   Distance, Depth, Latitude_Degrees, Longitude_Degrees, Date_Year):
    if not MODELS:
        return "No models loaded.", None

    inp = dict(zip(ALL_FEAT_COLS, [
        float(ClimSST), float(Temperature_Mean), float(Temperature_Minimum), float(Temperature_Maximum),
        float(SSTA), float(SSTA_DHW), float(TSA), float(TSA_DHW), float(TSA_DHW_Frequency),
        float(Windspeed), float(SSTA_Frequency), float(SSTA_Frequency_Standard_Deviation),
        float(Turbidity_ct), float(Turbidity), float(Cyclone_Frequency),
        float(Distance), float(Depth), float(Latitude_Degrees), float(Longitude_Degrees), float(Date_Year),
    ]))
    X_input = pd.DataFrame([inp])

    names_out = []; probs_out = []; preds_out = []
    for name, model in MODELS.items():
        pred, prob = predict_with_model(name, model, X_input)
        names_out.append(name); probs_out.append(prob); preds_out.append(pred)

    votes = sum(preds_out)
    avg   = np.mean(probs_out) * 100

    if   votes == 4: cons = "CRITICAL — All 4 predict BLEACHING"
    elif votes == 3: cons = "HIGH RISK — 3/4 predict bleaching"
    elif votes == 2: cons = "MODERATE — 2/4 predict bleaching"
    elif votes == 1: cons = "LOW RISK — 1/4 predict bleaching"
    else:            cons = "SAFE — All 4 predict NO bleaching"

    txt  = "=" * 50 + "\n  BLEACHING RISK PREDICTION\n" + "=" * 50 + "\n\n"
    for n, p, pred in zip(names_out, probs_out, preds_out):
        bar = "█" * int(p * 20) + "░" * (20 - int(p * 20))
        lbl = "BLEACHING" if pred == 1 else "SAFE"
        txt += f"  {n:<25}\n  [{bar}] {p*100:5.1f}%  ->  {lbl}\n\n"
    txt += "─" * 50
    txt += f"\n  Avg probability : {avg:.1f}%"
    txt += f"\n  Votes           : {votes}/{len(MODELS)}"
    txt += f"\n  CONSENSUS       : {cons}\n" + "=" * 50

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor("#0f172a")

    ax1 = axes[0]; ax1.set_facecolor("#1e293b")
    bcolors = [COLORS.get(n, "#888") for n in names_out]
    bars = ax1.barh(names_out, [p * 100 for p in probs_out], color=bcolors, height=0.5)
    ax1.axvline(50, color="white", ls="--", lw=1, alpha=0.5)
    ax1.set_xlim(0, 108)
    ax1.set_xlabel("Bleaching Probability (%)", color="white")
    ax1.set_title("Prediction by Model", color="white", fontweight="bold")
    ax1.tick_params(colors="white"); ax1.spines[:].set_visible(False)
    for bar, p in zip(bars, probs_out):
        ax1.text(min(p * 100 + 2, 103), bar.get_y() + bar.get_height() / 2,
                 f"{p*100:.1f}%", va="center", color="white", fontsize=10, fontweight="bold")

    ax2 = axes[1]; ax2.set_facecolor("#1e293b"); ax2.set_aspect("equal")
    gc = "#FF4444" if votes >= 3 else "#FFA500" if votes >= 2 else "#2ECC71"
    ax2.pie([avg, 100 - avg], colors=[gc, "#2d3748"],
            startangle=90, counterclock=False,
            wedgeprops=dict(width=0.45, edgecolor="#0f172a", linewidth=2))
    ax2.text(0,  0.08, f"{avg:.0f}%", ha="center", va="center",
             color="white", fontsize=22, fontweight="bold")
    ax2.text(0, -0.18, "Avg Risk", ha="center", color="#94a3b8", fontsize=9)
    ax2.text(0, -0.38, cons.split("—")[0].strip(),
             ha="center", color=gc, fontsize=9, fontweight="bold")
    ax2.set_title("Consensus Gauge", color="white", fontweight="bold")
    plt.tight_layout(pad=1.5)
    plt.close(fig)
    return txt, fig


def run_batch():
    if not MODELS:
        return "No models loaded.", None

    model_names = list(MODELS.keys())
    site_names  = list(SAMPLE_SITES.keys())
    all_probs   = {n: [] for n in model_names}

    for site, vals in SAMPLE_SITES.items():
        X_site = pd.DataFrame([dict(zip(ALL_FEAT_COLS, vals))])
        for name, model in MODELS.items():
            _, prob = predict_with_model(name, model, X_site)
            all_probs[name].append(prob * 100)

    short = ["LR", "RF", "XGB", "SVM"][:len(model_names)]
    txt   = "BATCH PREDICTION — ALL SAMPLE SITES\n" + "=" * 70 + "\n"
    txt  += "  " + f"{'Site':<35}" + "".join(f"{s:>10}" for s in short) + "  Consensus\n"
    txt  += "-" * 70 + "\n"
    for i, site in enumerate(site_names):
        probs = [all_probs[n][i] for n in model_names]
        votes = sum(1 for p in probs if p >= 50)
        if   votes == 4: c = "CRITICAL"
        elif votes == 3: c = "HIGH RISK"
        elif votes == 2: c = "MODERATE"
        elif votes == 1: c = "LOW RISK"
        else:            c = "SAFE"
        txt += f"  {site[:33]:<35}" + "".join(f"{p:>9.1f}%" for p in probs) + f"  {c}\n"
    txt += "=" * 70

    prob_matrix = np.array([all_probs[n] for n in model_names])
    fig, ax = plt.subplots(figsize=(14, max(5, len(site_names) * 0.55 + 1)))
    fig.patch.set_facecolor("#0f172a"); ax.set_facecolor("#1e293b")
    im = ax.imshow(prob_matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=100)
    ax.set_xticks(range(len(site_names)))
    ax.set_xticklabels([s[:28] for s in site_names], rotation=45, ha="right",
                       color="white", fontsize=8)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, color="white", fontsize=9)
    for i in range(len(model_names)):
        for j in range(len(site_names)):
            v = prob_matrix[i, j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                    color="white" if v > 55 else "black", fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Bleaching Prob (%)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    ax.set_title("Batch Prediction — All Sites x All Models",
                 color="white", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.close(fig)
    return txt, fig


def run_dataset_validation():
    """
    Downloads the real BCO-DMO dataset, rebuilds the exact same test split used
    during training (random_state=42, stratify=y), picks 12 balanced real records
    (6 bleaching + 6 no-bleaching from the held-out test set), runs all 4 models,
    and returns a scoreboard + accuracy/heatmap chart.
    """
    if not MODELS:
        return "No models loaded.", None

    import urllib.request
    from sklearn.model_selection import train_test_split

    LOCAL_CSV   = "global_bleaching_environmental.csv"
    PRIMARY_URL = ("https://datadocs.bco-dmo.org/dataset/773466/file/"
                   "B11vA82u7y2Owp/global_bleaching_environmental.csv")

    if not os.path.exists(LOCAL_CSV):
        try:
            urllib.request.urlretrieve(PRIMARY_URL, LOCAL_CSV)
        except Exception as e:
            return (f"Dataset download failed: {e}\n"
                    "Place global_bleaching_environmental.csv in the app folder.", None)

    # Rebuild preprocessing identical to training
    df = pd.read_csv(LOCAL_CSV, low_memory=False)
    df.replace("nd", np.nan, inplace=True)

    BLEACH_COL = next(
        (c for c in df.columns if "percent_bleach" in c.lower()),
        next((c for c in df.columns if "bleach" in c.lower() and
              pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.3), None))
    df[BLEACH_COL] = pd.to_numeric(df[BLEACH_COL], errors="coerce")
    df.dropna(subset=[BLEACH_COL], inplace=True)
    df["Bleaching_Binary"] = (df[BLEACH_COL] > 0).astype(int)

    CANDIDATES = [
        "ClimSST", "Temperature_Mean", "Temperature_Minimum", "Temperature_Maximum",
        "SSTA", "SSTA_DHW", "TSA", "TSA_DHW", "TSA_DHW_Frequency",
        "Windspeed", "SSTA_Frequency", "SSTA_Frequency_Standard_Deviation",
        "Turbidity_ct", "Turbidity", "Cyclone_Frequency",
        "Distance", "Depth", "Latitude_Degrees", "Longitude_Degrees", "Date_Year",
    ]
    feats = [c for c in CANDIDATES if c in df.columns and
             pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.2]

    df_m = df[feats + ["Bleaching_Binary"]].copy()
    for col in feats:
        df_m[col] = pd.to_numeric(df_m[col], errors="coerce")
        df_m[col].fillna(df_m[col].median(), inplace=True)
    for col in feats:
        Q1, Q3 = df_m[col].quantile(0.25), df_m[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:
            df_m[col] = df_m[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    corr  = df_m[feats].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.97)]
    feats = [f for f in feats if f not in to_drop]
    df_m.drop(columns=[c for c in to_drop if c in df_m.columns], inplace=True)

    _, X_test, _, y_test = train_test_split(
        df_m[feats], df_m["Bleaching_Binary"],
        test_size=0.2, random_state=42, stratify=df_m["Bleaching_Binary"])

    # Balanced sample: 6 bleaching + 6 no-bleaching
    rng   = np.random.default_rng(42)
    b_idx = y_test[y_test == 1].index
    n_idx = y_test[y_test == 0].index
    chosen = np.concatenate([
        rng.choice(b_idx, size=min(6, len(b_idx)), replace=False),
        rng.choice(n_idx, size=min(6, len(n_idx)), replace=False),
    ])
    rng.shuffle(chosen)
    X_val = X_test.loc[chosen]
    y_val = y_test.loc[chosen]

    # Run all models on every selected record
    model_names = list(MODELS.keys())
    results = []
    for idx in chosen:
        row_X  = X_val.loc[[idx]]
        true_y = int(y_val.loc[idx])
        entry  = {"true": true_y, "preds": {}, "probs": {}}
        for name, model in MODELS.items():
            pred, prob = predict_with_model(name, model, row_X)
            entry["preds"][name] = pred
            entry["probs"][name] = prob
        results.append(entry)

    # Scoreboard text
    SHORT   = {"Logistic Regression": "LR ", "Random Forest": "RF ",
               "XGBoost": "XGB", "SVM (RBF Kernel)": "SVM"}
    headers = [SHORT.get(n, n[:4]) for n in model_names]
    col_w   = 11

    txt  = "=" * 72 + "\n"
    txt += "  DATASET VALIDATION — REAL RECORDS vs MODEL PREDICTIONS\n"
    txt += f"  ({(y_val==1).sum()} bleaching + {(y_val==0).sum()} no-bleaching from held-out test set)\n"
    txt += "=" * 72 + "\n"
    txt += f"  {'#':<4} {'Actual':<10}" + "".join(f"{h:^{col_w}}" for h in headers) + "  Agreement\n"
    txt += "-" * 72 + "\n"

    correct_per_model = {n: 0 for n in model_names}
    all_agree = 0

    for i, res in enumerate(results, 1):
        true_lbl = "BLEACH  " if res["true"] == 1 else "NO-BLCH "
        cells = []; row_ok = 0
        for name in model_names:
            pred = res["preds"][name]
            prob = res["probs"][name]
            ok   = pred == res["true"]
            correct_per_model[name] += int(ok)
            row_ok += int(ok)
            cells.append(f"{'OK' if ok else 'XX'} {prob*100:4.0f}%")
        all_agree += int(row_ok == len(model_names))
        agree = ("ALL CORRECT" if row_ok == len(model_names) else
                 "ALL WRONG  " if row_ok == 0 else
                 f"{row_ok}/{len(model_names)} correct")
        txt += f"  {i:<4} {true_lbl:<10}" + "".join(f"{c:^{col_w}}" for c in cells) + f"  {agree}\n"

    txt += "-" * 72 + "\n"
    txt += f"  {'ACCURACY':<14}" + "".join(
        f"{correct_per_model[n]/len(results)*100:^{col_w}.1f}%" for n in model_names)
    txt += f"  {all_agree}/{len(results)} all-agree\n"
    txt += "=" * 72

    # Chart: accuracy bars + correctness heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.patch.set_facecolor("#0f172a")

    ax1 = axes[0]; ax1.set_facecolor("#1e293b")
    accs   = [correct_per_model[n] / len(results) * 100 for n in model_names]
    colors = [COLORS.get(n, "#888") for n in model_names]
    bars   = ax1.barh(model_names, accs, color=colors, height=0.5)
    ax1.axvline(50, color="white", ls="--", lw=1, alpha=0.4)
    ax1.set_xlim(0, 118)
    ax1.set_xlabel("Accuracy on real records (%)", color="white")
    ax1.set_title(f"Model Accuracy — {len(results)} Real Dataset Records",
                  color="white", fontweight="bold")
    ax1.tick_params(colors="white"); ax1.spines[:].set_visible(False)
    for bar, acc, n in zip(bars, accs, model_names):
        ax1.text(min(acc + 1.5, 114), bar.get_y() + bar.get_height() / 2,
                 f"{acc:.1f}%  ({correct_per_model[n]}/{len(results)})",
                 va="center", color="white", fontsize=9, fontweight="bold")

    ax2 = axes[1]; ax2.set_facecolor("#1e293b")
    grid = np.array([[int(res["preds"][n] == res["true"]) for res in results]
                     for n in model_names], dtype=float)
    ax2.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    xlabels = [f"{'B' if r['true']==1 else 'N'}{j+1}" for j, r in enumerate(results)]
    ax2.set_xticks(range(len(results)))
    ax2.set_xticklabels(xlabels, color="white", fontsize=8)
    ax2.set_yticks(range(len(model_names)))
    ax2.set_yticklabels([SHORT.get(n, n) for n in model_names], color="white", fontsize=9)
    for i, name in enumerate(model_names):
        for j, res in enumerate(results):
            ok = res["preds"][name] == res["true"]
            ax2.text(j, i, "v" if ok else "X", ha="center", va="center",
                     fontsize=12, color="white", fontweight="bold")
    ax2.set_title("Correct (v) / Wrong (X) per Record  (B=Bleaching, N=No-bleaching)",
                  color="white", fontweight="bold", fontsize=9)
    plt.tight_layout(pad=2)
    plt.close(fig)
    return txt, fig


# ── Slider config — Kelvin, DHW capped at 3.0 (training range 0.00–2.97) ──────
SLIDER_CFG = [
    ("ClimSST — Climatological SST (K)",      293.0, 307.0, 299.5,  0.1),
    ("Temperature_Mean — Mean SST (K)",        293.0, 308.0, 300.5,  0.1),
    ("Temperature_Minimum (K)",                290.0, 305.0, 297.8,  0.1),
    ("Temperature_Maximum (K)",                295.0, 310.0, 302.0,  0.1),
    ("SSTA — SST Anomaly (K)",                  -3.0,   5.0,   1.0,  0.1),
    ("SSTA_DHW — SST Anomaly DHW",               0.0,   3.0,   1.0, 0.05),
    ("TSA — Thermal Stress Anomaly (K)",         -2.0,   5.0,   1.2,  0.1),
    ("TSA_DHW — Thermal Stress DHW",             0.0,   3.0,   1.5, 0.05),
    ("TSA_DHW_Frequency",                        0.0,   1.0,   0.2, 0.01),
    ("Windspeed (m/s)",                          0.0,  15.0,   5.0,  0.1),
    ("SSTA_Frequency",                           0.0,   1.0,   0.3, 0.01),
    ("SSTA_Frequency_Standard_Deviation",        0.0,   0.5,   0.1, 0.01),
    ("Turbidity_ct",                             0.0,   1.0,  0.05, 0.01),
    ("Turbidity",                                0.0,   1.0,  0.05, 0.01),
    ("Cyclone_Frequency",                        0.0,   1.0,  0.05, 0.01),
    ("Distance — Distance to Land (km)",         0.0, 100.0,  10.0,  0.5),
    ("Depth — Reef Depth (m)",                   0.0,  50.0,   5.0,  0.5),
    ("Latitude_Degrees",                       -35.0,  35.0, -18.0,  0.1),
    ("Longitude_Degrees",                     -180.0, 180.0, 147.0,  0.1),
    ("Date_Year",                             1980.0,2024.0,2016.0,  1.0),
]


# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Base(primary_hue="green", neutral_hue="slate"),
    title="Coral Reef Bleaching Predictor",
    css=".result-box textarea{font-family:monospace;font-size:13px;}",
) as app:

    gr.Markdown("""
# Coral Reef Bleaching Risk Predictor
### ML Assignment - SLIIT | LR . Random Forest . XGBoost . SVM
> **Temperature inputs are in Kelvin** -- 297K = 24C | 300K = 27C | 303K = 30C | 306K = 33C
""")
    loaded_str = ", ".join(MODELS.keys()) if MODELS else "None — check models/ folder"
    gr.Markdown(f"> **Models loaded:** {loaded_str}")

    with gr.Tabs():

        # ── TAB 1: Single Prediction ──────────────────────────────────────────
        with gr.Tab("Single Prediction"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Ocean Conditions (temperatures in Kelvin)")
                    with gr.Accordion("Temperature (K)", open=True):
                        sliders_t = [gr.Slider(mn, mx, v, step=st, label=lbl)
                                     for lbl, mn, mx, v, st in SLIDER_CFG[:4]]
                    with gr.Accordion("Thermal Stress", open=True):
                        sliders_s = [gr.Slider(mn, mx, v, step=st, label=lbl)
                                     for lbl, mn, mx, v, st in SLIDER_CFG[4:9]]
                    with gr.Accordion("Wind and Frequency", open=False):
                        sliders_w = [gr.Slider(mn, mx, v, step=st, label=lbl)
                                     for lbl, mn, mx, v, st in SLIDER_CFG[9:12]]
                    with gr.Accordion("Water and Location", open=False):
                        sliders_l = [gr.Slider(mn, mx, v, step=st, label=lbl)
                                     for lbl, mn, mx, v, st in SLIDER_CFG[12:]]
                    ALL_SLIDERS = sliders_t + sliders_s + sliders_w + sliders_l
                    predict_btn = gr.Button("Predict Bleaching Risk",
                                            variant="primary", size="lg")
                with gr.Column(scale=1):
                    gr.Markdown("### Results")
                    result_txt   = gr.Textbox(label="Model Predictions", lines=16,
                                              elem_classes=["result-box"])
                    result_chart = gr.Plot(label="Risk Visualization")

            predict_btn.click(fn=run_prediction, inputs=ALL_SLIDERS,
                              outputs=[result_txt, result_chart])

        # ── TAB 2: Sample Reef Sites ──────────────────────────────────────────
        with gr.Tab("Sample Reef Sites"):
            gr.Markdown("### Select a pre-configured reef site and predict")
            site_dd    = gr.Dropdown(choices=list(SAMPLE_SITES.keys()),
                                     label="Reef Site",
                                     value=list(SAMPLE_SITES.keys())[0])
            load_btn   = gr.Button("Load Site and Predict", variant="primary")
            site_txt   = gr.Textbox(label="Prediction Results", lines=16,
                                    elem_classes=["result-box"])
            site_chart = gr.Plot(label="Risk Chart")

            def load_and_predict(site_name):
                vals = SAMPLE_SITES.get(site_name, [0.0] * 20)
                return run_prediction(*vals)

            load_btn.click(fn=load_and_predict, inputs=[site_dd],
                           outputs=[site_txt, site_chart])

            rows = ["| # | Site | ClimSST (K) | Temp_Mean (K) | SSTA | TSA_DHW | Year |",
                    "|---|---|---|---|---|---|---|"]
            for i, (site, vals) in enumerate(SAMPLE_SITES.items(), 1):
                rows.append(f"| {i} | {site} | {vals[0]} | {vals[1]}"
                             f" | {vals[4]} | {vals[7]} | {int(vals[19])} |")
            gr.Markdown("\n".join(rows))

        # ── TAB 3: Batch Prediction ───────────────────────────────────────────
        with gr.Tab("Batch Prediction"):
            gr.Markdown("### Run all 10 sample sites through all 4 models simultaneously")
            batch_btn   = gr.Button("Run Batch Prediction", variant="primary", size="lg")
            batch_txt   = gr.Textbox(label="Batch Results", lines=20,
                                     elem_classes=["result-box"])
            batch_chart = gr.Plot(label="Probability Heatmap — All Sites x All Models")
            batch_btn.click(fn=run_batch, inputs=[], outputs=[batch_txt, batch_chart])

        # ── TAB 4: Dataset Validation ─────────────────────────────────────────
        with gr.Tab("Dataset Validation"):
            gr.Markdown("""
### Real Records from the BCO-DMO Dataset
Downloads the actual dataset, rebuilds the **exact same test split** used during training
(`random_state=42`, `stratify=y`), picks 12 balanced real records
(6 bleaching + 6 no-bleaching from the held-out test set),
runs all 4 models, and shows whether each prediction is correct or wrong.
""")
            val_btn   = gr.Button("Run Dataset Validation", variant="primary", size="lg")
            val_txt   = gr.Textbox(label="Scoreboard", lines=22,
                                   elem_classes=["result-box"])
            val_chart = gr.Plot(label="Accuracy and Correctness Heatmap")
            val_btn.click(fn=run_dataset_validation, inputs=[],
                          outputs=[val_txt, val_chart])

        # ── TAB 5: Model Info ─────────────────────────────────────────────────
        with gr.Tab("Model Info"):
            gr.Markdown("### Loaded Models and Performance")
            info_rows = ["| Model | Type | Features | Test Accuracy | F1 | ROC-AUC |",
                         "|---|---|---|---|---|---|"]
            for name, meta in METADATA.items():
                mtype = "Pipeline (no scaler file)" if isinstance(MODELS[name], _SkPipeline) else "Standard"
                info_rows.append(
                    f"| {name} | {mtype} | {len(meta.get('features', []))} |"
                    f" {meta.get('test_accuracy', '—')} |"
                    f" {meta.get('test_f1', '—')} |"
                    f" {meta.get('test_roc_auc', '—')} |"
                )
            gr.Markdown("\n".join(info_rows))
            gr.Markdown("""
**Dataset:** Global Coral Bleaching Database 1980-2020 (BCO-DMO)
**DOI:** https://doi.org/10.26008/1912/bco-dmo.773466.2

**Model types and scaler files:**

| Model | Type | Scaler file |
|---|---|---|
| Logistic Regression | sklearn Pipeline (scaler + poly + LR inside) | None - built into Pipeline |
| Random Forest | RandomForestClassifier | None |
| XGBoost | XGBClassifier | None |
| SVM (RBF Kernel) | SVC | svm_scaler.pkl |

**Key bleaching risk indicators (Kelvin):**
- TSA_DHW > 2.5 = high bleaching risk (training range: 0.00-2.97)
- Temperature_Mean > 302K (~29C) + SSTA > 1.5 = elevated risk
- Low turbidity + high DHW = maximum risk

**Temperature reference:** 297K = 24C | 300K = 27C | 303K = 30C | 306K = 33C
""")

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)