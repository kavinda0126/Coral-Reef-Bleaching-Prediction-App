"""
Coral Reef Bleaching Prediction App
Hugging Face Spaces / Local Gradio App
Loads 4 models from models/ folder
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

# ── Load all 4 models from models/ ────────────────────────────────────────────
MODEL_PREFIXES = {
    "Logistic Regression": "lr",
    "Random Forest"       : "rf",
    "XGBoost"             : "xgb",
    "SVM (RBF Kernel)"    : "svm",
}

MODELS={}; SCALERS={}; FEATURES={}; METADATA={}

for name, prefix in MODEL_PREFIXES.items():
    mpath = f"models/{prefix}/{prefix}_model.pkl"
    fpath = f"models/{prefix}/{prefix}_features.pkl"
    spath = f"models/{prefix}/{prefix}_scaler.pkl"
    jpath = f"models/{prefix}/{prefix}_metadata.json"
    if not os.path.exists(mpath):
        print(f"WARNING: {mpath} not found — {name} skipped")
        continue
    MODELS[name]   = joblib.load(mpath)
    FEATURES[name] = joblib.load(fpath)
    SCALERS[name]  = joblib.load(spath) if os.path.exists(spath) else None
    if os.path.exists(jpath):
        with open(jpath) as f: METADATA[name]=json.load(f)
    print(f"Loaded: {name}")

COLORS = {
    "Logistic Regression": "#E74C3C",
    "Random Forest"       : "#2ECC71",
    "XGBoost"             : "#F39C12",
    "SVM (RBF Kernel)"    : "#3498DB",
}

ALL_FEAT_COLS = [
    "ClimSST","Temperature_Mean","Temperature_Minimum","Temperature_Maximum",
    "SSTA","SSTA_DHW","TSA","TSA_DHW","TSA_DHW_Frequency",
    "Windspeed","SSTA_Frequency","SSTA_Frequency_Standard_Deviation",
    "Turbidity_ct","Turbidity","Cyclone_Frequency",
    "Distance","Depth","Latitude_Degrees","Longitude_Degrees","Date_Year",
]

# ── Sample reef sites ──────────────────────────────────────────────────────────
SAMPLE_SITES = {
    "Great Barrier Reef — 2016 Bleaching (Severe)"    :[28.5,31.8,27.1,33.5,2.1,6.5,2.5,9.5,0.30,4.2,0.40,0.20,0.04,0.04,0.10,15.0,5.0,-18.3,147.2,2016],
    "Great Barrier Reef — 2020 (Mild)"                :[28.5,29.2,26.5,31.0,0.8,3.1,1.0,3.1,0.12,5.1,0.20,0.10,0.05,0.05,0.10,15.0,5.0,-18.3,147.2,2020],
    "Maldives — Healthy Reef"                          :[29.0,29.5,27.2,31.0,0.3,1.2,0.4,1.2,0.05,6.0,0.10,0.05,0.03,0.03,0.05,8.0,6.0,4.2,73.5,2019],
    "Maldives — 2016 Bleaching Event"                  :[29.0,32.1,27.5,34.0,2.8,10.2,3.1,10.2,0.40,3.5,0.50,0.25,0.03,0.03,0.05,8.0,6.0,4.2,73.5,2016],
    "Red Sea — Heat-Resilient Reef"                    :[27.5,28.0,25.5,30.0,0.1,0.5,0.2,0.5,0.02,7.0,0.05,0.03,0.08,0.08,0.02,20.0,8.0,22.5,37.8,2018],
    "Caribbean — Warm Season"                          :[28.8,30.9,26.8,32.5,1.7,6.8,2.0,6.8,0.25,5.5,0.35,0.15,0.06,0.06,0.08,10.0,4.0,17.5,-66.0,2015],
    "Sri Lanka — Indian Ocean Coast"                   :[29.2,30.5,27.0,32.0,1.2,4.5,1.5,4.5,0.18,4.8,0.25,0.12,0.07,0.07,0.06,5.0,3.0,6.9,81.5,2016],
    "Deep Protected Reef"                              :[26.0,26.5,24.5,28.5,-0.2,0.2,-0.1,0.2,0.01,8.0,0.02,0.01,0.12,0.12,0.01,30.0,25.0,-22.0,114.0,2017],
    "Turbid Coastal Reef (Natural Protection)"         :[28.0,30.2,26.0,32.0,1.9,7.1,2.2,7.1,0.28,3.2,0.38,0.18,0.35,0.35,0.09,2.0,3.0,-8.5,115.2,2016],
    "El Nino Hotspot — Critical Bleaching"             :[30.1,33.5,28.0,35.5,3.8,14.2,4.5,14.2,0.55,2.8,0.65,0.30,0.02,0.02,0.12,12.0,4.0,-16.0,145.5,2016],
}


def run_prediction(ClimSST,Temperature_Mean,Temperature_Minimum,Temperature_Maximum,
                   SSTA,SSTA_DHW,TSA,TSA_DHW,TSA_DHW_Frequency,Windspeed,
                   SSTA_Frequency,SSTA_Frequency_Standard_Deviation,
                   Turbidity_ct,Turbidity,Cyclone_Frequency,
                   Distance,Depth,Latitude_Degrees,Longitude_Degrees,Date_Year):
    if not MODELS:
        return "No models loaded.", None

    inp = dict(zip(ALL_FEAT_COLS,[
        float(ClimSST),float(Temperature_Mean),float(Temperature_Minimum),float(Temperature_Maximum),
        float(SSTA),float(SSTA_DHW),float(TSA),float(TSA_DHW),float(TSA_DHW_Frequency),
        float(Windspeed),float(SSTA_Frequency),float(SSTA_Frequency_Standard_Deviation),
        float(Turbidity_ct),float(Turbidity),float(Cyclone_Frequency),
        float(Distance),float(Depth),float(Latitude_Degrees),float(Longitude_Degrees),float(Date_Year)
    ]))

    names_out=[]; probs_out=[]; preds_out=[]
    for name,model in MODELS.items():
        feats=FEATURES[name]; scaler=SCALERS[name]
        row={f:inp.get(f,0.0) for f in feats}
        X=pd.DataFrame([row])[feats]
        if scaler: X=scaler.transform(X)
        else:      X=X.values
        pred=int(model.predict(X)[0])
        prob=float(model.predict_proba(X)[0][1])
        names_out.append(name); probs_out.append(prob); preds_out.append(pred)

    votes=sum(preds_out); avg=np.mean(probs_out)*100
    if votes==4:   cons="CRITICAL — All 4 predict BLEACHING"
    elif votes==3: cons="HIGH RISK — 3/4 predict bleaching"
    elif votes==2: cons="MODERATE — 2/4 predict bleaching"
    elif votes==1: cons="LOW RISK — 1/4 predict bleaching"
    else:          cons="SAFE — All 4 predict NO bleaching"

    txt ="="*50+"\n  BLEACHING RISK PREDICTION\n"+"="*50+"\n\n"
    for n,p,pred in zip(names_out,probs_out,preds_out):
        bar="█"*int(p*20)+"░"*(20-int(p*20))
        lbl="BLEACHING" if pred==1 else "SAFE"
        txt+=f"  {n:<25}\n  [{bar}] {p*100:5.1f}%  →  {lbl}\n\n"
    txt+="─"*50+f"\n  Avg probability : {avg:.1f}%"
    txt+=f"\n  Votes           : {votes}/{len(MODELS)}"
    txt+=f"\n  CONSENSUS       : {cons}\n"+"="*50

    # Chart
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    fig.patch.set_facecolor("#0f172a")
    ax1=axes[0]; ax1.set_facecolor("#1e293b")
    bcolors=[COLORS.get(n,"#888") for n in names_out]
    bars=ax1.barh(names_out,[p*100 for p in probs_out],color=bcolors,height=0.5)
    ax1.axvline(50,color="white",ls="--",lw=1,alpha=0.5)
    ax1.set_xlim(0,108); ax1.set_xlabel("Bleaching Probability (%)",color="white")
    ax1.set_title("Prediction by Model",color="white",fontweight="bold")
    ax1.tick_params(colors="white"); ax1.spines[:].set_visible(False)
    for bar,p in zip(bars,probs_out):
        ax1.text(min(p*100+2,103),bar.get_y()+bar.get_height()/2,
                 f"{p*100:.1f}%",va="center",color="white",fontsize=10,fontweight="bold")

    ax2=axes[1]; ax2.set_facecolor("#1e293b"); ax2.set_aspect("equal")
    gc="#FF4444" if votes>=3 else "#FFA500" if votes>=2 else "#2ECC71"
    ax2.pie([avg,100-avg],colors=[gc,"#2d3748"],startangle=90,counterclock=False,
            wedgeprops=dict(width=0.45,edgecolor="#0f172a",linewidth=2))
    ax2.text(0,0.08,f"{avg:.0f}%",ha="center",va="center",color="white",
             fontsize=22,fontweight="bold")
    ax2.text(0,-0.18,"Avg Risk",ha="center",color="#94a3b8",fontsize=9)
    ax2.text(0,-0.38,cons.split("—")[0].strip(),ha="center",color=gc,
             fontsize=9,fontweight="bold")
    ax2.set_title("Consensus Gauge",color="white",fontweight="bold")
    plt.tight_layout(pad=1.5)
    plt.close(fig)
    return txt, fig


def load_sample(site_name):
    vals=SAMPLE_SITES.get(site_name,[0.0]*20)
    return vals


def run_batch():
    if not MODELS: return "No models loaded.", None
    all_probs={n:[] for n in MODELS}
    site_names=list(SAMPLE_SITES.keys())
    for site,vals in SAMPLE_SITES.items():
        inp=dict(zip(ALL_FEAT_COLS,vals))
        for name,model in MODELS.items():
            feats=FEATURES[name]; scaler=SCALERS[name]
            row={f:inp.get(f,0.0) for f in feats}
            X=pd.DataFrame([row])[feats]
            if scaler: X=scaler.transform(X)
            else:      X=X.values
            prob=float(model.predict_proba(X)[0][1])
            all_probs[name].append(prob*100)

    txt="BATCH PREDICTION — ALL SAMPLE SITES\n"+"="*70+"\n"
    model_names=list(MODELS.keys())
    short=["LR","RF","XGB","SVM"][:len(model_names)]
    txt+="  "+f"{'Site':<35}"+"".join(f"{s:>10}" for s in short)+"  Consensus\n"
    txt+="-"*70+"\n"
    for i,site in enumerate(site_names):
        probs=[all_probs[n][i] for n in model_names]
        votes=sum(1 for p in probs if p>=50)
        if votes==4:   c="CRITICAL"
        elif votes==3: c="HIGH RISK"
        elif votes==2: c="MODERATE"
        elif votes==1: c="LOW RISK"
        else:          c="SAFE"
        row=f"  {site[:33]:<35}"+"".join(f"{p:>9.1f}%" for p in probs)+f"  {c}"
        txt+=row+"\n"
    txt+="="*70

    prob_matrix=np.array([all_probs[n] for n in model_names])
    fig,ax=plt.subplots(figsize=(14,max(5,len(site_names)*0.55+1)))
    fig.patch.set_facecolor("#0f172a"); ax.set_facecolor("#1e293b")
    im=ax.imshow(prob_matrix,aspect="auto",cmap="RdYlGn_r",vmin=0,vmax=100)
    ax.set_xticks(range(len(site_names)))
    ax.set_xticklabels([s[:28] for s in site_names],rotation=45,ha="right",
                       color="white",fontsize=8)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names,color="white",fontsize=9)
    for i in range(len(model_names)):
        for j in range(len(site_names)):
            v=prob_matrix[i,j]
            ax.text(j,i,f"{v:.0f}%",ha="center",va="center",fontsize=8,
                    color="white" if v>55 else "black",fontweight="bold")
    cbar=plt.colorbar(im,ax=ax,shrink=0.7)
    cbar.set_label("Bleaching Prob (%)",color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(),color="white")
    ax.set_title("Batch Prediction — All Sites × All Models",
                 color="white",fontweight="bold",fontsize=12)
    plt.tight_layout()
    plt.close(fig)
    return txt, fig


# ── Slider config ──────────────────────────────────────────────────────────────
SLIDER_CFG=[
    ("ClimSST — Climatological SST (°C)",        20.0,35.0,28.5,0.1),
    ("Temperature_Mean — Mean SST (°C)",          20.0,36.0,29.5,0.1),
    ("Temperature_Minimum (°C)",                  18.0,33.0,27.0,0.1),
    ("Temperature_Maximum (°C)",                  22.0,38.0,31.5,0.1),
    ("SSTA — SST Anomaly (°C)",                  -3.0, 5.0, 1.0,0.1),
    ("SSTA_DHW — SST Anomaly DHW",                0.0,16.0, 4.0,0.1),
    ("TSA — Thermal Stress Anomaly (°C)",        -2.0, 6.0, 1.2,0.1),
    ("TSA_DHW — Thermal Stress DHW",              0.0,18.0, 5.0,0.1),
    ("TSA_DHW_Frequency",                         0.0, 1.0, 0.2,0.01),
    ("Windspeed (m/s)",                           0.0,15.0, 5.0,0.1),
    ("SSTA_Frequency",                            0.0, 1.0, 0.3,0.01),
    ("SSTA_Frequency_Standard_Deviation",         0.0, 0.5, 0.1,0.01),
    ("Turbidity_ct",                              0.0, 1.0,0.05,0.01),
    ("Turbidity",                                 0.0, 1.0,0.05,0.01),
    ("Cyclone_Frequency",                         0.0, 1.0,0.05,0.01),
    ("Distance — Distance to Land (km)",          0.0,100.0,10.0,0.5),
    ("Depth — Reef Depth (m)",                    0.0,50.0, 5.0,0.5),
    ("Latitude_Degrees",                        -35.0,35.0,-18.0,0.1),
    ("Longitude_Degrees",                      -180.0,180.0,147.0,0.1),
    ("Date_Year",                              1980.0,2024.0,2016.0,1.0),
]


# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Base(primary_hue="green",neutral_hue="slate"),
    title="Coral Reef Bleaching Predictor",
    css=".result-box textarea{font-family:monospace;font-size:13px;}"
) as app:

    gr.Markdown("""
# 🪸 Coral Reef Bleaching Risk Predictor
### ML Assignment — SLIIT | LR · Random Forest · XGBoost · SVM
""")

    loaded_str = ", ".join(MODELS.keys()) if MODELS else "None — upload .pkl files"
    gr.Markdown(f"> **Models loaded:** {loaded_str}")

    with gr.Tabs():

        # ── TAB 1: Single Prediction ─────────────────────────────────────────
        with gr.Tab("🔍 Single Prediction"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Ocean Conditions")
                    with gr.Accordion("🌡️ Temperature", open=True):
                        sliders_t=[gr.Slider(mn,mx,v,step=st,label=lbl)
                                   for lbl,mn,mx,v,st in SLIDER_CFG[:4]]
                    with gr.Accordion("🔥 Thermal Stress", open=True):
                        sliders_s=[gr.Slider(mn,mx,v,step=st,label=lbl)
                                   for lbl,mn,mx,v,st in SLIDER_CFG[4:9]]
                    with gr.Accordion("💨 Wind & Frequency", open=False):
                        sliders_w=[gr.Slider(mn,mx,v,step=st,label=lbl)
                                   for lbl,mn,mx,v,st in SLIDER_CFG[9:12]]
                    with gr.Accordion("💧 Water & Location", open=False):
                        sliders_l=[gr.Slider(mn,mx,v,step=st,label=lbl)
                                   for lbl,mn,mx,v,st in SLIDER_CFG[12:]]
                    ALL_SLIDERS=sliders_t+sliders_s+sliders_w+sliders_l
                    predict_btn=gr.Button("🔍 Predict Bleaching Risk",
                                          variant="primary",size="lg")

                with gr.Column(scale=1):
                    gr.Markdown("### Results")
                    result_txt  =gr.Textbox(label="Model Predictions",lines=16,
                                            elem_classes=["result-box"])
                    result_chart=gr.Plot(label="Risk Visualization")

            predict_btn.click(fn=run_prediction,inputs=ALL_SLIDERS,
                               outputs=[result_txt,result_chart])

        # ── TAB 2: Sample Sites ───────────────────────────────────────────────
        with gr.Tab("🗺️ Sample Reef Sites"):
            gr.Markdown("### Select a pre-configured reef site to auto-fill sliders")
            site_dd  =gr.Dropdown(choices=list(SAMPLE_SITES.keys()),
                                   label="Reef Site",
                                   value=list(SAMPLE_SITES.keys())[0])
            load_btn =gr.Button("📍 Load Site & Predict",variant="primary")
            site_txt =gr.Textbox(label="Prediction Results",lines=16,
                                  elem_classes=["result-box"])
            site_chart=gr.Plot(label="Risk Chart")

            def load_and_predict(site_name):
                vals=SAMPLE_SITES.get(site_name,[0.0]*20)
                return run_prediction(*vals)

            load_btn.click(fn=load_and_predict,inputs=[site_dd],
                           outputs=[site_txt,site_chart])

            # Info table
            rows=["| # | Site | ClimSST | Temp | SSTA | TSA_DHW | Year |",
                  "|---|---|---|---|---|---|---|"]
            for i,(site,vals) in enumerate(SAMPLE_SITES.items(),1):
                rows.append(f"| {i} | {site} | {vals[0]} | {vals[1]} | {vals[4]} | {vals[7]} | {int(vals[19])} |")
            gr.Markdown("\n".join(rows))

        # ── TAB 3: Batch Prediction ───────────────────────────────────────────
        with gr.Tab("📋 Batch Prediction"):
            gr.Markdown("### Run all 10 sample sites through all 4 models simultaneously")
            batch_btn  =gr.Button("▶ Run Batch Prediction",variant="primary",size="lg")
            batch_txt  =gr.Textbox(label="Batch Results",lines=20,
                                    elem_classes=["result-box"])
            batch_chart=gr.Plot(label="Probability Heatmap — All Sites × All Models")
            batch_btn.click(fn=run_batch,inputs=[],outputs=[batch_txt,batch_chart])

        # ── TAB 4: Model Info ─────────────────────────────────────────────────
        with gr.Tab("ℹ️ Model Info"):
            gr.Markdown("### Loaded Models & Performance")
            info_rows=["| Model | Features | Test Accuracy | F1 | ROC-AUC |",
                        "|---|---|---|---|---|"]
            for name,meta in METADATA.items():
                info_rows.append(
                    f"| {name} | {len(meta.get('features',[]))} | "
                    f"{meta.get('test_accuracy','—')} | "
                    f"{meta.get('test_f1','—')} | "
                    f"{meta.get('test_roc_auc','—')} |")
            gr.Markdown("\n".join(info_rows))
            gr.Markdown("""
**Dataset:** Global Coral Bleaching Database 1980–2020 (BCO-DMO)
**DOI:** https://doi.org/10.26008/1912/bco-dmo.773466.2
**Models folder:** `models/` — contains 4 × `.pkl` files

**Key bleaching risk indicators:**
- `TSA_DHW > 8` → severe bleaching expected
- `Temperature_Mean > 30°C` + `SSTA > 1.5` → high risk
- Low turbidity + high DHW → maximum risk
""")

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)