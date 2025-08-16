# -*- coding: utf-8 -*-
"""
Diabetes Prediction Web App (Streamlit)
- Prediction (with proper scaling if scaler available)
- Patient-level advanced visualizations
- Dataset exploration (heatmap + feature distributions)
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =============== App Constants ===============
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(APP_DIR, "trained_model.sav")
DEFAULT_SCALER_PATH = os.path.join(APP_DIR, "scaler.sav")
DEFAULT_DATA_PATH = os.path.join(APP_DIR, "diabetes.csv")

FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# =============== Utils ===============
@st.cache_resource
def load_model_and_scaler(model_path: str = DEFAULT_MODEL_PATH, scaler_path: str = DEFAULT_SCALER_PATH):
    notes = []
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    obj = pickle.load(open(model_path, "rb"))
    model, scaler = None, None

    if isinstance(obj, dict) and ("model" in obj):
        model = obj["model"]
        scaler = obj.get("scaler", None)
        if scaler is None:
            notes.append("Model file loaded (dict), but no scaler inside. Will try separate scaler.sav.")
    else:
        model = obj
        notes.append("Model file loaded (plain). Will try separate scaler.sav.")

    if scaler is None and os.path.exists(scaler_path):
        scaler = pickle.load(open(scaler_path, "rb"))
        notes.append("Loaded scaler from scaler.sav.")

    if scaler is None:
        notes.append("⚠️ No scaler found. Prediction will use raw inputs (may reduce accuracy).")

    return model, scaler, notes


@st.cache_data
def load_data(default_path: str = DEFAULT_DATA_PATH, uploaded_file=None) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if os.path.exists(default_path):
        return pd.read_csv(default_path)
    raise FileNotFoundError("Could not find 'diabetes.csv'. Upload it or place it next to this script.")


def predict_diabetes(model, scaler, inputs_list):
    x = np.asarray(inputs_list, dtype=float).reshape(1, -1)
    if scaler is not None:
        x = scaler.transform(x)

    y_pred = model.predict(x)
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(x)[0]
        except Exception:
            proba = None

    label = '✅ The person is not diabetic' if int(y_pred[0]) == 0 else '⚠️ The person is diabetic'
    return label, int(y_pred[0]), proba


def input_row_as_list(P, G, BP, ST, I, BMI, DPF, AGE):
    return [P, G, BP, ST, I, BMI, DPF, AGE]


# =============== UI ===============
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Prediction Web App")

with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to", ["Prediction", "Data Visualizations", "About"])

# ===== Load model (once) =====
try:
    model, scaler, load_notes = load_model_and_scaler(DEFAULT_MODEL_PATH, DEFAULT_SCALER_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

if load_notes:
    for n in load_notes:
        st.info(n)

# =======================================
# Page: Prediction
# =======================================
if page == "Prediction":
    st.header("Enter Patient Details")

    # helper: free numeric input (now 5 chars max)
    def free_num_input(label, default="0"):
        val = st.text_input(label, default, max_chars=5)
        try:
            return float(val)
        except:
            return 0.0

    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = free_num_input('Number of Pregnancies', "0")
        Glucose = free_num_input('Glucose Level (mg/dL)', "120")
        BloodPressure = free_num_input('Blood Pressure (mm Hg)', "70")
        SkinThickness = free_num_input('Skin Thickness (mm)', "20")
    with col2:
        Insulin = free_num_input('Insulin (µU/mL)', "79")
        BMI = free_num_input('BMI', "25.0")
        DiabetesPedigreeFunction = free_num_input('Diabetes Pedigree Function', "0.5")
        Age = free_num_input('Age (years)', "33")

    if st.button("Diabetes Test Result"):
        inputs = input_row_as_list(Pregnancies, Glucose, BloodPressure, SkinThickness,
                                   Insulin, BMI, DiabetesPedigreeFunction, Age)
        diagnosis, yclass, proba = predict_diabetes(model, scaler, inputs)
        st.success(diagnosis)

        # ---------- Advanced Visuals ----------
        st.subheader("📍 Patient-Level Visualizations")

        # Basic bar chart
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.bar(FEATURE_NAMES, inputs, color="skyblue", edgecolor="black")
        ax1.set_ylabel("Value")
        ax1.set_title("Entered Features Overview")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig1)

        # Radar Chart
        st.subheader("📊 Radar Chart: Patient vs. Population")
        try:
            df = load_data(DEFAULT_DATA_PATH)
            avg_healthy = df[df["Outcome"] == 0][FEATURE_NAMES].mean().values
            avg_diabetic = df[df["Outcome"] == 1][FEATURE_NAMES].mean().values

            categories = FEATURE_NAMES + [FEATURE_NAMES[0]]
            N = len(FEATURE_NAMES)

            patient = inputs + [inputs[0]]
            healthy = avg_healthy.tolist() + [avg_healthy[0]]
            diabetic = avg_diabetic.tolist() + [avg_diabetic[0]]

            fig_radar = plt.figure(figsize=(6, 6))
            ax_radar = plt.subplot(111, polar=True)
            angles = np.linspace(0, 2 * np.pi, N + 1, endpoint=True)

            ax_radar.plot(angles, patient, "o-", linewidth=2, label="Patient")
            ax_radar.fill(angles, patient, alpha=0.25)
            ax_radar.plot(angles, healthy, "g--", linewidth=1.5, label="Avg Healthy")
            ax_radar.plot(angles, diabetic, "r--", linewidth=1.5, label="Avg Diabetic")

            ax_radar.set_thetagrids(angles[:-1] * 180/np.pi, categories)
            plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
            st.pyplot(fig_radar)
        except Exception as e:
            st.warning(f"Radar chart unavailable: {e}")

        # Glucose Gauge
        st.subheader("Glucose Level Status")
        fig2, ax2 = plt.subplots(figsize=(6, 1.5))
        ax2.barh(["Glucose"], [Glucose], color="orange" if Glucose > 140 else "green")
        ax2.axvline(x=140, color="red", linestyle="--", label="Risk Threshold (140)")
        ax2.set_xlim([0, 250])
        ax2.set_xlabel("mg/dL")
        ax2.legend()
        st.pyplot(fig2)

        # Probability Confidence
        if proba is not None and len(proba) == 2:
            st.subheader("Model Confidence (Probability)")
            fig3, ax3 = plt.subplots(figsize=(6, 1.5))
            ax3.barh(["Not Diabetic"], [proba[0]], color="green")
            ax3.barh(["Diabetic"], [proba[1]], color="red")
            ax3.set_xlim([0, 1])
            for i, v in enumerate(proba):
                ax3.text(v + 0.01, i, f"{v:.2f}", va="center")
            st.pyplot(fig3)

        # Patient vs Population
        st.subheader("Feature Comparison vs Population")
        try:
            avg_vals = df[FEATURE_NAMES].mean().values
            fig4, ax4 = plt.subplots(figsize=(8, 4))
            x = np.arange(len(FEATURE_NAMES))
            ax4.bar(x - 0.2, avg_vals, width=0.4, label="Population Avg", color="lightgray")
            ax4.bar(x + 0.2, inputs, width=0.4, label="Patient", color="dodgerblue")
            ax4.set_xticks(x)
            ax4.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right")
            ax4.set_title("Patient vs. Population Averages")
            ax4.legend()
            st.pyplot(fig4)
        except Exception:
            pass

        # Advice
        if yclass == 1:
            st.info("General tips: balanced diet, regular exercise, and monitor glucose. Consult a healthcare professional if needed.")
        else:
            st.info("Great! Maintain healthy lifestyle: balanced diet, activity, and regular checkups.")

# =======================================
# Page: Data Visualizations
# =======================================
elif page == "Data Visualizations":
    st.header("📊 Explore the Diabetes Dataset")

    uploaded = st.file_uploader("Upload diabetes.csv (optional)", type="csv")
    try:
        df = load_data(DEFAULT_DATA_PATH, uploaded_file=uploaded)
        st.success(f"Loaded dataset with shape: {df.shape}")
        if st.checkbox("Preview data"):
            st.dataframe(df.head())

        # Correlation heatmap
        if st.checkbox("Show Correlation Heatmap"):
            fig_hm, ax_hm = plt.subplots(figsize=(8, 6))
            corr = df.corr(numeric_only=True)
            im = ax_hm.imshow(corr, aspect="auto")
            ax_hm.set_xticks(range(len(corr.columns)))
            ax_hm.set_yticks(range(len(corr.index)))
            ax_hm.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax_hm.set_yticklabels(corr.index)
            for i in range(len(corr.index)):
                for j in range(len(corr.columns)):
                    ax_hm.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
            ax_hm.set_title("Correlation Heatmap")
            fig_hm.colorbar(im)
            st.pyplot(fig_hm)

        # Feature distribution
        if st.checkbox("Show Feature Distributions"):
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            feature = st.selectbox("Select a feature", num_cols, index=0 if num_cols else None)
            if feature:
                fig_hist, ax_hist = plt.subplots()
                ax_hist.hist(df[feature].dropna(), bins=30)
                ax_hist.set_title(f"Distribution: {feature}")
                st.pyplot(fig_hist)

    except FileNotFoundError as e:
        st.warning(str(e))
        st.info("Upload the dataset or place 'diabetes.csv' next to this script.")

# =======================================
# Page: About
# =======================================
else:
    st.header("About")
    st.write(
        """
        **Owner: Mirza Yasir Abdullah Baig**
        - I am software engineer and an AI/ML Engineer.
        - The purpose of this model is to provide the best results.
        - This is my first hackathon and i am very excited

        **Diabetes Prediction Web App**
        - Predicts diabetes using a trained ML model.
        - Provides patient-level advanced visualizations.
        - Dataset exploration with heatmap & distributions.
        
        **Files expected:**
        - `trained_model.sav` → model (and scaler if included)
        - `scaler.sav` (optional)
        - `diabetes.csv` (optional, else upload)

        *Disclaimer: For educational/demo use only. Not medical advice.*
        """
    )

