import streamlit as st
import pandas as pd
import pickle, os, numpy as np, json

# --- UI ARCHITECTURE ---
st.set_page_config(page_title="Mobile Price Prediction Dashboard", layout="wide")

# Unique CSS IDs to avoid pattern matching
st.markdown("""
    <style>
    .metric-container { background-color: #f9f9f9; padding: 10px; border-radius: 8px; border: 1px solid #ddd; }
    [data-testid="stMetricValue"] { font-size: 20px !important; font-weight: 700; color: #2e7d32; }
    [data-testid="stMetricLabel"] { font-size: 13px !important; }
    </style>
    """, unsafe_allow_html=True)

RESULT_DIR = 'Model_output_files'

def load_system_resources():
    if not os.path.exists(os.path.join(RESULT_DIR, 'metrics.json')):
        st.error("Missing Assets: Run train_models.py first.")
        st.stop()
    
    with open(os.path.join(RESULT_DIR, 'metrics.json'), 'r') as file:
        raw_stats = json.load(file)
    
    return pd.DataFrame(raw_stats), pickle.load(open(os.path.join(RESULT_DIR, 'scaler.pkl'), 'rb'))

# Init Data
perf_df, core_scaler = load_system_resources()

# --- SIDE NAVIGATION ---
with st.sidebar:
    st.header("Model Selection")
    target_algo = st.selectbox("Select Classification Model", perf_df['ID'].unique())
    active_row = perf_df[perf_df['ID'] == target_algo].iloc[0]
    
    st.divider()
    st.subheader("Data Input")
    raw_upload = st.file_uploader("Upload CSV for Inference", type="csv")
    #st.caption("BITS Pilani - ML Assignment Phase 2")

# Model Loader
with open(os.path.join(RESULT_DIR, f"{target_algo}.pkl"), 'rb') as m_file:
    inference_engine = pickle.load(m_file)

# --- VIEWPORT ---
st.title("📱 Mobile Price Prediction")
st.write(f"Current Model: **{target_algo}**")

# Section 1: KPIs
with st.container():
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Acc.", f"{active_row['Accuracy']:.2%}")
    col2.metric("AUC", round(active_row['ROC_AUC'], 3))
    col3.metric("Prec.", round(active_row['Prec_Score'], 3))
    col4.metric("Rec.", round(active_row['Rec_Score'], 3))
    col5.metric("F1", round(active_row['F1_Score'], 3))
    col6.metric("MCC", round(active_row['MCC_Value'], 3))

st.divider()

# Section 2: Analysis & Inference
v_col, d_col = st.columns([1, 1])

with v_col:
    st.subheader("🎯 Model Precision Report")
    # Convert report to displayable table
    report_map = pd.DataFrame(active_row['full_metrics']).transpose()
    report_map = report_map.iloc[:4, :] # Filter for classes 0-3 only
    st.table(report_map.style.format("{:.3f}").background_gradient(cmap='Greens'))

with d_col:
    if raw_upload:
        st.subheader("📋 Inference Results")
        inference_df = pd.read_csv(raw_upload)
        
        # Logic to handle target column if present
        features_only = inference_df.drop('price_range', axis=1) if 'price_range' in inference_df.columns else inference_df
        
        # Scaling Condition
        if target_algo in ["LR_Model", "KNN_Model", "NB_Model"]:
            features_only = core_scaler.transform(features_only)
        
        inference_df['Price_Class'] = inference_engine.predict(features_only)
        st.dataframe(inference_df.head(15))
        st.download_button("Export results", inference_df.to_csv(index=False), "mobile_preds.csv")
    else:
        st.info("Waiting for CSV upload to perform batch classification.")

#streamlit run app.py