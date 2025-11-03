import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --------------------------
# Streamlit Page Setup
# --------------------------
st.set_page_config(page_title="📊 Telecom Revenue Growth Prediction", layout="centered")

st.title("📈 Telecom Customer Segmentation & Revenue Growth")
st.write("""
This app predicts the **customer segment** (cluster) based on telecom usage behavior 
and spending patterns.  
It helps identify whether a customer is **well-monetized**, **under-monetized**, or **low-usage**.
""")

# --------------------------
# Model Training Section (Predefined Synthetic Data)
# --------------------------
st.sidebar.header("⚙️ Model Setup")

# Simulated historical customer data
np.random.seed(42)
data = pd.DataFrame({
    "MonthlyDataGB": np.random.uniform(5, 60, 200),
    "MonthlyVoiceMins": np.random.uniform(100, 2000, 200),
    "MonthlySpendRs": np.random.uniform(150, 1200, 200),
    "DataGB_Last3Mo_Avg": np.random.uniform(5, 55, 200),
    "SpendRs_Last3Mo_Avg": np.random.uniform(150, 1150, 200),
})

# Add derived ratio
data["DataGB_SpendRs_Ratio"] = data["MonthlyDataGB"] / data["MonthlySpendRs"]

# Standardize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Train K-Means clustering model
k = st.sidebar.slider("Select number of clusters (K)", 2, 6, 5)
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(scaled_data)

# --------------------------
# User Input Section
# --------------------------
st.subheader("🧍 Enter Customer Details")

col1, col2 = st.columns(2)
with col1:
    data_gb = st.number_input("Monthly Data Usage (GB)", 1.0, 100.0, 30.0, step=0.5)
    voice_mins = st.number_input("Monthly Voice Minutes", 50.0, 3000.0, 800.0, step=50.0)
    spend_rs = st.number_input("Monthly Spend (₹)", 100.0, 2000.0, 500.0, step=50.0)

with col2:
    data_last3 = st.number_input("Last 3 Months Avg Data (GB)", 1.0, 100.0, 28.0, step=0.5)
    spend_last3 = st.number_input("Last 3 Months Avg Spend (₹)", 100.0, 2000.0, 550.0, step=50.0)

# Compute derived ratio
ratio = data_gb / spend_rs

# Prepare input for model
new_customer = pd.DataFrame([{
    "MonthlyDataGB": data_gb,
    "MonthlyVoiceMins": voice_mins,
    "MonthlySpendRs": spend_rs,
    "DataGB_Last3Mo_Avg": data_last3,
    "SpendRs_Last3Mo_Avg": spend_last3,
    "DataGB_SpendRs_Ratio": ratio
}])

# Scale using same scaler
scaled_new = scaler.transform(new_customer)

# --------------------------
# Prediction
# --------------------------
if st.button("🔍 Predict Customer Segment"):
    cluster = kmeans.predict(scaled_new)[0]
    st.success(f"Predicted Segment: **Cluster {cluster}**")

    # Interpret the cluster based on average spend/usage
    cluster_summary = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=data.columns)
    cluster_summary = cluster_summary.round(2)

    st.subheader("📊 Cluster Insights")
    st.dataframe(cluster_summary)

    if cluster_summary.loc[cluster, "MonthlyDataGB"] > 40 and cluster_summary.loc[cluster, "MonthlySpendRs"] < 400:
        st.warning("⚠️ This is likely an **Under-Monetized Heavy User**.")
    elif cluster_summary.loc[cluster, "MonthlySpendRs"] > 700:
        st.success("💰 This is a **Well-Monetized Customer Segment**.")
    else:
        st.info("📱 This is a **Moderate Usage Segment**.")

else:
    st.info("👉 Enter details and click **Predict Customer Segment** to see the result.")
