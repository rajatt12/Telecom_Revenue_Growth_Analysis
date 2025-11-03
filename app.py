# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Telecom Cluster Prediction", layout="centered")
st.title("📊 Telecom Customer Segmentation (K-Means = 5)")
st.write("Enter customer details below to predict which cluster they belong to:")

# ----------------------------
# 1️⃣ User Inputs
# ----------------------------
MonthlyDataGB = st.number_input("Monthly Data Usage (GB)", min_value=0.0, max_value=500.0, value=25.0, step=0.5)
MonthlyVoiceMins = st.number_input("Monthly Voice Minutes", min_value=0.0, max_value=5000.0, value=800.0, step=10.0)
MonthlySpendRs = st.number_input("Monthly Spend (₹)", min_value=0.0, max_value=10000.0, value=400.0, step=10.0)
DataGB_Last3Mo_Avg = st.number_input("Data GB (Last 3 Months Avg)", min_value=0.0, max_value=500.0, value=28.0, step=0.5)
SpendRs_Last3Mo_Avg = st.number_input("Spend ₹ (Last 3 Months Avg)", min_value=0.0, max_value=10000.0, value=380.0, step=10.0)

# derived feature
DataGB_SpendRs_Ratio = 0.0 if MonthlySpendRs == 0 else MonthlyDataGB / MonthlySpendRs

# collect inputs
input_data = pd.DataFrame([{
    "MonthlyDataGB": MonthlyDataGB,
    "MonthlyVoiceMins": MonthlyVoiceMins,
    "MonthlySpendRs": MonthlySpendRs,
    "DataGB_Last3Mo_Avg": DataGB_Last3Mo_Avg,
    "SpendRs_Last3Mo_Avg": SpendRs_Last3Mo_Avg,
    "DataGB_SpendRs_Ratio": DataGB_SpendRs_Ratio
}])

st.write("### Input Data")
st.dataframe(input_data)

# ----------------------------
# 2️⃣ Dummy dataset (simulates your notebook data)
# ----------------------------
np.random.seed(42)
df = pd.DataFrame({
    "MonthlyDataGB": np.random.gamma(2.0, 10.0, 500) + np.random.uniform(0,5,500),
    "MonthlyVoiceMins": np.random.normal(600, 300, 500).clip(50, 3000),
    "MonthlySpendRs": (np.random.gamma(2.0, 200.0, 500) + 150).clip(50, 3000),
    "DataGB_Last3Mo_Avg": np.random.gamma(2.0, 9.0, 500) + np.random.uniform(0,3,500),
    "SpendRs_Last3Mo_Avg": (np.random.gamma(2.0, 180.0, 500) + 120).clip(50, 3000)
})
df["DataGB_SpendRs_Ratio"] = df["MonthlyDataGB"] / df["MonthlySpendRs"]

features = [
    "MonthlyDataGB", "MonthlyVoiceMins", "MonthlySpendRs",
    "DataGB_Last3Mo_Avg", "SpendRs_Last3Mo_Avg", "DataGB_SpendRs_Ratio"
]

# ----------------------------
# 3️⃣ Preprocess and train KMeans (k=5)
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# ----------------------------
# 4️⃣ Prediction for user input
# ----------------------------
input_scaled = scaler.transform(input_data[features])
cluster_pred = int(kmeans.predict(input_scaled)[0])

st.success(f"🎯 Predicted Cluster: **Cluster {cluster_pred}**")

# ----------------------------
# 5️⃣ Optional: Show cluster center for reference
# ----------------------------
centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)
st.write("### Cluster Centers (Approximate Values)")
st.dataframe(centers.round(2))
