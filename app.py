# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Telecom Revenue Growth - Segmentation", layout="wide")
st.title("📈 Telecom Customer Segmentation (Live KMeans)")

st.write("""
This app trains a small KMeans clustering model on a built-in synthetic telecom dataset,
selects the best number of clusters by silhouette score, and predicts which segment
a user-provided customer belongs to.
""")

# -------------------------
# 1) Build synthetic dataset
# -------------------------
@st.cache_data
def build_synthetic_data(n=500, seed=42):
    np.random.seed(seed)
    data = pd.DataFrame({
        "MonthlyDataGB": np.random.gamma(2.0, 10.0, n) + np.random.uniform(0,5,n),
        "MonthlyVoiceMins": np.random.normal(600, 300, n).clip(50, 3000),
        "MonthlySpendRs": (np.random.gamma(2.0, 200.0, n) + 150).clip(50, 3000),
        "DataGB_Last3Mo_Avg": np.random.gamma(2.0, 9.0, n) + np.random.uniform(0,3,n),
        "SpendRs_Last3Mo_Avg": (np.random.gamma(2.0, 180.0, n) + 120).clip(50, 3000),
    })
    data["DataGB_SpendRs_Ratio"] = (data["MonthlyDataGB"] / data["MonthlySpendRs"]).replace([np.inf, -np.inf], 0)
    # keep sensible numeric types
    return data.round(3)

data = build_synthetic_data()
st.sidebar.subheader("Data & Model Settings")
st.sidebar.write("Synthetic dataset preview:")
st.sidebar.dataframe(data.head())

# -------------------------
# 2) Preprocess & choose best k by silhouette
# -------------------------
features = ["MonthlyDataGB", "MonthlyVoiceMins", "MonthlySpendRs",
            "DataGB_Last3Mo_Avg", "SpendRs_Last3Mo_Avg", "DataGB_SpendRs_Ratio"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])

st.sidebar.subheader("KMeans selection")
k_min = st.sidebar.number_input("Min K", min_value=2, max_value=2, value=2, step=1)
k_max = st.sidebar.number_input("Max K", min_value=3, max_value=12, value=6, step=1)

# compute silhouette scores and pick best_k
best_k = None
best_score = -1
scores = {}
for k in range(int(k_min), int(k_max)+1):
    kmeans_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_tmp = kmeans_tmp.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels_tmp)
    scores[k] = score
    if score > best_score:
        best_score = score
        best_k = k

st.sidebar.write("Silhouette scores (k → score):")
st.sidebar.write(scores)
st.sidebar.success(f"Selected best_k = {best_k} (silhouette = {best_score:.3f})")

# train final KMeans with best_k
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans.fit(X_scaled)
data["Cluster"] = kmeans.labels_

# cluster summary (centers in original scale)
centers_orig = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features).round(2)
centers_orig["Cluster"] = centers_orig.index
centers_orig = centers_orig.set_index("Cluster")

# show cluster summary
st.subheader("Cluster centers (approx. in original units)")
st.dataframe(centers_orig)

# -------------------------
# 3) User input for a single customer
# -------------------------
st.subheader("Predict segment for a customer (enter features below)")

col1, col2 = st.columns(2)
with col1:
    MonthlyDataGB = st.number_input("Monthly Data (GB)", min_value=0.0, max_value=200.0, value=25.0, step=0.5)
    MonthlyVoiceMins = st.number_input("Monthly Voice Minutes", min_value=0.0, max_value=5000.0, value=600.0, step=10.0)
    MonthlySpendRs = st.number_input("Monthly Spend (₹)", min_value=0.0, max_value=10000.0, value=350.0, step=10.0)
with col2:
    DataGB_Last3Mo_Avg = st.number_input("Data GB (Last 3 mo avg)", min_value=0.0, max_value=200.0, value=24.0, step=0.5)
    SpendRs_Last3Mo_Avg = st.number_input("Spend ₹ (Last 3 mo avg)", min_value=0.0, max_value=10000.0, value=340.0, step=10.0)

DataGB_SpendRs_Ratio = (MonthlyDataGB / MonthlySpendRs) if MonthlySpendRs != 0 else 0.0

new_customer = pd.DataFrame([{
    "MonthlyDataGB": MonthlyDataGB,
    "MonthlyVoiceMins": MonthlyVoiceMins,
    "MonthlySpendRs": MonthlySpendRs,
    "DataGB_Last3Mo_Avg": DataGB_Last3Mo_Avg,
    "SpendRs_Last3Mo_Avg": SpendRs_Last3Mo_Avg,
    "DataGB_SpendRs_Ratio": round(DataGB_SpendRs_Ratio, 5)
}])

st.write("Input customer:")
st.table(new_customer.T)

# scale and predict
new_scaled = scaler.transform(new_customer[features])
pred_cluster = int(kmeans.predict(new_scaled)[0])

# -------------------------
# 4) Interpretation logic
# -------------------------
st.subheader("Prediction")
st.success(f"Predicted Cluster: **{pred_cluster}**")

# get center for predicted cluster
center = centers_orig.loc[pred_cluster]

# derive basic interpretation rules
data_center = center["MonthlyDataGB"]
spend_center = center["MonthlySpendRs"]
ratio_center = center["DataGB_SpendRs_Ratio"]

interpretation = []
# Under-monetized heavy user: high data, low spend
if (data_center >= centers_orig["MonthlyDataGB"].quantile(0.66)) and (spend_center <= centers_orig["MonthlySpendRs"].quantile(0.33)):
    interpretation.append("⚠️ Under-monetized heavy user — high data usage but relatively low spend. Consider upsell/campaigns.")
# Well-monetized heavy user: high data and high spend
elif (data_center >= centers_orig["MonthlyDataGB"].quantile(0.66)) and (spend_center >= centers_orig["MonthlySpendRs"].quantile(0.66)):
    interpretation.append("💰 Well-monetized heavy user — high usage and high spend. Focus on retention.")
# Low-usage low-spend
elif (data_center <= centers_orig["MonthlyDataGB"].quantile(0.33)) and (spend_center <= centers_orig["MonthlySpendRs"].quantile(0.33)):
    interpretation.append("📉 Low-usage / low-value customer — low data & low spend.")
else:
    interpretation.append("📱 Moderate or mixed behavior segment — neither extreme on spend nor usage.")

# Show numeric center values for this cluster
st.subheader("Cluster center values (representative)")
st.table(center.to_frame(name="Value"))

st.subheader("Interpretation")
for msg in interpretation:
    if "Under-monetized" in msg:
        st.warning(msg)
    elif "Well-monetized" in msg:
        st.success(msg)
    else:
        st.info(msg)

# -------------------------
# 5) Optional: show cluster distribution
# -------------------------
st.subheader("Cluster distribution in synthetic dataset")
dist = data["Cluster"].value_counts().sort_index()
st.bar_chart(dist)
