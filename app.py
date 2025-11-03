import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Telecom Customer Segmentation", layout="wide")
st.title("📊 Multi-Customer Cluster Prediction (K-Means = 5)")
st.markdown("Upload multiple customer records to predict their segments and see cluster-wise recommendations.")

# ----------------------------
# 1️⃣ Upload or sample data
# ----------------------------
uploaded_file = st.file_uploader("📤 Upload a CSV file with customer data", type=["csv"])

if uploaded_file:
    df_input = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data")
    st.dataframe(df_input.head())
else:
    st.info("No file uploaded. Using sample synthetic data for demo.")
    np.random.seed(42)
    df_input = pd.DataFrame({
        "MonthlyDataGB": np.random.gamma(2.0, 10.0, 50),
        "MonthlyVoiceMins": np.random.normal(600, 300, 50).clip(50, 3000),
        "MonthlySpendRs": (np.random.gamma(2.0, 200.0, 50) + 150).clip(50, 3000),
        "DataGB_Last3Mo_Avg": np.random.gamma(2.0, 9.0, 50),
        "SpendRs_Last3Mo_Avg": (np.random.gamma(2.0, 180.0, 50) + 120).clip(50, 3000)
    })
    df_input["DataGB_SpendRs_Ratio"] = df_input["MonthlyDataGB"] / df_input["MonthlySpendRs"]

# ----------------------------
# 2️⃣ Train Model (simulate your notebook)
# ----------------------------
np.random.seed(42)
df_train = pd.DataFrame({
    "MonthlyDataGB": np.random.gamma(2.0, 10.0, 500) + np.random.uniform(0,5,500),
    "MonthlyVoiceMins": np.random.normal(600, 300, 500).clip(50, 3000),
    "MonthlySpendRs": (np.random.gamma(2.0, 200.0, 500) + 150).clip(50, 3000),
    "DataGB_Last3Mo_Avg": np.random.gamma(2.0, 9.0, 500),
    "SpendRs_Last3Mo_Avg": (np.random.gamma(2.0, 180.0, 500) + 120).clip(50, 3000)
})
df_train["DataGB_SpendRs_Ratio"] = df_train["MonthlyDataGB"] / df_train["MonthlySpendRs"]

features = [
    "MonthlyDataGB", "MonthlyVoiceMins", "MonthlySpendRs",
    "DataGB_Last3Mo_Avg", "SpendRs_Last3Mo_Avg", "DataGB_SpendRs_Ratio"
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_train[features])
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# ----------------------------
# 3️⃣ Predict for uploaded dataset
# ----------------------------
if st.button("🔍 Predict Clusters for All Customers"):
    X_input_scaled = scaler.transform(df_input[features])
    df_input["Cluster"] = kmeans.predict(X_input_scaled)

    # Cluster labels
    cluster_map = {
        0: (" High-Value Customers", "Upsell premium plans and offer loyalty benefits."),
        1: (" Low-Usage Customers", "Send reactivation or low-cost bundle offers."),
        2: (" Data-Hungry Users", "Promote unlimited data or high-speed add-ons."),
        3: (" Voice-Focused Customers", "Recommend unlimited voice or hybrid plans."),
        4: (" Irregular Users", "Onboard or educate users to increase engagement.")
    }

    df_input["Cluster_Name"] = df_input["Cluster"].map(lambda x: cluster_map[x][0])
    df_input["Recommended_Action"] = df_input["Cluster"].map(lambda x: cluster_map[x][1])

    st.success("Clusters Predicted Successfully!")
    st.dataframe(df_input.head(10))

    # ----------------------------
    # 4️⃣ Group by Cluster and summarize
    # ----------------------------
    st.subheader("📊 Cluster Summary")
    cluster_summary = (
        df_input.groupby(["Cluster", "Cluster_Name"])
        .agg({
            "MonthlySpendRs": ["mean", "count"],
            "MonthlyDataGB": "mean",
            "MonthlyVoiceMins": "mean"
        })
        .reset_index()
    )
    cluster_summary.columns = ["Cluster", "Cluster_Name", "Avg Spend", "Count", "Avg Data GB", "Avg Voice Mins"]
    st.dataframe(cluster_summary.round(2))

    # ----------------------------
    # 5️⃣ Recommendations
    # ----------------------------
    st.subheader(" Recommended Actions per Cluster")
    for cluster_id, (name, action) in cluster_map.items():
        count = (df_input["Cluster"] == cluster_id).sum()
        st.markdown(f"**Cluster {cluster_id} – {name} ({count} customers)**")
        st.info(action)

