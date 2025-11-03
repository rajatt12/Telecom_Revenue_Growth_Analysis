import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -------------------------------
# Streamlit App Title
# -------------------------------
st.title("📊 Telecom Customer Segmentation (K=5 Clusters)")

st.write("""
Upload a customer dataset to group subscribers into meaningful clusters 
based on usage, spend, and engagement behavior.
""")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload your telecom dataset (CSV)", type=["csv"])

if uploaded_file:
    # Load the dataset
    df = pd.read_csv(uploaded_file)

    # Display the data
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # -------------------------------
    # Feature Selection (based on notebook)
    # -------------------------------
    features = [
        "MonthlyDataGB", "MonthlyVoiceMins", "MonthlySpendRs",
        "DataGB_Last3Mo_Avg", "SpendRs_Last3Mo_Avg", "DataGB_SpendRs_Ratio"
    ]

    X = df[features]

    # -------------------------------
    # Scaling + KMeans Model
    # -------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=5, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    # -------------------------------
    # Display Cluster Insights
    # -------------------------------
    st.subheader("📊 Cluster-wise Summary")

    cluster_summary = df.groupby("Cluster")[features].mean().round(2)
    st.dataframe(cluster_summary)

    # Cluster interpretation
    st.markdown("### 💡 Cluster Interpretation (Example)")
    cluster_descriptions = {
        0: "High Data, High Spend – Premium Users 💎",
        1: "Low Usage, Low Spend – Value Seekers 💰",
        2: "High Voice, Medium Spend – Talk-Time Lovers ☎️",
        3: "Medium Usage, High Spend – Upsell Candidates 📈",
        4: "Low Data, High Spend – Retention Focus 🔁"
    }

    st.write("Below are example interpretations — these can be fine-tuned after real analysis:")
    for cluster, desc in cluster_descriptions.items():
        st.markdown(f"**Cluster {cluster}:** {desc}")

    # -------------------------------
    # Display All Users Grouped by Cluster
    # -------------------------------
    st.subheader("👥 Customers Grouped by Cluster")
    for cluster in sorted(df["Cluster"].unique()):
        st.markdown(f"### Cluster {cluster} – {cluster_descriptions.get(cluster, 'Customer Segment')}")
        st.dataframe(df[df["Cluster"] == cluster][["SubscriberID"] + features + ["Cluster"]].head(10))

    # -------------------------------
    # Download Results
    # -------------------------------
    csv_download = df.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download Clustered Data", csv_download, "clustered_customers.csv", "text/csv")
else:
    st.info("👆 Please upload a CSV file to begin clustering.")
