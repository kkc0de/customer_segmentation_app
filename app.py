import streamlit as st
import pandas as pd
import joblib

preprocessor = joblib.load("preprocessor.pkl")
kmeans = joblib.load("customer_segmentation_model.pkl")

st.title("Customer Segmentation App")

uploaded_file = st.file_uploader("Upload customer CSV", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:", data.head())

    X_new = preprocessor.transform(data)
    clusters = kmeans.predict(X_new)
    data["Cluster"] = clusters

    st.write("Segmented Customers:", data.head())

    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button("Download Segmented CSV", csv, "segmented_customers.csv", "text/csv")