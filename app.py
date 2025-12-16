import streamlit as st
import pandas as pd
import joblib

# Load models
preprocessor = joblib.load("preprocessor.pkl")
kmeans = joblib.load("customer_segmentation_model.pkl")

st.title("Customer Segmentation App")

# --- Sample Data Download ---
# This lets users download a test file if they don't have their own
st.write("Don't have a file? Download this sample to test the app:")
with open("sample_data.csv", "rb") as file:
    st.download_button(
        label="Download Sample CSV",
        data=file,
        file_name="sample_data.csv",
        mime="text/csv"
    )
st.write("---") # Adds a visual separator line


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
