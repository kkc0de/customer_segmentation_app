import streamlit as st
import pandas as pd
import joblib

# Load models
preprocessor = joblib.load("preprocessor.pkl")
kmeans = joblib.load("customer_segmentation_model.pkl")

# --- STEP 1: DEFINE THE FEATURE ENGINEERING FUNCTION ---
def preprocess_data(df):
    """
    This function recreates the custom columns created in the Colab notebook.
    """
    # 1. Calculate Total Spent
    # We use .get() or fillna(0) just in case a column is empty, but standard addition works too
    df['Total_spent'] = (
        df['MntWines'] + 
        df['MntFruits'] + 
        df['MntMeatProducts'] + 
        df['MntFishProducts'] + 
        df['MntSweetProducts'] + 
        df['MntGoldProds']
    )
    
    # 2. Calculate Age
    df['Age'] = 2024 - df['Year_Birth']
    
    # 3. Calculate Total Kids
    df['total_kids'] = df['Kidhome'] + df['Teenhome']
    
    # Note: If you dropped columns like "ID" or "Dt_Customer" in your notebook BEFORE 
    # feeding data to the scaler, you might need to drop them here too. 
    # For now, we will keep them as the scaler usually ignores extra columns unless explicitly told not to.
    
    return df
# -----------------------------------------------------

st.title("Customer Segmentation App")

# Download Sample Button
st.write("Don't have a file? Download this sample to test the app:")
# Try-except block to prevent error if sample_data.csv isn't uploaded yet
try:
    with open("sample_data.csv", "rb") as file:
        st.download_button(
            label="Download Sample CSV",
            data=file,
            file_name="sample_data.csv",
            mime="text/csv"
        )
except FileNotFoundError:
    st.warning("⚠️ sample_data.csv not found in repo. Please upload it to GitHub.")

st.write("---")

uploaded_file = st.file_uploader("Upload customer CSV", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Raw Data Preview:", data.head())

    # --- STEP 2: APPLY THE CLEANING ---
    try:
        processed_data = preprocess_data(data)
        st.write("Data with new features (Age, Total_spent, etc.):", processed_data.head())
        
        # --- STEP 3: PREDICT ---
        # The preprocessor now sees the columns it expects ('Age', 'Total_spent', etc.)
        X_new = preprocessor.transform(processed_data)
        clusters = kmeans.predict(X_new)
        
        # Add clusters back to original data
        data["Cluster"] = clusters

        st.success("Segmentation Successful! 🎉")
        st.write("Segmented Customers:", data.head())

        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("Download Segmented CSV", csv, "segmented_customers.csv", "text/csv")
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.info("Check: Did you drop specific columns (like 'ID') in your notebook before scaling? If so, you might need to add `df.drop(...)` inside the preprocess_data function.")
