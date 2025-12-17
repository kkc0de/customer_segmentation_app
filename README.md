📌 Project Title

Customer Segmentation using K-Means Clustering

📖 Overview

This project performs customer segmentation using unsupervised machine learning (K-Means) on a real-world marketing dataset. The objective is to group customers based on demographic, household, purchasing behavior, and engagement patterns to support targeted marketing strategies.

🎯 Objectives

Identify meaningful customer segments

Understand purchasing behavior across clusters

Analyze campaign response patterns per segment

Visualize clusters for business interpretation

🧠 Techniques Used

Data Cleaning & Feature Engineering

One-Hot Encoding for categorical variables

Feature Scaling (StandardScaler)

K-Means Clustering

Elbow Method & Silhouette Score

PCA (for visualization only)

🗂 Dataset

Customer Personality Analysis (Marketing Campaign Dataset)
Contains customer demographics, purchase behavior, and marketing response data.

🛠️ Workflow

Data loading & cleaning

Feature selection (demographics + behavior)

Encoding & scaling using pipelines

Optimal cluster selection (Elbow & Silhouette)

K-Means clustering

PCA visualization

Cluster interpretation & insights

📊 Key Insights

Identified distinct customer groups based on spending and engagement

High-income clusters showed stronger response to campaigns

Certain clusters prefer online purchases over in-store

Campaign response varies significantly across segments

📌 Technologies

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

🚀 Future Improvements

Try DBSCAN / Hierarchical clustering

Apply PCA before clustering

Build a recommendation strategy per cluster
