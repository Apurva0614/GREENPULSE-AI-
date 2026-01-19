# GREENPULSE-AI-
GreenPulse is an AI framework monitoring India’s urban green cover. Using Global Forest Watch data and KNN imputation, it introduces the Green Deficit Index (GDI) to quantify ecological balance. Through Random Forest regression and classification, it predicts risk levels from "Excellent" to "High-Risk," supporting UN SDGs 11, 13, and 15.

🌿 GreenPulse – AI-Powered Urban Green Balance Monitor

GreenPulse is an end-to-end machine learning project designed to analyze, predict, and monitor urban green cover balance in Indian cities. Rapid urbanization and infrastructure development have led to significant tree cover loss, while existing monitoring systems remain fragmented and non-predictive. GreenPulse provides a data-driven, AI-based decision support system to quantify ecological impact and support sustainable urban planning.


---

📌 Project Objectives

Analyze historical tree cover data across Indian states and districts

Quantify urban green loss and gain using a domain-specific metric

Predict future green deficit using machine learning models

Categorize regions into ecological risk levels

Support data-driven environmental and urban planning decisions



---

🌍 Dataset

Source: Global Forest Watch (India)

Levels Used:

Subnational Level 1 (State)

Subnational Level 2 (District)


Key Attributes:

Area (ha)

Tree cover extent (2000, 2010)

Tree cover gain (2000–2020)

Annual tree cover loss (2001–2023)




---

🧹 Data Preprocessing

Column standardization and data cleaning

Missing value handling using K-Nearest Neighbors (KNN) Imputation

Encoding of categorical variables

Feature normalization and consistency checks



---

⚙️ Feature Engineering

Green Deficit Index (GDI)

A domain-specific metric introduced in this project:

GDI = (Tree Loss – Tree Gain) / (Tree Cover Extent 2000 + 1)

Positive GDI → Net green deficit

Negative GDI → Net green gain


GDI Categories

Category	Description

Excellent (Net Gain)	Strong green recovery
Acceptable	Stable green balance
Concerning	Moderate green loss
High-Risk	Severe green deficit



---

🤖 Machine Learning Models

Regression

Linear Regression

Random Forest Regressor


Classification

Logistic Regression

Random Forest Classifier


Random Forest models were selected for their robustness and ability to capture non-linear relationships.


---

📊 Model Evaluation

Regression Metrics: R² Score, MAE, MSE

Classification Metrics: Accuracy, Precision, Recall, F1-Score

Additional Analysis: Confusion Matrix and Prediction Confidence (predict_proba)



---

🖥️ Deployment

Interactive web interface for real-time prediction

Users can input regional values and receive:

Predicted GDI value

Ecological risk category

Model confidence score




---

🌱 Sustainability Alignment

This project aligns with the following UN Sustainable Development Goals (SDGs):

SDG 11: Sustainable Cities and Communities

SDG 13: Climate Action

SDG 15: Life on Land



---

📁 Project Structure

├── data/

├── feature_engineered_greenpulse/

├── preprocessing/

├── random_forest_classifier.pkl/

├── eda/

├── app/

├── README.md

├── ml

├── random_forest_regressor.pkl/

├── requirements/

├── retrain models/

├── visualization/

├── subnational_1_tree_cover_loss/

├── subnational_2_tree_cover_loss/


---

🚀 Future Scope

Integration with real-time Global Forest Watch APIs

Expansion to additional cities and regions

Advanced dashboards for policymakers

Incorporation of satellite imagery and deep learning models



---

👤 Author

Apurva
Artificial Intelligence & Machine Learning
Symbiosis Institute of Technology
