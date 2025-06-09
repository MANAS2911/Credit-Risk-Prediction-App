# 💳 Credit Risk Prediction App

A machine learning-powered web application built with **Streamlit** that predicts credit risk based on applicant data from the **German Credit dataset**. It features automated **data preprocessing**, **feature engineering**, **SMOTE oversampling**, **hyperparameter-tuned models**, and **SHAP-based model interpretation**.

---

## 🚀 Features

- 📊 Streamlit web interface for real-time credit risk prediction.
- ⚙️ Model training pipeline with **Random Forest**, **XGBoost**, and **Logistic Regression**.
- 🧠 Hyperparameter tuning via **GridSearchCV** and **Stratified K-Fold CV**.
- 🏷️ Custom `FeatureEngineer` transformer for domain-specific feature creation.
- 📈 SHAP summary plots for model interpretability.
- ✅ Imbalanced data handling with **SMOTE**.
- 💾 Model persistence using `joblib`.

---

## 📁 Project Structure

Credit-Risk-Prediction-App/
│
├── Credit.py # Main Streamlit app
├── feature_engineer.py # Feature engineering logic as custom transformer
├── german_credit_data.csv # Dataset
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

## 🛠️ Installation

1. **Clone the repository**
git clone https://github.com/your-username/Credit-Risk-Prediction-App.git
cd Credit-Risk-Prediction-App

2. **Create a virtual environment (optional but recommended)**
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install dependencies**
pip install -r requirements.txt

## Running the App
streamlit run Credit.py

## 📊 Model Details

Three models are trained using GridSearchCV with SMOTE and custom feature engineering:
- Random Forest
- XGBoost
- Logistic Regression
Each model is evaluated using F1-Score and ROC AUC. The best performing models are saved as .pkl files.

## 🧠 Model Explainability

- SHAP (SHapley Additive exPlanations) is used to visualize feature impact.
- Users can click "Show SHAP Summary" to view a plot explaining feature importance.

## 📌 Requirements

All dependencies are listed in requirements.txt. Key libraries include:
- streamlit
- scikit-learn
- xgboost
- shap
- imbalanced-learn
- pandas, numpy, matplotlib

## 🙋‍♂️ Author

Manas Choudhary,
Final Year Computer Engineering Student,
Project: Credit Risk Prediction System

Feel free to connect on [LinkedIn](www.linkedin.com/in/contactmanaschoudhary) or raise an issue or PR.

## ⭐ Star This Repository
If you like this project, give it a ⭐ to help others find it!

## App Link
[Credit Risk Predictor App](https://creditriskradar.streamlit.app/)
