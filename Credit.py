# CREDIT RISK PREDICTION STREAMLIT APP (FINAL IMPROVED VERSION with Hyperparameter Tuning) 
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import logging
import os

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# CONFIGURE LOGGING
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# DATA LOADER
class DataLoader:
    def __init__(self, filepath='C:/Users/Manas/OneDrive/Desktop/TCS/german_credit_data.csv'):
        self.filepath = filepath

    def load_data(self):
        df = pd.read_csv(self.filepath)
        df['Risk'] = np.where((df['Credit amount'] > 5000) & (df['Duration'] > 24), 'bad', 'good')
        return df

# FEATURE ENGINEERING
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['Debt/Duration'] = X['Credit amount'] / (X['Duration'] + 1e-5)
        X['Age_group'] = pd.cut(X['Age'], bins=[18, 30, 45, 60, 100],
                                labels=['18-30', '31-45', '46-60', '60+'])
        return X

# MODEL TRAINER 
class ModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y

        self.categorical = ['Sex', 'Housing', 'Purpose', 'Saving accounts', 'Checking account', 'Age_group']
        self.numerical = ['Age', 'Job', 'Credit amount', 'Duration', 'Debt/Duration']

        self.preprocessor = ColumnTransformer([
            ('num', StandardScaler(), self.numerical),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical)
        ])

        self.base_models = {
            'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
            'XGBoost': XGBClassifier(scale_pos_weight=np.sqrt(len(y[y==0])/len(y[y==1])), use_label_encoder=False, eval_metric='logloss', random_state=42),
            'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        }

        self.param_grids = {
            'Random Forest': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [5, 10, None],
                'classifier__min_samples_split': [2, 5]
            },
            'XGBoost': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 6],
                'classifier__learning_rate': [0.01, 0.1]
            },
            'Logistic Regression': {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__penalty': ['l2']
            }
        }

    def train_models(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)

        trained_models = {}
        results = []

        for name, base_model in self.base_models.items():
            pipeline = ImbPipeline([
                ('engineer', FeatureEngineer()),
                ('preprocessor', self.preprocessor),
                ('smote', SMOTE(random_state=42)),
                ('classifier', base_model)
            ])

            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=self.param_grids[name],
                scoring='f1_weighted',
                cv=StratifiedKFold(n_splits=5),
                n_jobs=-1,
                verbose=1
            )

            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            best_params = grid.best_params_

            y_pred = best_model.predict(X_test)
            f1_score_val = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
            roc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

            results.append({
                'Model': name,
                'Best Params': str(best_params),
                'Test F1': round(f1_score_val, 4),
                'ROC AUC': round(roc, 4)
            })

            trained_models[name] = best_model

        # Save all models
        for name, model in trained_models.items():
            joblib.dump(model, f'{name.replace(" ", "_")}_model.pkl')

        return pd.DataFrame(results)

# MODEL INTERPRETER 
class ModelInterpreter:
    def __init__(self, model):
        self.model = model

    def explain_model(self, df_sample):
        engineered = self.model.named_steps['engineer'].transform(df_sample)
        processed = self.model.named_steps['preprocessor'].transform(engineered)
        classifier = self.model.named_steps['classifier']

        feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()

        if isinstance(classifier, (RandomForestClassifier, XGBClassifier)):
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(processed)
        elif isinstance(classifier, LogisticRegression):
            explainer = shap.LinearExplainer(classifier, processed)
            shap_values = explainer.shap_values(processed)
        else:
            raise ValueError(f"Unsupported model type for SHAP: {type(classifier)}")

        plt.figure(figsize=(10, 5))
        shap.summary_plot(shap_values, processed, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig('shap_summary.png')
        plt.close()

# STREAMLIT APP 
def main():
    st.set_page_config(layout="wide")
    st.title('Credit Risk Prediction App')

    if 'df' not in st.session_state:
        st.session_state.df = DataLoader().load_data()

    df = st.session_state.df

    st.sidebar.header("Select Model and Input Features")
    model_choice = st.sidebar.selectbox("Choose Model for Prediction", ['Random Forest', 'XGBoost', 'Logistic Regression'])

    model_filename = f"{model_choice.replace(' ', '_')}_model.pkl"
    if not os.path.exists(model_filename):
        st.info("Training models because they don't exist yet...")
        trainer = ModelTrainer(
            df.drop(columns='Risk'),
            df['Risk'].map({'good': 0, 'bad': 1})
        )
        trainer.train_models()

    model = joblib.load(model_filename)

    # Sidebar inputs
    age = st.sidebar.slider('Age', 18, 100, 30)
    job = st.sidebar.slider('Job Category', 0, 3, 1)
    credit_amount = st.sidebar.slider('Credit Amount (€)', 250, 20000, 5000)
    duration = st.sidebar.slider('Duration (months)', 6, 72, 24)
    sex = st.sidebar.radio('Sex', ['male', 'female'])
    housing = st.sidebar.selectbox('Housing', ['own', 'rent', 'free'])
    purpose = st.sidebar.selectbox('Purpose', sorted(df['Purpose'].dropna().unique()))
    saving_acc = st.sidebar.selectbox('Saving Account', sorted(df['Saving accounts'].dropna().unique().tolist()) + ['unknown'])
    checking_acc = st.sidebar.selectbox('Checking Account', sorted(df['Checking account'].dropna().unique().tolist()) + ['unknown'])

    input_df = pd.DataFrame([[age, job, credit_amount, duration, sex, housing, purpose, saving_acc, checking_acc]],
                             columns=['Age', 'Job', 'Credit amount', 'Duration', 'Sex', 'Housing', 'Purpose', 'Saving accounts', 'Checking account'])

    if st.sidebar.button("Predict"):
        try:
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]
            st.success(f"Predicted Risk: {'High' if pred else 'Low'} — Probability of Risk: {proba:.2%}")
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            st.error(f"Prediction failed: {str(e)}")

    st.subheader("Model Explanation")
    if st.button("Show SHAP Summary"):
        try:
            sample = df.drop(columns='Risk').iloc[:50]
            interpreter = ModelInterpreter(model)
            interpreter.explain_model(sample)
            st.image('shap_summary.png')
        except Exception as e:
            st.error(f"SHAP explanation failed: {str(e)}")

    st.subheader("Model Training Performance")
    if st.button("Train and Compare Models"):
        trainer = ModelTrainer(
            df.drop(columns='Risk'),
            df['Risk'].map({'good': 0, 'bad': 1})
        )
        results = trainer.train_models()
        st.dataframe(results)
        st.success("Training complete. Models saved!")

# RUN APP
if __name__ == '__main__':
    main()






