# Telco Customer Churn Prediction & Machine Learning Pipeline

## 📌 Project Overview

Customer churn (customer attrition) is a critical metric for telecommunications companies.

This project focuses on analyzing customer data from a telecom provider, identifying factors associated with customer churn, and building machine learning classification models to predict whether a customer is likely to churn.

The project covers the full machine learning workflow, from data preprocessing and exploratory data analysis (EDA) to model training, evaluation, hyperparameter tuning, and deployment using Streamlit.

## 👥 Team Members

- Marina Melad Maken
- Nariman Ahmed Shawky
- Mennatallah Ahmed Abdel Salam

## 📊 Dataset

The project uses the **Telco Customer Churn** dataset (`Telco-Customer-Churn.csv`).

- **Rows:** 7,043
- **Columns:** 21

### Main Feature Groups

**Customer Information**
- customerID
- gender
- SeniorCitizen
- Partner
- Dependents

**Services Subscribed**
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies

**Account & Billing**
- tenure
- Contract
- PaperlessBilling
- PaymentMethod
- MonthlyCharges
- TotalCharges

**Target Variable**
- Churn (Yes / No)

## 🛠️ Project Workflow & Methodology

### 1. Data Preprocessing & Cleaning

- Converted `TotalCharges` to numeric values.
- Handled 11 missing/blank values in `TotalCharges` using median imputation.
- Removed `customerID` because it is an identifier and does not provide useful predictive information.
- Applied preprocessing and feature encoding to prepare the data for machine learning models.
- Used the IQR method to inspect numerical features (`tenure`, `MonthlyCharges`, and `TotalCharges`) for extreme outliers.

### 2. Exploratory Data Analysis (EDA)

The project includes exploratory analysis to understand the dataset and investigate relationships between customer characteristics and churn.

The analysis includes:
- Feature distributions
- Correlation analysis
- Numerical feature visualization
- Log-scaled box plots

### 3. Machine Learning Models

Several classification models were implemented and compared:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier
- LightGBM Classifier
- Stacking Classifier (Ensemble Learning)

### 4. Handling Class Imbalance

The dataset contains an imbalance between churned and non-churned customers.

To address this, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training data.

### 5. Hyperparameter Tuning

Model optimization was performed using:
- `RandomizedSearchCV`
- `StratifiedKFold`
- Hyperparameter tuning

### 6. Model Evaluation

The classification models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

## 🚀 Streamlit Deployment

The trained churn prediction model was deployed as an interactive **Streamlit web application**.

The application allows users to enter customer information and receive a churn-risk prediction.

### 🔗 Live Demo

https://telco-customer-churn-prediction1-jh2yzmyj98qe7eglf5vmh8.streamlit.app/

## 📁 Project Structure

```text
Telco-Customer-Churn-Prediction1/
│
├── churn_app/
│   ├── README.md
│   ├── Telco-Customer-Churn.csv
│   ├── app.py
│   ├── churn_core.py
│   ├── churn_model.joblib
│   ├── requirements.txt
│   └── train_model.py
│
├── Final_Project_ML (1).ipynb
├── Telco-Customer-Churn.csv
├── README.md
└── .gitignore
```

## 💻 How to Run the Project

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm
```

### Running the Notebook

1. Clone the repository:

```bash
git clone https://github.com/marinamelad-2/Telco-Customer-Churn-Prediction1.git
```

2. Open `Final_Project_ML (1).ipynb` in Google Colab or Jupyter Notebook.
3. Make sure `Telco-Customer-Churn.csv` is available in the working directory or Google Colab session.
4. Run the notebook cells sequentially.

## 🧰 Tools & Technologies

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Imbalanced Learning:** Imbalanced-learn (SMOTE)
- **Gradient Boosting:** XGBoost, LightGBM
- **Model Export:** Joblib
- **Deployment:** Streamlit
- **Version Control:** Git & GitHub

## 🎯 Project Outcome

This project demonstrates an end-to-end machine learning workflow for a customer churn prediction problem, including:

- Data cleaning and preprocessing
- Exploratory data analysis
- Handling class imbalance
- Training and comparing multiple classification models
- Hyperparameter tuning
- Model evaluation
- Saving a trained model
- Deploying an interactive prediction application with Streamlit

