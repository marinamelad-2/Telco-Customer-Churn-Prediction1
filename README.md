#  Telco Customer Churn Prediction & Machine Learning Pipeline

##  Project Overview
Customer churn (customer attrition) is a critical metric for telecommunications companies. This project focuses on analyzing customer data from a telecom provider to identify key factors contributing to churn and building robust machine learning classification models to predict whether a customer is likely to churn.

---

##  Team Members
- **Marina Melad maken**
- **Nariman Ahmed shawky**
- **Mennatallah Ahmed abdel salam**

---

##  Dataset Description
The dataset used in this project is the **Telco Customer Churn** dataset (`Telco-Customer-Churn.csv`). It contains **7,043 rows** and **21 columns**, covering demographic data, subscribed services, account information, and churn status.

### **Features Overview:**
- **Customer Information:** `customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Services Subscribed:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Account & Billing:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
- **Target Variable:** `Churn` (*Yes / No*)

---

## 🛠️ Project Workflow & Methodology

### **1. Data Preprocessing & Cleaning**
- **Hidden Nulls Treatment:** Discovered and handled 11 missing/blank values in `TotalCharges` by converting the column to numeric and imputing missing values with the **median**.
- **Feature Removal:** Dropped the identifier column `customerID` as it holds no predictive value.
- **Outlier Detection:** Used the Interquartile Range (IQR) method on numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) and confirmed no extreme outliers requiring truncation.

### **2. Exploratory Data Analysis (EDA)**
- Evaluated feature distributions and correlation with the target variable `Churn`.
- Visualized numerical variables using **Log-Scaled Box Plots**.

### **3. Machine Learning Models**
The project implements and compares multiple classification models:
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **XGBoost Classifier**
- **LightGBM Classifier**
- **Stacking Classifier** *(Ensemble Learning)*

### **4. Advanced Techniques**
- **Handling Class Imbalance:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the target classes.
- **Hyperparameter Tuning:** Optimized model performance using `RandomizedSearchCV` and **Cross-Validation** (`StratifiedKFold`).
- **Evaluation Metrics:** Evaluated models based on **Accuracy, Precision, Recall, F1-Score, and ROC-AUC**.

---

##  How to Run the Project

### **Prerequisites**
Make sure you have Python installed along with the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm
Running the Notebook
Download or clone this repository:

Bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
Upload the Telco-Customer-Churn.csv dataset to your working directory or Google Colab session.

Open the Notebook.ipynb file in Google Colab or Jupyter Notebook and run all cells sequentially.

 Tools & Technologies Used
Python (Pandas, NumPy)

Data Visualization: Matplotlib, Seaborn

Machine Learning & Preprocessing: Scikit-Learn, Imbalanced-Learn (SMOTE)

Gradient Boosting Frameworks: XGBoost, LightGBM

Deployment/Export: Pickle
