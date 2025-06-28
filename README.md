# Telco Customer Churn Prediction

## Project Overview

This project focuses on predicting customer churn in a telecommunications company using machine learning techniques. Customer churn, the act of customers discontinuing their service, is a critical business problem for telecommunication providers as acquiring new customers is often more expensive than retaining existing ones. By accurately predicting which customers are likely to churn, companies can implement proactive retention strategies.

This repository contains the full data science workflow, from exploratory data analysis (EDA) and data cleaning to feature engineering, model training, evaluation, and hyperparameter tuning.

## Table of Contents

1.  [Introduction](#1-introduction)
2.  [Dataset](#2-dataset)
3.  [Project Structure](#3-project-structure)
4.  [Methodology](#4-methodology)
    * [Data Loading & Initial Cleaning](#41-data-loading--initial-cleaning)
    * [Exploratory Data Analysis (EDA)](#42-exploratory-data-analysis-eda)
    * [Feature Engineering & Preprocessing](#43-feature-engineering--preprocessing)
    * [Model Training & Evaluation](#44-model-training--evaluation)
    * [Hyperparameter Tuning](#45-hyperparameter-tuning)
5.  [Key Findings & Model Performance](#5-key-findings--model-performance)
6.  [How to Run the Project](#6-how-to-run-the-project)
    * [Prerequisites](#prerequisites)
    * [Cloning the Repository](#cloning-the-repository)
    * [Setting up the Environment](#setting-up-the-environment)
    * [Running the Notebooks](#running-the-notebooks)
7.  [Technologies Used](#7-technologies-used)
8.  [Contact](#8-contact)

## 1. Introduction

The objective of this project is to build and evaluate machine learning models capable of predicting customer churn for a telecommunications company. The insights gained can help the company understand factors contributing to churn and enable targeted interventions to retain valuable customers. The project addresses the challenge of an imbalanced dataset, where churned customers are a minority.

## 2. Dataset

The dataset used in this project is the **Telco Customer Churn dataset**, publicly available. It contains information about a fictional telecommunications company's customers, including their services, account information, and whether they churned within the last month.

* **Source:** [Dataset Source - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Columns:** 21 columns
    * `customerID`: Customer ID
    * Demographic Info: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
    * Services: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
    * Account Info: `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `Tenure`
    * Target Variable: `Churn` (Yes/No)

## 3. Project Structure

The repository is organized as follows:
```
.
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Original dataset
│   └── processed/
│       └── telco_churn_processed_data.npz        # Processed train/test splits (ignored by Git)
├── notebooks/
│   ├── 1.0-EDA-DataCleaning.ipynb               # Exploratory Data Analysis and Initial Data Cleaning
│   ├── 2.0-FeatureEngineering-Preprocessing.ipynb # Feature Engineering and Data Preprocessing
│   └── 3.0-ModelTraining-Evaluation.ipynb       # Model Training, Evaluation, and Hyperparameter Tuning
├── .gitignore                                   # Specifies intentionally untracked files
└── README.md                                    # Project overview and instructions
```

## 4. Methodology

The project follows a standard machine learning pipeline:

### 4.1 Data Loading & Initial Cleaning

* Loaded the dataset and performed initial inspection (`.head()`, `.info()`, `.describe()`, `.shape`).
* Converted column names to lowercase for consistency.
* Handled the `totalcharges` column: Converted to numeric type, replacing empty strings with `NaN`s, then drop records with `NaN` as they are only few records.
* Converted the `churn` target variable from 'Yes'/'No' to binary `1`/`0`.
* Dropped the non-predictive `customerid` column.

### 4.2 Exploratory Data Analysis (EDA)

* Analyzed distributions of numerical features (`tenure`, `monthlycharges`, `totalcharges`).
* Visualized categorical feature distributions and their relationship with churn.
* Identified class imbalance in the `churn` target variable.
* Found that customers with month-to-month contracts and no online security were significantly more likely to churn."

### 4.3 Feature Engineering & Preprocessing

* Separated features (X) from the target (y).
* Identified numerical and categorical features.
* **One-Hot Encoding:** Applied to all categorical features using `sklearn.preprocessing.OneHotEncoder` with `drop='first'` to handle nominal categories and prevent multicollinearity. This was particularly important for features with 'No internet service' categories.
* **Feature Scaling:** Applied `sklearn.preprocessing.StandardScaler` to numerical features (`tenure`, `monthlycharges`, `totalcharges`) to normalize their scale, which is beneficial for many machine learning algorithms.
* **Train-Test Split:** Split the preprocessed data into 80% training and 20% testing sets using `sklearn.model_selection.train_test_split` with `stratify=y` to maintain the class distribution in both sets, crucial for the imbalanced target variable.
* Saved the processed `X_train`, `X_test`, `y_train`, `y_test` NumPy arrays into a `.npz` file in `data/processed/` for efficient loading in subsequent steps.

### 4.4 Model Training & Evaluation

* Established baseline performance using two common classification algorithms:
    * **Logistic Regression:** A linear model providing a good interpretable baseline.
    * **Random Forest Classifier:** An ensemble tree-based model known for its robustness.
* Both models were initialized with `class_weight='balanced'` to mitigate the impact of class imbalance on training.
* Models were evaluated using a suite of metrics relevant for imbalanced classification: Accuracy, Precision, Recall, F1-Score, and ROC AUC Score.
* Utility functions were created for modular evaluation: `print_metrics`, `plot_confusion_matrix`, and `plot_roc_curve`.

### 4.5 Hyperparameter Tuning

* To optimize model performance, hyperparameter tuning was performed for both Logistic Regression and Random Forest.
* **Logistic Regression:** Tuned `C` (regularization strength) and `penalty` (`l1`/`l2`) using `GridSearchCV`, optimizing for ROC AUC.
* **Random Forest:** Tuned `n_estimators`, `max_features`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `bootstrap` using `RandomizedSearchCV` (due to larger search space), also optimizing for ROC AUC.
* 5-fold cross-validation (`cv=5`) was used during the tuning process to ensure robust parameter selection.

## 5. Key Findings & Model Performance

After extensive preprocessing and hyperparameter tuning, the **Random Forest Classifier** emerged as the best-performing model for predicting customer churn in this dataset.

### Performance Summary of Tuned Models on Test Set:

| Metric          | Logistic Regression (Tuned) | Random Forest (Tuned) |
| :-------------- | :-------------------------- | :-------------------- |
| **Accuracy** | 0.7257                      | **0.7562** |
| **Precision** | 0.4902                      | **0.5287** |
| **Recall** | **0.7995** | 0.7647                |
| **F1-Score** | 0.6077                      | **0.6251** |
| **ROC AUC Score** | 0.8342                      | **0.8375** |

### Interpretation:

* The **Tuned Random Forest Classifier** achieved the highest Accuracy, Precision, F1-Score, and ROC AUC Score. Its improved F1-Score (0.6251) and ROC AUC (0.8375) indicate a better overall balance between correctly identifying churners and maintaining classification quality across various thresholds.
* While Tuned Logistic Regression had a slightly higher Recall (0.7995), the significant increase in Recall for the Tuned Random Forest (from 0.4893 baseline to 0.7647) makes it a more robust choice, particularly for a churn prediction problem where identifying as many potential churners as possible is critical. The Random Forest model provides a stronger balance of these key metrics, making it more effective for business intervention strategies.

## 6. How to Run the Project

To run this project locally, follow these steps:

### Prerequisites

* Python 3.8+
* Git

### Cloning the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/akallam/customer-churn-ml.git
cd customer-churn-ml
```

### Setting up the Environment
```
# Create a conda environment (if you have Anaconda/Miniconda)
conda create -n telco_churn python=3.9 pandas numpy scikit-learn matplotlib seaborn jupyter -y
conda activate telco_churn

# OR, if using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

```

### Running the Notebooks
1.  Start Jupyter Lab or Jupyter Notebook:
```
jupyter lab
#or
jupyter notebook
```
2. Navigate to the `notebooks/` directory
3. Open and run the notebooks in the following order
	* `1.0-EDA-DataCleaning.ipynb`
	* `2.0-FeatureEngineering-Preprocessing.ipynb`
	* `3.0-ModelTraining-Evaluation.ipynb`
	Ensure you run all cells in each notebook sequentially to generate the processed data and model outputs.

## 7. Technologies Used
* Python 3.x
* Pandas (for data manipulation and analysis)
* NumPy (for numerical operations)
* Scikit-learn (for machine learning models and preprocessing)
* Matplotlib (for data visualization)
* Seaborn (for enhanced data visualization)
* Jupyter Notebook

## 8. Contact
* Aditya Kallam
* GitHub: [https://github.com/akallam]
* LinkedIn: [https://www.linkedin.com/in/aditya-kallam/]






