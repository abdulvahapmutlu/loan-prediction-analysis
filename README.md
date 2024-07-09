# Loan Prediction Analysis
This repository contains a comprehensive analysis and machine learning modeling for predicting loan approval status using various classification algorithms. The project involves data preprocessing, exploratory data analysis, model training, hyperparameter tuning, and evaluation of multiple machine learning models.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Preprocessing](#data-preprocessing)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Model Training](#model-training)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Evaluation](#evaluation)
7. [Conclusion](#conclusion)

## Introduction

The goal of this project is to build predictive models that can determine whether a loan will be approved or not based on historical data. The dataset used for this analysis is `loan_train.csv`.
The dataset is taken from https://www.kaggle.com/datasets/vikasukani/loan-eligible-dataset

## Data Preprocessing

1. **Handling Missing Values**: Missing values in `Dependents`, `Credit_History`, `Loan_Amount_Term`, and `LoanAmount` were filled with their respective means.
2. **Outlier Removal**: Outliers in `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, and `Loan_Amount_Term` were removed using the IQR method.
3. **Binary Transformation**: Columns like `Gender`, `Married`, `Education`, `Self_Employed`, and `Loan_Status` were transformed into binary values.
4. **One-Hot Encoding**: The `Property_Area` column was one-hot encoded.
5. **Resampling**: The SMOTE technique was used to address class imbalance in the target variable.

## Exploratory Data Analysis

- Distribution plots for `Loan_Status`, `Gender`, and `Education`.
- Histograms for `ApplicantIncome` and `LoanAmount`.
- Boxplot for `ApplicantIncome` by `Education`.
- Pairplot for visualizing relationships between `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, and `Credit_History`.
- Correlation heatmap.
- Interactive scatter plot using Plotly for `ApplicantIncome` vs `LoanAmount` colored by `Loan_Status`.

## Model Training

Various machine learning models were trained:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes

## Hyperparameter Tuning

Hyperparameter tuning was performed using GridSearchCV for the following models:
- Random Forest
- Logistic Regression
- Decision Tree
- Gradient Boosting
- SVM
- KNN
- Naive Bayes

The best parameters were identified and used to retrain the models for optimal performance.

## Evaluation

The models were evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

Confusion matrices and ROC curves were plotted for the models with the best parameters.

## Conclusion

This project demonstrates the end-to-end process of building a machine learning model, from data preprocessing to model evaluation. The Random Forest and Gradient Boosting classifiers showed promising results with high accuracy and AUC scores. Best classifier for this project is the Random Forest.

## Usage

Requirements
Here are the requirements for your project:

Dependencies:

Pandas: For data manipulation and analysis.
Matplotlib: For creating static, animated, and interactive visualizations.
Seaborn: For statistical data visualization.
Plotly: For creating interactive plots.
Scikit-learn: For machine learning, including preprocessing, model selection, and evaluation.
Imbalanced-learn: For handling imbalanced datasets.
NumPy: For numerical operations.
You can install these dependencies using pip:

```
pip install pandas matplotlib seaborn plotly scikit-learn imbalanced-learn numpy
```
or 

```
pip install -r requirements.txt
```


To use this repository, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/loan-prediction-analysis.git
   ```
2. Install the required packages
   
3. Run the Jupyter notebook or Python scripts to execute the analysis.


Feel free to explore and modify the code as needed. Contributions are welcome!


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


### Repository Structure

```
loan-prediction-analysis/
│
├── data/
│   └── loan_train.csv
│
├── notebooks/
│   └── loan_prediction_analysis.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── data_visualization.py
│   ├── exploratory_data_analysis.py
│   ├── model_training.py
│   ├── hyperparameter_tuning.py
│   └── evaluation.py
│
├── requirements.txt
├── LICENSE
└── README.md
```

