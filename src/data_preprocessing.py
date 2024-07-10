import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings 

# Suppress warnings to keep the output clean
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('loan_train.csv')

# Display basic information about the DataFrame (data types, non-null counts, etc.)
df.info()

# Display statistical summary of the DataFrame
df.describe()

# Count non-null values for each column
df.count()

# Count the number of missing values for each column
df.isnull().sum()

# Convert 'Dependents' column to numeric, coercing errors to NaN
df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce')

# Fill missing values in 'Dependents' with the mean of the column
df['Dependents'].fillna(df['Dependents'].mean(), inplace=True)

# Fill missing values in 'Credit_History' with the mean of the column
df['Credit_History'].fillna(df['Credit_History'].mean(), inplace=True)

# Fill missing values in 'Loan_Amount_Term' with the mean of the column
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(), inplace=True)

# Fill missing values in 'LoanAmount' with the mean of the column
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)

# Drop any remaining rows with missing values
df = df.dropna()

# Define a function to remove outliers from specified columns
def remove_outliers(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# List of columns from which to remove outliers
columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

# Remove outliers from the DataFrame
df = remove_outliers(df, columns)

# Define mappings for binary categorical variables
binary_mappings = {
    'Gender': {'Male': 0, 'Female': 1},
    'Married': {'No': 0, 'Yes': 1},
    'Education': {'Not Graduate': 0, 'Graduate': 1},
    'Self_Employed': {'No': 0, 'Yes': 1},
    'Loan_Status': {'N': 0, 'Y': 1},
}

# Replace binary categorical values with the defined mappings
df.replace(binary_mappings, inplace=True)

# Convert categorical 'Property_Area' into dummy/indicator variables
df = pd.get_dummies(df, columns=['Property_Area'], prefix='Property_Area', drop_first=False)

# Drop the 'Loan_ID' column as it is not needed for modeling
df = df.drop(["Loan_ID"], axis=1)

# Define the feature set X (all columns except 'Loan_Status')
X = df.drop(["Loan_Status"], axis=1)

# Define the target variable y ('Loan_Status' column)
y = df["Loan_Status"]

# Ensure boolean columns are of integer type
bool_columns = ['Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban']
df[bool_columns] = df[bool_columns].astype('int64')

# Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
X, y = SMOTE().fit_resample(X, y)
