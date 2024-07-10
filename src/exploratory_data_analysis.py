import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('loan_train.csv')

# Display basic information about the DataFrame (data types, non-null counts, etc.)
print("DataFrame Info:")
df.info()

# Display statistical summary of the DataFrame
print("\nStatistical Summary:")
print(df.describe())

# Count the number of missing values for each column
print("\nMissing Values Count:")
print(df.isnull().sum())

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

# Visualizing the distribution of numerical features
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
for column in numerical_columns:
    plt.figure(figsize=(10, 4))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()

# Visualizing the relationship between numerical features and the target variable
for column in numerical_columns:
    plt.figure(figsize=(10, 4))
    sns.boxplot(x='Loan_Status', y=column, data=df)
    plt.title(f'{column} by Loan_Status')
    plt.show()

# Visualizing the count of categorical features
categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
for column in categorical_columns:
    plt.figure(figsize=(10, 4))
    sns.countplot(x=column, hue='Loan_Status', data=df)
    plt.title(f'Count of {column} by Loan_Status')
    plt.show()


