import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Set up the matplotlib figure
plt.figure(figsize=(20, 15))

# 1. Distribution of Loan Status
plt.subplot(2, 3, 1)
sns.countplot(data=df, x='Loan_Status', palette='Set2')
plt.title('Distribution of Loan Status')

# 2. Distribution of Gender
plt.subplot(2, 3, 2)
sns.countplot(data=df, x='Gender', palette='Set2')
plt.title('Distribution of Gender')

# 3. Distribution of Education
plt.subplot(2, 3, 3)
sns.countplot(data=df, x='Education', palette='Set2')
plt.title('Distribution of Education')

# 4. Distribution of Applicant Income
plt.subplot(2, 3, 4)
sns.histplot(data=df, x='ApplicantIncome', kde=True, bins=30, color='green')
plt.title('Distribution of Applicant Income')

# 5. Distribution of Loan Amount
plt.subplot(2, 3, 5)
sns.histplot(data=df, x='LoanAmount', kde=True, bins=30, color='blue')
plt.title('Distribution of Loan Amount')

# 6. Boxplot of Applicant Income by Education
plt.subplot(2, 3, 6)
sns.boxplot(data=df, x='Education', y='ApplicantIncome', palette='Set2')
plt.title('Applicant Income by Education')

plt.tight_layout()
plt.show()

# Pairplot to visualize relationships between variables
sns.pairplot(df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']], diag_kind='kde')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
corr = df[['Dependents', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']].corr()
sns.heatmap(corr, annot=True)
plt.title('Correlation Heatmap')
plt.show()

# Plotly scatter plot for Applicant Income vs Loan Amount colored by Loan Status
fig = px.scatter(df, x='ApplicantIncome', y='LoanAmount', color='Loan_Status', title='Applicant Income vs Loan Amount')
fig.show()

