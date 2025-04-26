import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def load_and_inspect_data(file_path):
# Load the dataset
    df = pd.read_csv(file_path)

# Display first few rows
    print("First 5 rows of the dataset:")
    print(df.head(), "\n")

# Get basic information about the dataset
    print("Dataset Info:")
    print(df.info(), "\n")

# Check for missing values
    print("Missing Values in Dataset:")
    print(df.isnull().sum(), "\n")

# Summary statistics
    print("Summary Statistics (for numerical columns):")
    print(df.describe(), "\n")

    return df

def plot_numerical_distribution(df):
    numerical_cols = df.select_dtypes(include=[np.number]).columns

    for col in numerical_cols:
        plt.figure(figsize=(10, 6))

        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()
def plot_categorical_distribution(df):
    categorical_cols = df.select_dtypes(include=[object]).columns

    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=col, data=df)
        plt.title(f'Count plot of {col}')
        plt.show()
def plot_correlation_matrix(df):
# Compute correlation matrix
    corr = df.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()
def plot_pairwise_relationships(df):
    sns.pairplot(df, diag_kind='kde')
    plt.show()

# Function to visualize boxplots to identify outliers
def plot_boxplots(df):
    numerical_cols = df.select_dtypes(include=[np.number]).columns

    for col in numerical_cols:

        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.show()
def plot_missing_values(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.show()

# Main EDA function
def perform_eda(file_path):
# Load dataset
    df = load_and_inspect_data(file_path)

# Visualize distributions of numerical features
    plot_numerical_distribution(df)

# Visualize distributions of categorical features
    plot_categorical_distribution(df)

# Plot correlation matrix for numerical features
    plot_correlation_matrix(df)

# Plot pairwise relationships between numerical features
    plot_pairwise_relationships(df)

# Visualize boxplots to identify outliers
    plot_boxplots(df)

# Visualize missing values
    plot_missing_values(df)
file_path = 'All Countries and Economies GDP (US) 1960-2023.csv' # Replace with your dataset path
perform_eda(file_path)  
