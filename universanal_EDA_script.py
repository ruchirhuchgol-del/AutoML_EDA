# ===================================================================
# GENERIC EXPLORATORY DATA ANALYSIS (EDA) SCRIPT
# ===================================================================
#
# Description: This script performs a comprehensive EDA on any tabular dataset.
#              It automatically identifies data types and generates relevant
#              visualizations and statistics.
#
# INSTRUCTIONS:
# 1. Install required libraries: pandas, numpy, matplotlib, seaborn
# 2. Update the FILE_PATH variable below to point to your dataset.
# 3. Run the script.
# ===================================================================

# IMPORT LIBRARIES 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
import os

filterwarnings('ignore')

# --- 2. USER CONFIGURATION ---
# ONLY CHANGE THIS SECTION
FILE_PATH = 'your_dataset.csv'  # <--- UPDATE THIS WITH YOUR FILE PATH
# You can also use an Excel file: 'your_dataset.xlsx'

# --- 3. FUNCTIONS FOR EDA ---

def load_data(file_path):
    """Loads data from a CSV or Excel file."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        print("Please update the FILE_PATH variable in the script.")
        return None
    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        print(" Data loaded successfully!")
        return df
    except Exception as e:
        print(f" Error loading data: {e}")
        return None

def get_column_types(df):
    """Separates columns into numerical and categorical."""
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    return numerical_cols, categorical_cols

def assess_data_quality(df):
    """Checks for missing values, duplicates, and basic info."""
    print("\n" + "="*50)
    print(" DATA QUALITY ASSESSMENT")
    print("="*50)

    # Basic Info
    print("\n--- Basic Information ---")
    print(f"Shape (rows, columns): {df.shape}")
    df.info()

    # Missing Values
    print("\n--- Missing Values ---")
    missing_vals = df.isnull().sum()
    missing_percent = (missing_vals / len(df)) * 100
    missing_df = pd.DataFrame({'Count': missing_vals, 'Percentage': missing_percent})
    missing_df = missing_df[missing_df['Count'] > 0].sort_values(by='Count', ascending=False)
    
    if not missing_df.empty:
        print(missing_df)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing_df.index, y=missing_df['Count'])
        plt.title('Missing Values Count per Column')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print(" No missing values found.")

    # Duplicate Rows
    duplicate_count = df.duplicated().sum()
    print(f"\n--- Duplicate Rows ---")
    if duplicate_count > 0:
        print(f" Found {duplicate_count} duplicate rows.")
    else:
        print(" No duplicate rows found.")

def univariate_analysis(df, numerical_cols, categorical_cols):
    """Performs univariate analysis with plots."""
    print("\n" + "="*50)
    print(" UNIVARIATE ANALYSIS")
    print("="*50)

    # Numerical Columns
    if numerical_cols:
        print(f"\n--- Analyzing {len(numerical_cols)} Numerical Columns ---")
        # Descriptive Statistics
        print(df[numerical_cols].describe())
        
        # Plots
        for col in numerical_cols:
            plt.figure(figsize=(14, 5))
            
            # Histogram
            plt.subplot(1, 2, 1)
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f'Distribution of {col}')
            
            # Box Plot
            plt.subplot(1, 2, 2)
            sns.boxplot(x=df[col])
            plt.title(f'Box Plot of {col}')
            
            plt.tight_layout()
            plt.show()

    # Categorical Columns
    if categorical_cols:
        print(f"\n--- Analyzing {len(categorical_cols)} Categorical Columns ---")
        for col in categorical_cols:
            # Skip columns with too many unique values for a clean plot
            if df[col].nunique() > 20:
                print(f"\nSkipping plot for '{col}' (has {df[col].nunique()} unique categories).")
                print(df[col].value_counts())
                continue

            plt.figure(figsize=(12, 6))
            
            # Count Plot
            sns.countplot(y=df[col], order=df[col].value_counts().index)
            plt.title(f'Count Plot of {col}')
            plt.xlabel('Count')
            plt.ylabel(col)
            
            plt.tight_layout()
            plt.show()

def bivariate_analysis(df, numerical_cols, categorical_cols):
    """Performs bivariate analysis to find relationships."""
    print("\n" + "="*50)
    print(" BIVARIATE ANALYSIS")
    print("="*50)

    # Numerical vs. Numerical (Correlation Heatmap)
    if len(numerical_cols) > 1:
        print("\n--- Correlation Matrix (Numerical Features) ---")
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
        plt.title('Correlation Heatmap of Numerical Features')
        plt.show()

    # Categorical vs. Categorical (Stacked Bar)
    if len(categorical_cols) > 1:
        print("\n--- Categorical Feature Relationships ---")
        # Limiting to top 2-3 categorical features to avoid too many plots
        for i, col1 in enumerate(categorical_cols[:2]):
            for col2 in categorical_cols[i+1:i+2]: # Analyze pairs
                if df[col1].nunique() * df[col2].nunique() < 50: # Avoid overly complex plots
                    plt.figure(figsize=(10, 6))
                    crosstab = pd.crosstab(df[col1], df[col2])
                    crosstab.plot(kind='bar', stacked=True, colormap='viridis')
                    plt.title(f'Relationship between {col1} and {col2}')
                    plt.ylabel('Count')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.show()

def generate_summary_report(df, numerical_cols, categorical_cols):
    """Generates a text-based summary of key findings."""
    print("\n" + "="*50)
    print(" SUMMARY REPORT")
    print("="*50)
    print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
    print(f"Identified {len(numerical_cols)} numerical and {len(categorical_cols)} categorical features.")
    
    # Most missing values
    missing_vals = df.isnull().sum()
    most_missing_col = missing_vals.idxmax()
    print(f"Column with most missing values: '{most_missing_col}' with {missing_vals.max()} missing entries.")
    
    # Highest correlation
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr().abs()
        # Get the pair with the highest correlation (excluding self-correlation)
        highest_corr = corr_matrix.unstack().sort_values(ascending=False)
        highest_corr = highest_corr[highest_corr < 1].head(1)
        if not highest_corr.empty:
            print(f"Highest correlation is between '{highest_corr.index[0][0]}' and '{highest_corr.index[0][1]}': {highest_corr.iloc[0]:.2f}")
    
    # Most varied numerical feature
    if numerical_cols:
        most_varied_col = df[numerical_cols].std().idxmax()
        print(f"Numerical feature with highest standard deviation: '{most_varied_col}'")
        
    # Categorical feature with most categories
    if categorical_cols:
        most_categories_col = df[categorical_cols].nunique().idxmax()
        print(f"Categorical feature with most unique categories: '{most_categories_col}' ({df[categorical_cols].nunique().max()} categories)")

    print("\n--- End of EDA ---")


# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    # Load the data
    df = load_data(FILE_PATH)

    if df is not None:
        # Get column types
        numerical_cols, categorical_cols = get_column_types(df)
        
        # Run the full EDA pipeline
        assess_data_quality(df)
        univariate_analysis(df, numerical_cols, categorical_cols)
        bivariate_analysis(df, numerical_cols, categorical_cols)
        generate_summary_report(df, numerical_cols, categorical_cols)
