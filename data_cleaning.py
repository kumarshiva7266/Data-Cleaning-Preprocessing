import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os

# Set random seed for reproducibility
np.random.seed(42)

# Create a sample dataset with various data types and issues
def create_sample_dataset():
    # Create a dictionary with sample data
    data = {
        'age': np.random.normal(35, 10, 1000),
        'salary': np.random.normal(50000, 15000, 1000),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD', None], 1000),
        'experience': np.random.normal(8, 4, 1000),
        'department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing', 'Sales', None], 1000),
        'performance_score': np.random.normal(75, 10, 1000)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Introduce some missing values
    df.loc[df.sample(frac=0.1).index, 'age'] = np.nan
    df.loc[df.sample(frac=0.15).index, 'salary'] = np.nan
    df.loc[df.sample(frac=0.2).index, 'experience'] = np.nan
    
    # Introduce some outliers
    df.loc[df.sample(frac=0.05).index, 'salary'] = df['salary'] * 3
    df.loc[df.sample(frac=0.05).index, 'age'] = df['age'] * 2
    
    return df

def explore_data(df):
    print("\n=== Basic Information ===")
    print("\nDataset Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nBasic Statistics:")
    print(df.describe())

def handle_missing_values(df):
    print("\n=== Handling Missing Values ===")
    
    # Create a copy of the dataframe
    df_cleaned = df.copy()
    
    # Handle numerical columns
    numerical_cols = ['age', 'salary', 'experience', 'performance_score']
    imputer = SimpleImputer(strategy='mean')
    df_cleaned[numerical_cols] = imputer.fit_transform(df[numerical_cols])
    
    # Handle categorical columns
    categorical_cols = ['education', 'department']
    for col in categorical_cols:
        df_cleaned[col] = df[col].fillna(df[col].mode()[0])
    
    print("\nMissing values after cleaning:")
    print(df_cleaned.isnull().sum())
    
    return df_cleaned

def encode_categorical_features(df):
    print("\n=== Encoding Categorical Features ===")
    
    # Create a copy of the dataframe
    df_encoded = df.copy()
    
    # Use Label Encoding for categorical columns
    categorical_cols = ['education', 'department']
    label_encoders = {}
    
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        df_encoded[col] = label_encoders[col].fit_transform(df_encoded[col])
    
    print("\nEncoded categorical columns:")
    for col in categorical_cols:
        print(f"\n{col} mapping:")
        for i, label in enumerate(label_encoders[col].classes_):
            print(f"{label}: {i}")
    
    return df_encoded

def normalize_features(df):
    print("\n=== Normalizing Features ===")
    
    # Create a copy of the dataframe
    df_normalized = df.copy()
    
    # Select numerical columns for normalization
    numerical_cols = ['age', 'salary', 'experience', 'performance_score']
    
    # Apply StandardScaler
    scaler = StandardScaler()
    df_normalized[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    print("\nNormalized data statistics:")
    print(df_normalized[numerical_cols].describe())
    
    return df_normalized

def visualize_outliers(df):
    print("\n=== Visualizing Outliers ===")
    
    # Create boxplots for numerical columns
    numerical_cols = ['age', 'salary', 'experience', 'performance_score']
    
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(data=df, y=col)
        plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()

def remove_outliers(df):
    print("\n=== Removing Outliers ===")
    
    # Create a copy of the dataframe
    df_no_outliers = df.copy()
    
    # Define numerical columns
    numerical_cols = ['age', 'salary', 'experience', 'performance_score']
    
    # Remove outliers using IQR method
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_no_outliers = df_no_outliers[
            (df_no_outliers[col] >= lower_bound) & 
            (df_no_outliers[col] <= upper_bound)
        ]
    
    print(f"\nOriginal shape: {df.shape}")
    print(f"Shape after removing outliers: {df_no_outliers.shape}")
    
    return df_no_outliers

def load_dataset(file_path=None):
    """
    Load a dataset from a CSV file or create a sample dataset if no file is provided.
    """
    if file_path and os.path.exists(file_path):
        print(f"Loading dataset from {file_path}")
        return pd.read_csv(file_path)
    else:
        print("No valid file path provided. Creating sample dataset...")
        return create_sample_dataset()

def save_processed_data(df, output_path='processed_data.csv'):
    """
    Save the processed DataFrame to a CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to {output_path}")

def process_dataset(df):
    """
    Process the dataset through all cleaning and preprocessing steps.
    Returns the final processed DataFrame.
    """
    print("\n=== Starting Data Processing ===")
    
    # Step 1: Handle missing values
    print("\nStep 1: Handling missing values...")
    df_cleaned = handle_missing_values(df)
    
    # Step 2: Encode categorical features
    print("\nStep 2: Encoding categorical features...")
    df_encoded = encode_categorical_features(df_cleaned)
    
    # Step 3: Normalize numerical features
    print("\nStep 3: Normalizing numerical features...")
    df_normalized = normalize_features(df_encoded)
    
    # Step 4: Remove outliers
    print("\nStep 4: Removing outliers...")
    df_final = remove_outliers(df_normalized)
    
    return df_final

def main():
    # Get user input for dataset
    print("Data Cleaning and Preprocessing Tool")
    print("===================================")
    print("1. Use sample dataset")
    print("2. Load your own dataset")
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "2":
        file_path = input("Enter the path to your CSV file: ")
        df = load_dataset(file_path)
    else:
        df = load_dataset()
    
    # Explore initial data
    print("\n=== Initial Data Exploration ===")
    explore_data(df)
    
    # Process the dataset
    df_processed = process_dataset(df)
    
    # Show final results
    print("\n=== Final Processed Dataset Information ===")
    print("\nFinal shape:", df_processed.shape)
    print("\nFinal data types:")
    print(df_processed.dtypes)
    print("\nFinal statistics:")
    print(df_processed.describe())
    
    # Save processed data
    save_choice = input("\nDo you want to save the processed data? (y/n): ")
    if save_choice.lower() == 'y':
        output_path = input("Enter output file path (default: processed_data.csv): ") or 'processed_data.csv'
        save_processed_data(df_processed, output_path)

if __name__ == "__main__":
    main() 