import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, SimpleImputer
from sklearn.impute import SimpleImputer as SKSimpleImputer

# Step 1: Extract - Load data from CSV
def extract_data(file_path):
    """
    Extract data from the input CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        print("Extraction complete. Raw data shape:", df.shape)
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please create the input file.")
        return None

# Step 2: Transform/Preprocess - Clean, encode, scale, and engineer features
def transform_data(df):
    """
    Transform the data:
    - Handle missing values (impute numerical with median, categorical with mode).
    - Encode categorical columns (one-hot).
    - Scale numerical columns (StandardScaler).
    - Feature engineering: Add 'age_income_ratio' = age / (income + 1) to avoid division by zero.
    """
    if df is None or df.empty:
        return None
    
    # Identify column types
    numerical_features = ['age', 'income', 'purchase_amount']
    categorical_features = ['gender', 'region']
    
    # Drop ID column if not needed for modeling (keep for loading)
    df_processed = df.drop('customer_id', axis=1)  # Assuming ID not needed in features
    
    # Create preprocessing pipeline using Scikit-learn
    # Imputer for numerical (median) and categorical (most frequent)
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Apply preprocessing (fit and transform)
    # Note: For target 'purchase_amount', we impute but don't encode/scale it here (treat as label)
    # Separate features and target for clarity
    X = df_processed.drop('purchase_amount', axis=1)
    y = df_processed['purchase_amount']
    
    # Impute y separately (median)
    y_imputer = SimpleImputer(strategy='median')
    y = pd.Series(y_imputer.fit_transform(y.values.reshape(-1, 1)).flatten(), index=y.index)
    
    # Preprocess features
    X_preprocessed = preprocessor.fit_transform(X)
    
    # Convert back to DataFrame for easier handling
    # Get feature names after preprocessing
    num_features = numerical_features[:-1]  # Exclude target from num features
    cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
    all_feature_names = list(num_features) + list(cat_feature_names)
    
    X_df = pd.DataFrame(X_preprocessed, columns=all_feature_names, index=X.index)
    
    # Feature engineering: age_income_ratio (using original imputed values for simplicity)
    # Reconstruct original imputed numericals for engineering
    X_imputed = pd.DataFrame(
        preprocessor.named_transformers_['num']['imputer'].transform(X[numerical_features[:-1]]),
        columns=num_features,
        index=X.index
    )
    X_imputed['age_income_ratio'] = X_imputed['age'] / (X_imputed['income'] + 1)  # Avoid div by zero
    
    # Combine everything
    transformed_df = pd.concat([X_df, X_imputed['age_income_ratio'], y.rename('purchase_amount')], axis=1)
    
    print("Transformation complete. Processed data shape:", transformed_df.shape)
    print(transformed_df.head())
    print("\nSummary stats after transformation:")
    print(transformed_df.describe())
    
    return transformed_df

# Step 3: Load - Save processed data to output CSV
def load_data(df, output_file):
    """
    Load the transformed data to a new CSV file.
    """
    if df is None or df.empty:
        print("No data to load.")
        return
    
    # Re-add customer_id if needed (from original df, assuming it's available)
    # For simplicity, we'll add a sequential ID if not present
    if 'customer_id' not in df.columns:
        df['customer_id'] = range(1, len(df) + 1)
    
    df.to_csv(output_file, index=False)
    print(f"Loading complete. Data saved to {output_file}")

# Main ETL Pipeline Function
def run_etl_pipeline(input_file='input_data.csv', output_file='output_data.csv'):
    """
    Run the full ETL pipeline.
    """
    print("=== Starting ETL Pipeline ===")
    
    # Extract
    raw_df = extract_data(input_file)
    
    # Transform
    processed_df = transform_data(raw_df)
    
    # Load
    load_data(processed_df, output_file)
    
    print("=== ETL Pipeline Complete ===")

# Run the pipeline if script is executed directly
if __name__ == "__main__":
    run_etl_pipeline()
