import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import os

def load_and_preprocess_data(raw_data_path, random_seed=42):
    """Loads CSV, splits features/target, applies scaling/encoding, and applies SMOTE."""
    df = pd.read_csv(raw_data_path)
    
    X = df.drop(columns=['Condition Rating'])
    y = df['Condition Rating'] - 1 # Shift labels from 1-5 to 0-4 for XGBoost

    # Define feature groups based on data
    num_features = ['Age', 'Diameter', 'Slope', 'Depth', 'Length', 'Soil PH']
    cat_features = ['Material', 'Soil Type', 'Road Type']

    # Create preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)

    # Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Balance training data with SMOTE
    smote = SMOTE(random_state=random_seed)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

    return X_train_resampled, X_test_processed, y_train_resampled, y_test, preprocessor