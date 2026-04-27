import os
import pandas as pd
from data_prep import load_and_preprocess_data
from train import train_model, save_model
from eval import evaluate_model
from visualize import plot_confusion_matrix, plot_feature_importance
from robustness_check import evaluate_bootstrap_robustness, evaluate_noise_robustness

def main():
    # Define Paths
    RAW_DATA_PATH = os.path.join('data', 'raw', 'pipe_condition_class_synthetic.csv')
    MODEL_OUT_PATH = os.path.join('output', 'models', 'best_xgb_model.joblib')
    RESULTS_DIR = os.path.join('output', 'results')
    FIG_DIR = os.path.join('output', 'figures')

    # 1. Data Prep & SMOTE
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(RAW_DATA_PATH)

    # 2. Model Training
    model = train_model(X_train, y_train)
    save_model(model, preprocessor, MODEL_OUT_PATH)

    # 3. Evaluation
    evaluate_model(model, X_test, y_test, RESULTS_DIR)

    # 4. Visualization
    predictions = model.predict(X_test)
    plot_confusion_matrix(y_test, predictions, os.path.join(FIG_DIR, 'confusion_matrix.png'))
    plot_feature_importance(model, preprocessor, os.path.join(FIG_DIR, 'feature_importance.png'))

    # 5. Robustness Checks
    # Make sure to add 'import pandas as pd' at the very top of run_pipeline.py!

    # 5. Robustness Checks
    feature_names = preprocessor.get_feature_names_out()
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    features_to_perturb = ['num__Age', 'num__Diameter', 'num__Soil PH']
    evaluate_bootstrap_robustness(model, X_test_df, y_test)
    evaluate_noise_robustness(model, X_test_df, y_test, numerical_features=features_to_perturb) 

if __name__ == '__main__':
    main()