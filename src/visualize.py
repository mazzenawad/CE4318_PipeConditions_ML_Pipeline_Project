import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import xgboost as xgb
import os

def plot_confusion_matrix(y_test, predictions, output_path):
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5])
    plt.title('Confusion Matrix: Pipe Condition Rating')
    plt.xlabel('Predicted Rating')
    plt.ylabel('Actual Rating')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_feature_importance(model, preprocessor, output_path):
    # Retrieve clean feature names from OneHotEncoder
    feature_names = preprocessor.get_feature_names_out()
    
    model.get_booster().feature_names = list(feature_names)
    
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model.get_booster(), importance_type='gain', max_num_features=15)
    plt.title('Top 15 Feature Importances (Gain)')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()