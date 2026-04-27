#Robustness Check using 1. Bootstrap Evaluation and 2. Noise Perturbation Check
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.utils import resample

def evaluate_bootstrap_robustness(model, X_test, y_test, n_iterations=100, random_state=42):
    print(f"Running Bootstrap Evaluation ({n_iterations} iterations)")
    np.random.seed(random_state)
    scores = []
    
    for _ in range(n_iterations):
        X_boot, y_boot = resample(X_test, y_test)
        
        y_pred = model.predict(X_boot)
        score = f1_score(y_boot, y_pred, average='macro')
        scores.append(score)
  
    lower_bound = np.percentile(scores, 2.5)
    upper_bound = np.percentile(scores, 97.5)
    mean_score = np.mean(scores)
    
    print(f"Bootstrap Macro F1-Score: {mean_score:.4f} (95% CI: [{lower_bound:.4f}, {upper_bound:.4f}])")
    plt.figure(figsize=(8, 5))
    sns.histplot(scores, kde=True, bins=20, color='royalblue')
    plt.axvline(mean_score, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_score:.4f}')
    plt.title('Bootstrap Robustness: F1-Score Distribution')
    plt.xlabel('Macro F1-Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    
    return {'mean': mean_score, 'lower_ci': lower_bound, 'upper_ci': upper_bound, 'all_scores': scores}

def evaluate_noise_robustness(model, X_test, y_test, numerical_features, noise_levels=[0.0, 0.05, 0.10, 0.20, 0.30]):
    results = []
    
    for level in noise_levels:
        X_noisy = X_test.copy()
        
        if level > 0:
            for feature in numerical_features:
                std_dev = X_noisy[feature].std()
                
                noise = np.random.normal(loc=0.0, scale=std_dev * level, size=len(X_noisy))

                X_noisy[feature] = X_noisy[feature] + noise
                
        y_pred = model.predict(X_noisy)
        score = f1_score(y_test, y_pred, average='macro')
        
        results.append({'Noise Level (%)': int(level * 100), 'Macro F1-Score': score})
        
    results_df = pd.DataFrame(results)
   
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=results_df, x='Noise Level (%)', y='Macro F1-Score', marker='o', linewidth=2, color='darkorange')
    plt.title('Noise Robustness: Performance Degradation')
    plt.xlabel('Noise Injected (% of Feature Std Dev)')
    plt.ylabel('Macro F1-Score')
    plt.ylim(0, results_df['Macro F1-Score'].max() + 0.1)
    plt.grid(True)
    plt.show()
    
    return results_df

def run_robustness_suite(data_path, model_path):
    df = pd.read_csv(data_path)
   
    X = df.drop('Condition Rating', axis=1)
    y = df['Condition Rating']
    if y.min() > 0:
        y = y - y.min() 
        
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
    model = joblib.load(model_path)
  
    print("-" * 50)
    evaluate_bootstrap_robustness(model, X_test, y_test)
  
    print("-" * 50)
    features_to_perturb = ['Age', 'Diameter', 'Soil PH']
    evaluate_noise_robustness(model, X_test, y_test, numerical_features=features_to_perturb)

if __name__ == "__main__":
    DATA_FILE = 'pipe_condition_class_synthetic.csv'
    MODEL_FILE = 'best_xgb_model.joblib' 
    
    run_robustness_suite(DATA_FILE, MODEL_FILE)