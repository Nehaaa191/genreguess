import os
import sys
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.common.split_utils import load_splits

def load_and_preprocess_data(csv_path: str, split_path: str):
    df = pd.read_csv(csv_path)
    
    # We only want train split to fit scaler and models (actually CV will further split it)
    splits = load_splits(split_path)
    train_filenames = splits['train']
    
    # GTZAN CSV filenames in features_30_sec.csv look like: "blues.00000.wav"
    # But in our split JSON they look like "blues\\blues.00000.wav" or "blues/blues.00000.wav".
    # Let's extract basenames for mapping.
    
    train_basenames = [os.path.basename(f) for f in train_filenames]
    
    # Filter for training set
    train_df = df[df['filename'].isin(train_basenames)].copy()
    
    if len(train_df) == 0:
        raise ValueError("No matching files found in train split. Check filename formats.")
    
    # Drop filename, length, and label to get features
    X_train = train_df.drop(columns=['filename', 'length', 'label'])
    y_train = train_df['label']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Save scaler for later evaluation
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(scaler, 'artifacts/classical_scaler.pkl')
    
    return X_train_scaled, y_train, scaler

def train_logistic_regression(X, y):
    print("Training Logistic Regression...")
    param_grid = {'C': [0.1, 1.0, 10.0]}
    lr = LogisticRegression(multi_class='multinomial', max_iter=2000, random_state=42)
    grid = GridSearchCV(lr, param_grid, cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)
    
    print(f"Best LR Params: {grid.best_params_}")
    print(f"Best LR CV Accuracy: {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_score_

def train_random_forest(X, y):
    print("Training Random Forest...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [1, 2]
    }
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)
    
    print(f"Best RF Params: {grid.best_params_}")
    print(f"Best RF CV Accuracy: {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_score_

def main():
    csv_path = "data/raw/features_30_sec.csv"
    split_path = "data/splits/split_seed42.json"
    
    if not os.path.exists(csv_path):
        print(f"CSV not found at {csv_path}")
        return
        
    X_train, y_train, scaler = load_and_preprocess_data(csv_path, split_path)
    
    lr_model, lr_score = train_logistic_regression(X_train, y_train)
    rf_model, rf_score = train_random_forest(X_train, y_train)
    
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(lr_model, 'artifacts/logistic_regression.pkl')
    joblib.dump(rf_model, 'artifacts/random_forest.pkl')
    
    # Save CV scores to report
    os.makedirs('reports/classical_ml', exist_ok=True)
    with open('reports/classical_ml/cv_scores.txt', 'w') as f:
        f.write(f"Logistic Regression CV Accuracy: {lr_score:.4f}\n")
        f.write(f"Random Forest CV Accuracy: {rf_score:.4f}\n")

if __name__ == "__main__":
    main()
