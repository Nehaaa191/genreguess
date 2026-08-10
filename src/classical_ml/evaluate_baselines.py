import os
import sys
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.common.split_utils import load_splits

def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    csv_path = "data/raw/features_30_sec.csv"
    split_path = "data/splits/split_seed42.json"
    
    if not os.path.exists(csv_path):
        return
        
    df = pd.read_csv(csv_path)
    splits = load_splits(split_path)
    test_filenames = splits['test']
    test_basenames = [os.path.basename(f) for f in test_filenames]
    
    test_df = df[df['filename'].isin(test_basenames)].copy()
    X_test = test_df.drop(columns=['filename', 'length', 'label'])
    y_test = test_df['label']
    
    scaler = joblib.load('artifacts/classical_scaler.pkl')
    X_test_scaled = scaler.transform(X_test)
    
    lr_model = joblib.load('artifacts/logistic_regression.pkl')
    rf_model = joblib.load('artifacts/random_forest.pkl')
    
    labels = sorted(y_test.unique())
    os.makedirs('reports/classical_ml', exist_ok=True)
    
    with open('reports/classical_ml/test_metrics.txt', 'w') as f:
        f.write("--- Logistic Regression ---\n")
        lr_preds = lr_model.predict(X_test_scaled)
        f.write(f"Accuracy: {accuracy_score(y_test, lr_preds):.4f}\n\n")
        f.write(classification_report(y_test, lr_preds))
        
        plot_confusion_matrix(y_test, lr_preds, labels, 
                              "Logistic Regression Confusion Matrix", 
                              "reports/classical_ml/lr_confusion_matrix.png")
                              
        f.write("\n\n--- Random Forest ---\n")
        rf_preds = rf_model.predict(X_test_scaled)
        f.write(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}\n\n")
        f.write(classification_report(y_test, rf_preds))
        
        plot_confusion_matrix(y_test, rf_preds, labels, 
                              "Random Forest Confusion Matrix", 
                              "reports/classical_ml/rf_confusion_matrix.png")

if __name__ == "__main__":
    main()
