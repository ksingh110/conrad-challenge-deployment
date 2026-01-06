"""
Evaluate Best Model on Evaluation Dataset

Loads the best trained model and evaluates it on the ml_eval_df.csv dataset.

Usage:
    python evaluate_best_model.py [model_folder]
    
    If model_folder is not specified, defaults to 'lfc_5'
"""

import sys
import pandas as pd
import numpy as np
import pickle
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Best Model Evaluation on ml_eval_df.csv")
print("=" * 80)

# ============================================================================
# 1. Load Best Model and Metadata
# ============================================================================
print("\n[1/5] Loading best model and metadata...")

# Get model folder from command line argument or use default
if len(sys.argv) > 1:
    model_folder = sys.argv[1]
    print(f"Using specified model folder: {model_folder}")
else:
    model_folder = 'lfc_5'
    print(f"Using default model folder: {model_folder}")

best_model_path = os.path.join(model_folder, 'best_immunotherapy_model.pkl')
feature_names_path = os.path.join(model_folder, 'feature_names.pkl')
metadata_path = os.path.join(model_folder, 'best_model_metadata.json')

# Check if model exists
if not os.path.exists(best_model_path):
    print(f"\nERROR: Best model not found at: {best_model_path}")
    print("\nPlease do one of the following:")
    print("  1. Run immunotherapy_response_model_training.py first to train the model")
    print("  2. Specify a different model folder: python evaluate_best_model.py <model_folder>")
    print("  3. If you have a model file elsewhere, create a folder and place:")
    print("     - best_immunotherapy_model.pkl (the trained model)")
    print("     - feature_names.pkl (list of feature names used during training)")
    print("     - best_model_metadata.json (optional, model metadata)")
    exit(1)

# Load model
with open(best_model_path, 'rb') as f:
    best_model = pickle.load(f)
print(f"Best model loaded from: {best_model_path}")

# Load feature names
if os.path.exists(feature_names_path):
    with open(feature_names_path, 'rb') as f:
        training_features = pickle.load(f)
    print(f"Training feature names loaded: {len(training_features)} features")
else:
    print("WARNING: Feature names file not found. Will infer from model.")
    training_features = None

# Load metadata
if os.path.exists(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"Model type: {metadata['model_type']}")
    print(f"Training ROC-AUC: {metadata['roc_auc']:.4f}")
else:
    print("WARNING: Metadata file not found.")
    metadata = None

# ============================================================================
# 2. Load Evaluation Dataset
# ============================================================================
print("\n[2/5] Loading evaluation dataset...")

eval_data_path = r'G:\My Drive\Holland Lab\Head and Neck Cancer\Data\ml_eval_df.csv'
eval_df = pd.read_csv(eval_data_path)

print(f"Evaluation dataset shape: {eval_df.shape}")
print(f"Columns: {list(eval_df.columns[:5])}... (and {len(eval_df.columns) - 5} more)")

# Check if response column exists
if 'response01' not in eval_df.columns:
    print("WARNING: 'response01' column not found. Evaluation metrics will not be calculated.")
    has_ground_truth = False
    y_true = None
else:
    has_ground_truth = True
    y_true = eval_df['response01'].copy()
    print(f"\nGround truth distribution:")
    print(y_true.value_counts().sort_index())
    print(f"Response rate: {y_true.mean():.3f}")

# ============================================================================
# 3. Prepare Features
# ============================================================================
print("\n[3/5] Preparing features...")

# Identify non-feature columns (metadata columns)
non_feature_cols = ['Run', 'Sample.Name', 'response01']
# Also check for other common metadata columns
for col in eval_df.columns:
    if col.lower() in ['rowname', 'patient_id', 'sample_id', 'sample_name']:
        if col not in non_feature_cols:
            non_feature_cols.append(col)

# Get feature columns from evaluation data
eval_feature_cols = [col for col in eval_df.columns if col not in non_feature_cols]
print(f"Evaluation dataset has {len(eval_feature_cols)} feature columns")

# If we have training features, align them
if training_features is not None:
    print(f"Training model expects {len(training_features)} features")
    
    # Check which features are missing in evaluation data
    missing_features = set(training_features) - set(eval_feature_cols)
    if missing_features:
        print(f"WARNING: {len(missing_features)} features from training are missing in evaluation data:")
        print(f"  Missing features: {list(missing_features)[:10]}..." if len(missing_features) > 10 else f"  Missing features: {list(missing_features)}")
    
    # Check which features are extra in evaluation data
    extra_features = set(eval_feature_cols) - set(training_features)
    if extra_features:
        print(f"INFO: {len(extra_features)} features in evaluation data are not in training data (will be ignored)")
    
    # Use only features that are in both training and evaluation
    common_features = [f for f in training_features if f in eval_feature_cols]
    print(f"Using {len(common_features)} common features for prediction")
    
    if len(common_features) < len(training_features):
        print(f"WARNING: Only {len(common_features)}/{len(training_features)} training features available!")
        print("Model performance may be degraded.")
    
    # Extract features in the same order as training
    X_eval = eval_df[common_features].copy()
    feature_columns = common_features
else:
    # If we don't have training features, use all features from eval data
    print("Using all features from evaluation dataset")
    X_eval = eval_df[eval_feature_cols].copy()
    feature_columns = eval_feature_cols

print(f"Final feature matrix shape: {X_eval.shape}")

# Check for missing values
if X_eval.isnull().any().any():
    missing_count = X_eval.isnull().sum().sum()
    print(f"WARNING: Found {missing_count} missing values. Filling with 0.")
    X_eval = X_eval.fillna(0)

# ============================================================================
# 4. Make Predictions
# ============================================================================
print("\n[4/5] Making predictions...")

# Make predictions
y_pred = best_model.predict(X_eval)
y_pred_proba = best_model.predict_proba(X_eval)[:, 1]

print(f"Predictions made for {len(y_pred)} samples")
print(f"Predicted distribution:")
pred_counts = pd.Series(y_pred).value_counts().sort_index()
print(f"  Class 0 (No Response): {pred_counts.get(0, 0)}")
print(f"  Class 1 (Response): {pred_counts.get(1, 0)}")
print(f"Predicted response rate: {y_pred.mean():.3f}")

# ============================================================================
# 5. Evaluate and Save Results
# ============================================================================
print("\n[5/5] Evaluating and saving results...")

# Create results dataframe
results_df = eval_df[non_feature_cols].copy()
results_df['predicted_response'] = y_pred
results_df['predicted_probability'] = y_pred_proba

# Save predictions
output_folder = 'model_evaluation'
os.makedirs(output_folder, exist_ok=True)
results_path = os.path.join(output_folder, 'evaluation_predictions.csv')
results_df.to_csv(results_path, index=False)
print(f"Predictions saved to: {results_path}")

# Calculate metrics if ground truth is available
if has_ground_truth:
    print("\n" + "="*80)
    print("Evaluation Metrics")
    print("="*80)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Negative  Positive")
    print(f"Actual Negative    {cm[0,0]:2d}       {cm[0,1]:2d}")
    print(f"        Positive    {cm[1,0]:2d}       {cm[1,1]:2d}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Response', 'Response']))
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'n_samples': int(len(y_true)),
        'n_features_used': int(len(feature_columns)),
        'n_features_expected': int(len(training_features)) if training_features else None,
        'confusion_matrix': {
            'true_negative': int(cm[0, 0]),
            'false_positive': int(cm[0, 1]),
            'false_negative': int(cm[1, 0]),
            'true_positive': int(cm[1, 1])
        }
    }
    
    if metadata:
        metrics['training_roc_auc'] = metadata['roc_auc']
        metrics['model_type'] = metadata['model_type']
    
    metrics_path = os.path.join(output_folder, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC Curve
    ax1 = axes[0]
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    ax1.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', linewidth=2, color='steelblue')
    ax1.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curve - Evaluation Dataset', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Confusion Matrix Heatmap
    ax2 = axes[1]
    im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax2.figure.colorbar(im, ax=ax2)
    ax2.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['No Response', 'Response'],
           yticklabels=['No Response', 'Response'],
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    viz_path = os.path.join(output_folder, 'evaluation_results.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {viz_path}")
    
    # Save detailed metrics to text file
    metrics_text_path = os.path.join(output_folder, 'evaluation_metrics.txt')
    with open(metrics_text_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Model Evaluation Metrics on ml_eval_df.csv\n")
        f.write("="*80 + "\n\n")
        
        if metadata:
            f.write(f"Model Type: {metadata['model_type']}\n")
            f.write(f"Training ROC-AUC: {metadata['roc_auc']:.4f}\n\n")
        
        f.write(f"Evaluation Dataset:\n")
        f.write(f"  Number of samples: {len(y_true)}\n")
        f.write(f"  Number of features used: {len(feature_columns)}\n")
        if training_features:
            f.write(f"  Number of features expected: {len(training_features)}\n")
        f.write(f"\n")
        
        f.write("Performance Metrics:\n")
        f.write(f"  Accuracy:  {accuracy:.4f}\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall:    {recall:.4f}\n")
        f.write(f"  F1-Score:  {f1:.4f}\n")
        f.write(f"  ROC-AUC:   {roc_auc:.4f}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(f"                Predicted\n")
        f.write(f"              Negative  Positive\n")
        f.write(f"Actual Negative    {cm[0,0]:2d}       {cm[0,1]:2d}\n")
        f.write(f"        Positive    {cm[1,0]:2d}       {cm[1,1]:2d}\n\n")
        
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=['No Response', 'Response']))
    
    print(f"Detailed metrics saved to: {metrics_text_path}")
else:
    print("\nNo ground truth available. Only predictions have been saved.")

print("\n" + "="*80)
print("Evaluation completed successfully!")
print("="*80)
print(f"\nResults saved in: {output_folder}/")
print(f"  - evaluation_predictions.csv: Predictions for all samples")
if has_ground_truth:
    print(f"  - evaluation_metrics.json: Performance metrics")
    print(f"  - evaluation_metrics.txt: Detailed performance report")
    print(f"  - evaluation_results.png: ROC curve and confusion matrix")

