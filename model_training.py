"""
Immunotherapy Response Model Training Script

Trains XGBoost, Penalized Logistic Regression, and Random Forest binary classifiers 
to predict immunotherapy response based on gene expression features.
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
# np.random.seed(42)

print("=" * 80)
print("Immunotherapy Response Model Training")
print("=" * 80)

# Define helper function for model evaluation
def evaluate_model_cv(model, X, y, model_name, cv):
    """Evaluate model using cross-validation and return metrics"""
    # Cross-validation scores
    cv_scores_accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    cv_scores_precision = cross_val_score(model, X, y, cv=cv, scoring='precision')
    cv_scores_recall = cross_val_score(model, X, y, cv=cv, scoring='recall')
    cv_scores_f1 = cross_val_score(model, X, y, cv=cv, scoring='f1')
    cv_scores_roc_auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    
    # Get CV predictions for ROC curve and confusion matrix
    y_cv_pred = cross_val_predict(model, X, y, cv=cv, method='predict')
    y_cv_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
    
    # Metrics from CV predictions
    cv_accuracy = accuracy_score(y, y_cv_pred)
    cv_precision = precision_score(y, y_cv_pred)
    cv_recall = recall_score(y, y_cv_pred)
    cv_f1 = f1_score(y, y_cv_pred)
    cv_roc_auc = roc_auc_score(y, y_cv_proba)
    
    # Confusion matrix from CV predictions
    cm = confusion_matrix(y, y_cv_pred)
    
    print(f"\n{'='*60}")
    print(f"{model_name} - Cross-Validation Scores (Mean ± Std across folds)")
    print(f"{'='*60}")
    print(f"Accuracy:  {cv_scores_accuracy.mean():.4f} ± {cv_scores_accuracy.std():.4f}")
    print(f"Precision: {cv_scores_precision.mean():.4f} ± {cv_scores_precision.std():.4f}")
    print(f"Recall:    {cv_scores_recall.mean():.4f} ± {cv_scores_recall.std():.4f}")
    print(f"F1-Score:  {cv_scores_f1.mean():.4f} ± {cv_scores_f1.std():.4f}")
    print(f"ROC-AUC:   {cv_scores_roc_auc.mean():.4f} ± {cv_scores_roc_auc.std():.4f}")
    
    print(f"\n{model_name} - Overall CV Performance (from aggregated predictions)")
    print(f"{'='*60}")
    print(f"Accuracy:  {cv_accuracy:.4f}")
    print(f"Precision: {cv_precision:.4f}")
    print(f"Recall:    {cv_recall:.4f}")
    print(f"F1-Score:  {cv_f1:.4f}")
    print(f"ROC-AUC:   {cv_roc_auc:.4f}")
    
    print(f"\n{model_name} - Confusion Matrix (from CV predictions)")
    print(f"{'='*60}")
    print(f"                Predicted")
    print(f"              Negative  Positive")
    print(f"Actual Negative    {cm[0,0]:2d}       {cm[0,1]:2d}")
    print(f"        Positive    {cm[1,0]:2d}       {cm[1,1]:2d}")
    
    print(f"\n{model_name} - Classification Report (from CV predictions)")
    print(f"{'='*60}")
    print(classification_report(y, y_cv_pred, target_names=['No Response', 'Response']))
    
    return {
        'cv_scores': {
            'accuracy': cv_scores_accuracy,
            'precision': cv_scores_precision,
            'recall': cv_scores_recall,
            'f1': cv_scores_f1,
            'roc_auc': cv_scores_roc_auc
        },
        'cv_metrics': {
            'accuracy': cv_accuracy,
            'precision': cv_precision,
            'recall': cv_recall,
            'f1': cv_f1,
            'roc_auc': cv_roc_auc
        },
        'y_cv_proba': y_cv_proba,
        'y_cv_pred': y_cv_pred,
        'confusion_matrix': cm
    }

# Define datasets to process
datasets = ['lfc_5']

# Loop over each dataset
for dataset in datasets:
    print("\n" + "=" * 80)
    print(f"Processing Dataset: ml_df_filtered_{dataset}")
    print("=" * 80)
    
    # Create output folder for this dataset
    output_folder = dataset
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder}/")
    
    # ============================================================================
    # 1. Data Loading and Preprocessing
    # ============================================================================
    print("\n[1/9] Loading and preprocessing data...")
    
    # Load the dataset
    data_path = f'G:\\My Drive\\Holland Lab\\Head and Neck Cancer\\Data\\ml_df_filtered_{dataset}.csv'
    df = pd.read_csv(data_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Response distribution:\n{df['response01'].value_counts()}")

    # Extract features (all columns except rowname, patient_id, and response01)
    feature_columns = [col for col in df.columns if col not in ['rowname', 'patient_id', 'response01']]
    X = df[feature_columns].copy()
    y = df['response01'].copy()

    print(f"\nNumber of features: {len(feature_columns)}")
    print(f"Number of samples: {len(X)}")
    print(f"Class distribution: {y.value_counts().to_dict()}")

    # Calculate class weights for imbalance handling
    class_counts = y.value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1]  # negative / positive
    print(f"Class imbalance ratio (negative/positive): {scale_pos_weight:.3f}")

    # ============================================================================
    # 2. Cross-Validation Setup
    # ============================================================================
    print("\n[2/9] Setting up cross-validation...")
    print(f"Using all {len(X)} samples for cross-validation evaluation")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    print("\nNote: With only 30 samples, cross-validation provides more reliable")
    print("      performance estimates than a separate test set.")

    # ============================================================================
    # 3. XGBoost Model with Enhanced Hyperparameter Optimization
    # ============================================================================
    print("\n[3/9] Training XGBoost model with enhanced hyperparameter optimization...")

    # Define expanded parameter distribution for XGBoost
    # Using distributions for RandomizedSearchCV to efficiently explore larger space
    xgb_param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3, 0.5],
        'reg_alpha': [0, 0.1, 0.5, 1.0, 2.0],  # L1 regularization
        'reg_lambda': [1, 1.5, 2.0, 3.0],  # L2 regularization
        'scale_pos_weight': [scale_pos_weight, scale_pos_weight * 1.2, scale_pos_weight * 1.5, scale_pos_weight * 2.0]
    }

    # Create XGBoost classifier
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False
    )

    # Stratified K-Fold for cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Randomized search with cross-validation for efficient exploration of large parameter space
    # Using 100 iterations to balance thoroughness with computation time
    print("Performing randomized search with 100 iterations (this may take several minutes)...")
    print("Exploring expanded hyperparameter space for optimal performance...")
    xgb_random_search = RandomizedSearchCV(
        xgb_clf,
        xgb_param_dist,
        n_iter=100,  # Number of parameter settings sampled
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    xgb_random_search.fit(X, y)

    print(f"\nBest XGBoost parameters: {xgb_random_search.best_params_}")
    print(f"Best XGBoost CV score (ROC-AUC): {xgb_random_search.best_score_:.4f}")

    # Train final model with best parameters on all data
    best_xgb = xgb_random_search.best_estimator_
    best_xgb.fit(X, y)

    # ============================================================================
    # 4. Penalized Logistic Regression Model with Hyperparameter Optimization
    # ============================================================================
    print("\n[4/9] Training Penalized Logistic Regression (Elastic Net) model with hyperparameter optimization...")

    # Define expanded parameter distribution for Elastic Net Logistic Regression
    # Expanded to improve recall and ROC-AUC performance
    lr_param_dist = {
        'C': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0],
        'l1_ratio': [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],  # Includes pure L2 (0.0) and pure L1 (1.0)
        'penalty': ['elasticnet'],
        'solver': ['saga'],
        'max_iter': [500, 1000, 1500, 2000, 3000, 4000, 5000, 7500, 10000],
        'tol': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],  # Tolerance for stopping criteria
        'class_weight': [
            'balanced',
            {0: 1.0, 1: scale_pos_weight},
            {0: 1.0, 1: scale_pos_weight * 1.2},
            {0: 1.0, 1: scale_pos_weight * 1.5},
            {0: 1.0, 1: scale_pos_weight * 2.0},
            {0: 1.0, 1: scale_pos_weight * 2.5},
            {0: 1.0, 1: scale_pos_weight * 3.0},
            {0: 0.8, 1: scale_pos_weight},
            {0: 0.9, 1: scale_pos_weight * 1.2}
        ]
    }

    # Create Logistic Regression classifier
    lr_clf = LogisticRegression(
        # random_state=42,
        n_jobs=-1
    )

    # Randomized search with cross-validation
    # Increased iterations to explore larger hyperparameter space
    print("Performing randomized search with 150 iterations (this may take several minutes)...")
    print("Exploring expanded hyperparameter space for optimal Elastic Net regularization...")
    print("Focusing on improving recall and ROC-AUC performance...")
    lr_random_search = RandomizedSearchCV(
        lr_clf,
        lr_param_dist,
        n_iter=150,  # Increased from 50 to explore larger parameter space
        cv=cv,
        scoring='roc_auc',  # Primary metric
        n_jobs=-1,
        verbose=1,
        # random_state=42
    )

    lr_random_search.fit(X, y)

    print(f"\nBest Penalized Logistic Regression parameters: {lr_random_search.best_params_}")
    print(f"Best Penalized Logistic Regression CV score (ROC-AUC): {lr_random_search.best_score_:.4f}")

    # Train final model with best parameters on all data
    best_lr = lr_random_search.best_estimator_
    best_lr.fit(X, y)

    # ============================================================================
    # 5. Random Forest Model with Grid Search
    # ============================================================================
    print("\n[5/9] Training Random Forest model with GridSearchCV...")

    # Define parameter grid for Random Forest
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced']
    }

    # Create Random Forest classifier
    rf_clf = RandomForestClassifier(random_state=42)

    # Grid search with cross-validation
    print("Performing grid search (this may take a few minutes)...")
    rf_grid_search = GridSearchCV(
        rf_clf,
        rf_param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    rf_grid_search.fit(X, y)

    print(f"\nBest Random Forest parameters: {rf_grid_search.best_params_}")
    print(f"Best Random Forest CV score (ROC-AUC): {rf_grid_search.best_score_:.4f}")

    # Train final model with best parameters on all data
    best_rf = rf_grid_search.best_estimator_
    best_rf.fit(X, y)

    # ============================================================================
    # 6. Model Evaluation
    # ============================================================================
    print("\n[6/9] Evaluating models using cross-validation...")

    # Evaluate XGBoost
    xgb_results = evaluate_model_cv(best_xgb, X, y, "XGBoost", cv)

    # Evaluate Penalized Logistic Regression
    lr_results = evaluate_model_cv(best_lr, X, y, "Penalized Logistic Regression", cv)

    # Evaluate Random Forest
    rf_results = evaluate_model_cv(best_rf, X, y, "Random Forest", cv)

    # ============================================================================
    # 7. Feature Importance
    # ============================================================================
    print("\n[7/9] Extracting feature importance...")

    # XGBoost feature importance
    xgb_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_xgb.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nXGBoost - Top 20 Most Important Features:")
    print("="*60)
    for i, row in xgb_importance.head(20).iterrows():
        print(f"{row['feature']:20s}  {row['importance']:.6f}")

    # Penalized Logistic Regression feature importance (using absolute coefficients)
    lr_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': np.abs(best_lr.coef_[0])
    }).sort_values('importance', ascending=False)

    print("\nPenalized Logistic Regression - Top 20 Most Important Features:")
    print("="*60)
    for i, row in lr_importance.head(20).iterrows():
        print(f"{row['feature']:20s}  {row['importance']:.6f}")

    # Random Forest feature importance
    rf_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nRandom Forest - Top 20 Most Important Features:")
    print("="*60)
    for i, row in rf_importance.head(20).iterrows():
        print(f"{row['feature']:20s}  {row['importance']:.6f}")

    # ============================================================================
    # 8. Visualizations
    # ============================================================================
    print("\n[8/9] Creating visualizations...")

    # Create figure with subplots (3 rows, 3 columns)
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    # ROC Curve (from CV predictions) - Top left (spanning 3 columns)
    ax1 = axes[0, 0]
    fpr_xgb, tpr_xgb, _ = roc_curve(y, xgb_results['y_cv_proba'])
    fpr_lr, tpr_lr, _ = roc_curve(y, lr_results['y_cv_proba'])
    fpr_rf, tpr_rf, _ = roc_curve(y, rf_results['y_cv_proba'])

    ax1.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {xgb_results['cv_metrics']['roc_auc']:.3f})", linewidth=2, color='steelblue')
    ax1.plot(fpr_lr, tpr_lr, label=f"Penalized LR (AUC = {lr_results['cv_metrics']['roc_auc']:.3f})", linewidth=2, color='coral')
    ax1.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {rf_results['cv_metrics']['roc_auc']:.3f})", linewidth=2, color='forestgreen')
    ax1.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curves - Cross-Validation', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(alpha=0.3)

    # XGBoost Feature Importance - Top middle
    ax2 = axes[0, 1]
    top_xgb = xgb_importance.head(20)
    ax2.barh(range(len(top_xgb)), top_xgb['importance'].values, color='steelblue')
    ax2.set_yticks(range(len(top_xgb)))
    ax2.set_yticklabels(top_xgb['feature'].values, fontsize=9)
    ax2.set_xlabel('Importance', fontsize=12)
    ax2.set_title('XGBoost - Top 20 Feature Importance', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(alpha=0.3, axis='x')

    # Penalized Logistic Regression Feature Importance - Top right
    ax3 = axes[0, 2]
    top_lr = lr_importance.head(20)
    ax3.barh(range(len(top_lr)), top_lr['importance'].values, color='coral')
    ax3.set_yticks(range(len(top_lr)))
    ax3.set_yticklabels(top_lr['feature'].values, fontsize=9)
    ax3.set_xlabel('Importance (|Coefficient|)', fontsize=12)
    ax3.set_title('Penalized LR - Top 20 Feature Importance', fontsize=14, fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(alpha=0.3, axis='x')

    # Random Forest Feature Importance - Middle left
    ax4 = axes[1, 0]
    top_rf = rf_importance.head(20)
    ax4.barh(range(len(top_rf)), top_rf['importance'].values, color='forestgreen')
    ax4.set_yticks(range(len(top_rf)))
    ax4.set_yticklabels(top_rf['feature'].values, fontsize=9)
    ax4.set_xlabel('Importance', fontsize=12)
    ax4.set_title('Random Forest - Top 20 Feature Importance', fontsize=14, fontweight='bold')
    ax4.invert_yaxis()
    ax4.grid(alpha=0.3, axis='x')

    # XGBoost Performance Metrics - Middle center
    ax5 = axes[1, 1]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    xgb_vals = [
        xgb_results['cv_metrics']['accuracy'],
        xgb_results['cv_metrics']['precision'],
        xgb_results['cv_metrics']['recall'],
        xgb_results['cv_metrics']['f1'],
        xgb_results['cv_metrics']['roc_auc']
    ]

    x = np.arange(len(metrics))
    ax5.bar(x, xgb_vals, alpha=0.8, color='steelblue')
    ax5.set_xlabel('Metrics', fontsize=12)
    ax5.set_ylabel('Score', fontsize=12)
    ax5.set_title('XGBoost - Cross-Validation Metrics', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics, rotation=45, ha='right')
    ax5.set_ylim([0, 1.1])
    ax5.grid(alpha=0.3, axis='y')

    # Penalized Logistic Regression Performance Metrics - Middle right
    ax6 = axes[1, 2]
    lr_vals = [
        lr_results['cv_metrics']['accuracy'],
        lr_results['cv_metrics']['precision'],
        lr_results['cv_metrics']['recall'],
        lr_results['cv_metrics']['f1'],
        lr_results['cv_metrics']['roc_auc']
    ]

    ax6.bar(x, lr_vals, alpha=0.8, color='coral')
    ax6.set_xlabel('Metrics', fontsize=12)
    ax6.set_ylabel('Score', fontsize=12)
    ax6.set_title('Penalized LR - Cross-Validation Metrics', fontsize=14, fontweight='bold')
    ax6.set_xticks(x)
    ax6.set_xticklabels(metrics, rotation=45, ha='right')
    ax6.set_ylim([0, 1.1])
    ax6.grid(alpha=0.3, axis='y')

    # Random Forest Performance Metrics - Bottom left
    ax7 = axes[2, 0]
    rf_vals = [
        rf_results['cv_metrics']['accuracy'],
        rf_results['cv_metrics']['precision'],
        rf_results['cv_metrics']['recall'],
        rf_results['cv_metrics']['f1'],
        rf_results['cv_metrics']['roc_auc']
    ]

    ax7.bar(x, rf_vals, alpha=0.8, color='forestgreen')
    ax7.set_xlabel('Metrics', fontsize=12)
    ax7.set_ylabel('Score', fontsize=12)
    ax7.set_title('Random Forest - Cross-Validation Metrics', fontsize=14, fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels(metrics, rotation=45, ha='right')
    ax7.set_ylim([0, 1.1])
    ax7.grid(alpha=0.3, axis='y')

    # Model Comparison - Bottom center and right (spanning 2 columns)
    ax8 = axes[2, 1]
    ax9 = axes[2, 2]
    ax8.remove()  # Remove the middle subplot
    ax9.remove()  # Remove the right subplot
    
    # Create comparison plot spanning 2 columns using subplot2grid
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    ax_compare = fig.add_subplot(gs[2, 1:3])  # Span columns 1 and 2 in row 2
    x_pos = np.arange(len(metrics))
    width = 0.25
    ax_compare.bar(x_pos - width, xgb_vals, width, label='XGBoost', alpha=0.8, color='steelblue')
    ax_compare.bar(x_pos, lr_vals, width, label='Penalized LR', alpha=0.8, color='coral')
    ax_compare.bar(x_pos + width, rf_vals, width, label='Random Forest', alpha=0.8, color='forestgreen')
    ax_compare.set_xlabel('Metrics', fontsize=12)
    ax_compare.set_ylabel('Score', fontsize=12)
    ax_compare.set_title('Model Comparison - Cross-Validation Metrics', fontsize=14, fontweight='bold')
    ax_compare.set_xticks(x_pos)
    ax_compare.set_xticklabels(metrics, rotation=45, ha='right')
    ax_compare.set_ylim([0, 1.1])
    ax_compare.legend(loc='upper right', fontsize=10)
    ax_compare.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    viz_path = os.path.join(output_folder, 'model_evaluation_results.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    print(f"Visualization saved as '{viz_path}'")

    # ============================================================================
    # 9. Save Models
    # ============================================================================
    print("\n[9/9] Saving models...")

    # Save XGBoost model
    xgb_model_path = os.path.join(output_folder, 'xgboost_immunotherapy_model.pkl')
    with open(xgb_model_path, 'wb') as f:
        pickle.dump(best_xgb, f)
    print(f"XGBoost model saved to: {xgb_model_path}")

    # Save Penalized Logistic Regression model
    lr_model_path = os.path.join(output_folder, 'penalized_lr_immunotherapy_model.pkl')
    with open(lr_model_path, 'wb') as f:
        pickle.dump(best_lr, f)
    print(f"Penalized Logistic Regression model saved to: {lr_model_path}")

    # Save Random Forest model
    rf_model_path = os.path.join(output_folder, 'random_forest_immunotherapy_model.pkl')
    with open(rf_model_path, 'wb') as f:
        pickle.dump(best_rf, f)
    print(f"Random Forest model saved to: {rf_model_path}")

    # Save feature names
    feature_names_path = os.path.join(output_folder, 'feature_names.pkl')
    with open(feature_names_path, 'wb') as f:
        pickle.dump(feature_columns, f)
    print(f"Feature names saved to: {feature_names_path}")

    # Save feature importance dataframes
    xgb_importance_path = os.path.join(output_folder, 'xgboost_feature_importance.csv')
    xgb_importance.to_csv(xgb_importance_path, index=False)
    print(f"XGBoost feature importance saved to CSV file")

    lr_importance_path = os.path.join(output_folder, 'penalized_lr_feature_importance.csv')
    lr_importance.to_csv(lr_importance_path, index=False)
    print(f"Penalized Logistic Regression feature importance saved to CSV file")

    rf_importance_path = os.path.join(output_folder, 'random_forest_feature_importance.csv')
    rf_importance.to_csv(rf_importance_path, index=False)
    print(f"Random Forest feature importance saved to CSV file")

    # Determine and save the best model based on ROC-AUC
    model_performance = {
        'XGBoost': xgb_results['cv_metrics']['roc_auc'],
        'Penalized LR': lr_results['cv_metrics']['roc_auc'],
        'Random Forest': rf_results['cv_metrics']['roc_auc']
    }
    best_model_name = max(model_performance, key=model_performance.get)
    best_model_roc_auc = model_performance[best_model_name]
    
    # Select the best model object
    best_model_map = {
        'XGBoost': best_xgb,
        'Penalized LR': best_lr,
        'Random Forest': best_rf
    }
    best_model = best_model_map[best_model_name]
    
    # Save the best model
    best_model_path = os.path.join(output_folder, 'best_immunotherapy_model.pkl')
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\nBest model ({best_model_name}) saved to: {best_model_path}")
    print(f"Best model ROC-AUC: {best_model_roc_auc:.4f}")
    
    # Save metadata about the best model
    best_model_metadata = {
        'model_type': best_model_name,
        'roc_auc': float(best_model_roc_auc),
        'all_model_performances': {k: float(v) for k, v in model_performance.items()}
    }
    metadata_path = os.path.join(output_folder, 'best_model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(best_model_metadata, f, indent=2)
    print(f"Best model metadata saved to: {metadata_path}")

    # Print comparative summary
    print("\n" + "="*100)
    print("Model Comparison Summary")
    print("="*100)
    print(f"{'Metric':<20} {'XGBoost':<15} {'Penalized LR':<15} {'Random Forest':<15} {'Best':<15}")
    print("-"*100)
    
    # Determine best model for each metric
    metrics_dict = {
        'ROC-AUC': {
            'XGBoost': xgb_results['cv_metrics']['roc_auc'],
            'Penalized LR': lr_results['cv_metrics']['roc_auc'],
            'Random Forest': rf_results['cv_metrics']['roc_auc']
        },
        'Accuracy': {
            'XGBoost': xgb_results['cv_metrics']['accuracy'],
            'Penalized LR': lr_results['cv_metrics']['accuracy'],
            'Random Forest': rf_results['cv_metrics']['accuracy']
        },
        'Precision': {
            'XGBoost': xgb_results['cv_metrics']['precision'],
            'Penalized LR': lr_results['cv_metrics']['precision'],
            'Random Forest': rf_results['cv_metrics']['precision']
        },
        'Recall': {
            'XGBoost': xgb_results['cv_metrics']['recall'],
            'Penalized LR': lr_results['cv_metrics']['recall'],
            'Random Forest': rf_results['cv_metrics']['recall']
        },
        'F1-Score': {
            'XGBoost': xgb_results['cv_metrics']['f1'],
            'Penalized LR': lr_results['cv_metrics']['f1'],
            'Random Forest': rf_results['cv_metrics']['f1']
        }
    }
    
    for metric_name, values in metrics_dict.items():
        best_model = max(values, key=values.get)
        print(f"{metric_name:<20} {values['XGBoost']:<15.4f} {values['Penalized LR']:<15.4f} {values['Random Forest']:<15.4f} {best_model:<15}")
    print("="*100)

    # Save metrics to text file
    metrics_file_path = os.path.join(output_folder, 'model_metrics.txt')
    with open(metrics_file_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("Model Performance Metrics\n")
        f.write("="*100 + "\n\n")
        
        # XGBoost metrics
        f.write("XGBoost Model Metrics\n")
        f.write("-"*100 + "\n")
        f.write("Cross-Validation Scores (Mean ± Std across folds):\n")
        f.write(f"  Accuracy:  {xgb_results['cv_scores']['accuracy'].mean():.4f} ± {xgb_results['cv_scores']['accuracy'].std():.4f}\n")
        f.write(f"  Precision: {xgb_results['cv_scores']['precision'].mean():.4f} ± {xgb_results['cv_scores']['precision'].std():.4f}\n")
        f.write(f"  Recall:    {xgb_results['cv_scores']['recall'].mean():.4f} ± {xgb_results['cv_scores']['recall'].std():.4f}\n")
        f.write(f"  F1-Score:  {xgb_results['cv_scores']['f1'].mean():.4f} ± {xgb_results['cv_scores']['f1'].std():.4f}\n")
        f.write(f"  ROC-AUC:   {xgb_results['cv_scores']['roc_auc'].mean():.4f} ± {xgb_results['cv_scores']['roc_auc'].std():.4f}\n\n")
        f.write("Overall CV Performance (from aggregated predictions):\n")
        f.write(f"  Accuracy:  {xgb_results['cv_metrics']['accuracy']:.4f}\n")
        f.write(f"  Precision: {xgb_results['cv_metrics']['precision']:.4f}\n")
        f.write(f"  Recall:    {xgb_results['cv_metrics']['recall']:.4f}\n")
        f.write(f"  F1-Score:  {xgb_results['cv_metrics']['f1']:.4f}\n")
        f.write(f"  ROC-AUC:   {xgb_results['cv_metrics']['roc_auc']:.4f}\n\n")
        
        # Penalized Logistic Regression metrics
        f.write("Penalized Logistic Regression Model Metrics\n")
        f.write("-"*100 + "\n")
        f.write("Cross-Validation Scores (Mean ± Std across folds):\n")
        f.write(f"  Accuracy:  {lr_results['cv_scores']['accuracy'].mean():.4f} ± {lr_results['cv_scores']['accuracy'].std():.4f}\n")
        f.write(f"  Precision: {lr_results['cv_scores']['precision'].mean():.4f} ± {lr_results['cv_scores']['precision'].std():.4f}\n")
        f.write(f"  Recall:    {lr_results['cv_scores']['recall'].mean():.4f} ± {lr_results['cv_scores']['recall'].std():.4f}\n")
        f.write(f"  F1-Score:  {lr_results['cv_scores']['f1'].mean():.4f} ± {lr_results['cv_scores']['f1'].std():.4f}\n")
        f.write(f"  ROC-AUC:   {lr_results['cv_scores']['roc_auc'].mean():.4f} ± {lr_results['cv_scores']['roc_auc'].std():.4f}\n\n")
        f.write("Overall CV Performance (from aggregated predictions):\n")
        f.write(f"  Accuracy:  {lr_results['cv_metrics']['accuracy']:.4f}\n")
        f.write(f"  Precision: {lr_results['cv_metrics']['precision']:.4f}\n")
        f.write(f"  Recall:    {lr_results['cv_metrics']['recall']:.4f}\n")
        f.write(f"  F1-Score:  {lr_results['cv_metrics']['f1']:.4f}\n")
        f.write(f"  ROC-AUC:   {lr_results['cv_metrics']['roc_auc']:.4f}\n\n")
        
        # Random Forest metrics
        f.write("Random Forest Model Metrics\n")
        f.write("-"*100 + "\n")
        f.write("Cross-Validation Scores (Mean ± Std across folds):\n")
        f.write(f"  Accuracy:  {rf_results['cv_scores']['accuracy'].mean():.4f} ± {rf_results['cv_scores']['accuracy'].std():.4f}\n")
        f.write(f"  Precision: {rf_results['cv_scores']['precision'].mean():.4f} ± {rf_results['cv_scores']['precision'].std():.4f}\n")
        f.write(f"  Recall:    {rf_results['cv_scores']['recall'].mean():.4f} ± {rf_results['cv_scores']['recall'].std():.4f}\n")
        f.write(f"  F1-Score:  {rf_results['cv_scores']['f1'].mean():.4f} ± {rf_results['cv_scores']['f1'].std():.4f}\n")
        f.write(f"  ROC-AUC:   {rf_results['cv_scores']['roc_auc'].mean():.4f} ± {rf_results['cv_scores']['roc_auc'].std():.4f}\n\n")
        f.write("Overall CV Performance (from aggregated predictions):\n")
        f.write(f"  Accuracy:  {rf_results['cv_metrics']['accuracy']:.4f}\n")
        f.write(f"  Precision: {rf_results['cv_metrics']['precision']:.4f}\n")
        f.write(f"  Recall:    {rf_results['cv_metrics']['recall']:.4f}\n")
        f.write(f"  F1-Score:  {rf_results['cv_metrics']['f1']:.4f}\n")
        f.write(f"  ROC-AUC:   {rf_results['cv_metrics']['roc_auc']:.4f}\n\n")
        
        # Model comparison summary
        f.write("="*100 + "\n")
        f.write("Model Comparison Summary\n")
        f.write("="*100 + "\n")
        f.write(f"{'Metric':<20} {'XGBoost':<15} {'Penalized LR':<15} {'Random Forest':<15} {'Best':<15}\n")
        f.write("-"*100 + "\n")
        for metric_name, values in metrics_dict.items():
            best_model = max(values, key=values.get)
            f.write(f"{metric_name:<20} {values['XGBoost']:<15.4f} {values['Penalized LR']:<15.4f} {values['Random Forest']:<15.4f} {best_model:<15}\n")
        f.write("="*100 + "\n")
    
    print(f"Model metrics saved to: {metrics_file_path}")

    print("\n" + "="*80)
    print(f"Model training completed successfully for {dataset}!")
    print("="*80)

print("\n" + "="*80)
print("All datasets processed successfully!")
print("="*80)
