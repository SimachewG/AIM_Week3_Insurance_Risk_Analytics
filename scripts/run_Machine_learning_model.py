import pandas as pd
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.Machine_learning_model import (
    prepare_data,
    get_models,
    evaluate_regression_models,
    evaluate_classification_models,
    save_risk_based_premium,
    shap_analysis,
    save_feature_importance,
    save_model_comparison_report
)

# Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Load preprocessed data
data = pd.read_csv("data/processed/cleaned_data.csv")

# Add binary target for classification
data['HasClaim'] = (data['TotalClaims'] > 0).astype(int)

# Subsets for different tasks
severity_data = data[data['TotalClaims'] > 0].copy()  # Only rows with claims

# Prepare data
X_sev_train, X_sev_test, y_sev_train, y_sev_test = prepare_data(severity_data, 'TotalClaims')
X_prem_train, X_prem_test, y_prem_train, y_prem_test = prepare_data(data, 'CalculatedPremiumPerTerm')
X_clf_train, X_clf_test, y_clf_train, y_clf_test = prepare_data(data, 'HasClaim')

# Load models
reg_models, clf_models = get_models()

# Evaluate models
severity_results = evaluate_regression_models(reg_models, X_sev_train, X_sev_test, y_sev_train, y_sev_test, "Claim Severity")
premium_results = evaluate_regression_models(reg_models, X_prem_train, X_prem_test, y_prem_train, y_prem_test, "Premium Prediction")
classification_results = evaluate_classification_models(clf_models, X_clf_train, X_clf_test, y_clf_train, y_clf_test)

# Save model performance comparison report
save_model_comparison_report(severity_results, premium_results, classification_results)

# Select best models for pipeline (use XGBoost as an example)
best_clf = clf_models["XGBoost"]
best_clf.fit(X_clf_train, y_clf_train)

best_sev = reg_models["XGBoost"]
best_sev.fit(X_sev_train, y_sev_train)

# Save risk-based premium predictions
#save_risk_based_premium(best_clf, best_sev, X_clf_test)
#save_risk_based_premium(best_clf, best_sev, X_sev_test)
save_risk_based_premium(best_clf, best_sev, X_clf_test, X_sev_test)


# Run SHAP analysis for interpretability
shap_analysis(best_sev, X_sev_test)

# Save feature importances
save_feature_importance(best_sev, X_sev_test, "XGBoost_Severity")
save_feature_importance(best_clf, X_clf_test, "XGBoost_Classifier")

# Save best models
joblib.dump(best_clf, "models/classification_xgb_model.pkl")
joblib.dump(best_sev, "models/severity_xgb_model.pkl")

print("✅ All modeling tasks completed successfully.")

