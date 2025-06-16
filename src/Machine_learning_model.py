import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

import shap

def prepare_data(df, target_column):
    """Prepare data by handling missing values, feature engineering, label encoding, and splitting."""
    df = df.copy()

    # 1. Handle missing data
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna('Missing', inplace=True)  # for categorical

    # 2. Feature engineering (example)
    # premiums per claims ratio
    df['PremiumToClaimsRatio'] = df['TotalPremium'] / (df['TotalClaims'] + 1)
    
    # 3. Label Encoding for categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object', 'category', 'bool']).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    # 4. Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 5. Train-test split
    return train_test_split(X, y, test_size=0.2, random_state=42)


def get_models():
    regression_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    classification_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
    }

    return regression_models, classification_models

def tune_hyperparameters(model, param_grid, X_train, y_train, is_classification=True, scoring=None):
    scoring = scoring or ('f1' if is_classification else 'neg_mean_squared_error')
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring=scoring, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print(f"✅ Best Parameters for {model.__class__.__name__}: {grid_search.best_params_}")
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_regression_models(models, X_train, X_test, y_train, y_test, label, tune=False, param_grids=None):
    print(f"\n--- {label} ---")
    results = {}

    for name, model in models.items():
        try:
            print(f"Training {name}...")
            if tune and param_grids and name in param_grids:
                model, _ = tune_hyperparameters(model, param_grids[name], X_train, y_train, is_classification=False)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            results[name] = {'RMSE': rmse, 'R2': r2}
            print(f"{name}:\n  RMSE = {rmse:.2f}, R2 = {r2:.3f}")
        except Exception as e:
            print(f"❌ Error in {name}: {e}")
    return results

def evaluate_classification_models(models, X_train, X_test, y_train, y_test, tune=False, param_grids=None):
    print(f"\n--- Claim Probability Classification ---")
    results = {}

    for name, model in models.items():
        try:
            print(f"Training {name}...")
            if tune and param_grids and name in param_grids:
                model, _ = tune_hyperparameters(model, param_grids[name], X_train, y_train, is_classification=True)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            print(f"{name}:\n  Accuracy = {acc:.3f}, Precision = {prec:.3f}, Recall = {rec:.3f}, F1-score = {f1:.3f}")
            print("  Classification Report:\n", classification_report(y_test, y_pred))

            results[name] = {
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1
            }
        except Exception as e:
            print(f"❌ Error in {name}: {e}")

    return results

#def save_risk_based_premium(classifier, severity_model, X_test, output_path="outputs/risk_based_premium.csv"):
#    prob_claim = classifier.predict_proba(X_test)[:, 1]
#    predicted_severity = severity_model.predict(X_test)
#    risk_based_premium = prob_claim * predicted_severity + 100

#    df = pd.DataFrame({
#        'PredictedProbability': prob_claim,
#        'PredictedSeverity': predicted_severity,
#        'RiskBasedPremium': risk_based_premium
#    })
#    df.to_csv(output_path, index=False)
#    print(f"✅ Risk-Based Premium saved to {output_path}")

def save_risk_based_premium(classifier, severity_model, X_clf_test, X_sev_test, output_path="outputs/risk_based_premium.csv"):
    """Predict risk-based premium as: P(claim) * E[claim severity] + margin."""

    # Step 1: Predict probability of a claim
    prob_claim = classifier.predict_proba(X_clf_test)[:, 1]

    # Step 2: Predict claim severity
    predicted_severity = severity_model.predict(X_sev_test)

    # Step 3: Align lengths (optional depending on design)
    n = min(len(prob_claim), len(predicted_severity))
    risk_premium = prob_claim[:n] * predicted_severity[:n]

    # Step 4: Save results
    df_out = pd.DataFrame({
        "prob_claim": prob_claim[:n],
        "predicted_severity": predicted_severity[:n],
        "risk_based_premium": risk_premium
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"Saved risk-based premium predictions to {output_path}")


def shap_analysis(model, X_test, output_img="outputs/shap_summary.png", top_features_csv="outputs/top_features.csv"):
    print("\n--- SHAP Analysis ---")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig(output_img)
    print(f"SHAP plot saved to {output_img}")

    if hasattr(model, 'feature_importances_'):
        feat_imp_df = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        feat_imp_df.head(10).to_csv(top_features_csv, index=False)
        print(f"Top features saved to {top_features_csv}")
        return feat_imp_df
    else:
        print("Model does not support feature importances.")
        return None

def save_feature_importance(model, X, model_name, output_path="outputs"):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        features = X.columns
        feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        feat_df = feat_df.sort_values(by='Importance', ascending=False)

        output_file = os.path.join(output_path, f"{model_name}_feature_importance.csv")
        feat_df.to_csv(output_file, index=False)
        print(f"✅ Feature importance saved to {output_file}")
        return feat_df
    else:
        print(f"⚠️ Model {model_name} does not support feature importance.")
        return None

def save_model_comparison_report(severity_results, premium_results, classification_results, output="outputs/model_comparison.csv"):
    rows = []
    for model_name, metrics in severity_results.items():
        rows.append({'Model': model_name, 'Task': 'Claim Severity', **metrics})
    for model_name, metrics in premium_results.items():
        rows.append({'Model': model_name, 'Task': 'Premium Prediction', **metrics})
    for model_name, metrics in classification_results.items():
        rows.append({'Model': model_name, 'Task': 'Claim Classification', **metrics})

    df_report = pd.DataFrame(rows)
    df_report.to_csv(output, index=False)
    print(f"📊 Model comparison report saved to {output}")




