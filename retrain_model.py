"""
Retrain EMI Eligibility Classifier with corrected labels and class balancing.

Root cause of original bug:
  - The original emi_eligibility labels in training data were generated with logic
    that made ~88% of Rented-house applicants "Not_Eligible" regardless of their
    actual financial ratios (DTI, EMI-to-income, etc.).
  - Result: the ML model learned to predict based on house_type instead of finances.

Fix:
  - Re-derive eligibility labels using industry-standard financial criteria.
  - Train XGBoost Pipeline with the corrected labels.
"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

print("=" * 60)
print("EMI Eligibility Model Retraining (with corrected labels)")
print("=" * 60)

# ── 1. Load data ─────────────────────────────────────────────
df = pd.read_csv("data/processed_emi_data.csv")
print(f"\nLoaded {len(df):,} records")

INPUT_COLS = [
    "age", "gender", "marital_status", "education", "monthly_salary",
    "employment_type", "years_of_employment", "company_type", "house_type",
    "monthly_rent", "family_size", "dependents", "school_fees", "college_fees",
    "travel_expenses", "groceries_utilities", "other_monthly_expenses",
    "existing_loans", "current_emi_amount", "credit_score", "bank_balance",
    "emergency_fund", "emi_scenario", "requested_amount", "requested_tenure"
]
NUM_COLS = [
    "age", "monthly_salary", "years_of_employment", "monthly_rent",
    "family_size", "dependents", "school_fees", "college_fees",
    "travel_expenses", "groceries_utilities", "other_monthly_expenses",
    "current_emi_amount", "credit_score", "bank_balance", "emergency_fund",
    "requested_amount", "requested_tenure"
]
CAT_COLS = [
    "gender", "marital_status", "education", "employment_type",
    "company_type", "house_type", "existing_loans", "emi_scenario"
]

# ── 2. Re-derive fair eligibility labels ─────────────────────
print("\nRe-deriving eligibility labels using financial criteria...")

df['requested_emi'] = df['requested_amount'] / df['requested_tenure'].clip(lower=1)
df['total_monthly_expenses'] = (
    df['monthly_rent'] + df['school_fees'] + df['college_fees'] +
    df['travel_expenses'] + df['groceries_utilities'] + df['other_monthly_expenses']
)
df['total_obligation'] = df['total_monthly_expenses'] + df['current_emi_amount'] + df['requested_emi']
df['net_disposable'] = df['monthly_salary'] - df['total_obligation']
df['req_emi_ratio'] = df['requested_emi'] / df['monthly_salary'].clip(lower=1) * 100
df['total_obligation_ratio'] = df['total_obligation'] / df['monthly_salary'].clip(lower=1) * 100
df['existing_dti'] = df['current_emi_amount'] / df['monthly_salary'].clip(lower=1) * 100

# Industry-standard eligibility rules:
#  Eligible   : total obligation < 50% income AND new EMI < 30% income
#               AND credit score >= 650 AND net disposable > 0
#  High_Risk  : (total obligation 50-70% OR new EMI 30-45%) AND credit 600-649
#               AND net disposable >= -5000
#  Not_Eligible: everything else

conds_eligible = (
    (df['total_obligation_ratio'] < 50) &
    (df['req_emi_ratio'] < 30) &
    (df['credit_score'] >= 650) &
    (df['net_disposable'] > 0)
)
conds_high_risk = (
    ~conds_eligible &
    (df['total_obligation_ratio'] < 70) &
    (df['req_emi_ratio'] < 45) &
    (df['credit_score'] >= 580) &
    (df['net_disposable'] >= -5000)
)

df['fair_eligibility'] = 'Not_Eligible'
df.loc[conds_high_risk, 'fair_eligibility'] = 'High_Risk'
df.loc[conds_eligible, 'fair_eligibility'] = 'Eligible'

print("\nOriginal label distribution:")
print(df['emi_eligibility'].value_counts())
print("\nFair label distribution:")
print(df['fair_eligibility'].value_counts())

X = df[INPUT_COLS].copy()
y = df['fair_eligibility'].copy()

# ── 3. Encode target ──────────────────────────────────────────
le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"\nClasses: {dict(enumerate(le.classes_))}")

# ── 4. Train / test split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

# ── 5. Preprocessor ───────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), NUM_COLS),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
])

# ── 6. Sample weights for any residual imbalance ─────────────
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

# ── 7. XGBoost ────────────────────────────────────────────────
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss",
    n_jobs=-1,
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", xgb),
])

# ── 8. Train ──────────────────────────────────────────────────
print("\nTraining XGBoost...")
pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)
print("Done.")

# ── 9. Evaluate ───────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── 10. Sanity-check the user's example applicant ────────────
test_applicant = pd.DataFrame([{
    "age": 35, "gender": "Male", "marital_status": "Single",
    "education": "High School", "monthly_salary": 75000,
    "employment_type": "Private", "years_of_employment": 5.0,
    "company_type": "Startup", "house_type": "Rented", "monthly_rent": 15000,
    "family_size": 3, "dependents": 1, "school_fees": 5000, "college_fees": 0,
    "travel_expenses": 3000, "groceries_utilities": 10000,
    "other_monthly_expenses": 5000, "existing_loans": "Yes",
    "current_emi_amount": 10000, "credit_score": 720, "bank_balance": 500000,
    "emergency_fund": 200000, "emi_scenario": "E-commerce Shopping EMI",
    "requested_amount": 500000, "requested_tenure": 72
}])

# Manually verify using our rules first
req_emi = 500000 / 72
expenses = 15000 + 5000 + 0 + 3000 + 10000 + 5000
total_obl = expenses + 10000 + req_emi
req_emi_ratio = req_emi / 75000 * 100
total_obl_ratio = total_obl / 75000 * 100
net_disp = 75000 - total_obl
print(f"\n=== Manual check for test applicant ===")
print(f"Requested EMI/month = ₹{req_emi:.0f} ({req_emi_ratio:.1f}% of salary)")
print(f"Total obligations = ₹{total_obl:.0f} ({total_obl_ratio:.1f}% of salary)")
print(f"Net disposable after all obligations = ₹{net_disp:.0f}")
print(f"Expected label (by rules): {'Eligible' if total_obl_ratio < 50 and req_emi_ratio < 30 and net_disp > 0 else 'Not_Eligible'}")

pred_enc = pipeline.predict(test_applicant)[0]
pred_proba = pipeline.predict_proba(test_applicant)[0]
pred_label = le.classes_[pred_enc]
print(f"\nModel Prediction: {pred_label}")
for cls, prob in zip(le.classes_, pred_proba):
    print(f"  {cls}: {prob*100:.1f}%")

# ── 11. Save ──────────────────────────────────────────────────
models_dir = Path("models")
joblib.dump(pipeline, models_dir / "best_classifier_xgboost.pkl")
joblib.dump(preprocessor, models_dir / "preprocessor.pkl")
joblib.dump(le,          models_dir / "label_encoder.pkl")

feature_names = {
    "feature_names": list(pipeline.named_steps["preprocessor"].get_feature_names_out()),
    "numerical_cols": [f"num__{c}" for c in NUM_COLS],
    "categorical_cols": list(
        pipeline.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .get_feature_names_out()
    )
}
with open(models_dir / "feature_names.json", "w") as f:
    json.dump(feature_names, f, indent=2)

print("\n✅ Models saved successfully.")
print("   Restart the Streamlit app to load the new model.")

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

print("=" * 60)
print("EMI Eligibility Model Retraining")
print("=" * 60)

# ── 1. Load data ─────────────────────────────────────────────
df = pd.read_csv("data/processed_emi_data.csv")
print(f"\nDataset shape: {df.shape}")

INPUT_COLS = [
    "age", "gender", "marital_status", "education", "monthly_salary",
    "employment_type", "years_of_employment", "company_type", "house_type",
    "monthly_rent", "family_size", "dependents", "school_fees", "college_fees",
    "travel_expenses", "groceries_utilities", "other_monthly_expenses",
    "existing_loans", "current_emi_amount", "credit_score", "bank_balance",
    "emergency_fund", "emi_scenario", "requested_amount", "requested_tenure"
]
TARGET_COL = "emi_eligibility"

X = df[INPUT_COLS].copy()
y = df[TARGET_COL].copy()

# ── 2. Encode target ──────────────────────────────────────────
le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"\nClasses: {dict(enumerate(le.classes_))}")
print(f"Class distribution:\n{pd.Series(y_enc).value_counts().rename(index=dict(enumerate(le.classes_)))}")

# ── 3. Train/test split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"\nTrain size: {len(X_train):,} | Test size: {len(X_test):,}")

# ── 4. Preprocessor ───────────────────────────────────────────
NUM_COLS = [
    "age", "monthly_salary", "years_of_employment", "monthly_rent",
    "family_size", "dependents", "school_fees", "college_fees",
    "travel_expenses", "groceries_utilities", "other_monthly_expenses",
    "current_emi_amount", "credit_score", "bank_balance", "emergency_fund",
    "requested_amount", "requested_tenure"
]
CAT_COLS = [
    "gender", "marital_status", "education", "employment_type",
    "company_type", "house_type", "existing_loans", "emi_scenario"
]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), NUM_COLS),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
])

# ── 5. Compute sample weights to fix class imbalance ─────────
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

# ── 6. XGBoost Classifier with class weighting ───────────────
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="mlogloss",
    n_jobs=-1,
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", xgb),
])

# ── 7. Train ──────────────────────────────────────────────────
print("\nTraining XGBoost with balanced class weights...")
pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)
print("Training complete.")

# ── 8. Evaluate ───────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── 9. Sanity-check with your example applicant ──────────────
test_applicant = pd.DataFrame([{
    "age": 35, "gender": "Male", "marital_status": "Single",
    "education": "High School", "monthly_salary": 75000,
    "employment_type": "Private", "years_of_employment": 5.0,
    "company_type": "Startup", "house_type": "Rented", "monthly_rent": 15000,
    "family_size": 3, "dependents": 1, "school_fees": 5000, "college_fees": 0,
    "travel_expenses": 3000, "groceries_utilities": 10000,
    "other_monthly_expenses": 5000, "existing_loans": "Yes",
    "current_emi_amount": 10000, "credit_score": 720, "bank_balance": 500000,
    "emergency_fund": 200000, "emi_scenario": "E-commerce Shopping EMI",
    "requested_amount": 500000, "requested_tenure": 72
}])

pred_enc = pipeline.predict(test_applicant)[0]
pred_proba = pipeline.predict_proba(test_applicant)[0]
pred_label = le.classes_[pred_enc]
print(f"\n=== Sanity check (your test applicant) ===")
print(f"Prediction: {pred_label}")
for cls, prob in zip(le.classes_, pred_proba):
    print(f"  {cls}: {prob*100:.1f}%")

# ── 10. Save models ───────────────────────────────────────────
models_dir = Path("models")
joblib.dump(pipeline, models_dir / "best_classifier_xgboost.pkl")
joblib.dump(preprocessor, models_dir / "preprocessor.pkl")
joblib.dump(le, models_dir / "label_encoder.pkl")

# Save feature names
feature_names = {
    "feature_names": list(pipeline.named_steps["preprocessor"].get_feature_names_out()),
    "numerical_cols": [f"num__{c}" for c in NUM_COLS],
    "categorical_cols": list(
        pipeline.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .get_feature_names_out([f"cat__{c}" for c in CAT_COLS])
    )
}
with open(models_dir / "feature_names.json", "w") as f:
    json.dump(feature_names, f, indent=2)

print("\n✅ Models saved to models/ directory:")
print("   - best_classifier_xgboost.pkl (balanced Pipeline)")
print("   - preprocessor.pkl")
print("   - label_encoder.pkl")
print("   - feature_names.json")
print("\nRestart the Streamlit app to load the new models.")
