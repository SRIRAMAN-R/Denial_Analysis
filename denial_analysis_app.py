# denial_analysis_app.py

# ===========================
# STEP 1: DATA PREPROCESSING
# ===========================
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTENC

# Load and clean the input Excel
df = pd.read_excel("AR - performance review - input.xlsx", header=2)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Clean currency columns
df['payment_amount'] = df['payment_amount'].replace('[\$,]', '', regex=True).astype(float)
df['balance'] = df['balance'].replace('[\$,]', '', regex=True).astype(float)
df['denied'] = df['denial_reason'].notnull().astype(int)

# Fill missing values
for col in ['insurance_company', 'physician_name', 'cpt_code']:
    df[col].fillna(df[col].mode()[0], inplace=True)
df['payment_amount'].fillna(df['payment_amount'].median(), inplace=True)
df['balance'].fillna(df['balance'].median(), inplace=True)

# Encode categorical features
le_insurance = LabelEncoder()
le_physician = LabelEncoder()
le_cpt = LabelEncoder()
df['insurance_company'] = le_insurance.fit_transform(df['insurance_company'])
df['physician_name'] = le_physician.fit_transform(df['physician_name'])
df['cpt_code'] = le_cpt.fit_transform(df['cpt_code'])

# Define X, y and SMOTENC resampling
X = df[['insurance_company', 'physician_name', 'cpt_code', 'payment_amount', 'balance']]
y = df['denied']
cat_indices = [0, 1, 2]
smote_nc = SMOTENC(categorical_features=cat_indices, random_state=42)
X_resampled, y_resampled = smote_nc.fit_resample(X, y)

# Create synthetic dataframe
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled['denied'] = y_resampled

# Reformat values
df_resampled['payment_amount'] = (df_resampled['payment_amount'] / 10).round() * 10

df_resampled['balance'] = (df_resampled['balance'] / 10).round() * 10
df_resampled['payment_amount'] = df_resampled['payment_amount'].astype(int)
df_resampled['balance'] = df_resampled['balance'].astype(int)

# Decode categorical values
df_resampled['insurance_company'] = le_insurance.inverse_transform(df_resampled['insurance_company'].astype(int))
df_resampled['physician_name'] = le_physician.inverse_transform(df_resampled['physician_name'].astype(int))
df_resampled['cpt_code'] = le_cpt.inverse_transform(df_resampled['cpt_code'].astype(int))

# Add denial reasons randomly from original dataset
denial_reasons = df['denial_reason'].dropna().astype(str).unique().tolist()
if "" not in denial_reasons:
    denial_reasons.append("")
df_resampled['denial_reason'] = "Not Denied"
df_resampled.loc[df_resampled['denied'] == 1, 'denial_reason'] = np.random.choice(denial_reasons, size=(df_resampled['denied'] == 1).sum(), replace=True)
df_resampled['denial_reason'] = df_resampled['denial_reason'].replace('', 'Not Denied').fillna("Not Denied")

# Final sample to limit rows
df_final = df_resampled[['cpt_code', 'insurance_company', 'physician_name', 'payment_amount', 'balance', 'denial_reason']].sample(n=500, replace=True, random_state=42)

# Save synthetic dataset
df_final.to_excel("AR_performance_review_synthetic.xlsx", index=False)
print("✅ Synthetic dataset generated.")

# ========================
# STEP 2: MODEL TRAINING
# ========================
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load synthetic dataset
df = pd.read_excel("AR_performance_review_synthetic.xlsx")
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
df['denied'] = (df['denial_reason'] != "Not Denied").astype(int)

# Label encoding
categorical_cols = ['cpt_code', 'insurance_company', 'physician_name']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col])
    label_encoders[col] = le

# Prepare features and target
features = ['cpt_code_enc', 'insurance_company_enc', 'physician_name_enc', 'payment_amount', 'balance']
X = df[features]
y = df['denied']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train binary model
clf = RandomForestClassifier(random_state=42, class_weight="balanced")
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Binary Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))

# Save model & encoders
joblib.dump(clf, 'denial_model.pkl')
for col in categorical_cols:
    joblib.dump(label_encoders[col], f'{col}_encoder.pkl')
print("✅ Model and encoders saved.")

