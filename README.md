# 📊 DENIAL_ANALYSIS

A comprehensive end-to-end solution for Revenue Cycle Management (RCM) analysts to identify, predict, and reduce healthcare claim denials using machine learning and an interactive Streamlit interface.

This dashboard allows users to upload Excel files, view predictions for denial likelihood, analyze denial trends, and receive actionable recommendations.


---

## 📁 Data Input Format

Your Excel file must contain the following columns:

| Column Name         | Description                          |
|---------------------|--------------------------------------|
| CPT Code            | Procedure code for the billed service |
| Insurance Company   | Name of the payer                    |
| Physician Name      | The rendering provider               |
| Payment Amount      | Amount paid by insurance (numeric, in $)       |
| Balance             | Unpaid amount (in $)                 |
| Denial Reason       | (Optional) Textual reason for denial |

---


## ✅ System

### 1. Upload & Classify Claims
- Upload Excel file and get instant denial predictions
- Use pre-trained ML model with categorical encoding
- Display predictions alongside original data

### 2. Identify Top Denied CPT Codes

- Rank CPTs by frequency of denial
- Visualize top codes and their denial rate

### Breakdown by Payer and Provider

- Identify payers and providers with high denial volume
- Highlight revenue leakage by source
  
### 4. Root Cause Analysis
- Extract top denial reasons
-- Flag issues like:
- Missing modifiers
- Non-covered services
- Credentialing/documentation errors

### 5. Predictions
- Binary Classification – Predict: Denied vs Not Denied
- Multiclass Classification – Predict: Specific Denial Reason

### 6. Visual Reporting with Charts
- CPTs by denial frequency and rate
- Insurance and providers by denials
- Lost revenue per CPT
- All visualized via bar plots in Streamlit

---

## 🚀 Final Output (via Streamlit App)

Features:
- Upload Excel files
- Clean & normalize data
- Append to historical data
- Run ML model
- View predictions
- Display charts and root causes

---

## 🔁 Project Workflow

###  Step 1: Synthetic Dataset Generation
- Resample initial dataset using SMOTENC to balance classes
- Generate 500 rows with CPT, payer, provider, payment & balance
- Append randomly selected denial reasons

###  Step 2: ML Model Training
- Train RandomForestClassifier using encoded features
- Save model and encoders for reuse
- Evaluate accuracy and classification report

###  Step 3: Streamlit Application
Interactive interface to:
- Upload files
- Append and process data
- Run predictions and visualize results

---

## 🧠 Streamlit Button Logic

| Sidebar Option            | Action                          |
|---------------------------|--------------------------------------|
| ➕ Add Additional Data    | Upload Excel, clean/encode/append to existing dataset |
| 🚀 Run Pipeline   | Train/test models, display metrics & graphs                |
| 🔄 Start Over      | Reset session to allow new uploads or restart process            |

---

## 🔍 Step-by-Step Breakdown

### 📤 Add Additional Data
- Upload any Excel file with billing data
- Cleans headers, formats currency columns
- Fills missing values
- Merges uploaded data into a master dataset
- Prediction runs only on uploaded file
  
  <img width="1905" height="858" alt="image" src="https://github.com/user-attachments/assets/a2ccd05b-fa76-45bc-8404-224e8f584bdb" />


## ⚙️ Run Pipeline and Show Results
### Tab 1: ML Prediction
- Binary classifier: Claim Denied or Not
- Multiclass classifier: Specific Denial Reason

#### Outputs:
- Accuracy
- Classification Report

<img width="1869" height="844" alt="image" src="https://github.com/user-attachments/assets/06606e9b-156a-49de-bc3a-346db3369425" />

### Tab 2: Data Analysis
##### Charts:
- Top CPTs denied
- Denials by payer and provider
- Denial rate (%)
- Lost revenue by CPT

  <img width="1885" height="898" alt="image" src="https://github.com/user-attachments/assets/fd78821b-2190-4b9a-9ff9-610600fce1d4" />

### Tab 3: Root Cause & Recommendations
- Lists denial reasons
- Displays potential causes and recommended fixes

---

##  How to Run Locally

```bash
git clone https://github.com/SRIRAMAN-R/Denial_Analysis.git
cd Denial_Analysis
pip install -r requirements.txt
python denial_analysis_app.py
streamlit run app.py

```



