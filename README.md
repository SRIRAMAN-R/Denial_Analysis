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
  
<img width="1867" height="816" alt="image" src="https://github.com/user-attachments/assets/8a45ec37-db99-4628-8a00-453cb1955d69" />


## ⚙️ Run Pipeline and Show Results
### Tab 1: ML Prediction
- Binary classifier: Claim Denied or Not
- Multiclass classifier: Specific Denial Reason

#### Outputs:
- Accuracy
- Classification Report

<img width="1854" height="826" alt="image" src="https://github.com/user-attachments/assets/1fdf323e-3bc2-4bf8-9377-4d786e7522ed" />

### Tab 2: Data Analysis
##### Charts:
- Top CPTs denied
- Denials by payer and provider
- Denial rate (%)
- Lost revenue by CPT

<img width="1820" height="851" alt="image" src="https://github.com/user-attachments/assets/5efc40fa-8e0e-496d-ad5a-a96243ec91a9" />

### Tab 3: Root Cause & Recommendations
- Lists denial reasons
- Displays potential causes and recommended fixes

<img width="1729" height="771" alt="image" src="https://github.com/user-attachments/assets/4b9b3707-2352-4f74-8d0d-c8b999d4b2aa" />


---

##  How to Run Locally

```bash
git clone https://github.com/SRIRAMAN-R/Denial_Analysis.git
cd Denial_Analysis
pip install -r requirements.txt
python denial_analysis_app.py
streamlit run app.py

```



