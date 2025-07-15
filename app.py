# app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Denial Analysis", layout="wide")
st.markdown("""
    <h1 style='text-align: center; color: #8d5996; font-size: 50px;'>Denial Analysis </h1>
    <hr style='border: 2px solid #8d5996;'>
""", unsafe_allow_html=True)

# ======================
# SESSION STATE INIT
# ======================
if 'add_data_clicked' not in st.session_state:
    st.session_state.add_data_clicked = False

# ======================
# SIDEBAR CONTROLS
# ======================
st.sidebar.title("Options")

if st.sidebar.button("➕ Add additional data"):
    st.session_state.add_data_clicked = True

if st.sidebar.button("🚀 Run pipeline and show results"):
    st.session_state.run_pipeline = True
else:
    st.session_state.run_pipeline = False

if st.sidebar.button("🔄 Start Over"):
    st.session_state.add_data_clicked = False
    st.rerun()

# ======================
# UTILITY FUNCTIONS
# ======================
def save_barplot(data, x, y, title, xlabel, ylabel, palette):
    fig, ax = plt.subplots(figsize=(6, 4))
    data[y] = data[y].astype(str)
    top_data = data.head(10)
    sns.barplot(x=x, y=y, data=top_data, palette=palette, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', label_type='edge', fontsize=8, padding=2)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    return fig

def classify(df):
    categorical_cols = ['cpt_code', 'insurance_company', 'physician_name']
    for col in categorical_cols:
        encoder = joblib.load(f"{col}_encoder.pkl")
        df[col + '_enc'] = encoder.transform(df[col].astype(str))

    features = ['cpt_code_enc', 'insurance_company_enc', 'physician_name_enc', 'payment_amount', 'balance']
    model = joblib.load("denial_model.pkl")

    if st.button("Run Model"):
        st.write("### Prediction Results:")
        df['Predicted_Denied'] = model.predict(df[features])
        df['Predicted_Denied'] = df['Predicted_Denied'].map({1: "Denied", 0: "Not Denied"})

        display_cols = ['cpt_code', 'insurance_company', 'physician_name', 'payment_amount', 'balance', 'Predicted_Denied', 'denial_reason']
        st.dataframe(df[display_cols].head(50))

# ======================
# MAIN TABS
# ======================
def tabs():
    tab1, tab2, tab3 = st.tabs(["ML Prediction", "Data Analysis", "Solution and Root Cause"])

    with tab1:
        df = pd.read_excel("AR_performance_review_synthetic.xlsx")
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        df['denial_reason'] = df['denial_reason'].fillna("Not Denied")
        df['denied'] = (df['denial_reason'] != "Not Denied").astype(int)

        categorical_cols = ['cpt_code', 'insurance_company', 'physician_name']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col + '_enc'] = le.fit_transform(df[col])

        df['payment_amount'] = pd.to_numeric(df['payment_amount'], errors='coerce').fillna(0)
        df['balance'] = pd.to_numeric(df['balance'], errors='coerce').fillna(0)

        features = ['cpt_code_enc', 'insurance_company_enc', 'physician_name_enc', 'payment_amount', 'balance']
        X = df[features]
        y_binary = df['denied']
        y_multiclass = df[df['denied'] == 1]['denial_reason']
        X_multiclass = df[df['denied'] == 1][features]

        X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X, y_binary, test_size=0.2, random_state=42)
        X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multiclass, y_multiclass, test_size=0.2, random_state=42)

        clf_bin = RandomForestClassifier(random_state=42, class_weight="balanced")
        clf_bin.fit(X_train_bin, y_train_bin)
        y_pred_bin = clf_bin.predict(X_test_bin)

        st.write("### Binary Classification (Denied or Not Denied)")
        st.write(f"Accuracy: {accuracy_score(y_test_bin, y_pred_bin):.4f}")
        st.table(pd.DataFrame(classification_report(y_test_bin, y_pred_bin, output_dict=True)).transpose())

        le_reason = LabelEncoder()
        y_train_multi_enc = le_reason.fit_transform(y_train_multi)
        y_test_multi_enc = le_reason.transform(y_test_multi)
        clf_multi = RandomForestClassifier(random_state=42, class_weight="balanced")
        clf_multi.fit(X_train_multi, y_train_multi_enc)
        y_pred_multi = clf_multi.predict(X_test_multi)

        st.write("### Multiclass Classification (Denial Reason)")
        st.write(f"Accuracy: {accuracy_score(y_test_multi_enc, y_pred_multi):.4f}")
        st.table(pd.DataFrame(classification_report(y_test_multi_enc, y_pred_multi, target_names=le_reason.classes_, output_dict=True)).transpose())

    with tab2:
        df = pd.read_excel("AR_performance_review_synthetic.xlsx")
        df['denied'] = df['denial_reason'].notna()

        denials_by_cpt = df[df['denied']].groupby('cpt_code').size().reset_index(name='denial_count')
        total_by_cpt = df.groupby('cpt_code').size().reset_index(name='total_count')
        merged = pd.merge(denials_by_cpt, total_by_cpt, on='cpt_code')
        merged['denial_rate'] = (merged['denial_count'] / len(df)) * 100
        merged_filtered = merged[(merged['denial_rate'] > 0) & (merged['denial_rate'] < 100)]

        denials_by_payer = df[df['denied']].groupby('insurance_company').size().reset_index(name='denial_count')
        denials_by_provider = df[df['denied']].groupby('physician_name').size().reset_index(name='denial_count')
        lost_revenue = df[df['denied']].groupby('cpt_code')['balance'].sum().reset_index()

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(save_barplot(denials_by_cpt.sort_values('denial_count', ascending=False), 'denial_count', 'cpt_code', "Top Denied CPT Codes", "Denial Count", "CPT Code", 'Reds_r'))
        with col2:
            st.pyplot(save_barplot(merged_filtered.sort_values('denial_rate', ascending=False), 'denial_rate', 'cpt_code', "Top CPTs by Denial Rate (%)", "Denial Rate (%)", "CPT Code", 'Greens_r'))

        col3, col4 = st.columns(2)
        with col3:
            st.pyplot(save_barplot(denials_by_payer, 'denial_count', 'insurance_company', "Top Payers with Most Denials", "Denial Count", "Insurance Company", 'Purples_r'))
        with col4:
            st.pyplot(save_barplot(denials_by_provider, 'denial_count', 'physician_name', "Top Providers with Most Denials", "Denial Count", "Physician", 'Oranges_r'))

        st.pyplot(save_barplot(lost_revenue, 'balance', 'cpt_code', "Top CPTs by Lost Revenue", "Total Unpaid Balance ($)", "CPT Code", 'Blues_r'))

    with tab3:
        df = pd.read_excel("AR_performance_review_synthetic.xlsx")
        df['denial_reason'] = df['denial_reason'].fillna("Not Denied")
        section_issue = df['denial_reason'].dropna().astype(str).unique().tolist()

        section_solution = """
            ####  Recommendations:

            **1. Missing information**
            -  Use complete and correct details.
            -  Train staff on payer-specific documentation.

            **2. Charge exceeds fee schedule**
            -  Compare charges with payer schedules regularly.
            -  Appeal denials with proper justification.

            **3. Non-covered service**
            -  Check plan coverage before providing care.
            -  Inform patients of potential out-of-pocket costs.
        """

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Root Cause of Denial")
            filtered_reasons = [r for r in section_issue if r != 'Not Denied']
            for reason in filtered_reasons:
                if ' - ' in reason:
                    code, desc = reason.split(" - ", 1)
                else:
                    code, desc = reason, ''
                st.markdown(f"""
                <div style='color:white; padding:4px; font-size:16px'>
                    <b>{code}</b>: {desc}
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("### Recommendations for Denials")
            st.markdown(section_solution)

# ======================
# FILE UPLOAD & PROCESS
# ======================
if st.session_state.add_data_clicked:
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

    def load_excel_dynamic(file):
        raw = pd.read_excel(file, header=None)
        for i, row in raw.iterrows():
            if row.notna().sum() >= 3:
                return pd.read_excel(file, skiprows=i)

    if uploaded_file:
        try:
            df = load_excel_dynamic(uploaded_file)
            df.drop(columns=['#'], errors='ignore', inplace=True)
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            df['payment_amount'] = df['payment_amount'].replace('[\$,]', '', regex=True).astype(float)
            df['balance'] = df['balance'].replace('[\$,]', '', regex=True).astype(float)
            df['denial_reason'].fillna("Not Denied", inplace=True)
            df['payment_amount'].fillna(df['payment_amount'].median(), inplace=True)
            df['balance'].fillna(df['balance'].median(), inplace=True)

            existing = pd.read_excel("AR_performance_review_synthetic.xlsx")
            combined = pd.concat([existing, df], ignore_index=True)
            combined.to_excel("AR_performance_review_synthetic.xlsx", index=False)
            classify(df)
            st.success("✅ Data appended and prediction complete.")
        except Exception as e:
            st.error(f"Error: {e}")

# ======================
# PIPELINE EXECUTION
# ======================
if st.session_state.run_pipeline:
    tabs()
