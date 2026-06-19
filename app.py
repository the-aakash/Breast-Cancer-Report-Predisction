import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('patients.db')
    c = conn.cursor()
    # Table banana agar nahi bani hai toh 
    c.execute('''CREATE TABLE IF NOT EXISTS patient_records 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, diagnosis TEXT, confidence REAL)''')
    conn.commit()
    conn.close()

def save_to_db(name, result, conf):
    conn = sqlite3.connect('patients.db')
    c = conn.cursor()
    c.execute("INSERT INTO patient_records (name, diagnosis, confidence) VALUES (?, ?, ?)", (name, result, conf))
    conn.commit()
    conn.close()

init_db()

# --- MODEL TRAINING ---
@st.cache_resource
def train_model():
    # Dataset load karna [cite: 60, 562]
    df = pd.read_csv("cancer.csv")
    cols_to_drop = ['id', 'Unnamed: 32']
    # Target variable 'diagnosis' ko baki features se alag karna [cite: 325, 576, 942]
    X = df.drop([c for c in cols_to_drop if c in df.columns] + ['diagnosis'], axis=1)
    y = df['diagnosis'].map({'M': 1, 'B': 0})
    
    # Feature Scaling: Machine learning algorithms ke liye zaruri hai [cite: 571, 617]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Random Forest Classifier ka use: High accuracy ke liye [cite: 62, 775, 872]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, X.columns

model, scaler, feature_names = train_model()

# --- WEB UI SETUP ---
st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")
st.title("🩺 Breast Cancer Diagnostic System")
st.write("Aapki report ke patterns ke hisab se cancer predict karne ka system.")

# Patient Name Section
patient_name = st.text_input("Enter the full name Patient: ", placeholder="E.g. Mrs. Sharma")

st.write("---")
st.subheader("Enter your Medical Report Details in the given Input Form-)")

# Input Function: Data collection ke liye [cite: 269, 276]
def user_input_features():
    inputs = {}
    col1, col2, col3 = st.columns(3)
    
    for i, name in enumerate(feature_names):
        if i % 3 == 0:
            with col1:
                inputs[name] = st.number_input(f"{name}", value=0.0, format="%.4f")
        elif i % 3 == 1:
            with col2:
                inputs[name] = st.number_input(f"{name}", value=0.0, format="%.4f")
        else:
            with col3:
                inputs[name] = st.number_input(f"{name}", value=0.0, format="%.4f")
            
    return pd.DataFrame([inputs])

input_df = user_input_features()

# --- PREDICTION & SAVING ---
if st.button("Predict & Save Data"):
    if not patient_name:
        st.warning("Firstly please input the patient's name!")
    else:
        # User input ko scale karna [cite: 571, 617]
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        prob = model.predict_proba(input_scaled)
        
        diag = "Malignant" if prediction[0] == 1 else "Benign"
        conf_score = np.max(prob) * 100
        
        # Result Display (M=Cancer, B=Normal) [cite: 76, 325, 576]
        if prediction[0] == 1:
            st.error(f"⚠️ Result of {patient_name} : {diag} (Cancerous)")
            st.write("The patterns in your report indicate cancer. Please contact a doctor.[cite: 740].")
        else:
            st.success(f"✅ Result of {patient_name} : {diag} (Non-Cancerous)")
            st.write("The patterns in the report indicate a benign condition.")
        
        st.write(f"**Confidence Level:** {conf_score:.2f}%")
        
        # History save karna 
        save_to_db(patient_name, diag, conf_score)
        st.info(f"Data for {patient_name} has been saved successfully!")

# --- VIEW HISTORY ---
st.write("---")
if st.checkbox("Show Saved Patient History"):
    st.subheader("Saved Records (Local Database)")
    conn = sqlite3.connect('patients.db')
    try:
        history_df = pd.read_sql_query("SELECT * FROM patient_records", conn)
        st.dataframe(history_df, use_container_width=True)
    except:
        st.write("No records have been saved yet.")
    conn.close()

st.info("Disclaimer: Ye system Machine Learning algorithms par based hai aur educational use ke liye hai[cite: 279, 866].")