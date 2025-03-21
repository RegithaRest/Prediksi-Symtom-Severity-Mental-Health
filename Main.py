import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load model
model = pickle.load(open('model_xgb.pkl', 'rb'))

# Title
st.title('Prediksi Symptom Severity')
st.write('Symptom Severity adalah tingkat keparahan suatu gejala yang dialami oleh seseorang. Ini biasanya digunakan dalam dunia medis atau penelitian untuk mengukur seberapa serius atau mengganggu suatu gejala terhadap kehidupan seseorang.')

st.write('Silahkan isi form berikut: ')

col1, col2 = st.columns(2)

# **Input pengguna**
with col1:
    Gender_options = {
        1: "Laki-laki",
        0: "Perempuan"
    }
    Jenis_Kelamin = st.selectbox("Pilih Jenis Kelamin", options=list(Gender_options.keys()), format_func=lambda x: Gender_options[x])

    Age = st.number_input("Masukan umur", value=0)

    MoodScore = st.number_input("Rate mood anda saat ini (Skala 1.0 - 10.0)", value=0.0, max_value=10.00)

    SleepQuality = st.number_input("Rate kualitas tidur anda saat ini (Skala 1.0 - 10.0)", value=0.0, max_value=10.00)

with col2:
    Diagnosis_options = {
        0: "Bipolar Disorder",
        1: "Generalized Anxiety",
        2: "Major Depresive Disorder",
        3: "Panic Disorder"
    }
    Jenis_Diagnosis = st.selectbox("Jenis Diagnosa Mental", options=list(Diagnosis_options.keys()), format_func=lambda x: Diagnosis_options[x])

    Medication_options = {
        0: "Antidepressants",
        1: "Antipsychotics",
        2: "Anxiolytics",
        3: "Benzodiazepines",
        4: "Mood Stabilizers",
        5: "SSRIs"
    }
    Jenis_Pengobatan = st.selectbox("Jenis Pengobatan Mental", options=list(Medication_options.keys()), format_func=lambda x: Medication_options[x])

    TherapyType_options = {
        0: "Cognitive Behavioral Therapy",
        1: "Dialectical Behavioral Therapy",
        2: "Interpersonal Therapy",
        3: "Mindfulness Based Therapy"
    }
    Jenis_Therapy = st.selectbox("Jenis Terapi", options=list(TherapyType_options.keys()), format_func=lambda x: TherapyType_options[x])

# **Prediksi**
if st.button('Prediksi'):
    input_data = np.array([[
        Jenis_Kelamin, 
        Age, 
        Jenis_Diagnosis, 
        Jenis_Pengobatan, 
        MoodScore,  
        SleepQuality,  
        Jenis_Therapy
    ]], dtype=float)

    pred_symptom_severity = model.predict(input_data)

    st.write(f'Hasil Prediksi: {pred_symptom_severity[0]}')
