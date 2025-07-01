import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

@st.cache_resource
def load_model():
    df = pd.read_csv("data/asuransi_kebakaran_fraud.csv")
    le_loc = LabelEncoder()
    df['property_location'] = le_loc.fit_transform(df['property_location'])
    le_const = LabelEncoder()
    df['construction_type'] = le_const.fit_transform(df['construction_type'])
    df['fire_alarm_installed'] = df['fire_alarm_installed'].map({'Ya': 1, 'Tidak': 0})
    X = df.drop(['policy_id', 'is_fraud'], axis=1)
    y = df['is_fraud']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, le_loc, le_const

model, scaler, le_loc, le_const = load_model()

st.title("🔥 Deteksi Fraud Asuransi Kebakaran")
st.write("Masukkan data klaim untuk mendeteksi kemungkinan fraud.")

age = st.number_input("Usia Nasabah", min_value=18, max_value=100, value=30)
property_value = st.number_input("Nilai Properti (juta IDR)", min_value=100, max_value=10000, value=1000)
premium = st.number_input("Premi Dibayar (juta IDR)", min_value=1.0, max_value=1000.0, value=50.0)
location = st.selectbox("Lokasi Properti", ['Urban', 'Suburban', 'Rural'])
construction = st.selectbox("Tipe Konstruksi", ['Beton', 'Kayu', 'Campuran'])
alarm = st.radio("Ada Alarm Kebakaran?", ['Ya', 'Tidak'])
claim = st.number_input("Jumlah Klaim (juta IDR)", min_value=0.0, max_value=10000.0, value=0.0)

if st.button("Prediksi Fraud"):
    input_data = pd.DataFrame([[
        age,
        property_value,
        premium,
        le_loc.transform([location])[0],
        le_const.transform([construction])[0],
        1 if alarm == 'Ya' else 0,
        claim
    ]], columns=[
        'customer_age', 'property_value', 'premium_amount',
        'property_location', 'construction_type',
        'fire_alarm_installed', 'claim_amount'
    ])
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]
    if pred == 1:
        st.error(f"❌ Klaim dicurigai sebagai **FRAUD** dengan probabilitas {proba:.2%}")
    else:
        st.success(f"✅ Klaim valid. Probabilitas fraud: {proba:.2%}")
