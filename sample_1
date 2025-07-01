import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# ---------- Load atau Train Model ----------
@st.cache_resource
def load_model():
    # Load data
    df = pd.read_csv("asuransi_kebakaran_fraud.csv")

    # Encode kategori
    le_loc = LabelEncoder()
    df['property_location'] = le_loc.fit_transform(df['property_location'])
    le_const = LabelEncoder()
    df['construction_type'] = le_const.fit_transform(df['construction_type'])
    df['fire_alarm_installed'] = df['fire_alarm_installed'].map({'Ya': 1, 'Tidak': 0})

    # Fitur dan label
    X = df.drop(['policy_id', 'is_fraud'], axis=1)
    y = df['is_fraud']

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler, le_loc, le_const

model, scaler, le_loc, le_const = load_model()

# ---------- Streamlit UI ----------
st.title("🔥 Deteksi Fraud Asuransi Kebakaran")
st.markdown("Masukkan data klaim untuk mendeteksi apakah ada potensi fraud atau tidak.")

# Form input
age = st.number_input("Usia Nasabah", min_value=18, max_value=100, value=30)
property_value = st.number_input("Nilai Properti (juta IDR)", min_value=100, max_value=10000, value=1000)
premium = st.number_input("Jumlah Premi (juta IDR)", min_value=1.0, max_value=1000.0, value=50.0)

location = st.selectbox("Lokasi Properti", ['Urban', 'Suburban', 'Rural'])
construction = st.selectbox("Tipe Konstruksi", ['Beton', 'Kayu', 'Campuran'])
alarm = st.radio("Alarm Kebakaran Terpasang?", ['Ya', 'Tidak'])

claim_amount = st.number_input("Jumlah Klaim (juta IDR)", min_value=0.0, max_value=10000.0, value=0.0)

if st.button("Prediksi Fraud"):
    # Encode input
    input_data = pd.DataFrame([[
        age,
        property_value,
        premium,
        le_loc.transform([location])[0],
        le_const.transform([construction])[0],
        1 if alarm == 'Ya' else 0,
        claim_amount
    ]], columns=[
        'customer_age', 'property_value', 'premium_amount',
        'property_location', 'construction_type', 'fire_alarm_installed',
        'claim_amount'
    ])

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediksi
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    # Output
    if pred == 1:
        st.error(f"⚠️ Prediksi: **FRAUD** dengan probabilitas {proba:.2%}")
    else:
        st.success(f"✅ Prediksi: **Valid** dengan probabilitas fraud {proba:.2%}")
