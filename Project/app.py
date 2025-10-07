import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn

# Load model yang sudah di-training
with open('best_gradientboosting_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.set_page_config(page_title="Daegu Apartment Price Prediction", page_icon="🏙️")

st.title("🏙️ Daegu Apartment Price Prediction")
st.write("Aplikasi ini memprediksi harga jual apartemen di Daegu berdasarkan beberapa karakteristik properti.")

st.sidebar.header("Masukkan Data Apartemen")

# Input form
hallway = st.sidebar.selectbox("Tipe Hallway", ['terraced', 'mixed', 'corridor'])
timesubway = st.sidebar.selectbox("Jarak ke Subway", ['0-5min', '5min~10min', '10min~15min', '15min~20min', 'no_bus_stop_nearby'])
station = st.sidebar.selectbox("Stasiun Terdekat", ['Kyungbuk_uni_hospital', 'Chil-sung-market', 'Bangoge', 'Sin-nam'])
n_fac_etc = st.sidebar.number_input("Jumlah Fasilitas Lainnya (ETC)", min_value=0.0)
n_fac_public = st.sidebar.number_input("Jumlah Fasilitas Kantor Publik", min_value=0.0)
n_school_univ = st.sidebar.number_input("Jumlah Universitas Terdekat", min_value=0.0)
n_parking = st.sidebar.number_input("Jumlah Basement Parking Lot", min_value=0.0)
year_built = st.sidebar.number_input("Tahun Dibangun", min_value=1978, max_value=2025, value=2005)
n_fac_inapt = st.sidebar.number_input("Jumlah Fasilitas di Apartemen", min_value=0)
size = st.sidebar.number_input("Ukuran Apartemen (sqft)", min_value=135, max_value=2500, value=1000)

# Buat dataframe input
input_data = pd.DataFrame({
    'HallwayType': [hallway],
    'TimeToSubway': [timesubway],
    'SubwayStation': [station],
    'N_FacilitiesNearBy(ETC)': [n_fac_etc],
    'N_FacilitiesNearBy(PublicOffice)': [n_fac_public],
    'N_SchoolNearBy(University)': [n_school_univ],
    'N_Parkinglot(Basement)': [n_parking],
    'YearBuilt': [year_built],
    'N_FacilitiesInApt': [n_fac_inapt],
    'Size(sqf)': [size]
})

input_data['AptAge'] = 2025 - input_data['YearBuilt']


# Prediksi
if st.sidebar.button("Prediksi Harga"):
    prediction = model.predict(input_data)
    predicted_price = prediction[0]

    st.subheader("💰 Hasil Prediksi")
    st.write(f"Perkiraan harga apartemen: **₩{predicted_price:,.0f}**")
    st.caption("Model: Gradient Boosting Regressor (Tuned)")

st.markdown("---")
st.markdown("📊 **Catatan:** Model ini memprediksi harga apartemen di Daegu berdasarkan data tahun 1985–2007.")
