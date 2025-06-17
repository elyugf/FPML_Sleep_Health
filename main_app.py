import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

# ========== TRAINING & CACHING ==========
@st.cache_resource
def train_models():
    df = pd.read_csv("Sleep_Health.csv")
    # Ganti label BMI dari "Normal" menjadi "Underweight"
    df['BMI Category'] = df['BMI Category'].replace({
        'Normal': 'Underweight',
        'Normal Weight': 'Normal'
    })


    X = df[['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Physical Activity Level',
            'Stress Level', 'BMI Category', 'Heart Rate', 'Daily Steps',
            'Blood Pressure (systolic)', 'Blood Pressure (diastolic)']]

    y1 = df['Sleep Disorder']
    y2 = df['Quality of Sleep']

    label_encoder = LabelEncoder()
    y1_encoded = label_encoder.fit_transform(y1)

    unique_categories = {
        'gender': sorted(df['Gender'].dropna().unique().tolist()),
        'occupation': sorted(df['Occupation'].dropna().unique().tolist()),
        'bmi': ['Underweight', 'Normal', 'Overweight', 'Obese']
    }

    X['Gender'] = X['Gender'].astype(str)
    X['Occupation'] = X['Occupation'].astype(str)
    X['BMI Category'] = X['BMI Category'].astype(str)
    X = pd.get_dummies(X)

    X_train_cls, _, y_train_cls, _ = train_test_split(X, y1_encoded, test_size=0.2, random_state=42)
    X_train_reg, _, y_train_reg, _ = train_test_split(X, y2, test_size=0.2, random_state=42)

    clf_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    clf_model.fit(X_train_cls, y_train_cls)

    reg_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    reg_model.fit(X_train_reg, y_train_reg)

    conf_matrix = confusion_matrix(y_train_cls, clf_model.predict(X_train_cls))
    report_dict = classification_report(y_train_cls, clf_model.predict(X_train_cls), output_dict=True)

    return clf_model, reg_model, label_encoder, unique_categories, conf_matrix, report_dict, X.columns.tolist()

# ========== LOAD ATAU LATIH ==========
try:
    clf_model = joblib.load('clf_model.pkl')
    reg_model = joblib.load('reg_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    unique_categories = joblib.load('unique_categories.pkl')
    conf_matrix = joblib.load('conf_matrix.pkl')
    report_dict = joblib.load('classification_report.pkl')
    feature_names = joblib.load('feature_names.pkl')
except:
    clf_model, reg_model, label_encoder, unique_categories, conf_matrix, report_dict, feature_names = train_models()
    joblib.dump(clf_model, 'clf_model.pkl')
    joblib.dump(reg_model, 'reg_model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    joblib.dump(unique_categories, 'unique_categories.pkl')
    joblib.dump(conf_matrix, 'conf_matrix.pkl')
    joblib.dump(report_dict, 'classification_report.pkl')
    joblib.dump(feature_names, 'feature_names.pkl')

# ========== KONFIGURASI HALAMAN ==========
st.set_page_config(page_title="Prediksi Tidur", layout="centered", page_icon="🌙")

# Tambahkan gambar ilustrasi tidur
image = Image.open("Tidur.jpg")
st.image(image, use_column_width=True)

# Judul dan Subjudul
st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>🌙 Sleep Health Analyzer</h1>
    <h4 style='text-align: center; color: #555555;'>Prediksi Gangguan dan Kualitas Tidur</h4>
""", unsafe_allow_html=True)

# =================== FORM INPUT ===================
gender_options = unique_categories['gender']
occupation_options = unique_categories['occupation']
bmi_options = ['Underweight', 'Normal', 'Overweight', 'Obese']


col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Jenis Kelamin", gender_options)
    age = st.number_input("Usia", 1, 90, 30)
    occupation = st.selectbox("Pekerjaan", occupation_options)
    sleep_duration = st.number_input("Durasi Tidur (Jam)", 0.0, 10.0, 7.0, 0.1)
    physical_activity_level = st.slider("Aktivitas Fisik (Menit/Hari)", 1, 100, 50)
    stress_level = st.slider("Tingkat Stres (1-10)", 1, 10, 5)

with col2:
    st.markdown("**Kategori BMI**", unsafe_allow_html=True)
    st.markdown("""
    <ul style='margin-top: -10px; margin-bottom: 5px;'>
      <li>Underweight: ≤ 18.49 kg/m²</li>
      <li>Normal: 18.5–24.9 kg/m²</li>
      <li>Overweight: > 25–27 kg/m²</li>
      <li>Obese: > 27 kg/m²</li>
    </ul>
    """, unsafe_allow_html=True)
    bmi_category = st.selectbox("", bmi_options)
    systolic_bp = st.number_input("Tekanan Darah Sistolik", 80, 200, 120)
    diastolic_bp = st.number_input("Tekanan Darah Diastolik", 40, 150, 80)
    heart_rate = st.number_input("Detak Jantung (Beats per Minute)", 40, 120, 70)
    daily_steps = st.number_input("Langkah Harian", 1000, 15000, 5000)

# =================== PREDIKSI ===================
if st.button("🌙 Lihat Hasil Analisis Tidur"):
    input_dict = {
        'Gender': [gender],
        'Age': [age],
        'Occupation': [occupation],
        'Sleep Duration': [sleep_duration],
        'Physical Activity Level': [physical_activity_level],
        'Stress Level': [stress_level],
        'BMI Category': [bmi_category],
        'Heart Rate': [heart_rate],
        'Daily Steps': [daily_steps],
        'Blood Pressure (systolic)': [systolic_bp],
        'Blood Pressure (diastolic)': [diastolic_bp]
    }

    input_df = pd.DataFrame(input_dict)
    input_encoded = pd.get_dummies(input_df)

    # Pastikan semua fitur ada
    missing_cols = set(feature_names) - set(input_encoded.columns)
    for col in missing_cols:
        input_encoded[col] = 0
    input_encoded = input_encoded[feature_names]

    try:
        pred_cls = clf_model.predict(input_encoded)[0]
        pred_label = label_encoder.inverse_transform([pred_cls])[0]
        pred_quality = reg_model.predict(input_encoded)[0]

        if pred_label != "None" and pred_quality > 6:
            pred_quality = 6.0

        # HASIL PREDIKSI
        # st.markdown("Hasil Prediksi Anda:")
        st.markdown("""
        <h2 style="font-size: 30px; font-weight: bold; margin-top: 20px;">Hasil Prediksi Anda:</h2>
        <div style="background-color: #E0ECF8; padding: 20px; border-radius: 10px;">
            <p style='font-size: 18px;'>🧠 <strong>Gangguan Tidur:</strong> <span style='color: #4B8BBE;'>{}</span></p>
            <p style='font-size: 18px;'>🛏 <strong>Kualitas Tidur:</strong> <span style='color: #4B8BBE;'>{:.2f} / 10</span></p>
        </div>
        """.format(pred_label, pred_quality), unsafe_allow_html=True)

        # DONUT CHART - Prediksi Kualitas Tidur
        fig, ax = plt.subplots(figsize=(1.8, 1.8))

        quality_pct = pred_quality * 10  # misalnya 7.5 => 75%

        # Donut chart (pakai wedgeprops untuk bikin bolong tengah)
        ax.pie(
            [quality_pct, 100 - quality_pct],
            colors=['#4B8BBE', '#D3D3D3'],
            startangle=90,
            counterclock=False,
            labels=['', ''],  # kosongkan label
            wedgeprops={'width': 0.4}
        )

        # Teks persentase di tengah donat
        ax.text(0, 0, f"{quality_pct:.0f}%", ha='center', va='center', fontsize=10, fontweight='bold', color='#4B8BBE')

        st.markdown("### 📊 Prediksi Kualitas Tidur")
        st.pyplot(fig)



        # INTERPRETASI
        if pred_quality >= 8:
            st.success("🟢 Kualitas Tidur Anda Sangat Baik.")
        elif pred_quality >= 6:
            st.info("🔵 Kualitas Tidur Anda Cukup Baik, bisa lebih optimal.")
        elif pred_quality >= 4:
            st.warning("🟠 Kualitas Tidur Anda Kurang. Coba evaluasi gaya hidup Anda.")
        else:
            st.error("🔴 Kualitas Tidur Anda Sangat Rendah. Disarankan konsultasi ke ahli.")

        # SARAN GANGGUAN TIDUR
        if pred_label == "Sleep Apnea":
            st.warning("⚠ Kemungkinan Sleep Apnea. Disarankan konsultasi ke dokter.")
        elif pred_label == "Insomnia":
            st.info("🧘 Coba teknik relaksasi dan perbaiki kebiasaan tidur.")
        else:
            if pred_quality >= 8:
                st.balloons()
            st.success("🎉 Anda tampaknya tidak memiliki gangguan tidur.")
            st.balloons()

    except Exception as e:
        st.error(f"Kesalahan saat prediksi: {e}")
