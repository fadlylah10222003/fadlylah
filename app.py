import streamlit as st
import pandas as pd
import numpy as np
import joblib
from lime.lime_tabular import LimeTabularExplainer
from database import init_db, simpan, load_data

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="Prediksi Gizi", layout="wide")

# =============================
# STYLE GLOBAL
# =============================
st.markdown("""
<style>
.card {
    background-color: #111827;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# =============================
# HEADER (SEMUA HALAMAN)
# =============================
def header():
    st.markdown("""
    <div style='
    background: linear-gradient(90deg, #1e3c72, #2a5298);
    padding: 20px;
    border-radius: 12px;
    color: white;
    margin-bottom: 20px;
    '>
    <h2>📊 Sistem Prediksi Status Gizi</h2>
    <p>Menggunakan SVM dan LIME untuk analisis kesehatan siswa</p>
    </div>
    """, unsafe_allow_html=True)

# =============================
# INIT
# =============================
init_db()

model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

data_train = pd.read_csv("data_final_ml.csv")
data_train.columns = data_train.columns.str.strip()

fitur_model = [
    'jenis_kelamin','berat_kg','tinggi_cm','lingkar_perut_cm',
    'frekuensi_makan','sarapan','konsumsi_sayur_buah','konsumsi_daging',
    'konsumsi_susu','fast_food',
    'Bagaimana dengan minuman kemasan berasa/ bersoda (selain susu) yang ananda konsumsi?',
    'nafsu_makan_rumah','nafsu_makan_sekolah','air_minum','lama_tidur'
]

X_train = data_train[fitur_model].astype(float).fillna(0)

menu = st.sidebar.selectbox("Menu", ["Dashboard", "Prediksi", "Riwayat", "Tentang"])

# =============================
# DASHBOARD
# =============================
if menu == "Dashboard":

    header()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Data", len(load_data()))
    col2.metric("Model", "SVM")
    col3.metric("Status", "Aktif")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("### 📌 Tentang Aplikasi")
    st.write("Aplikasi ini digunakan untuk memprediksi status gizi siswa secara otomatis.")

    st.write("### 🚀 Fitur")
    st.write("✔ Prediksi status gizi")
    st.write("✔ Penjelasan LIME")
    st.write("✔ Riwayat data")

    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# PREDIKSI
# =============================
elif menu == "Prediksi":

    header()

    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        jk = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        jenis_kelamin = 1 if jk == "Laki-laki" else 0

        berat = st.number_input("Berat badan (kg)", 10.0,150.0,40.0)
        tinggi = st.number_input("Tinggi badan (cm)", 80.0,200.0,150.0)

    with col2:
        lingkar = st.number_input("Lingkar perut (cm)", 30.0,150.0,70.0)

        frek_map = {"1 kali":1,"2 kali":2,"3 kali":3,"Lebih dari 3 kali":4}
        frek = frek_map[st.selectbox("Frekuensi makan", list(frek_map.keys()))]

    sarapan = 1 if st.selectbox("Sarapan setiap hari?", ["Tidak","Ya"]) == "Ya" else 0

    opsi = ["Tidak pernah","Kadang","Sering"]
    sayur = opsi.index(st.selectbox("Sayur & buah", opsi))
    daging = opsi.index(st.selectbox("Daging", opsi))
    susu = opsi.index(st.selectbox("Susu", opsi))
    fastfood = opsi.index(st.selectbox("Fast food", opsi))
    soda = opsi.index(st.selectbox("Minuman bersoda", opsi))

    opsi2 = ["Kurang","Normal"]
    nafsu_rumah = opsi2.index(st.selectbox("Nafsu makan di rumah", opsi2))
    nafsu_sekolah = opsi2.index(st.selectbox("Nafsu makan di sekolah", opsi2))

    air = ["Kurang","Cukup","Banyak"].index(st.selectbox("Air minum", ["Kurang","Cukup","Banyak"]))
    tidur = ["<6 jam","6-8 jam",">8 jam"].index(st.selectbox("Lama tidur", ["<6 jam","6-8 jam",">8 jam"]))

    if st.button("Prediksi"):

        data_input = pd.DataFrame([[jenis_kelamin,berat,tinggi,lingkar,frek,sarapan,
        sayur,daging,susu,fastfood,soda,nafsu_rumah,nafsu_sekolah,air,tidur]],
        columns=fitur_model).astype(float)

        data_scaled = scaler.transform(data_input)

        pred_svm = model.predict(data_scaled)[0]
        prob = model.predict_proba(data_scaled)[0]

        bmi = berat / ((tinggi/100)**2)

        if bmi < 18.5:
            pred = "Kurus"
        elif bmi < 25:
            pred = "Normal"
        elif bmi < 30:
            pred = "Gemuk"
        else:
            pred = "Obesitas"

        st.markdown('</div>', unsafe_allow_html=True)

        # =============================
        # HASIL UTAMA (BESAR)
        # =============================
        st.markdown(f"""
        <div style='
        padding:20px;
        border-radius:12px;
        background:#111827;
        text-align:center;
        margin-bottom:20px;
        '>
        <h1 style='color:#22c55e;'>Status: {pred}</h1>
        <h3 style='color:white;'>BMI: {bmi:.2f}</h3>
        </div>
        """, unsafe_allow_html=True)

        # =============================
        # SVM
        # =============================
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("### 🤖 Hasil Model SVM")

        st.write(f"Hasil SVM: {pred_svm}")

        st.dataframe(pd.DataFrame({
            "Kelas": model.classes_,
            "Probabilitas": prob
        }))

        st.markdown('</div>', unsafe_allow_html=True)

        # =============================
        # LIME
        # =============================
        explainer = LimeTabularExplainer(
            X_train.values,
            feature_names=fitur_model,
            class_names=model.classes_,
            mode="classification",
            discretize_continuous=False
        )

        exp = explainer.explain_instance(
            data_input.values[0],
            lambda x: model.predict_proba(scaler.transform(x))
        )

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("🔍 Penjelasan Model (LIME)")
        st.pyplot(exp.as_pyplot_figure())

        st.subheader("🧠 Penjelasan Sederhana")

        for f, v in exp.as_list():
            if v > 0:
                st.write(f"✔ {f} meningkatkan kemungkinan hasil")
            else:
                st.write(f"✖ {f} menurunkan kemungkinan hasil")

        st.markdown('</div>', unsafe_allow_html=True)

        simpan(berat, tinggi, lingkar, pred)

# =============================
# RIWAYAT
# =============================
elif menu == "Riwayat":

    header()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("### 📂 Riwayat Prediksi")

    df = pd.DataFrame(load_data(), columns=["ID","Berat","Tinggi","Lingkar","Hasil"])
    st.dataframe(df)

    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# TENTANG
# =============================
elif menu == "Tentang":

    header()

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.write("### ℹ️ Tentang Aplikasi")
    st.write("""
    Aplikasi ini menggunakan:
    - Support Vector Machine (SVM)
    - LIME untuk interpretasi
    
    Tujuan:
    Membantu memahami status gizi secara mudah dan interaktif.
    """)

    st.markdown('</div>', unsafe_allow_html=True)