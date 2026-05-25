import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib
import sqlite3
from lime.lime_tabular import LimeTabularExplainer
from database import init_db, simpan_riwayat, hapus_riwayat, load_data

# Menyembunyikan tombol Share & GitHub, tapi TETAP memunculkan menu titik tiga
st.markdown("""
    <style>
    /* Sembunyikan seluruh komponen header atas */
    header[data-testid="stHeader"] {
        visibility: hidden;
    }
    /* Paksa tombol titik tiga (Main Menu) agar tetap kelihatan */
    header[data-testid="stHeader"] [data-testid="stMainMenu"] {
        visibility: visible;
    }
    </style>
""", unsafe_allow_html=True)

# =============================
# CONFIG
# =============================
# WAJIB ditaruh di baris pertama setelah import, sebelum ada kode st. lainnya!
st.set_page_config(
    page_title="Sistem Prediksi Status Gizi",
    page_icon="🥗",
    layout="wide",  # <--- Ini kuncinya biar layar otomatis melebar di 100%
    initial_sidebar_state="expanded"
)
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
    'jenis_kelamin',
    'berat_kg',
    'tinggi_cm',
    'lingkar_perut_cm',
    'frekuensi_makan',
    'sarapan',
    'konsumsi_sayur_buah',
    'konsumsi_daging',
    'konsumsi_susu',
    'fast_food',
    'minuman_bersoda',
    'nafsu_makan_rumah',
    'nafsu_makan_sekolah',
    'air_minum',
    'lama_tidur',
    'olahraga',
    'aktivitas_fisik',
    'BMI'
]

X_train = data_train[fitur_model].astype(float).fillna(0)

# =============================================================
# INI DI TARUH DI ATAS (Menggantikan selectbox/radio yang lama)
# =============================================================
tab_dashboard, tab_prediksi, tab_riwayat, tab_tentang = st.tabs([
    "📊 Dashboard", 
    "🤖 Prediksi Gizi", 
    "📜 Riwayat", 
    "ℹ️ Tentang"
])


# =============================
# DASHBOARD (Menggunakan Tab)
# =============================
with tab_dashboard:
    header()
    
    # Ambil data dari database
    df_dash = load_data()
    total_data = len(df_dash) if not df_dash.empty else 0
    
    # 1. Tampilan Metrik Atas (Teks dipersingkat agar TIDAK TERPOTONG di 100%)
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Total Pemeriksaan", f"{total_data} Siswa")
    # Parameter 'help' akan memunculkan info balon balon teks saat di-hover kursor
    col2.metric("🤖 Model Klasifikasi", "SVM", help="Support Vector Machine")
    col3.metric("🟢 Koneksi Database", "Terhubung")
    
    st.markdown("---")
    
    # 2. PANEL GRAFIK MENGGUNAKAN TABS (Solusi agar grafik tidak gepeng di zoom 100%)
    if not df_dash.empty:
        st.write("### 📊 Analisis & Distribusi Data Real-Time")
        
        # Membuat 2 Tab interaktif di dalam Dashboard
        tab_gizi, tab_jk = st.tabs(["🥗 Distribusi Kategori Status Gizi", "🧑‍🤝‍🧑 Proporsi Jenis Kelamin"])
        
        with tab_gizi:
            st.write("#### Grafik Batang Kategori Gizi")
            kolom_prediksi = 'hasil_prediksi' if 'hasil_prediksi' in df_dash.columns else (df_dash.columns[-1] if len(df_dash.columns) > 0 else None)
            
            if kolom_prediksi and kolom_prediksi in df_dash.columns:
                distribusi_gizi = df_dash[kolom_prediksi].value_counts()
                st.bar_chart(distribusi_gizi)
            else:
                st.info("Kolom status gizi belum terdeteksi di database.")
                
        with tab_jk:
            st.write("#### Grafik Batang Jenis Kelamin")
            if 'jenis_kelamin' in df_dash.columns:
                df_jk = df_dash.copy()
                df_jk['jenis_kelamin'] = df_jk['jenis_kelamin'].map({1: "Laki-laki", 0: "Perempuan"}).fillna(df_jk['jenis_kelamin'])
                distribusi_jk = df_jk['jenis_kelamin'].value_counts()
                st.bar_chart(distribusi_jk)
            else:
                st.info("Kolom jenis kelamin tidak ditemukan di database.")
                
        st.markdown("---")
    else:
        st.info("💡 **Informasi Visualisasi:** Grafik distribusi data akan muncul di sini secara otomatis setelah Anda melakukan pengujian prediksi pertama kali!")
        st.markdown("---")
        
    # 3. Bagian Informasi & Fitur Utama (Berdampingan rapi)
    col_info, col_fitur = st.columns([1.3, 1])
    
    with col_info:
        st.write("### 📌 Tentang Aplikasi")
        st.write("""
        Aplikasi ini dikembangkan sebagai sistem cerdas pendukung keputusan untuk memprediksi 
        dan menganalisis **Status Gizi Siswa** secara otomatis berbasis Machine Learning. 
        
        Dengan menggabungkan keunggulan **Model SVM** untuk akurasi klasifikasi tingkat tinggi, 
        serta pendekatan **Explainable AI (LIME)**, aplikasi ini mampu memaparkan transparansi 
        faktor klinis (seperti pola makan, aktivitas fisik, dan riwayat biologis) yang menjadi pemicu 
        utama dari hasil diagnosis gizi siswa.
        """)
        
    with col_fitur:
        st.write("### 🚀 Fitur Utama Sistem")
        st.markdown("👁️‍**Prediksi Status Gizi:** Klasifikasi otomatis ragam kondisi gizi siswa.")
        st.markdown("🔍 **Penjelasan LIME:** Transparansi model guna melihat kontribusi parameter klinis.")
        st.markdown("🗃️ **Manajemen Riwayat:** Perekaman rekam medis digital hasil prediksi beserta fitur hapus data.")
        st.markdown("📥 **Ekspor Laporan:** Unduh laporan menyeluruh dalam format standar `.CSV` untuk kebutuhan Tenaga Kesehatan (Nakes).")


# =============================
# PREDIKSI (Menggunakan Tab)
# =============================
with tab_prediksi:
    header()

    # ==========================================================
    # 🟢 PERUBAHAN 1: INISIALISASI MEMORI SESSION STATE
    # ==========================================================
    if 'sudah_prediksi' not in st.session_state:
        st.session_state['sudah_prediksi'] = False
    if 'saved_pred_svm' not in st.session_state:
        st.session_state['saved_pred_svm'] = None
    if 'saved_status_imt' not in st.session_state:
        st.session_state['saved_status_imt'] = None
    if 'saved_bmi' not in st.session_state:
        st.session_state['saved_bmi'] = 0.0
    if 'saved_warna_status' not in st.session_state:
        st.session_state['saved_warna_status'] = "#ffffff"
    if 'saved_prob' not in st.session_state:
        st.session_state['saved_prob'] = None
    if 'saved_pred_index' not in st.session_state:
        st.session_state['saved_pred_index'] = None
    if 'saved_exp' not in st.session_state:
        st.session_state['saved_exp'] = None
    if 'saved_data_input' not in st.session_state:
        st.session_state['saved_data_input'] = None

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("### 📝 Form Pemeriksaan Kesehatan Siswa")
    
    # Input Dipecah Menjadi Sub-Tab Internal
    tab1, tab2, tab3 = st.tabs(["📏 Data Fisik", "🍔 Pola Makan", "🏃 Aktivitas & Tidur"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            jk = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
            jenis_kelamin = 1 if jk == "Laki-laki" else 0
            berat = st.number_input("Berat badan (kg)", 10.0, 150.0, 40.0)
        with col2:
            tinggi = st.number_input("Tinggi badan (cm)", 80.0, 200.0, 150.0)
            lingkar = st.number_input("Lingkar perut (cm)", 30.0, 150.0, 70.0)

    with tab2:
        col3, col4 = st.columns(2)
        with col3:
            frek_map = {"1 kali": 1, "2 kali": 2, "3 kali": 3, "Lebih dari 3 kali": 4}
            frek = frek_map[st.selectbox("Frekuensi makan sehari", list(frek_map.keys()))]
            
            sarapan = 1 if st.selectbox("Apakah ananda sarapan setiap hari?", ["Tidak", "Ya / Kadang-kadang"]) == "Ya / Kadang-kadang" else 0
            
            opsi_konsumsi = ["Tidak pernah", "Kadang-kadang", "Setiap hari"]
            sayur = opsi_konsumsi.index(st.selectbox("Konsumsi sayur & buah", opsi_konsumsi))
            daging = opsi_konsumsi.index(st.selectbox("Konsumsi daging", opsi_konsumsi))
        with col4:
            susu = opsi_konsumsi.index(st.selectbox("Konsumsi susu", opsi_konsumsi))
            fastfood = opsi_konsumsi.index(st.selectbox("Konsumsi fast food", opsi_konsumsi))
            soda = opsi_konsumsi.index(st.selectbox("Minuman bersoda", opsi_konsumsi))
            
            nafsu_map_app = {
                "Normal (Tidak mengalami mual/tidak nafsu makan)": 0,
                "Kurang (Ya, mengalami mual/tidak nafsu makan)": 1
            }
            nafsu_rumah_tampil = st.selectbox("Nafsu makan di rumah", list(nafsu_map_app.keys()))
            nafsu_rumah = nafsu_map_app[nafsu_rumah_tampil]
            
            nafsu_sekolah_tampil = st.selectbox("Nafsu makan di sekolah", list(nafsu_map_app.keys()))
            nafsu_sekolah = nafsu_map_app[nafsu_sekolah_tampil]

    with tab3:
        col5, col6 = st.columns(2)
        with col5:
            air_map = {
                "Kurang dari 5 gelas per hari": 0,
                "5 sampai dengan 8 gelas per hari": 1,
                "Lebih dari 8 gelas per hari": 2
            }
            air_tampil = st.selectbox("Jumlah air minum (air putih) dalam sehari", list(air_map.keys()))
            air = air_map[air_tampil]
            
            tidur_map = {
                "Kurang dari 6 jam": 0,
                "6 sampai dengan 8 jam": 1,
                "Lebih dari 8 jam": 2
            }
            tidur_tampil = st.selectbox("Lama waktu tidur dalam sehari", list(tidur_map.keys()))
            tidur = tidur_map[tidur_tampil]

        with col6:
            olahraga_map = {
                "0 sampai 1 hari per minggu": 0,
                "2 sampai 4 hari per minggu": 1,
                "5 sampai 7 hari per minggu": 2
            }
            olahraga_tampil = st.selectbox("Berapa hari ananda melakukan olahraga dalam satu minggu?", list(olahraga_map.keys()))
            olahraga = olahraga_map[olahraga_tampil]
            
            aktivitas_map = {
                "Kurang dari 15 menit": 0,
                "15 sampai dengan 30 menit": 1,
                "Lebih dari 30 menit": 2
            }
            aktivitas_tampil = st.selectbox("Aktivitas fisik (olahraga) dalam satu hari", list(aktivitas_map.keys()))
            aktivitas = aktivitas_map[aktivitas_tampil]

    st.markdown("<br>", unsafe_allow_html=True)
    tombol_prediksi = st.button("Mulai Analisis Prediksi", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    if tombol_prediksi:
        bmi = berat / ((tinggi / 100) ** 2)

        data_input = pd.DataFrame([[
            jenis_kelamin, berat, tinggi, lingkar, frek, sarapan,
            sayur, daging, susu, fastfood, soda, nafsu_rumah,
            nafsu_sekolah, air, tidur, olahraga, aktivitas, bmi
        ]], columns=fitur_model).astype(float)

        data_scaled = scaler.transform(data_input)
        pred_svm = model.predict(data_scaled)[0]
        prob = model.predict_proba(data_scaled)[0]

        if bmi < 18.5:
            status_imt = "Kurus"
        elif bmi < 25:
            status_imt = "Normal"
        elif bmi < 30:
            status_imt = "Gemuk"
        else:
            status_imt = "Obesitas"

        if pred_svm == "Kurus":
            warna_status = "#ef4444"
        elif pred_svm == "Normal":
            warna_status = "#22c55e"
        elif pred_svm == "Gemuk":
            warna_status = "#eab308"
        else:
            warna_status = "#ef4444"

        explainer = LimeTabularExplainer(
            X_train.values,
            feature_names=fitur_model,
            class_names=model.classes_,
            mode="classification",
            discretize_continuous=False
        )

        pred_index = list(model.classes_).index(pred_svm)
        exp = explainer.explain_instance(
            data_input.values[0],
            lambda x: model.predict_proba(scaler.transform(x)),
            labels=[pred_index]
        )

        st.session_state['sudah_prediksi'] = True
        st.session_state['saved_pred_svm'] = pred_svm
        st.session_state['saved_status_imt'] = status_imt
        st.session_state['saved_bmi'] = bmi
        st.session_state['saved_warna_status'] = warna_status
        st.session_state['saved_prob'] = prob
        st.session_state['saved_pred_index'] = pred_index
        st.session_state['saved_exp'] = exp
        st.session_state['saved_data_input'] = data_input

        simpan_riwayat(
            jenis_kelamin, berat, tinggi, lingkar, 
            frek, sarapan, sayur, daging, susu, fastfood, soda, nafsu_rumah, nafsu_sekolah, 
            air, tidur, olahraga, aktivitas, 
            bmi, pred_svm
        )
        
        st.success("✅ Data pemeriksaan lengkap dan pola hidup berhasil disimpan ke database!")

    if st.session_state['sudah_prediksi']:
        pred_svm = st.session_state['saved_pred_svm']
        status_imt = st.session_state['saved_status_imt']
        bmi = st.session_state['saved_bmi']
        warna_status = st.session_state['saved_warna_status']
        prob = st.session_state['saved_prob']
        pred_index = st.session_state['saved_pred_index']
        exp = st.session_state['saved_exp']
        data_input = st.session_state['saved_data_input']

        col_res1, col_res2 = st.columns([1, 1])

        with col_res1:
            st.markdown(f"""
            <div style='
            padding:25px;
            border-radius:12px;
            background:#111827;
            text-align:center;
            border: 2px solid {warna_status};
            height: 100%;
            '>
            <p style='color:#9ca3af; margin:0; font-size:14px; font-weight:bold;'>🤖 PREDIKSI MODEL AI (SVM)</p>
            <h1 style='color:{warna_status}; margin:5px 0 15px 0; font-size:40px;'>{pred_svm}</h1>
            
            <hr style='border-color:#374151; margin:15px 0;'>
            
            <p style='color:#9ca3af; margin:0; font-size:13px;'>📊 REFERENSI IMT STANDAR</p>
            <p style='color:white; margin:5px 0 0 0; font-size:16px;'>Kategori IMT: <b>{status_imt}</b></p>
            <p style='color:#9ca3af; margin:0; font-size:14px;'>Nilai BMI: {bmi:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        with col_res2:
            st.markdown('<div class="card" style="height:100%;">', unsafe_allow_html=True)
            st.write("### 🤖 Validasi Probabilitas Model SVM")
            
            df_prob = pd.DataFrame({
                "Kategori Kelas": model.classes_,
                "Tingkat Keyakinan Model": [f"{p*100:.2f}%" for p in prob]
            })
            st.dataframe(df_prob, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔍 Penjelasan Faktor Klinis (Analisis LIME)")
        
        def terjemahkan_fitur_ke_bahasa_awam(teks):
            teks_lower = teks.lower()
            try:
                val = data_input[teks].values[0]
            except:
                val = None

            is_positive = False
            if val is not None:
                if val == 1 or val == 1.0 or str(val).lower() in ['1', '1.0', 'ya', 'rutin', 'sering', 'laki-laki', 'aktif', 'cukup', 'baik', 'teratur']:
                    is_positive = True

            if 'jenis_kelamin' in teks_lower:
                return "Siswa Laki-laki" if is_positive else "Siswa Perempuan"
            elif 'sarapan' in teks_lower:
                return "Rutin Sarapan Pagi" if is_positive else "Jarang/Tidak Pernah Sarapan"
            elif 'fast_food' in teks_lower or 'fastfood' in teks_lower:
                return "Sering Makanan Fast Food" if is_positive else "Jarang Makanan Fast Food"
            elif 'minuman_bersoda' in teks_lower or 'soda' in teks_lower:
                return "Sering Minum Bersoda/Manis" if is_positive else "Jarang Minum Bersoda"
            elif 'susu' in teks_lower:
                return "Rutin Minum Susu" if is_positive else "Jarang Minum Susu"
            elif 'sayur' in teks_lower:
                return "Cukup Konsumsi Sayur & Buah" if is_positive else "Kurang Konsumsi Sayur & Buah"
            elif 'daging' in teks_lower:
                return "Sering Konsumsi Lauk Berprotein (Daging)" if is_positive else "Jarang Cult Daging"
            elif 'olahraga' in teks_lower:
                return "Aktif Berolahraga" if is_positive else "Kurang/Jarang Berolahraga"
            elif 'tidur' in teks_lower:
                return "Durasi Tidur Cukup (≥6-8 Jam)" if is_positive else "Kurang Istirahat / Bergadang"
            elif 'air' in teks_lower:
                return "Kecukupan Air Putih Terpenuhi" if is_positive else "Kurang Minum Air Putih"
            elif 'frek' in teks_lower:
                return "Frekuensi Makan Teratur" if is_positive else "Frekuensi Makan Tidak Teratur"
            elif 'nafsu_rumah' in teks_lower:
                return "Nafsu Makan Rumah Baik" if is_positive else "Nafsu Makan Rumah Menurun"
            elif 'nafsu_sekolah' in teks_lower:
                return "Nafsu Makan Sekolah Baik" if is_positive else "Nafsu Makan Sekolah Menurun"
            elif 'aktivitas' in teks_lower:
                return "Aktivitas Fisik Harian Tinggi" if is_positive else "Aktivitas Fisik Rendah (Sedenter)"
            elif 'bmi' in teks_lower:
                return "Indeks Massa Tubuh (IMT) Siswa"
            elif 'berat' in teks_lower:
                return "Faktor Berat Badan (kg)"
            elif 'tinggi' in teks_lower:
                return "Faktor Tinggi Badan (cm)"
            elif 'lingkar' in teks_lower:
                return "Faktor Lingkar Perut (cm)"
            
            return teks

        col_lime1, col_lime2 = st.columns([4, 3])
        
        with col_lime1:
            list_eksplorasi = exp.as_list(label=pred_index)
            df_lime = pd.DataFrame(list_eksplorasi, columns=['Fitur_Mentah', 'Kekuatan'])
            df_lime['Faktor Pola Hidup'] = df_lime['Fitur_Mentah'].apply(terjemahkan_fitur_ke_bahasa_awam)
            df_lime['Pengaruh'] = df_lime['Kekuatan'].apply(lambda x: 'Mendukung Diagnosis' if x > 0 else 'Menahan/Memperkecil')
            
            fig_lime = px.bar(
                df_lime, x='Kekuatan', y='Faktor Pola Hidup', orientation='h', color='Pengaruh',
                color_discrete_map={'Mendukung Diagnosis': '#10b981', 'Menahan/Memperkecil': '#ef4444'}, text_auto='.4f'
            )
            
            fig_lime.update_layout(
                xaxis_title="← Menahan Peluang | Kekuatan Pengaruh Fitur | Mendukung Peluang →",
                yaxis_title=None, legend_title="Keterangan Pengaruh:", height=450,
                margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            fig_lime.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='#64748b', gridcolor='#334155')
            st.plotly_chart(fig_lime, use_container_width=True)
            
        with col_lime2:
            st.write("📊 **Interpretasi Teks Klinis:**")
            for f, v in exp.as_list(label=pred_index):
                f_human = terjemahkan_fitur_ke_bahasa_awam(f)
                if v > 0:
                    st.success(f"**{f_human}** menjadi faktor pendorong utama model mendiagnosis status gizi ini.")
                else:
                    st.error(f"**{f_human}** menghambat peluang model untuk berpindah ke status gizi lainnya.")

        st.markdown('</div>', unsafe_allow_html=True)


# =============================
# RIWAYAT (Menggunakan Tab)
# =============================
with tab_riwayat:
    header()
    st.write("## 🗃️ Manajemen Riwayat Prediksi Status Gizi")
    
    df_riwayat = load_data()
    
    if not df_riwayat.empty:
        df_display = df_riwayat.copy()
        
        if 'jenis_kelamin' in df_display.columns:
            df_display['jenis_kelamin'] = df_display['jenis_kelamin'].map({1: "Laki-laki", 0: "Perempuan"})
            
        if 'frekuensi_makan' in df_display.columns:
            df_display['frekuensi_makan'] = df_display['frekuensi_makan'].map({1: "1 kali", 2: "2 kali", 3: "3 kali", 4: "Lebih dari 3 kali"})
            
        if 'sarapan' in df_display.columns:
            df_display['sarapan'] = df_display['sarapan'].map({1: "Ya / Kadang-kadang", 0: "Tidak"})
            
        map_konsumsi = {0: "Tidak pernah", 1: "Kadang-kadang", 2: "Setiap hari"}
        kolom_konsumsi = ['konsumsi_sayur_buah', 'konsumsi_daging', 'konsumsi_susu', 'fast_food', 'minuman_bersoda']
        for col in kolom_konsumsi:
            if col in df_display.columns:
                df_display[col] = df_display[col].map(map_konsumsi)
                
        map_nafsu = {0: "Normal", 1: "Kurang"}
        for col in ['nafsu_makan_rumah', 'nafsu_makan_sekolah']:
            if col in df_display.columns:
                df_display[col] = df_display[col].map(map_nafsu)

        if 'jumlah_air' in df_display.columns:
            df_display['jumlah_air'] = df_display['jumlah_air'].map({0: "Kurang dari 5 gelas", 1: "5-8 gelas", 2: "Lebih dari 8 gelas"})
            
        if 'lama_tidur' in df_display.columns:
            df_display['lama_tidur'] = df_display['lama_tidur'].map({0: "Kurang dari 6 jam", 1: "6-8 jam", 2: "Lebih dari 8 jam"})
            
        if 'olahraga' in df_display.columns:
            df_display['olahraga'] = df_display['olahraga'].map({0: "0-1 hari/minggu", 1: "2-4 hari/minggu", 2: "5-7 hari/minggu"})
            
        if 'aktivitas_fisik' in df_display.columns:
            df_display['aktivitas_fisik'] = df_display['aktivitas_fisik'].map({0: "Kurang dari 15 menit", 1: "15-30 menit", 2: "Lebih dari 30 menit"})

        st.write(f"Total data terekam: `{len(df_display)}` pemeriksaan.")
        st.dataframe(df_display, use_container_width=True)
        
        csv = df_riwayat.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Unduh Laporan Riwayat (.CSV)", data=csv, file_name="riwayat_gizi.csv", mime="text/csv")
        
        st.markdown("---")
        
        st.write("### 🗑️ Panel Hapus Riwayat Data")
        list_id = df_riwayat['id'].tolist()
        col_select, col_btn = st.columns([1, 2])
        
        with col_select:
            id_terpilih = st.selectbox("Pilih ID Data:", list_id)
            
        with col_btn:
            st.write("") 
            st.write("") 
            if st.button("🔴 Hapus Data Terpilih", use_container_width=True):
                hapus_riwayat(id_terpilih)
                st.success(f"✅ Data dengan ID {id_terpilih} berhasil dihapus!")
                st.rerun() 
                
    else:
        st.info("Belum ada data riwayat pemeriksaan.")


# =============================
# TENTANG (Menggunakan Tab)
# =============================
with tab_tentang:
    header()
    
    st.write("## ℹ️ Informasi Sistem Utama")
    st.write("Aplikasi ini dikembangkan sebagai implementasi riset Tugas Akhir untuk mendeteksi dini kelainan gizi populasi anak sekolah secara cepat, akurat, dan transparan.")
    
    st.markdown("---")
    
    st.write("### 🧠 Pendekatan Metodologi")
    col_svm, col_lime = st.columns(2)
    
    with col_svm:
        st.markdown("""
        <div style="background-color: #1e293b; padding: 20px; border-radius: 10px; border-left: 5px solid #3b82f6; height: 100%;">
            <h4 style="margin-top:0; color: #3b82f6;">🤖 Support Vector Machine (SVM)</h4>
            <p style="font-size: 14px; color: #cbd5e1;">
                Bertindak sebagai <b>core engine classifier</b> yang bertanggung jawab menghasilkan 
                prediksi klasifikasi multi-kelas (Kurus, Normal, Gemuk, Obesitas) secara optimal. 
                SVM dipilih karena kemampuannya yang sangat baik dalam menangani dataset medis/klinis 
                dengan pola yang kompleks.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_lime:
        st.markdown("""
        <div style="background-color: #1e293b; padding: 20px; border-radius: 10px; border-left: 5px solid #10b981; height: 100%;">
            <h4 style="margin-top:0; color: #10b981;">🔍 Explainable AI (LIME)</h4>
            <p style="font-size: 14px; color: #cbd5e1;">
                Bertindak sebagai <b>pemberi transparansi</b> (Local Interpretable Model-agnostic Explanations). 
                LIME memecah sifat <i>black-box</i> pada AI dengan memaparkan kontribusi fitur/parameter klinis 
                lokal per siswa, sehingga hasil prediksi dapat dipertanggungjawabkan secara medis.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("🎯 **Tujuan Riset:** Membantu Tenaga Kesehatan (Nakes) maupun Lingkungan Sekolah dalam melakukan *screening* gizi awal agar intervensi kesehatan dapat dilakukan lebih dini.")
    st.markdown("---")
