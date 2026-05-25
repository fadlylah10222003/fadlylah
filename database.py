import sqlite3
from datetime import datetime
import pandas as pd
import streamlit as st

def load_data():
    # Sesuaikan "riwayat_gizi.db" dengan nama file database yang kamu buat di init_db()
    conn = sqlite3.connect("riwayat_gizi.db") 
    
    try:
        # Sesuaikan "riwayat" dengan nama tabel yang kamu gunakan di database kamu
        df = pd.read_sql_query("SELECT * FROM riwayat", conn)
    except Exception as e:
        # Jika tabel belum terbentuk/error, return DataFrame kosong agar dashboard tidak crash
        df = pd.DataFrame()
    finally:
        conn.close()
        
    return df
# 1. FUNGSI INISIALISASI DATABASE (Membuat tabel jika belum ada)
def init_db():
    conn = sqlite3.connect("riwayat_gizi.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS riwayat (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tanggal TEXT,
            jenis_kelamin INTEGER,
            berat_kg REAL,
            tinggi_cm REAL,
            lingkar_perut_cm REAL,
            frekuensi_makan INTEGER,
            sarapan INTEGER,
            konsumsi_sayur_buah INTEGER,
            konsumsi_daging INTEGER,
            konsumsi_susu INTEGER,
            fast_food INTEGER,
            minuman_bersoda INTEGER,
            nafsu_makan_rumah INTEGER,
            nafsu_makan_sekolah INTEGER,
            air_minum INTEGER,
            lama_tidur INTEGER,
            olahraga INTEGER,
            aktivitas_fisik INTEGER,
            bmi REAL,
            hasil_prediksi TEXT
        )
    """)
    conn.commit()
    conn.close()

# Jalankan init_db di awal agar tabel selalu siap
init_db()

# 2. FUNGSI SIMPAN (Menampung semua 19 parameter pola hidup)
def simpan_riwayat(jenis_kelamin, berat, tinggi, lingkar, frek, sarapan, sayur, daging, susu, fastfood, soda, nafsu_rumah, nafsu_sekolah, air, tidur, olahraga, aktivitas, bmi, hasil_prediksi):
    conn = sqlite3.connect("riwayat_gizi.db")
    cursor = conn.cursor()
    tanggal_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute("""
        INSERT INTO riwayat (
            tanggal, jenis_kelamin, berat_kg, tinggi_cm, lingkar_perut_cm,
            frekuensi_makan, sarapan, konsumsi_sayur_buah, konsumsi_daging,
            konsumsi_susu, fast_food, minuman_bersoda, nafsu_makan_rumah,
            nafsu_makan_sekolah, air_minum, lama_tidur, olahraga, aktivitas_fisik,
            bmi, hasil_prediksi
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        tanggal_sekarang, jenis_kelamin, berat, tinggi, lingkar,
        frek, sarapan, sayur, daging, susu, fastfood, soda, nafsu_rumah, nafsu_sekolah,
        air, tidur, olahraga, aktivitas, bmi, hasil_prediksi
    ))
    conn.commit()
    conn.close()

# 3. FUNGSI HAPUS DATA (Fitur Manajemen Data baru)
def hapus_riwayat(id_data):
    conn = sqlite3.connect("riwayat_gizi.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM riwayat WHERE id = ?", (id_data,))
    conn.commit()
    conn.close()