import sqlite3

def init_db():
    conn = sqlite3.connect("gizi.db")
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS riwayat (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        berat REAL,
        tinggi REAL,
        lingkar REAL,
        hasil TEXT
    )
    """)
    conn.commit()
    conn.close()

def simpan(berat, tinggi, lingkar, hasil):
    conn = sqlite3.connect("gizi.db")
    c = conn.cursor()
    c.execute("INSERT INTO riwayat (berat, tinggi, lingkar, hasil) VALUES (?,?,?,?)",
              (berat, tinggi, lingkar, hasil))
    conn.commit()
    conn.close()

def load_data():
    conn = sqlite3.connect("gizi.db")
    c = conn.cursor()
    c.execute("SELECT * FROM riwayat")
    data = c.fetchall()
    conn.close()
    return data
