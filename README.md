# GRU + XAI — Sesi Input Jawaban Interaktif (versi Web)

Ini adalah versi web dari `input_jawaban()` / `list_history()` / `tampilkan_penjelasan()`
/ `export_history_csv()` di notebook Anda. Prediksi dijalankan oleh model GRU
yang **sungguh-sungguh Anda latih** di pipeline — bukan simulasi atau data palsu.

## 1. Salin artefak model ke folder ini

Letakkan file-file berikut sejajar dengan `app.py` (folder yang sama):

```
gru_web/
├── app.py
├── requirements.txt
├── templates/index.html
├── gru_optik_model.h5      <- dari notebook (cell "Save Artifacts")
├── tokenizer.pkl           <- dari notebook
├── id2label.json           <- dari notebook
└── dataset_gru.csv         <- opsional, hanya untuk penjelasan SHAP yang lebih akurat
```

Jika Anda melatih modelnya di Google Colab, unduh keempat file di atas dari
panel File Colab (klik kanan → Download) dan pindahkan ke folder `gru_web/`.

## 2. Install dependency (sekali saja)

Buka terminal di folder `gru_web/`, lalu:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Instalasi TensorFlow bisa memakan waktu beberapa menit — ini normal.

## 3. Jalankan servernya

```bash
python app.py
```

Anda akan melihat log `Memuat model...` lalu `Model dimuat. maxlen=...`.
Setelah itu buka browser ke:

```
http://127.0.0.1:5000
```

Itu saja — tidak perlu Live Server atau ekstensi lain, Flask sudah
menyajikan halamannya sendiri di alamat itu.

## Yang bisa dilakukan di halaman ini

- **Kirim Jawaban** — ketik jawaban siswa, model asli langsung memprediksi
  label (Paham Konsep / Miskonsepsi / Tidak Paham) beserta tingkat keyakinan
  untuk setiap kelas.
- **Riwayat jawaban** — setara `list_history()`, bisa dicari dengan kotak
  pencarian.
- **Jelaskan** (per baris) — setara `tampilkan_penjelasan()`, menghitung
  kontribusi kata (SHAP) untuk jawaban tersebut. Ini butuh beberapa detik
  karena SHAP menjalankan banyak simulasi prediksi di belakang layar — sama
  seperti di notebook.
- **Ekspor CSV** — setara `export_history_csv()`, mengunduh riwayat.
- **Hapus Riwayat** — setara `clear_history(confirm=True)`, dengan
  konfirmasi agar tidak tidak sengaja terhapus.

## Untuk keperluan video

- Halaman ini berjalan sepenuhnya lokal (`127.0.0.1`) — aman direkam tanpa
  koneksi internet, kecuali font Google Fonts (kosmetik saja; jika offline,
  browser otomatis memakai font sistem).
- Jika ingin proses "Jelaskan" terasa lebih cepat saat direkam, kecilkan
  `nsamples` di `app.py` bagian `/api/explain` (default 60) — semakin kecil,
  semakin cepat, namun estimasi SHAP sedikit lebih kasar.
- Riwayat jawaban tersimpan di memori server selama `python app.py` berjalan.
  Menutup terminal/server akan mengosongkan riwayat (sesuai perilaku notebook
  aslinya yang juga berbasis variabel session, bukan database).

## Troubleshooting

- **"File artefak tidak ditemukan"** saat menjalankan `app.py` → pastikan
  ketiga file (`gru_optik_model.h5`, `tokenizer.pkl`, `id2label.json`) benar-benar
  ada di folder yang sama dengan `app.py`.
- **Error saat memuat model (`.h5`)** → biasanya karena versi TensorFlow di
  komputer Anda berbeda dari versi di Colab. Coba `pip install tensorflow`
  tanpa versi tertentu dulu; jika masih gagal, muat ulang model di Colab lalu
  simpan ulang dengan `final_model.save("gru_optik_model.keras")` (format
  baru) dan ubah `MODEL_PATH` di `app.py` menyesuaikan.
- **Tombol "Jelaskan" menampilkan pesan SHAP nonaktif** → paket `shap` gagal
  ter-install atau gagal diimpor; jalankan `pip install shap` lagi dan lihat
  pesan error di terminal tempat `app.py` berjalan. Fitur prediksi utama tetap
  berfungsi normal walau ini gagal.
