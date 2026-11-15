# AVESONIC: dashboard pengenalan spesies burung Indonesia dari suara

Dashboard Streamlit untuk tugas Data Mining. Lima halaman:

| Halaman | Isi |
|---|---|
| Beranda | Ringkasan dan tiga temuan utama dalam bahasa sehari-hari |
| Kenali Datanya | Sepuluh spesies (audio + spektrogram), komposisi data, peta lokasi, pembagian data, kualitas klip |
| Coba Tebak | Tebak spesies dari klip contoh, unggahan, atau rekaman mikrofon |
| Hasil Analisis | Perbandingan enam metode, uji selisih, hasil per spesies, confusion matrix, peragaan DTW |
| Catatan & Sumber | Batasan, kamus istilah, daftar perekam, lisensi, ringkasan metode |

## Menjalankan di komputer sendiri

Butuh Python 3.12.

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

Lalu buka http://localhost:8501.

## Memasang di Streamlit Community Cloud

1. Unggah seluruh isi folder ini ke sebuah repositori GitHub.
2. Di https://share.streamlit.io pilih repositori itu, berkas utama `app.py`, dan Python 3.12.
3. `requirements.txt` dan `packages.txt` dibaca otomatis.

Pemasangan di Streamlit Community Cloud belum diuji; aplikasi ini baru dijalankan dan diuji secara lokal.

## Catatan penting

- **Model BirdNET diunduh otomatis.** Saat pertama kali ada unggahan atau rekaman, aplikasi mengunduh model BirdNET 2.4 (sekitar 77 MB) dari Zenodo ke `aset/birdnet/`. Klip contoh tidak membutuhkan unduhan ini. Model BirdNET berlisensi CC BY-NC-SA 4.0.
- **Versi scikit-learn harus sama** dengan yang tertulis di `requirements.txt`, karena model SVM disimpan dalam format joblib.
- **Rekaman panjang** dianalisis 60 detik pertamanya, paling banyak 8 potongan 3 detik yang paling bersuara.
- **Klip contoh** diambil dari data uji, hanya dari rekaman berlisensi BY-NC-SA atau lebih longgar, dipilih berdasarkan kejernihan tanpa melihat benar-salahnya tebakan model.

## Struktur

```
app.py                  titik masuk + navigasi
avesonic/tema.py           CSS, komponen, gaya grafik
avesonic/mesin.py          aset, pengolahan audio, BirdNET + SVM, MFCC/DTW peragaan
avesonic/halaman/*.py      lima halaman
aset/                   model SVM, ringkasan hasil, metadata, 30 klip contoh, logo
alat/siapkan_aset.py    membangun ulang isi aset/ dari hasil analisis
```

## Sumber

Rekaman berasal dari Xeno-canto (https://xeno-canto.org) dan tetap milik perekamnya masing-masing di bawah lisensi Creative Commons. Nama Indonesia spesies mengikuti "Daftar nama burung di Indonesia" di Wikipedia bahasa Indonesia.
