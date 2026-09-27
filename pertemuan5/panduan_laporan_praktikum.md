# PANDUAN LAPORAN PRAKTIKUM
## Mata Kuliah: Machine Learning
### Pertemuan ke-1 Minggu ke-2: Supervised Learning — Prediksi Defect Produk Manufaktur

**Program Studi:** D4 Teknologi Rekayasa Informatika dan Komputer (TRIN)
**Institusi:** Politeknik Manufaktur Bandung (Polman Bandung)
**Topik Praktikum:** Supervised Learning — Klasifikasi Kualitas Produk Manufaktur
**Sifat Pengerjaan:** Mandiri (studi kasus + latihan) & Kelompok (tugas kelompok)
**Batas Pengumpulan:** Hari ke-4 (Pertemuan ke-4) melalui Google Drive


## DAFTAR ISI

1. [Informasi Umum Praktikum](#1-informasi-umum-praktikum)
2. [Ketentuan Umum Laporan](#2-ketentuan-umum-laporan)
3. [Struktur Laporan Praktikum Mandiri](#3-struktur-laporan-praktikum-mandiri)
4. [Struktur Laporan Latihan Mandiri](#4-struktur-laporan-latihan-mandiri)
5. [Struktur Laporan Tugas Kelompok](#5-struktur-laporan-tugas-kelompok)
6. [Pembagian Tugas Kelompok](#6-pembagian-tugas-kelompok)
7. [Format Penamaan File dan Struktur Folder](#7-format-penamaan-file-dan-struktur-folder)
8. [Mekanisme Pengumpulan](#8-mekanisme-pengumpulan)
9. [Rubrik Penilaian](#9-rubrik-penilaian)
10. [Template Laporan (Siap Pakai)](#10-template-laporan-siap-pakai)
11. [Checklist Sebelum Mengumpulkan](#11-checklist-sebelum-mengumpulkan)
12. [Sanksi dan Ketentuan Keterlambatan](#12-sanksi-dan-ketentuan-keterlambatan)
13. [Lampiran: Contoh Halaman Judul dan Lembar Pengesahan](#13-lampiran-contoh-halaman-judul-dan-lembar-pengesahan)


## 1. INFORMASI UMUM PRAKTIKUM

### 1.1 Konteks Minggu ke-2

Minggu ke-2 merupakan **minggu tematik Machine Learning** yang terdiri dari **4 pertemuan**:

| Pertemuan | Topik | Jenis Praktikum |
|---|---|---|
| **Pertemuan 1** | **Supervised Learning — Prediksi Defect Produk Manufaktur** | Studi Kasus + Latihan Mandiri + Tugas Kelompok |
| Pertemuan 2 | Unsupervised Learning | Studi Kasus + Latihan Mandiri + Tugas Kelompok |
| Pertemuan 3 | Reinforcement Learning | Studi Kasus + Latihan Mandiri + Tugas Kelompok |
| Pertemuan 4 | Evaluasi Mingguan (Proyek + Teori) | Presentasi + Evaluasi |

### 1.2 Ketentuan Pengerjaan Pertemuan 1

| Komponen | Sifat | Dikerjakan | Dikumpulkan |
|---|---|---|---|
| **Studi Kasus** (Prediksi Defect Manufaktur) | Mandiri | Hari ini (Pertemuan 1) | Hari ke-4 |
| **Latihan Mandiri** | Mandiri | Hari ini (Pertemuan 1) | Hari ke-4 |
| **Tugas Kelompok** | Kelompok | Di rumah | Hari ke-4 |

### 1.3 Tujuan Praktikum

Setelah mengikuti praktikum ini, mahasiswa diharapkan mampu:

1. **Memahami** konsep supervised learning untuk klasifikasi kualitas produk manufaktur.
2. **Melakukan** eksplorasi data (EDA) pada dataset defect produksi.
3. **Melakukan** preprocessing data: encoding, scaling, handling class imbalance.
4. **Melatih** minimal 4 algoritma klasifikasi (Decision Tree, KNN, Naive Bayes, Logistic Regression, Random Forest).
5. **Mengevaluasi** performa model dengan metrik yang tepat (Accuracy, Precision, Recall, F1-Score, AUC-ROC).
6. **Menginterpretasikan** hasil dan mengidentifikasi fitur paling berpengaruh terhadap defect.
7. **Mengaitkan** hasil ML dengan konteks nyata industri manufaktur.


## 2. KETENTUAN UMUM LAPORAN

### 2.1 Ketentuan Format

| Aspek | Ketentuan |
|---|---|
| **Ukuran Kertas** | A4 (21 × 29.7 cm) |
| **Margin** | Kiri 4 cm, Kanan 3 cm, Atas 3 cm, Bawah 3 cm |
| **Font** | Times New Roman 12 pt (isi), 14 pt (judul bab), 12 pt bold (subbab) |
| **Spasi** | 1.5 (isi), 1.0 (kode program, tabel, gambar) |
| **Penomoran Halaman** | Kanan bawah, dimulai dari BAB I |
| **Bahasa** | Indonesia baku (sesuai PUEBI) |
| **Jumlah Halaman** | Laporan Mandiri: 15–25 halaman; Laporan Kelompok: 20–35 halaman |

### 2.2 Ketentuan Konten

1. **Setiap kode program** yang ditulis harus disertai **penjelasan** (bukan hanya copy-paste).
2. **Setiap gambar/tabel** harus diberi **nomor** dan **caption** (misal: Gambar 3.1 Distribusi IPK).
3. **Setiap hasil** harus disertai **interpretasi** — bukan hanya "akurasi 0.89" tetapi "akurasi 0.89 menunjukkan model mampu memprediksi dengan benar 89% dari data testing".
4. **Referensi** harus dicantumkan dengan format APA 7th edition.
5. **Tidak boleh** ada plagiarisme. Kutipan langsung harus diberi tanda kutip dan sumber.

### 2.3 Ketentuan Notebook Jupyter

1. Notebook harus **bersih** — semua cell sudah dijalankan dan output terlihat.
2. Gunakan **Markdown cell** untuk menjelaskan setiap tahap.
3. **Nomori** setiap section (1, 2, 3, ...) agar mudah diikuti.
4. **Komentari** kode yang kompleks.
5. Simpan dalam format `.ipynb` **dan** ekspor ke `.html` atau `.pdf` sebagai backup.


## 3. STRUKTUR LAPORAN PRAKTIKUM MANDIRI

Laporan Praktikum Mandiri adalah laporan atas **Studi Kasus** yang dikerjakan **sendiri** oleh setiap mahasiswa.

### 3.1 Halaman Judul

Berisi:
- Judul laporan: **"Laporan Praktikum Machine Learning: Prediksi Defect Produk Manufaktur"**
- Nama mahasiswa, NIM, kelas
- Nama mata kuliah, program studi, institusi
- Logo Polman Bandung
- Tahun akademik

### 3.2 Lembar Pengesahan (opsional untuk mandiri)

Berisi tanda tangan mahasiswa dan dosen pengampu.

### 3.3 Kata Pengantar

Ucapan syukur dan terima kasih, serta penjelasan singkat isi laporan.

### 3.4 Daftar Isi, Daftar Gambar, Daftar Tabel

Dibuat otomatis menggunakan fitur Table of Contents di Word/Google Docs.

### 3.5 BAB I — PENDAHULUAN

#### 1.1 Latar Belakang
Jelaskan mengapa prediksi defect produk penting dalam industri manufaktur. Kaitkan dengan:
- Kebutuhan kualitas produk
- Efisiensi biaya
- Dukungan keputusan berbasis data
- Relevansi dengan Polman Bandung

**Contoh kalimat:**
> "Dalam industri manufaktur modern, setiap defect yang lolos ke pasar dapat menyebabkan kerugian finansial yang signifikan. Dengan Machine Learning, kita dapat membangun model yang mampu memprediksi probabilitas defect sebelum produk selesai diproduksi, sehingga tindakan preventif dapat segera dilakukan."

#### 1.2 Rumusan Masalah
Tulis 3–5 pertanyaan yang akan dijawab, contoh:
1. Bagaimana karakteristik dataset defect produksi manufaktur?
2. Algoritma supervised learning mana yang memberikan performa terbaik?
3. Fitur apa yang paling berpengaruh terhadap status defect?
4. Bagaimana menginterpretasikan hasil model untuk stakeholder manufaktur?

#### 1.3 Tujuan Praktikum
Tulis tujuan yang ingin dicapai (3–5 poin).

#### 1.4 Manfaat Praktikum
Tulis manfaat teoretis dan praktis.

#### 1.5 Batasan Masalah
Contoh:
- Dataset yang digunakan adalah *Predicting Manufacturing Defects Dataset* dari Kaggle.
- Algoritma yang dibandingkan: Decision Tree, KNN, Naive Bayes, Logistic Regression, Random Forest.
- Tidak membahas deployment ke produksi.

### 3.6 BAB II — LANDASAN TEORI

#### 2.1 Machine Learning
Definisi, jenis (supervised, unsupervised, reinforcement).

#### 2.2 Supervised Learning
Definisi, konsep fitur dan label, contoh aplikasi di manufaktur.

#### 2.3 Klasifikasi
Perbedaan klasifikasi vs regresi, contoh algoritma.

#### 2.4 Algoritma yang Digunakan
Jelaskan **masing-masing** algoritma:
- **Decision Tree:** konsep, Gini Impurity, hyperparameter
- **KNN:** konsep, jarak Euclidean, pemilihan K
- **Naive Bayes:** konsep, Teorema Bayes, asumsi independensi
- **Logistic Regression:** konsep, fungsi sigmoid, interpretasi koefisien
- **Random Forest:** konsep ensemble, bagging, feature importance

#### 2.5 Evaluasi Model
Jelaskan: Confusion Matrix, Accuracy, Precision, Recall, F1-Score, AUC-ROC.

#### 2.6 Class Imbalance
Jelaskan mengapa defect biasanya jarang, dan strategi penanganannya (SMOTE, class_weight).

#### 2.7 Studi Kasus Manufaktur
Penjelasan konteks, fitur yang relevan, tantangan.

### 3.7 BAB III — METODOLOGI

#### 3.1 Alur Kerja (Flowchart)
Buat flowchart ML Pipeline:
```
Load Data → EDA → Preprocessing → Split → Training → Testing → Evaluasi → Kesimpulan
```

#### 3.2 Dataset
- **Nama:** Predicting Manufacturing Defects Dataset
- **Sumber:** https://www.kaggle.com/datasets/rabieelkharoua/predicting-manufacturing-defects-dataset
- **Jumlah data:** 3.240 entri
- **Jumlah fitur:** 16 kolom
- **Target:** DefectStatus (0 = Low Defect, 1 = High Defect)
- **Deskripsi kolom:** (tabel)

#### 3.3 Tools dan Library
- Anaconda (Python 3.11)
- Jupyter Notebook
- pandas, numpy, scikit-learn, matplotlib, seaborn, imbalanced-learn

#### 3.4 Tahapan Praktikum
Jelaskan langkah-langkah yang akan dilakukan secara ringkas.

### 3.8 BAB IV — HASIL DAN PEMBAHASAN

#### 4.1 Eksplorasi Data (EDA)

**4.1.1 Informasi Umum Dataset**
Sertakan output `df.info()` dan `df.describe()` + penjelasan.

**4.1.2 Pengecekan Missing Value dan Duplikat**
Sertakan output dan interpretasi.

**4.1.3 Visualisasi Distribusi Fitur**
Sertakan minimal **3 visualisasi**:
- Histogram DefectRate berdasarkan status
- Boxplot QualityScore
- Countplot DefectStatus

Setiap gambar diberi caption dan interpretasi.

**4.1.4 Analisis Korelasi**
Sertakan heatmap korelasi + interpretasi.

#### 4.2 Preprocessing

**4.2.1 Feature Engineering**
Jelaskan fitur baru yang dibuat (Total_Cost, Efficiency_Ratio, dll).

**4.2.2 Handling Missing Value dan Duplikat**
Jelaskan strategi yang digunakan.

**4.2.3 Encoding Data Kategorikal**
Sertakan kode dan penjelasan (jika ada).

**4.2.4 Split Data**
Sertakan kode, jelaskan fungsi `stratify=y` dan `random_state`.

**4.2.5 Feature Scaling**
Sertakan kode, jelaskan mengapa `fit` hanya pada training.

**4.2.6 Penanganan Class Imbalance (jika diterapkan)**
Jelaskan SMOTE atau class_weight yang digunakan.

#### 4.3 Modeling

**4.3.1 Decision Tree**
- Kode
- Parameter yang digunakan
- Hasil akurasi, precision, recall, F1, AUC

**4.3.2 K-Nearest Neighbors**
- Kode
- Nilai K yang digunakan
- Hasil metrik

**4.3.3 Naive Bayes**
- Kode
- Hasil metrik

**4.3.4 Logistic Regression**
- Kode
- Hasil metrik

**4.3.5 Random Forest**
- Kode
- Hasil metrik

#### 4.4 Evaluasi Model

**4.4.1 Tabel Perbandingan Metrik**
Sertakan tabel:

| Model | Train Acc | Test Acc | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|---|
| Decision Tree | ... | ... | ... | ... | ... | ... |
| KNN | ... | ... | ... | ... | ... | ... |
| Naive Bayes | ... | ... | ... | ... | ... | ... |
| Logistic Regression | ... | ... | ... | ... | ... | ... |
| Random Forest | ... | ... | ... | ... | ... | ... |

**4.4.2 Confusion Matrix Model Terbaik**
Sertakan gambar + interpretasi TP, TN, FP, FN.

**4.4.3 Classification Report**
Sertakan precision, recall, f1-score per kelas.

**4.4.4 ROC Curve**
Sertakan gambar ROC curve untuk semua model.

**4.4.5 Visualisasi Perbandingan**
Sertakan bar chart perbandingan.

#### 4.5 Pembahasan

**4.5.1 Analisis Model Terbaik**
Mengapa model X menjadi yang terbaik? Kaitkan dengan karakteristik data.

**4.5.2 Analisis Overfitting**
Apakah ada model yang overfitting? Bagaimana mengatasinya?

**4.5.3 Fitur Paling Berpengaruh**
Berdasarkan feature importance, fitur mana yang paling berpengaruh?

**4.5.4 Implikasi untuk Quality Control**
Bagaimana hasil ini bisa diterapkan di lini produksi?

**4.5.5 Keterbatasan**
Apa keterbatasan dari praktikum ini?

### 3.9 BAB V — KESIMPULAN DAN SARAN

#### 5.1 Kesimpulan
Jawab rumusan masalah di BAB I secara ringkas (3–5 poin).

#### 5.2 Saran
Saran untuk pengembangan selanjutnya (misal: gunakan XGBoost, tambah data, terapkan SMOTE).

### 3.10 DAFTAR PUSTAKA

Format APA 7th edition. Minimal **5 referensi** (buku, jurnal, dokumentasi).

### 3.11 LAMPIRAN

- **Lampiran A:** Notebook Jupyter lengkap (`.ipynb` atau `.pdf`)
- **Lampiran B:** Kode program lengkap
- **Lampiran C:** Dataset (sample)


## 4. STRUKTUR LAPORAN LATIHAN MANDIRI

Laporan Latihan Mandiri adalah laporan atas **soal latihan** yang dikerjakan **sendiri** oleh setiap mahasiswa. Berbeda dengan studi kasus, laporan ini lebih ringkas dan fokus pada jawaban soal.

### 4.1 Halaman Judul

Sama seperti laporan mandiri, tetapi judulnya: **"Laporan Latihan Mandiri Machine Learning: Prediksi Defect Produk Manufaktur"**

### 4.2 Identitas

- Nama, NIM, Kelas
- Nama Mata Kuliah
- Pertemuan ke-1 Minggu ke-2

### 4.3 Soal dan Jawaban

Untuk **setiap soal latihan**, tuliskan:

**Soal 1:** [Tulis soal]

**Jawaban:**
- Kode program
- Output
- Penjelasan/interpretasi

**Soal 2:** [Tulis soal]

**Jawaban:**
- ...

...dan seterusnya.

### 4.4 Soal Latihan yang Harus Dikerjakan

#### Latihan 1: Eksperimen Hyperparameter Decision Tree

Ubah `max_depth` dari 3, 5, 7, 10, dan 15. Catat F1-Score dan AUC-ROC untuk setiap nilai. Buat tabel dan grafik. Jelaskan nilai mana yang optimal dan mengapa.

#### Latihan 2: Eksperimen Nilai K pada KNN

Ubah `n_neighbors` dari 3, 5, 7, 9, 11. Catat F1-Score dan Recall untuk setiap nilai K. Buat tabel dan grafik. Jelaskan nilai K optimal.

#### Latihan 3: Visualisasi Tambahan

Buat boxplot `DefectRate` berdasarkan `DefectStatus`. Buat scatter plot antara `ProductionVolume` dan `DefectRate` dengan warna berdasarkan `DefectStatus`. Interpretasikan hasilnya.

#### Latihan 4: Pertanyaan Konsep

Jawab pertanyaan berikut:
1. Mengapa accuracy tidak cukup untuk data imbalanced? Berikan contoh.
2. Apa perbedaan Precision dan Recall? Dalam konteks defect produk, mana yang lebih penting?
3. Jelaskan mengapa `stratify=y` penting pada `train_test_split`.
4. Apa itu overfitting? Bagaimana mendeteksinya dari tabel perbandingan?
5. Mengapa Random Forest sering lebih baik dari Decision Tree tunggal?

#### Latihan 5: Refleksi

Tulis refleksi pribadi (minimal 200 kata) tentang:
- Apa yang Anda pelajari dari praktikum ini?
- Bagian mana yang paling sulit?
- Bagian mana yang paling menarik?
- Bagaimana Anda akan menerapkan ML di industri manufaktur?

### 4.5 Kesimpulan Latihan

Ringkasan hasil latihan dalam 1 paragraf.


## 5. STRUKTUR LAPORAN TUGAS KELOMPOK

Laporan Tugas Kelompok adalah laporan atas **tugas kelompok** yang dikerjakan **bersama** oleh anggota kelompok.

### 5.1 Halaman Judul

Judul: **"Laporan Tugas Kelompok Machine Learning: Prediksi Defect Produk Manufaktur"**

Sertakan:
- Nama kelompok
- Daftar anggota (nama + NIM)
- Pembagian tugas
- Institusi dan tahun

### 5.2 Lembar Pembagian Tugas

| No | Nama | NIM | Tugas | Bobot Kontribusi |
|---|---|---|---|---|
| 1 | ... | ... | Koordinator + Bagian A | 20% |
| 2 | ... | ... | Bagian B | 25% |
| 3 | ... | ... | Bagian C | 25% |
| 4 | ... | ... | Bagian D + Presentasi | 20% |
| 5 | ... | ... | Editor + Reviewer | 10% |

### 5.3 BAB I — PENDAHULUAN

Sama seperti laporan mandiri, tetapi ditambah:
- **Peran masing-masing anggota** dalam proyek.

### 5.4 BAB II — LANDASAN TEORI

Ditambah teori tentang:
- **XGBoost**
- **SMOTE** (Synthetic Minority Over-sampling Technique)
- **SHAP** (SHapley Additive exPlanations)
- **Hyperparameter Tuning** (GridSearchCV)

### 5.5 BAB III — METODOLOGI

#### 3.1 Dataset yang Digunakan
Jelaskan dataset utama + dataset alternatif (jika ada).

#### 3.2 Alur Kerja Kelompok
Flowchart yang menggambarkan alur kerja tim.

#### 3.3 Pembagian Tugas
Detail pembagian tugas + timeline.

### 5.6 BAB IV — HASIL DAN PEMBAHASAN

#### 4.1 Bagian A: Eksplorasi Dataset Alternatif
- Deskripsi dataset (AI4I 2020, Steel Plates Faults, atau Tool Wear)
- EDA singkat
- Minimal 2 visualisasi
- Temuan

#### 4.2 Bagian B: Perbandingan 5 Algoritma

Tabel perbandingan **5 algoritma**:

| Model | Train Acc | Test Acc | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|---|
| Decision Tree | ... | ... | ... | ... | ... | ... |
| KNN | ... | ... | ... | ... | ... | ... |
| Naive Bayes | ... | ... | ... | ... | ... | ... |
| Logistic Regression | ... | ... | ... | ... | ... | ... |
| Random Forest | ... | ... | ... | ... | ... | ... |

**Analisis:** Model mana yang terbaik? Mengapa? Apakah ada yang overfitting?

#### 4.3 Bagian C: Feature Importance

- Analisis feature importance dari model terbaik.
- Identifikasi 5 fitur paling berpengaruh.
- Jelaskan mengapa fitur-fitur tersebut penting dalam konteks manufaktur.
- Buat rekomendasi untuk monitoring di lini produksi.

#### 4.4 Bagian D: Rekomendasi Model

Rekomendasi model terbaik untuk sistem quality control + alasan.

### 5.7 BAB V — KESIMPULAN DAN SARAN

Kesimpulan kelompok + saran pengembangan.

### 5.8 DAFTAR PUSTAKA

Minimal **8 referensi** (lebih banyak dari laporan mandiri).

### 5.9 LAMPIRAN

- Notebook kelompok
- Slide presentasi
- Log aktivitas kelompok (opsional)
- Dokumentasi rapat (opsional)


## 6. PEMBAGIAN TUGAS KELOMPOK

### 6.1 Struktur Tim yang Direkomendasikan

Untuk kelompok berisi **4–5 mahasiswa**, berikut pembagian tugas yang direkomendasikan:

| Peran | Tanggung Jawab | Output |
|---|---|---|
| **Koordinator** | Mengatur jadwal, memimpin diskusi, memastikan semua bagian selesai tepat waktu, mengintegrasikan laporan | Timeline, laporan final |
| **Data Engineer** | Mencari dataset alternatif, EDA, cleaning data | Bagian A laporan |
| **Model Engineer 1** | Melatih Decision Tree, KNN, Naive Bayes | Bagian B (DT, KNN, NB) |
| **Model Engineer 2** | Melatih Logistic Regression, Random Forest | Bagian B (LR, RF) |
| **Analis & Presenter** | Analisis perbandingan, feature importance, presentasi | Bagian C + Slide |

### 6.2 Aturan Pembagian Tugas

1. **Setiap anggota wajib berkontribusi** — tidak boleh ada yang hanya "titip nama".
2. **Setiap anggota harus memahami keseluruhan laporan**, bukan hanya bagiannya.
3. **Bobot kontribusi** harus disepakati bersama dan dicantumkan di lembar pembagian tugas.
4. **Log aktivitas** (opsional) bisa dilampirkan untuk membuktikan kontribusi.
5. **Jika ada anggota yang tidak berkontribusi**, laporkan ke dosen dengan bukti.

### 6.3 Timeline yang Direkomendasikan

| Hari | Aktivitas |
|---|---|
| Hari 1 (Pertemuan 1) | Pembentukan kelompok, pembagian tugas, diskusi awal |
| Hari 1–2 (di rumah) | Masing-masing mengerjakan bagiannya |
| Hari 3 | Integrasi laporan, review bersama |
| Hari 4 (Pertemuan 4) | Presentasi + pengumpulan |


## 7. FORMAT PENAMAAN FILE DAN STRUKTUR FOLDER

### 7.1 Format Penamaan File

**Laporan Mandiri:**
```
LaporanMandiri_[NamaLengkap]_[NIM]_TRIN_[Kelas].pdf
```
Contoh:
```
LaporanMandiri_BudiSantoso_2212345_TRIN_2A.pdf
```

**Notebook Mandiri:**
```
NotebookMandiri_[NamaLengkap]_[NIM]_TRIN_[Kelas].ipynb
```

**Laporan Latihan Mandiri:**
```
LatihanMandiri_[NamaLengkap]_[NIM]_TRIN_[Kelas].pdf
```

**Laporan Kelompok:**
```
LaporanKelompok_[NamaKelompok]_TRIN_[Kelas].pdf
```

**Slide Presentasi Kelompok:**
```
SlideKelompok_[NamaKelompok]_TRIN_[Kelas].pdf
```

**Notebook Kelompok:**
```
NotebookKelompok_[NamaKelompok]_TRIN_[Kelas].ipynb
```

### 7.2 Struktur Folder di Google Drive

Buat folder dengan struktur berikut:

```
📁 Machine Learning - Minggu 2 - [Kelas]
│
├── 📁 Laporan Mandiri
│   ├── 📁 [NIM]_[Nama]_Mandiri
│   │   ├── 📄 LaporanMandiri_[Nama]_[NIM].pdf
│   │   ├── 📄 NotebookMandiri_[Nama]_[NIM].ipynb
│   │   ├── 📄 LatihanMandiri_[Nama]_[NIM].pdf
│   │   └── 📁 Dataset
│   │       └── 📄 manufacturing_defect_dataset.csv
│
├── 📁 Laporan Kelompok
│   ├── 📁 [NamaKelompok]
│   │   ├── 📄 LaporanKelompok_[NamaKelompok].pdf
│   │   ├── 📄 NotebookKelompok_[NamaKelompok].ipynb
│   │   ├── 📄 SlideKelompok_[NamaKelompok].pdf
│   │   └── 📁 Lampiran
│   │       ├── 📄 LogAktivitas.pdf
│   │       └── 📄 DokumentasiRapat.pdf
│
└── 📄 README.txt (petunjuk pengumpulan)
```

### 7.3 Aturan Penamaan Folder

- Gunakan **huruf tanpa spasi** (gunakan underscore `_`).
- Sertakan **NIM** untuk folder mandiri.
- Sertakan **nama kelompok** untuk folder kelompok.
- **Jangan** gunakan karakter khusus (`!@#$%^&*`).


## 8. MEKANISME PENGUMPULAN

### 8.1 Waktu Pengumpulan

| Komponen | Deadline | Tempat |
|---|---|---|
| Laporan Mandiri | Hari ke-4, pukul 23:59 WIB | Google Drive |
| Latihan Mandiri | Hari ke-4, pukul 23:59 WIB | Google Drive |
| Laporan Kelompok | Hari ke-4, pukul 23:59 WIB | Google Drive |
| Presentasi Kelompok | Hari ke-4, saat pertemuan | Ruang kelas |

### 8.2 Link Google Drive

Link Google Drive akan diberikan oleh dosen pengampu melalui:
- Google Classroom
- Grup WhatsApp kelas
- Email kampus

### 8.3 Langkah Pengumpulan

**Langkah 1:** Buka link Google Drive yang diberikan.

**Langkah 2:** Buat folder sesuai struktur di atas (bagian 7.2).

**Langkah 3:** Upload semua file ke folder yang sesuai.

**Langkah 4:** Pastikan semua file **tidak dalam mode "view only"** — atur sharing menjadi "Anyone with the link can view".

**Langkah 5:** Screenshot folder Anda sebagai bukti pengumpulan.

**Langkah 6:** Konfirmasi di Google Classroom bahwa Anda sudah mengumpulkan.

### 8.4 Hal yang Tidak Boleh Dilakukan

❌ Mengumpulkan file dalam format `.rar` atau `.zip` (kecuali diminta).
❌ Mengumpulkan hanya link tanpa file.
❌ Mengumpulkan setelah deadline tanpa izin.
❌ Mengumpulkan file kosong atau rusak.
❌ Mengumpulkan file dengan nama yang tidak sesuai format.


## 9. RUBRIK PENILAIAN

### 9.1 Rubrik Laporan Mandiri (Bobot 20%)

| Komponen | Bobot | Kriteria Penilaian | Skor |
|---|---|---|---|
| **Kelengkapan Struktur** | 10% | Semua bab ada dan lengkap | 0–100 |
| **Kualitas EDA** | 15% | Visualisasi informatif, interpretasi tajam | 0–100 |
| **Preprocessing** | 15% | Encoding, scaling, split dilakukan dengan benar | 0–100 |
| **Modeling** | 20% | 5 algoritma dilatih dengan parameter yang tepat | 0–100 |
| **Evaluasi** | 20% | Metrik lengkap, interpretasi benar | 0–100 |
| **Kesimpulan** | 10% | Menjawab rumusan masalah, ada saran | 0–100 |
| **Kualitas Penulisan** | 10% | Bahasa baku, rapi, referensi lengkap | 0–100 |

**Nilai Akhir = Σ (Bobot × Skor)**

### 9.2 Rubrik Latihan Mandiri (Bobot 10%)

| Komponen | Bobot | Kriteria | Skor |
|---|---|---|---|
| **Kelengkapan Jawaban** | 40% | Semua soal dijawab | 0–100 |
| **Ketepatan Jawaban** | 30% | Jawaban benar dan relevan | 0–100 |
| **Kedalaman Analisis** | 20% | Ada interpretasi, bukan hanya output | 0–100 |
| **Refleksi Pribadi** | 10% | Refleksi mendalam, autentik | 0–100 |

### 9.3 Rubrik Laporan Kelompok (Bobot 30%)

| Komponen | Bobot | Kriteria | Skor |
|---|---|---|---|
| **Bagian A: EDA Dataset Alternatif** | 10% | EDA mendalam, visualisasi informatif | 0–100 |
| **Bagian B: Perbandingan 5 Algoritma** | 15% | Tabel lengkap, analisis tajam | 0–100 |
| **Bagian C: Feature Importance** | 10% | Analisis tepat, rekomendasi logis | 0–100 |
| **Bagian D: Rekomendasi** | 5% | Rekomendasi logis dan beralasan | 0–100 |
| **Pembagian Tugas** | 5% | Jelas, adil, ada bukti kontribusi | 0–100 |
| **Kualitas Penulisan** | 5% | Rapi, konsisten, referensi lengkap | 0–100 |

### 9.4 Rubrik Presentasi Kelompok (Bobot 10%)

| Komponen | Bobot | Kriteria | Skor |
|---|---|---|---|
| **Kejelasan Penyampaian** | 30% | Mudah dipahami, tidak membaca teks | 0–100 |
| **Kelengkapan Materi** | 30% | Semua bagian tercakup | 0–100 |
| **Visual Slide** | 20% | Menarik, tidak terlalu padat | 0–100 |
| **Kemampuan Menjawab** | 20% | Menjawab pertanyaan dengan baik | 0–100 |

### 9.5 Konversi Nilai

| Nilai Angka | Huruf | Keterangan |
|---|---|---|
| 85–100 | A | Sangat Baik |
| 75–84 | B | Baik |
| 65–74 | C | Cukup |
| 55–64 | D | Kurang |
| < 55 | E | Sangat Kurang |


## 10. TEMPLATE LAPORAN (SIAP PAKAI)

### 10.1 Template Halaman Judul

```
═══════════════════════════════════════════════════════════
                    LAPORAN PRAKTIKUM
                    MACHINE LEARNING
        "Prediksi Defect Produk Manufaktur"
═══════════════════════════════════════════════════════════

                    [LOGO POLMAN BANDUNG]
                    (ukuran 4×4 cm)


Disusun oleh:
Nama    : [Nama Lengkap]
NIM     : [NIM]
Kelas   : [Kelas]

Dosen Pengampu:
[Nama Dosen]


PROGRAM STUDI D4 TEKNOLOGI REKAYASA
INFORMATIKA DAN KOMPUTER
POLITEKNIK MANUFAKTUR BANDUNG
2025
═══════════════════════════════════════════════════════════
```

### 10.2 Template Lembar Pembagian Tugas Kelompok

```
═══════════════════════════════════════════════════════════
              LEMBAR PEMBAGIAN TUGAS KELOMPOK
═══════════════════════════════════════════════════════════

Nama Kelompok : [Nama Kelompok]
Kelas         : [Kelas]
Topik         : Prediksi Defect Produk Manufaktur

┌────┬─────────────────┬──────────┬────────────────────┬──────────┐
│ No │ Nama            │ NIM      │ Tugas              │ Bobot    │
├────┼─────────────────┼──────────┼────────────────────┼──────────┤
│ 1  │ [Nama 1]        │ [NIM]    │ Koordinator +      │ 20%      │
│    │                 │          │ Bagian A           │          │
│ 2  │ [Nama 2]        │ [NIM]    │ Bagian B (DT, KNN) │ 25%      │
│ 3  │ [Nama 3]        │ [NIM]    │ Bagian B (LR, RF)  │ 25%      │
│ 4  │ [Nama 4]        │ [NIM]    │ Bagian C           │ 20%      │
│ 5  │ [Nama 5]        │ [NIM]    │ Editor + Presenter │ 10%      │
└────┴─────────────────┴──────────┴────────────────────┴──────────┘

Disepakati pada: [Tanggal]

Tanda tangan:
1. [Nama 1]    (..............)
2. [Nama 2]    (..............)
3. [Nama 3]    (..............)
4. [Nama 4]    (..............)
5. [Nama 5]    (..............)
═══════════════════════════════════════════════════════════
```

### 10.3 Template Tabel Perbandingan Model

```
┌──────────────────────┬─────────┬─────────┬───────────┬────────┬──────────┬─────────┐
│ Model                │ Train   │ Test    │ Precision │ Recall │ F1-Score │ AUC-ROC │
│                      │ Acc     │ Acc     │           │        │          │         │
├──────────────────────┼─────────┼─────────┼───────────┼────────┼──────────┼─────────┤
│ Decision Tree        │ 0.xxx   │ 0.xxx   │ 0.xxx     │ 0.xxx  │ 0.xxx    │ 0.xxx   │
│ KNN                  │ 0.xxx   │ 0.xxx   │ 0.xxx     │ 0.xxx  │ 0.xxx    │ 0.xxx   │
│ Naive Bayes          │ 0.xxx   │ 0.xxx   │ 0.xxx     │ 0.xxx  │ 0.xxx    │ 0.xxx   │
│ Logistic Regression  │ 0.xxx   │ 0.xxx   │ 0.xxx     │ 0.xxx  │ 0.xxx    │ 0.xxx   │
│ Random Forest        │ 0.xxx   │ 0.xxx   │ 0.xxx     │ 0.xxx  │ 0.xxx    │ 0.xxx   │
└──────────────────────┴─────────┴─────────┴───────────┴────────┴──────────┴─────────┘
```

### 10.4 Template Refleksi Pribadi

```
═══════════════════════════════════════════════════════════
                    REFLEKSI PRIBADI
═══════════════════════════════════════════════════════════

Nama    : [Nama]
NIM     : [NIM]
Tanggal : [Tanggal]

1. Apa yang saya pelajari dari praktikum ini?
   [Tulis 3–5 poin]

2. Bagian mana yang paling sulit?
   [Tulis 2–3 poin + cara mengatasinya]

3. Bagian mana yang paling menarik?
   [Tulis 2–3 poin]

4. Bagaimana saya akan menerapkan ML di industri manufaktur?
   [Tulis 1 paragraf]

5. Rencana perbaikan untuk praktikum selanjutnya:
   [Tulis 2–3 poin]
═══════════════════════════════════════════════════════════
```


## 11. CHECKLIST SEBELUM MENGUMPULKAN

### 11.1 Checklist Laporan Mandiri

- [ ] Halaman judul lengkap
- [ ] Kata pengantar
- [ ] Daftar isi, daftar gambar, daftar tabel
- [ ] BAB I: Pendahuluan (latar belakang, rumusan masalah, tujuan, manfaat, batasan)
- [ ] BAB II: Landasan teori (minimal 5 subbab)
- [ ] BAB III: Metodologi (flowchart, dataset, tools, tahapan)
- [ ] BAB IV: Hasil dan pembahasan
  - [ ] EDA (minimal 3 visualisasi)
  - [ ] Preprocessing
  - [ ] Modeling (5 algoritma)
  - [ ] Evaluasi (tabel + confusion matrix + classification report + ROC)
  - [ ] Pembahasan (analisis, overfitting, fitur penting, implikasi, keterbatasan)
- [ ] BAB V: Kesimpulan dan saran
- [ ] Daftar pustaka (minimal 5 referensi)
- [ ] Lampiran (notebook)
- [ ] Format: A4, margin, font, spasi
- [ ] Penomoran halaman
- [ ] Caption gambar dan tabel
- [ ] Tidak ada typo
- [ ] Nama file sesuai format

### 11.2 Checklist Latihan Mandiri

- [ ] Halaman judul
- [ ] Identitas mahasiswa
- [ ] Soal 1–5 dijawab lengkap
- [ ] Setiap jawaban disertai kode, output, penjelasan
- [ ] Refleksi pribadi (minimal 200 kata)
- [ ] Kesimpulan latihan
- [ ] Nama file sesuai format

### 11.3 Checklist Laporan Kelompok

- [ ] Halaman judul dengan daftar anggota
- [ ] Lembar pembagian tugas (ditandatangani)
- [ ] BAB I: Pendahuluan (+ peran anggota)
- [ ] BAB II: Landasan teori (+ XGBoost, SMOTE, SHAP)
- [ ] BAB III: Metodologi
- [ ] BAB IV: Hasil dan pembahasan
  - [ ] Bagian A: EDA dataset alternatif
  - [ ] Bagian B: Perbandingan 5 algoritma
  - [ ] Bagian C: Feature importance
  - [ ] Bagian D: Rekomendasi model
- [ ] BAB V: Kesimpulan dan saran
- [ ] Daftar pustaka (minimal 8 referensi)
- [ ] Lampiran (notebook, slide, log aktivitas)
- [ ] Nama file sesuai format
- [ ] Semua anggota sudah review laporan
- [ ] Siap presentasi

### 11.4 Checklist Pengumpulan

- [ ] Semua file sudah di-upload ke Google Drive
- [ ] Struktur folder sesuai ketentuan
- [ ] Nama file sesuai format
- [ ] Sharing link sudah "Anyone with the link can view"
- [ ] Sudah screenshot bukti pengumpulan
- [ ] Sudah konfirmasi di Google Classroom
- [ ] Deadline: Hari ke-4, pukul 23:59 WIB


## 12. SANKSI DAN KETENTUAN KETERLAMBATAN

### 12.1 Ketentuan Keterlambatan

| Keterlambatan | Sanksi |
|---|---|
| 0–6 jam | Pengurangan nilai 10% |
| 6–24 jam | Pengurangan nilai 25% |
| 24–48 jam | Pengurangan nilai 50% |
| > 48 jam | Nilai 0 (kecuali ada izin resmi) |

### 12.2 Ketentuan Plagiarisme

- **Plagiarisme ringan** (mengutip tanpa sumber): pengurangan nilai 20%.
- **Plagiarisme sedang** (menyalin sebagian besar): nilai 0 untuk komponen tersebut.
- **Plagiarisme berat** (menyalin seluruh laporan): nilai 0 untuk mata kuliah + sanksi akademik.

### 12.3 Ketentuan Anggota Kelompok Tidak Aktif

1. Anggota yang tidak berkontribusi **wajib dilaporkan** oleh koordinator dengan bukti (log aktivitas, chat, dll).
2. Anggota yang dilaporkan akan dinilai **secara individual** berdasarkan kontribusinya.
3. Jika terbukti tidak berkontribusi sama sekali, nilai kelompok untuk anggota tersebut adalah **0**.

### 12.4 Ketentuan Perizinan

Jika tidak bisa mengumpulkan tepat waktu karena alasan darurat (sakit, musibah), ajukan izin **sebelum deadline** dengan bukti (surat dokter, dll). Izin yang disetujui akan diberikan perpanjangan waktu maksimal 3 hari.


## 13. LAMPIRAN: CONTOH HALAMAN JUDUL DAN LEMBAR PENGESAHAN

### 13.1 Contoh Halaman Judul Lengkap

```
═══════════════════════════════════════════════════════════════════
                        LAPORAN PRAKTIKUM MANDIRI
                          MACHINE LEARNING

            "Prediksi Defect Produk Manufaktur
        Menggunakan Algoritma Supervised Learning"

═══════════════════════════════════════════════════════════════════

                          [LOGO POLMAN]
                          (4 × 4 cm)


    Disusun sebagai salah satu syarat kelulusan praktikum
    Mata Kuliah Machine Learning
    Pertemuan ke-1, Minggu ke-2


    Disusun oleh:
    ┌─────────────────────────────────────┐
    │ Nama    : Budi Santoso              │
    │ NIM     : 2212345                   │
    │ Kelas   : D4 TRIN 2A                │
    │ Program : D4 Teknologi Rekayasa     │
    │           Informatika dan Komputer  │
    └─────────────────────────────────────┘


    Dosen Pengampu:
    [Nama Dosen Pengampu]


    PROGRAM STUDI D4 TEKNOLOGI REKAYASA
    INFORMATIKA DAN KOMPUTER
    POLITEKNIK MANUFAKTUR BANDUNG
    TAHUN 2025
═══════════════════════════════════════════════════════════════════
```

### 13.2 Contoh Lembar Pengesahan

```
═══════════════════════════════════════════════════════════════════
                         LEMBAR PENGESAHAN
═══════════════════════════════════════════════════════════════════

Laporan Praktikum Mandiri dengan judul:

    "Prediksi Defect Produk Manufaktur
        Menggunakan Algoritma Supervised Learning"

Disusun oleh:
    Nama    : Budi Santoso
    NIM     : 2212345
    Kelas   : D4 TRIN 2A

Telah diperiksa dan disetujui pada:

    Hari/Tanggal : [Hari, Tanggal]
    Tempat       : Bandung


    Mahasiswa,                          Dosen Pengampu,




    (Budi Santoso)                      ([Nama Dosen])
    NIM. 2212345                        NIP. [NIP Dosen]
═══════════════════════════════════════════════════════════════════
```

### 13.3 Contoh Log Aktivitas Kelompok

```
═══════════════════════════════════════════════════════════════════
                    LOG AKTIVITAS KELOMPOK
═══════════════════════════════════════════════════════════════════

Nama Kelompok : Alpha Team
Kelas         : D4 TRIN 2A

┌────┬────────────┬──────────────────────┬──────────┬───────────┐
│ No │ Tanggal    │ Aktivitas            │ Anggota  │ Durasi    │
├────┼────────────┼──────────────────────┼──────────┼───────────┤
│ 1  │ 2025-01-15 │ Diskusi awal,        │ Semua    │ 2 jam     │
│    │            │ pembagian tugas      │          │           │
│ 2  │ 2025-01-16 │ Mencari dataset      │ Budi,    │ 3 jam     │
│    │            │ alternatif           │ Ani      │           │
│ 3  │ 2025-01-16 │ Latih Decision Tree  │ Citra    │ 2 jam     │
│ 4  │ 2025-01-17 │ Latih Random Forest  │ Dedi     │ 2 jam     │
│ 5  │ 2025-01-17 │ Feature importance   │ Eka      │ 3 jam     │
│ 6  │ 2025-01-18 │ Integrasi laporan    │ Semua    │ 4 jam     │
│ 7  │ 2025-01-19 │ Review & presentasi  │ Semua    │ 2 jam     │
└────┴────────────┴──────────────────────┴──────────┴───────────┘

Total jam kerja kelompok: 18 jam

Bandung, [Tanggal]

Koordinator,


(Budi Santoso)
NIM. 2212345
═══════════════════════════════════════════════════════════════════
```


## PENUTUP

Panduan laporan praktikum ini disusun untuk membantu mahasiswa D4 TRIN Polman Bandung dalam menyelesaikan dan melaporkan praktikum Machine Learning minggu ke-2, pertemuan ke-1 tentang **Supervised Learning — Prediksi Defect Produk Manufaktur**.

**Poin-poin kunci:**

1. **Studi kasus** dan **latihan mandiri** dikerjakan **secara mandiri** oleh setiap mahasiswa.
2. **Tugas kelompok** dikerjakan **bersama** dengan pembagian tugas yang jelas.
3. **Semua laporan** dikumpulkan pada **hari ke-4** melalui **Google Drive**.
4. **Setiap komponen** memiliki rubrik penilaian yang jelas.
5. **Kualitas lebih penting** daripada kuantitas — fokus pada pemahaman dan interpretasi.

**Pesan untuk Mahasiswa:**
> "Laporan bukan sekadar formalitas. Ini adalah kesempatan Anda untuk **berpikir seperti data scientist** — menganalisis masalah, memilih pendekatan, dan mengomunikasikan hasil dengan jelas. Dalam konteks manufaktur, kemampuan ini sangat berharga."

**Pesan untuk Dosen:**
> "Panduan ini bisa disesuaikan dengan tingkat pemahaman mahasiswa dan ketersediaan waktu. Untuk kelas yang lebih besar, pertimbangkan untuk menambah asisten praktikum."

---

**Dokumen ini disusun untuk keperluan edukasi di Politeknik Manufaktur Bandung (Polman Bandung).**
**Versi:** 1.0 | **Minggu ke-2, Pertemuan ke-1 dari 4**
**Terakhir Diperbarui:** 2026
