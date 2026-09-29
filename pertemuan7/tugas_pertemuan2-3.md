# PANDUAN TUGAS KELOMPOK — PERTEMUAN 3 & 4 MINGGU KE-2
## Proyek Implementasi Sistem Machine Learning Terintegrasi

**Mata Kuliah:** Machine Learning
**Program Studi:** D4 Teknologi Rekayasa Informatika dan Komputer (TRIN)
**Institusi:** Politeknik Manufaktur Bandung (Polman Bandung)
**Periode:** Pertemuan 3 (Hari ke-3) & Pertemuan 4 (Hari ke-4) Minggu ke-2
**Durasi:** 2 Hari Kerja
**Sifat:** Tugas Kelompok (wajib untuk semua anggota)
**Output:** Sistem ML berbasis Web + Laporan Progres Harian + Git Repository


## DAFTAR ISI

1. [Ringkasan Tugas](#1-ringkasan-tugas)
2. [Tujuan Pembelajaran](#2-tujuan-pembelajaran)
3. [Pembagian Kelompok dan Dataset](#3-pembagian-kelompok-dan-dataset)
4. [Minimum Requirement Sistem](#4-minimum-requirement-sistem)
5. [Struktur Tim dan Tanggung Jawab](#5-struktur-tim-dan-tanggung-jawab)
6. [Timeline 2 Hari](#6-timeline-2-hari)
7. [Format Laporan Progres Harian](#7-format-laporan-progres-harian)
8. [Panduan Git dan Repository](#8-panduan-git-dan-repository)
9. [Kriteria Penilaian](#9-kriteria-penilaian)
10. [Tips dan Trik](#10-tips-dan-trik)
11. [FAQ (Pertanyaan yang Sering Diajukan)](#11-faq-pertanyaan-yang-sering-diajukan)
12. [Lampiran: Template dan Contoh](#12-lampiran-template-dan-contoh)


## 1. RINGKASAN TUGAS

### 1.1 Apa yang Harus Dibuat?

Setiap kelompok harus membangun **satu sistem Machine Learning berbasis web** yang mengintegrasikan minimal **2 dari 3 paradigma ML** yang telah dipelajari:

- **Supervised Learning** (Klasifikasi/Regresi)
- **Unsupervised Learning** (Clustering)
- **Reinforcement Learning** (Q-Learning atau sejenisnya)

Sistem ini harus:
1. **Berjalan** — dapat diakses melalui browser.
2. **Fungsional** — dapat menerima input dan memberikan output prediksi.
3. **Terintegrasi** — minimal 2 modul ML dalam satu aplikasi.
4. **Terdefinisi** — dataset berasal dari Kaggle/UCI/Open Source.

### 1.2 Mengapa Tugas Ini Penting?

Di dunia industri, model ML **tidak berdiri sendiri**. Ia harus diintegrasikan ke dalam sistem yang dapat digunakan oleh operator, engineer, atau manajer. Tugas ini melatih Anda untuk:

- Bekerja dalam tim dengan peran yang jelas.
- Mengintegrasikan berbagai model ML.
- Membangun aplikasi web end-to-end.
- Menggunakan Git untuk kolaborasi.
- Melaporkan progres secara profesional.

### 1.3 Output yang Dikumpulkan

| Output | Format | Deadline |
|---|---|---|
| **Laporan Progres Hari ke-3** | `.doc` / `.docx` | Hari ke-3, pukul 23:59 WIB |
| **Link Dataset** | Link | Hari ke-3, simpan dalam laporan harian |
| **Laporan Progres Hari ke-4** | `.doc` / `.docx` | Hari ke-4, pukul 13.00 WIB |
| **Source Code** | Git Repository | Hari ke-4, pukul 13.00 WIB |
| **Slide Presentasi** | `.pptx` | Hari ke-4, saat presentasi |

link pengumpulan : https://forms.gle/nHujXpbpar3dmc4Q7
## 2. TUJUAN PEMBELAJARAN

Setelah menyelesaikan tugas ini, mahasiswa diharapkan mampu:

1. **Merancang** sistem ML terintegrasi dari permasalahan nyata.
2. **Membangun** pipeline ML end-to-end (data → model → web).
3. **Berkolaborasi** dalam tim dengan pembagian peran yang jelas.
4. **Menggunakan Git** untuk version control dan kolaborasi.
5. **Melaporkan** progres kerja secara terstruktur dan profesional.
6. **Mempresentasikan** hasil proyek kepada stakeholder.


## 3. PEMBAGIAN KELOMPOK DAN DATASET

### 3.1 Pembagian Kelompok

Bagi kelompok kelas menjadi **beberapa kelompok** (4–5 mahasiswa per kelompok). Setiap kelompok **wajib** menggunakan dataset yang **berbeda** — tidak boleh ada duplikasi antar kelompok.

### 3.2 Daftar Dataset per Kelompok

Setiap kelompok memilih **satu** dataset dari daftar berikut (first come, first served — koordinator kelas akan mengelola pendaftaran):

| Kelompok | Dataset | Link | Paradigma ML | Jumlah Data |
|---|---|---|---|---|
| **1** | AI4I 2020 Predictive Maintenance | [Kaggle](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020) | Supervised + Unsupervised | 10.000 × 14 |
| **2** | Steel Plates Faults | [Kaggle](https://www.kaggle.com/datasets/uciml/faulty-steel-plates) | Supervised + Unsupervised | 1.941 × 27 |
| **3** | Tool Wear Detection in CNC Mill | [Kaggle](https://www.kaggle.com/datasets/shasun/tool-wear-detection-in-cnc-mill) | Supervised + Unsupervised | ~2.500 × 15 |
| **4** | Bearing Fault Detection (CWRU) | [Kaggle](https://www.kaggle.com/datasets/brjapon/cwru-bearing-datasets) | Supervised + Unsupervised | ~10.000 × 10 |
| **5** | Wafer Defect Classification | [Kaggle](https://www.kaggle.com/datasets/meruvakodandasuraj/semiconductor-wafer-defect-classification-dataset) | Supervised + Unsupervised | 5.000 × 10 |
| **6** | Energy Consumption (PJM) | [Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) | Supervised + Unsupervised | ~145.000 × 2 |
| **7** | Smart Manufacturing IoT-Cloud | [Kaggle](https://www.kaggle.com/datasets/ziya07/smart-manufacturing-iot-cloud-monitoring-dataset) | Supervised + RL | ~10.000 × 10 |
| **8** | Batch Reactor Anomaly Data | [Kaggle](https://www.kaggle.com/datasets/aimindteams/batch-reactor-anomaly-data-sample) | RL + Supervised | 5.000 × 7 |

**Catatan:**
- Jika kelompok Anda ingin menggunakan dataset **lain** yang tidak ada di daftar, **ajukan ke dosen** terlebih dahulu.
- Dataset **harus ringan** (< 100 MB) agar dapat diolah pada laptop RAM 8GB.
- Dataset **harus memiliki label** (untuk supervised) **atau** struktur yang jelas (untuk unsupervised).

### 3.3 Cara Mendaftar Dataset

1. **Diskusi kelompok** — pilih dataset yang diminati.
2. **Hubungi koordinator kelas** — untuk memastikan dataset belum dipakai kelompok lain.
3. **Daftar resmi** — koordinator kelas akan mengirim daftar final ke dosen.
4. **Konfirmasi** — dosen akan mengonfirmasi dataset yang disetujui.

**Deadline pendaftaran:** Hari ke-3, pukul 12:00 WIB.


## 4. MINIMUM REQUIREMENT SISTEM

Sistem yang dibangun **wajib** memenuhi requirement berikut. Setiap requirement memiliki bobot penilaian.

### 4.1 Requirement Wajib (Harus Ada)

| No | Requirement | Deskripsi | Bobot |
|---|---|---|---|
| **R1** | **Aplikasi Web Berjalan** | Dapat diakses melalui browser (`http://localhost:5000`) | 15% |
| **R2** | **Minimal 2 Modul ML** | Integrasi minimal 2 paradigma ML (Supervised + Unsupervised, atau Supervised + RL, atau ketiganya) | 20% |
| **R3** | **Input & Output Form** | Form input untuk parameter, output prediksi yang jelas | 15% |
| **R4** | **Visualisasi** | Minimal 1 grafik (Chart.js, Matplotlib embedded, atau sejenisnya) | 10% |
| **R5** | **Git Repository Aktif** | Setiap anggota push minimal 3 commit (hari 3) + 3 commit (hari 4) | 15% |
| **R6** | **Laporan Progres** | 2 laporan harian (hari ke-3 & hari ke-4) dalam format `.docx` | 15% |
| **R7** | **Presentasi** | Slide + demo aplikasi saat pertemuan ke-4 | 10% |

### 4.2 Requirement Opsional (Bonus Nilai)

| No | Requirement | Deskripsi | Bonus |
|---|---|---|---|
| **B1** | **Modul Ketiga ML** | Integrasi ketiga paradigma (Supervised + Unsupervised + RL) | +5% |
| **B2** | **Database** | Simpan history prediksi ke SQLite/PostgreSQL | +3% |
| **B3** | **Authentication** | Login/logout dengan Flask-Login | +3% |
| **B4** | **Deployment Cloud** | Deploy ke Heroku/Railway/Render | +4% |
| **B5** | **Docker** | Aplikasi dapat dijalankan dengan Docker | +3% |
| **B6** | **Dokumentasi API** | Swagger/OpenAPI docs | +2% |

### 4.3 Detail Setiap Requirement

#### R1 — Aplikasi Web Berjalan

- **Framework:** Flask (rekomendasi), Django, atau FastAPI.
- **Port:** 5000 (default Flask) atau 8000 (FastAPI).
- **Akses:** `http://localhost:5000` atau `http://127.0.0.1:5000`.
- **Halaman Minimal:**
  - Home page
  - Halaman modul ML pertama
  - Halaman modul ML kedua
  - About page

#### R2 — Minimal 2 Modul ML

**Contoh Kombinasi:**

| Kombinasi | Modul 1 | Modul 2 |
|---|---|---|
| A | Klasifikasi Defect (Supervised) | Clustering State (Unsupervised) |
| B | Klasifikasi Defect (Supervised) | Kontrol Reaktor (RL) |
| C | Clustering State (Unsupervised) | Kontrol Reaktor (RL) |
| D | Klasifikasi Defect | Clustering State + Kontrol Reaktor |

#### R3 — Input & Output Form

- **Input:** Form HTML dengan field yang sesuai dataset.
- **Output:** Hasil prediksi/cluster/action yang ditampilkan jelas.
- **Validasi:** Input harus divalidasi (tidak boleh kosong, harus angka, dll.).

#### R4 — Visualisasi

Minimal **1 visualisasi** dari pilihan berikut:
- **Chart.js:** Doughnut chart untuk probabilitas, bar chart untuk Q-values.
- **Matplotlib embedded:** Heatmap korelasi, confusion matrix.
- **Plotly:** Scatter plot interaktif, 3D plot.

#### R5 — Git Repository Aktif

- **Platform:** GitHub, GitLab, atau Bitbucket.
- **Branch:** `main` (production) dan `dev` (development).
- **Commit:** Setiap anggota minimal 3 commit/hari.
- **Commit Message:** Format `[TIPE] Deskripsi` (contoh: `[FEAT] Tambah form input prediksi`).

#### R6 — Laporan Progres

- **Format:** `.doc` atau `.docx`.
- **Isi:** Lihat bagian 7.
- **Deadline:** Hari ke-3 pukul 23:59 WIB dan Hari ke-4 pukul 13.00 WIB.

#### R7 — Presentasi

- **Durasi:** 15 menit (10 menit presentasi + 5 menit Q&A).
- **Slide:** 8–12 slide.
- **Demo:** Aplikasi harus dapat didemonstrasikan live.


## 5. STRUKTUR TIM DAN TANGGUNG JAWAB

### 5.1 Peran Wajib dalam Kelompok

Setiap kelompok (4–5 mahasiswa) **wajib** memiliki peran berikut:

| No | Peran | Tanggung Jawab | Deliverable |
|---|---|---|---|
| **1** | **Koordinator / Project Manager** | - Mengatur jadwal dan deadline internal<br>- Memimpin diskusi<br>- Memastikan semua anggota berkontribusi<br>- Menyusun laporan progres harian | - Timeline<br>- Laporan progres<br>- Konflik resolution |
| **2** | **Data Engineer** | - Mencari dan mengunduh dataset<br>- Melakukan EDA<br>- Preprocessing data<br>- Feature engineering | - Notebook EDA<br>- Dataset bersih<br>- `preprocessing.py` |
| **3** | **ML Engineer 1** | - Melatih model ML pertama (Supervised atau Unsupervised)<br>- Evaluasi model<br>- Export model ke `.pkl` | - Notebook training<br>- Model `.pkl`<br>- Metrik evaluasi |
| **4** | **ML Engineer 2** | - Melatih model ML kedua (paradigma berbeda)<br>- Evaluasi model<br>- Export model ke `.pkl` | - Notebook training<br>- Model `.pkl`<br>- Metrik evaluasi |
| **5** | **Backend & Frontend Developer** | - Membangun Flask API<br>- Membangun frontend HTML/CSS/JS<br>- Integrasi model ke web<br>- Testing | - `app.py`<br>- Templates HTML<br>- Static files |

**Catatan:**
- Jika kelompok hanya 4 orang, **Backend & Frontend Developer** dapat digabung dengan **ML Engineer 2**, atau **Koordinator** merangkap **Backend Developer**.
- Setiap anggota **wajib memahami keseluruhan sistem**, bukan hanya bagiannya.

### 5.2 Contoh Pembagian Tugas (5 Orang)

| Nama | NIM | Peran | Tugas Spesifik |
|---|---|---|---|
| **Budi** | 2212345 | Koordinator | Mengatur timeline, memimpin daily standup, menyusun laporan progres |
| **Ani** | 2212346 | Data Engineer | Download dataset, EDA, preprocessing, feature engineering |
| **Citra** | 2212347 | ML Engineer 1 | Train model Supervised (Random Forest), evaluasi, export `.pkl` |
| **Dedi** | 2212348 | ML Engineer 2 | Train model Unsupervised (K-Means), evaluasi, export `.pkl` |
| **Eka** | 2212349 | Backend & Frontend Dev | Flask API, HTML/CSS/JS, integrasi, testing, deployment |

### 5.3 Aturan Kontribusi

1. **Setiap anggota wajib push minimal 3 commit per hari** ke repository kelompok.
2. **Setiap anggota wajib hadir** di daily standup (minimal 1x/hari).
3. **Setiap anggota wajib memahami** keseluruhan sistem, bukan hanya bagiannya.
4. **Jika ada anggota tidak berkontribusi**, laporkan ke dosen dengan bukti (log commit, chat).

### 5.4 Daily Standup (Wajib)

Setiap hari, kelompok **wajib** melakukan **daily standup** minimal 15 menit. Agenda:

1. **Apa yang sudah dikerjakan kemarin?**
2. **Apa yang akan dikerjakan hari ini?**
3. **Apakah ada hambatan?**
4. **Update timeline.**

**Bukti:** notulensi singkat dan simpan di laporan progres masing masing anggota.


## 6. TIMELINE 2 HARI

### 6.1 Hari ke-3 (Pertemuan 3)

| Waktu | Aktivitas | PIC | Output |
|---|---|---|---|
| **08:00–09:00** | Pembentukan kelompok, pendaftaran dataset | Koordinator | Daftar kelompok + dataset |
| **09:00–10:00** | Diskusi requirement & pembagian peran | Semua | Lembar pembagian tugas |
| **10:00–12:00** | Setup repository Git, setup environment | Backend Dev | Repo aktif |
| **12:00–13:00** | ISHOMA | — | — |
| **13:00–15:00** | Data Engineer: EDA + Preprocessing | Data Engineer | Notebook EDA |
| **13:00–15:00** | ML Engineer 1: Train model pertama | ML Engineer 1 | Model `.pkl` |
| **15:00–17:00** | ML Engineer 2: Train model kedua | ML Engineer 2 | Model `.pkl` |
| **17:00–19:00** | ISHOMA | — | — |
| **19:00–21:00** | Backend Dev: Flask API + integrasi model | Backend Dev | `app.py` |
| **21:00–22:00** | Daily standup + evaluasi hari ke-3 | Semua | Notulensi |
| **22:00–23:00** | Menyusun laporan progres hari ke-3 masing masing anggota | Koordinator | Laporan hari ke-3 |
| **23:59** | **DEADLINE Laporan Progres Hari ke-3** | — | Upload ke Google Drive |

Berikut revisi jadwal **Hari ke-4 (Pertemuan 4)** dengan ketentuan:

- **Maksimal pengumpulan laporan progres: 13.00**
- **Setelah 13.00: paparan masing-masing kelompok sampai 15.40**

| Waktu | Aktivitas | PIC | Output |
|---|---|---|---|
| **08:00–09:00** | Daily standup + review hari ke-3 | Semua | Notulensi |
| **09:00–11:00** | Frontend Dev: HTML/CSS/JS | Backend Dev | Templates + static |
| **09:00–11:00** | Data Engineer: Dokumentasi dataset | Data Engineer | `README_data.md` |
| **11:00–12:00** | Integrasi frontend + backend + finalisasi laporan progres | Backend Dev & Koordinator | Aplikasi berjalan + laporan siap |
| **12:00–13:00** | ISHOMA | — | — |
| **13:00** | **DEADLINE PENGUMPULAN LAPORAN PROGRES** | Koordinator | Laporan progres terkirim |
| **13:00–15:40** | Paparan masing-masing kelompok | Semua | Slide + feedback |
| **15:40–16:00** | Penutup, arahan, dan tindak lanjut | Koordinator | Notulensi penutup |
| **16:00–17:00** | Bug fixing + polish | Backend Dev | Aplikasi stabil |
| **17:00–19:00** | ISHOMA | — | — |
| **19:00–21:00** | Rekap feedback + finalisasi laporan akhir | Koordinator | Laporan akhir |
| **21:00–22:00** | Push final ke Git + bersihkan repository | Semua | Repo final |
| **22:00–23:00** | Upload semua deliverables ke Google Drive | Koordinator | Deliverables |
| **23:59** | **DEADLINE SEMUA OUTPUT** | — | — |

Catatan: durasi paparan tiap kelompok dapat dibagi menyesuaikan jumlah kelompok dalam rentang **13.00–15.40**.

### 6.3 Checklist Timeline

**Hari ke-3:**
- [ ] Kelompok terbentuk
- [ ] Dataset terdaftar dan disetujui
- [ ] Repository Git aktif
- [ ] Environment siap (Anaconda + Flask)
- [ ] EDA selesai
- [ ] Model 1 dilatih
- [ ] Model 2 dilatih
- [ ] Flask API dasar selesai
- [ ] Daily standup dilakukan
- [ ] Laporan progres hari ke-3 diupload

**Hari ke-4:**
- [ ] Frontend selesai
- [ ] Integrasi selesai
- [ ] Testing selesai
- [ ] Bug fixing selesai
- [ ] Slide presentasi selesai
- [ ] Latihan presentasi selesai
- [ ] Laporan progres hari ke-4 diupload
- [ ] Push final ke Git
- [ ] Semua deliverables diupload


## 7. FORMAT LAPORAN PROGRES HARIAN

### 7.1 Ketentuan Umum

| Aspek | Ketentuan |
|---|---|
| **Format File** | `.doc` atau `.docx` (bukan `.pdf`) |
| **Nama File** | `LaporanProgres_Hari[3/4]_[NamaKelompok]_TRIN[Kelas].docx` |
| **Ukuran Kertas** | A4 |
| **Font** | Times New Roman 12 pt |
| **Spasi** | 1.5 |
| **Jumlah Halaman** | 2–4 halaman per laporan |
| **Deadline** | Hari ke-3 & Hari ke-4 |

### 7.2 Struktur Laporan Progres

#### Halaman Judul

```
═══════════════════════════════════════════════════════════
                LAPORAN PROGRES HARIAN
                PROYEK MACHINE LEARNING
═══════════════════════════════════════════════════════════

Hari ke-    : [3 / 4]
Tanggal     : [Tanggal]
Nama Kelompok: [Nama Kelompok]
Kelas       : [Kelas]
Dataset     : [Nama Dataset]

Anggota Kelompok:
1. [Nama] — [NIM] — [Peran]
2. [Nama] — [NIM] — [Peran]
3. [Nama] — [NIM] — [Peran]
4. [Nama] — [NIM] — [Peran]
5. [Nama] — [NIM] — [Peran]

PROGRAM STUDI D4 TEKNOLOGI REKAYASA
INFORMATIKA DAN KOMPUTER
POLITEKNIK MANUFAKTUR BANDUNG
2026
═══════════════════════════════════════════════════════════
```

#### Bagian 1: Ringkasan Progres

```
1. RINGKASAN PROGRES

Persentase penyelesaian proyek: [XX]%

Status keseluruhan: [On Track / Behind Schedule / Ahead of Schedule]

Ringkasan: [2-3 kalimat tentang progres hari ini]
```

#### Bagian 2: Aktivitas per Anggota

```
2. AKTIVITAS PER ANGGOTA

┌────┬─────────────┬──────────────────────────┬──────────┬──────────┐
│ No │ Nama        │ Aktivitas                │ Durasi   │ Status   │
├────┼─────────────┼──────────────────────────┼──────────┼──────────┤
│ 1  │ [Nama]      │ [Deskripsi aktivitas]    │ [X jam]  │ [Done]   │
│ 2  │ [Nama]      │ [Deskripsi aktivitas]    │ [X jam]  │ [Done]   │
│ 3  │ [Nama]      │ [Deskripsi aktivitas]    │ [X jam]  │ [Progress]│
│ 4  │ [Nama]      │ [Deskripsi aktivitas]    │ [X jam]  │ [Done]   │
│ 5  │ [Nama]      │ [Deskripsi aktivitas]    │ [X jam]  │ [Done]   │
└────┴─────────────┴──────────────────────────┴──────────┴──────────┘

Total jam kerja kelompok: [XX] jam
```

#### Bagian 3: Bukti Git Commit

```
3. BUKTI GIT COMMIT

Link Repository: [URL]

┌────┬─────────────┬──────────────────────────────┬──────────────────┐
│ No │ Nama        │ Commit Message               │ Hash             │
├────┼─────────────┼──────────────────────────────┼──────────────────┤
│ 1  │ [Nama]      │ [FEAT] Tambah form input     │ abc123...        │
│ 2  │ [Nama]      │ [FIX] Perbaiki bug scaling   │ def456...        │
│ 3  │ [Nama]      │ [DOCS] Update README         │ ghi789...        │
└────┴─────────────┴──────────────────────────────┴──────────────────┘

Screenshot Git Graph:
[Tempel screenshot network graph dari GitHub]
```

#### Bagian 4: Screenshot Progress

```
4. SCREENSHOT PROGRESS

Screenshot 1: [Deskripsi, misal "Halaman Home Aplikasi"]
[Tempel gambar]

Screenshot 2: [Deskripsi, misal "Hasil Prediksi Supervised"]
[Tempel gambar]
```

#### Bagian 5: Hambatan dan Solusi

```
5. HAMBATAN DAN SOLUSI

┌────┬──────────────────────────────┬──────────────────────────────┐
│ No │ Hambatan                     │ Solusi                       │
├────┼──────────────────────────────┼──────────────────────────────┤
│ 1  │ [Deskripsi hambatan]         │ [Deskripsi solusi]           │
│ 2  │ [Deskripsi hambatan]         │ [Deskripsi solusi]           │
└────┴──────────────────────────────┴──────────────────────────────┘
```

#### Bagian 6: Rencana Hari Berikutnya

```
6. RENCANA HARI BERIKUTNYA

┌────┬──────────────────────────────┬──────────────┬──────────────┐
│ No │ Aktivitas                    │ PIC          │ Deadline     │
├────┼──────────────────────────────┼──────────────┼──────────────┤
│ 1  │ [Deskripsi aktivitas]        │ [Nama]       │ [Jam]        │
│ 2  │ [Deskripsi aktivitas]        │ [Nama]       │ [Jam]        │
└────┴──────────────────────────────┴──────────────┴──────────────┘
```

#### Bagian 7: Notulensi Daily Standup

```
7. NOTULENSI DAILY STANDUP

Tanggal: [Tanggal]
Waktu: [Jam]
Durasi: [X menit]
Media: [Google Meet / Zoom / Tatap Muka]

Agenda:
1. Apa yang sudah dikerjakan?
2. Apa yang akan dikerjakan?
3. Hambatan?

Notulensi:
[Deskripsi singkat hasil diskusi]

Screenshot Meeting:
[Tempel screenshot meeting]
```


## 8. PANDUAN GIT DAN REPOSITORY

### 8.1 Setup Repository

**Langkah 1: Buat Repository di GitHub**

1. Buka [github.com](https://github.com) dan login.
2. Klik tombol **"New"** untuk membuat repository baru.
3. Isi:
   - **Repository name:** `smart-ml-[nama-kelompok]` (contoh: `smart-ml-alpha`)
   - **Description:** Proyek Machine Learning — [Nama Dataset]
   - **Visibility:** Public (agar dosen bisa akses)
   - **Initialize:** Centang "Add a README file"
4. Klik **"Create repository"**.

**Langkah 2: Clone Repository ke Lokal**

```bash
# Buka Anaconda Prompt
cd Documents
git clone https://github.com/[username]/smart-ml-[nama-kelompok].git
cd smart-ml-[nama-kelompok]
```

**Langkah 3: Setup Struktur Folder**

```bash
mkdir models data training static templates tests
mkdir static\css static\js static\img
```

**Langkah 4: Tambahkan Anggota Tim**

1. Buka repository di GitHub.
2. Klik **Settings** → **Collaborators** → **Add people**.
3. Masukkan username GitHub setiap anggota.
4. Setiap anggota akan menerima undangan via email.

### 8.2 Alur Kerja Git untuk Tim

**Setiap anggota wajib mengikuti alur ini:**

```
1. Pull update terbaru
   ↓
2. Buat branch baru untuk fitur
   ↓
3. Kerjakan fitur
   ↓
4. Commit dengan message jelas
   ↓
5. Push ke branch
   ↓
6. Buat Pull Request
   ↓
7. Review oleh anggota lain
   ↓
8. Merge ke main
```

### 8.3 Perintah Git yang Wajib Dikuasai

```bash
# 1. Cek status
git status

# 2. Pull update terbaru
git pull origin main

# 3. Buat branch baru
git checkout -b feature/nama-fitur

# 4. Tambahkan file yang diubah
git add .
# atau
git add nama_file.py

# 5. Commit dengan message
git commit -m "[FEAT] Tambah form input prediksi"

# 6. Push ke branch
git push origin feature/nama-fitur

# 7. Pindah branch
git checkout main

# 8. Merge branch
git merge feature/nama-fitur

# 9. Push ke main
git push origin main
```

### 8.4 Format Commit Message

**Format:** `[TIPE] Deskripsi singkat`

| Tipe | Deskripsi | Contoh |
|---|---|---|
| **FEAT** | Fitur baru | `[FEAT] Tambah halaman prediksi defect` |
| **FIX** | Perbaikan bug | `[FIX] Perbaiki error saat input kosong` |
| **DOCS** | Dokumentasi | `[DOCS] Update README dengan cara install` |
| **STYLE** | Format kode | `[STYLE] Rapikan indentasi app.py` |
| **REFACTOR** | Refactor kode | `[REFACTOR] Pisahkan logic prediksi ke fungsi` |
| **TEST** | Testing | `[TEST] Tambah unit test untuk API` |
| **CHORE** | Tugas rutin | `[CHORE] Update requirements.txt` |

**Aturan:**
- **Wajib** menggunakan format `[TIPE] Deskripsi`.
- **Deskripsi** harus jelas dan singkat (maks 50 karakter).
- **Jangan** commit dengan message `"update"`, `"fix"`, `"asdf"`, dll.
- **Jangan** commit file yang tidak perlu (`.pyc`, `__pycache__`, `.ipynb_checkpoints`).

### 8.5 File `.gitignore`

Buat file `.gitignore` di root repository:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Dataset (jika besar)
data/*.csv
!data/.gitkeep

# Models (jika besar)
models/*.pkl
!models/.gitkeep

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Logs
*.log
```

### 8.6 Bukti Commit untuk Penilaian

Setiap hari, **setiap anggota wajib**:
1. Push minimal **3 commit** ke repository.
2. Screenshot **network graph** dari GitHub.
3. Lampirkan screenshot di laporan progres.

**Cara screenshot network graph:**
1. Buka repository di GitHub.
2. Klik tab **"Insights"** → **"Network"**.
3. Screenshot grafik yang muncul.
4. Tempel di laporan progres.

### 8.7 Troubleshooting Git

| Masalah | Solusi |
|---|---|
| `fatal: not a git repository` | Pastikan Anda berada di folder repository yang benar |
| `error: failed to push` | Lakukan `git pull` terlebih dahulu |
| `merge conflict` | Buka file yang konflik, selesaikan manual, lalu commit |
| `permission denied` | Pastikan Anda sudah di-add sebagai collaborator |
| `large file` | Tambahkan file ke `.gitignore` |


## 9. KRITERIA PENILAIAN

### 9.1 Bobot Penilaian

| Komponen | Bobot | Deskripsi |
|---|---|---|
| **Aplikasi Web** | 25% | Berjalan, fungsional, sesuai requirement |
| **Model ML** | 20% | Akurasi, preprocessing, evaluasi |
| **Git Repository** | 20% | Commit aktif, struktur rapi, kolaborasi |
| **Laporan Progres** | 15% | 2 laporan, lengkap, tepat waktu |
| **Presentasi** | 10% | Jelas, demo berjalan, Q&A |
| **Kerja Sama Tim** | 10% | Pembagian tugas, kontribusi merata |

### 9.2 Rubrik Detail

#### Aplikasi Web (25%)

| Kriteria | Skor |
|---|---|
| Aplikasi berjalan, 3 modul ML, UI menarik | 85–100 |
| Aplikasi berjalan, 2 modul ML, UI baik | 75–84 |
| Aplikasi berjalan, 2 modul ML, UI sederhana | 65–74 |
| Aplikasi berjalan, 1 modul ML | 55–64 |
| Aplikasi tidak berjalan | < 55 |

#### Model ML (20%)

| Kriteria | Skor |
|---|---|
| Model akurat (> 85%), preprocessing lengkap, evaluasi detail | 85–100 |
| Model akurat (75–85%), preprocessing lengkap | 75–84 |
| Model akurat (65–75%), preprocessing dasar | 65–74 |
| Model akurat (< 65%) | 55–64 |
| Model tidak dilatih | < 55 |

#### Git Repository (20%)

| Kriteria | Skor |
|---|---|
| Setiap anggota > 6 commit, struktur rapi, README lengkap | 85–100 |
| Setiap anggota 4–6 commit, struktur baik | 75–84 |
| Setiap anggota 3 commit, struktur dasar | 65–74 |
| Beberapa anggota < 3 commit | 55–64 |
| Tidak ada commit | < 55 |

#### Laporan Progres (15%)

| Kriteria | Skor |
|---|---|
| 2 laporan, lengkap, tepat waktu, screenshot jelas | 85–100 |
| 2 laporan, lengkap, tepat waktu | 75–84 |
| 2 laporan, ada yang kurang | 65–74 |
| 1 laporan | 55–64 |
| Tidak ada laporan | < 55 |

#### Presentasi (10%)

| Kriteria | Skor |
|---|---|
| Slide menarik, demo berjalan, Q&A lancar | 85–100 |
| Slide baik, demo berjalan | 75–84 |
| Slide sederhana, demo berjalan | 65–74 |
| Demo tidak berjalan | 55–64 |
| Tidak presentasi | < 55 |

#### Kerja Sama Tim (10%)

| Kriteria | Skor |
|---|---|
| Semua anggota berkontribusi, komunikasi lancar | 85–100 |
| Mayoritas anggota berkontribusi | 75–84 |
| Beberapa anggota pasif | 65–74 |
| Banyak anggota pasif | 55–64 |
| Tidak ada kerja sama | < 55 |

### 9.3 Konversi Nilai

| Nilai Angka | Huruf | Keterangan |
|---|---|---|
| 85–100 | A | Sangat Baik |
| 75–84 | B | Baik |
| 65–74 | C | Cukup |
| 55–64 | D | Kurang |
| < 55 | E | Sangat Kurang |


## 10. TIPS DAN TRIK

### 10.1 Tips Manajemen Waktu

1. **Mulai dari yang mudah** — selesaikan model ML dulu, baru web.
2. **Gunakan template** — jangan mulai dari nol.
3. **Commit sering** — jangan tunggu selesai baru commit.
4. **Daily standup singkat** — maksimal 15 menit.
5. **Fokus pada MVP** — selesaikan minimum requirement dulu, baru bonus.

### 10.2 Tips Kolaborasi

1. **Komunikasi aktif** — buat grup WhatsApp untuk koordinasi.
2. **Bagi tugas jelas** — setiap orang tahu apa yang harus dikerjakan.
3. **Saling review** — periksa pekerjaan anggota lain sebelum merge.
4. **Dokumentasi** — tulis komentar di kode dan README.
5. **Hindari konflik** — jika ada perbedaan pendapat, diskusikan dengan kepala dingin.

### 10.3 Tips Teknis

1. **Gunakan virtual environment** — hindari konflik dependency.
2. **Test API dengan Postman** — sebelum integrasi ke frontend.
3. **Gunakan Bootstrap** — untuk UI yang cepat dan responsif.
4. **Simpan model** — gunakan `joblib` untuk serialisasi.
5. **Backup** — push ke Git setiap selesai fitur.

### 10.4 Tips Laporan

1. **Screenshot lengkap** — setiap progress harus ada bukti.
2. **Tulis dengan jelas** — hindari bahasa yang ambigu.
3. **Update setiap hari** — jangan menumpuk di akhir.
4. **Cek format** — pastikan sesuai template.
5. **Submit tepat waktu** — jangan mepet deadline.

### 10.5 Tools Rekomendasi

| Tools | Fungsi | Link |
|---|---|---|
| **VS Code** | Code editor | code.visualstudio.com |
| **GitHub Desktop** | GUI Git | desktop.github.com |
| **Postman** | API testing | postman.com |
| **Figma** | UI design | figma.com |
| **Trello** | Task management | trello.com |
| **Notion** | Dokumentasi | notion.so |


## 11. FAQ (PERTANYAAN YANG SERING DIAJUKAN)

**Q1: Apakah boleh menggunakan dataset yang tidak ada di daftar?**

A: Boleh, asalkan:
- Di-approve oleh dosen terlebih dahulu.
- Ukuran < 100 MB.
- Memiliki label (untuk supervised) atau struktur jelas (untuk unsupervised).
- Tidak sama dengan dataset kelompok lain.

**Q2: Apakah wajib menggunakan Flask?**

A: Tidak wajib, tetapi **direkomendasikan**. Alternatif: Django, FastAPI, Streamlit. Namun, Flask paling mudah untuk pemula.

**Q3: Bagaimana jika laptop saya tidak kuat menjalankan aplikasi?**

A: Gunakan dataset yang lebih kecil (< 10.000 baris). Untuk training, gunakan Google Colab gratis.

**Q4: Apakah boleh menggunakan library tambahan?**

A: Boleh, asalkan tercatat di `requirements.txt`.

**Q5: Bagaimana cara mengatasi konflik Git?**

A:
1. Lakukan `git pull` terlebih dahulu.
2. Buka file yang konflik (biasanya ada tanda `<<<<<<<`).
3. Pilih kode yang benar, hapus tanda konflik.
4. `git add .` lalu `git commit`.

**Q6: Apakah setiap anggota wajib presentasi?**

A: Tidak harus semua, tetapi **minimal 2 orang** harus presentasi. Semua anggota harus siap menjawab pertanyaan.

**Q7: Bagaimana jika ada anggota yang tidak berkontribusi?**

A:
1. Dokumentasikan (screenshot chat, log commit).
2. Laporkan ke dosen dengan bukti.
3. Anggota tersebut akan dinilai secara individual.

**Q8: Apakah boleh menggunakan template dari internet?**

A: Boleh untuk **referensi**, tetapi **wajib dimodifikasi** dan disesuaikan. Jangan copy-paste mentah.

**Q9: Berapa lama waktu presentasi?**

A: 15 menit (10 menit presentasi + 5 menit Q&A).

**Q10: Apakah laporan harus dalam bahasa Indonesia?**

A: Ya, laporan dalam **bahasa Indonesia baku**. Kode dan komentar boleh dalam bahasa Inggris.


## 12. LAMPIRAN: TEMPLATE DAN CONTOH

### 12.1 Template Lembar Pembagian Tugas

```
═══════════════════════════════════════════════════════════
              LEMBAR PEMBAGIAN TUGAS KELOMPOK
═══════════════════════════════════════════════════════════

Nama Kelompok : [Nama Kelompok]
Kelas         : [Kelas]
Dataset       : [Nama Dataset]

┌────┬─────────────┬──────────┬────────────────────┬──────────┐
│ No │ Nama        │ NIM      │ Peran              │ Bobot    │
├────┼─────────────┼──────────┼────────────────────┼──────────┤
│ 1  │ [Nama 1]    │ [NIM]    │ Koordinator        │ 20%      │
│ 2  │ [Nama 2]    │ [NIM]    │ Data Engineer      │ 20%      │
│ 3  │ [Nama 3]    │ [NIM]    │ ML Engineer 1      │ 20%      │
│ 4  │ [Nama 4]    │ [NIM]    │ ML Engineer 2      │ 20%      │
│ 5  │ [Nama 5]    │ [NIM]    │ Backend & Frontend │ 20%      │
└────┴─────────────┴──────────┴────────────────────┴──────────┘

Disepakati pada: [Tanggal]

Tanda tangan:
1. [Nama 1]    (..............)
2. [Nama 2]    (..............)
3. [Nama 3]    (..............)
4. [Nama 4]    (..............)
5. [Nama 5]    (..............)
═══════════════════════════════════════════════════════════
```

### 12.2 Template README Repository

```markdown
# Smart ML — [Nama Kelompok]

Proyek Machine Learning Terintegrasi untuk [Nama Dataset].

## Anggota Kelompok

| Nama | NIM | Peran |
|---|---|---|
| [Nama 1] | [NIM] | Koordinator |
| [Nama 2] | [NIM] | Data Engineer |
| [Nama 3] | [NIM] | ML Engineer 1 |
| [Nama 4] | [NIM] | ML Engineer 2 |
| [Nama 5] | [NIM] | Backend & Frontend |

## Dataset

- **Nama:** [Nama Dataset]
- **Sumber:** [Link]
- **Jumlah Data:** [X baris × Y kolom]
- **Target:** [Nama kolom target]

## Fitur

- [Fitur 1]
- [Fitur 2]
- [Fitur 3]

## Teknologi

- Python 3.11
- Flask
- Scikit-learn
- Pandas, NumPy
- Bootstrap 5, Chart.js

## Cara Menjalankan

1. Clone repository:
   ```bash
   git clone https://github.com/[username]/smart-ml-[nama-kelompok].git
   cd smart-ml-[nama-kelompok]
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Jalankan aplikasi:
   ```bash
   python app.py
   ```

4. Buka browser: `http://localhost:5000`

## Struktur Folder

```
smart-ml-[nama-kelompok]/
├── app.py
├── requirements.txt
├── models/
├── data/
├── training/
├── static/
├── templates/
└── tests/
```

## Screenshot

[Tempel screenshot aplikasi]

## Lisensi

Proyek ini dibuat untuk keperluan edukasi di Politeknik Manufaktur Bandung.
```

### 12.3 Contoh Commit Message yang Baik

```
[FEAT] Tambah halaman prediksi defect
[FEAT] Tambah form input sensor readings
[FIX] Perbaiki error saat input kosong
[FIX] Perbaiki scaling pada preprocessing
[DOCS] Update README dengan cara install
[DOCS] Tambah komentar di app.py
[STYLE] Rapikan indentasi templates
[REFACTOR] Pisahkan logic prediksi ke fungsi
[TEST] Tambah unit test untuk API
[CHORE] Update requirements.txt
```

### 12.4 Contoh Laporan Progres Hari ke-3 (Ringkas)

```
═══════════════════════════════════════════════════════════
                LAPORAN PROGRES HARIAN
                PROYEK MACHINE LEARNING
═══════════════════════════════════════════════════════════

Hari ke-     : 3
Tanggal      : 15 Januari 2026
Nama Kelompok: Alpha Team
Kelas        : D4 TRIN 2A
Dataset      : AI4I 2020 Predictive Maintenance

Anggota Kelompok:
1. Budi Santoso — 2212345 — Koordinator
2. Ani Wijaya — 2212346 — Data Engineer
3. Citra Dewi — 2212347 — ML Engineer 1
4. Dedi Pratama — 2212348 — ML Engineer 2
5. Eka Putri — 2212349 — Backend & Frontend

───────────────────────────────────────────────────────────
1. RINGKASAN PROGRES

Persentase penyelesaian: 60%
Status: On Track
Ringkasan: Hari ini kelompok berhasil menyelesaikan EDA,
preprocessing, training 2 model ML, dan setup Flask API dasar.

───────────────────────────────────────────────────────────
2. AKTIVITAS PER ANGGOTA

┌────┬─────────────┬──────────────────────────┬──────────┬──────────┐
│ No │ Nama        │ Aktivitas                │ Durasi   │ Status   │
├────┼─────────────┼──────────────────────────┼──────────┼──────────┤
│ 1  │ Budi        │ Koordinasi, laporan      │ 4 jam    │ Done     │
│ 2  │ Ani         │ EDA + preprocessing      │ 6 jam    │ Done     │
│ 3  │ Citra       │ Train Random Forest      │ 5 jam    │ Done     │
│ 4  │ Dedi        │ Train K-Means            │ 5 jam    │ Done     │
│ 5  │ Eka         │ Setup Flask + API dasar  │ 6 jam    │ Done     │
└────┴─────────────┴──────────────────────────┴──────────┴──────────┘

Total jam kerja: 26 jam

───────────────────────────────────────────────────────────
3. BUKTI GIT COMMIT

Link Repository: https://github.com/alphateam/smart-ml-alpha

┌────┬─────────────┬──────────────────────────────┬──────────────────┐
│ No │ Nama        │ Commit Message               │ Hash             │
├────┼─────────────┼──────────────────────────────┼──────────────────┤
│ 1  │ Budi        │ [DOCS] Tambah README         │ a1b2c3d...       │
│ 2  │ Ani         │ [FEAT] EDA notebook          │ e4f5g6h...       │
│ 3  │ Citra       │ [FEAT] Train RF model        │ i7j8k9l...       │
│ 4  │ Dedi        │ [FEAT] Train K-Means         │ m0n1o2p...       │
│ 5  │ Eka         │ [FEAT] Setup Flask API       │ q3r4s5t...       │
└────┴─────────────┴──────────────────────────────┴──────────────────┘

[Screenshot network graph]

───────────────────────────────────────────────────────────
4. SCREENSHOT PROGRESS

[Screenshot 1: EDA notebook]
[Screenshot 2: Flask API response]

───────────────────────────────────────────────────────────
5. HAMBATAN DAN SOLUSI

┌────┬──────────────────────────────┬──────────────────────────────┐
│ No │ Hambatan                     │ Solusi                       │
├────┼──────────────────────────────┼──────────────────────────────┤
│ 1  │ Data imbalance 96:4          │ Gunakan class_weight         │
│ 2  │ Flask error import           │ Install ulang Flask          │
└────┴──────────────────────────────┴──────────────────────────────┘

───────────────────────────────────────────────────────────
6. RENCANA HARI BERIKUTNYA

┌────┬──────────────────────────────┬──────────────┬──────────────┐
│ No │ Aktivitas                    │ PIC          │ Deadline     │
├────┼──────────────────────────────┼──────────────┼──────────────┤
│ 1  │ Frontend HTML/CSS/JS         │ Eka          │ 11:00        │
│ 2  │ Integrasi frontend-backend   │ Eka          │ 12:00        │
│ 3  │ Testing end-to-end           │ Semua        │ 14:00        │
│ 4  │ Slide presentasi             │ Budi         │ 16:00        │
└────┴──────────────────────────────┴──────────────┴──────────────┘

───────────────────────────────────────────────────────────
7. NOTULENSI DAILY STANDUP

Tanggal: 15 Januari 2026
Waktu: 21:00 WIB
Durasi: 20 menit
Media: Google Meet

Notulensi:
- Semua anggota melaporkan progres masing-masing.
- Ani menyelesaikan EDA dengan insight penting: data imbalanced.
- Citra berhasil train RF dengan F1 0.92.
- Dedi berhasil train K-Means dengan K=3.
- Eka berhasil setup Flask API, belum integrasi frontend.
- Rencana besok: frontend + integrasi.

[Screenshot meeting]

═══════════════════════════════════════════════════════════
```


## PENUTUP

Tugas ini adalah **simulasi dunia kerja nyata** — Anda akan bekerja dalam tim, menggunakan Git, membangun sistem ML, dan melaporkan progres harian. Semua keterampilan ini sangat berharga untuk karier Anda sebagai lulusan D4 TRIN Polman Bandung.

**Poin-poin kunci:**

1. **Kerja sama tim** adalah kunci sukses.
2. **Git commit aktif** adalah bukti kontribusi Anda.
3. **Laporan progres harian** melatih kedisiplinan.
4. **MVP dulu, bonus kemudian** — selesaikan minimum requirement sebelum menambah fitur.
5. **Komunikasi aktif** dengan anggota tim dan dosen.

**Pesan untuk Mahasiswa:**
> "Jangan takut untuk mencoba dan membuat kesalahan. Setiap commit adalah langkah maju. Setiap bug adalah pelajaran. Setiap laporan adalah cerminan profesionalisme Anda. Selamat mengerjakan — ini adalah kesempatan Anda untuk membangun portofolio yang membanggakan!"

**Pesan untuk Dosen:**
> "Panduan ini dapat disesuaikan dengan jumlah kelompok dan tingkat kemampuan mahasiswa. Untuk kelas dengan waktu terbatas, fokus pada minimum requirement saja. Untuk kelas yang lebih mahir, tantang dengan bonus requirements."

---

**Dokumen ini disusun untuk keperluan edukasi di Politeknik Manufaktur Bandung (Polman Bandung).**
**Versi:** 1.0 | **Pertemuan 3 & 4 Minggu ke-2**
**Terakhir Diperbarui:** 2026
