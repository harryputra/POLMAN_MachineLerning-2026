# PANDUAN LENGKAP TUGAS EDA SPRINT 1 HARI

## “Eksplorasi Bebas di Pagi Hari, Aksi Nyata di Sore Hari”

---

### A. ATURAN MAIN & JADWAL (HARI-H)

| Waktu | Durasi | Sesi | Jenis | Target Akhir |
| :--- | :--- | :--- | :--- | :--- |
| **08.00 – 12.00** | 4 Jam | **Mandiri (Individu)** | Eksplorasi Teori + Fungsi | Membuat **"Ensiklopedia EDA"** pribadi (tanpa batasan jumlah fungsi). |
| **12.00 – 13.00** | 1 Jam | Istirahat & Koordinasi | Kelompok | Bagi peran (Coder, Visualisator, Narator, Statistikawan) |
| **13.00 – 17.00** | 4 Jam | **Kelompok (Tim)** | Praktik Full EDA | Mengubah data kotor menjadi **"Data Siap Modeling"** |
| **17.00 – 17.30** | 30 Menit | **Pleno Kilat** | Presentasi | Setiap kelompok tunjukkan 3 insight paling mencengangkan dari data. |

---

## BAGIAN 1: TUGAS MANDIRI (PAGI) – “BERBURU ILMU TANPA BATAS”

*(Waktu: 08.00 – 12.00 | Output: 1 File Ensiklopedia + 1 File Refleksi)*

**Tujuan:** Mahasiswa membangun **pemahaman konseptual yang kuat** dan **mengumpulkan sebanyak mungkin "senjata" (fungsi/command)** untuk EDA. Tidak ada batasan jumlah fungsi! Semakin banyak dan bervariasi yang ditemukan, semakin kaya bekal mereka.

---

### TAHAP 1: EKSPLORASI KONSEP (08.00 – 09.30)

**Instruksi:** Cari tahu jawaban dari **5 Pertanyaan Wajib** di bawah ini melalui mesin pencari, blog, atau jurnal. Tulis jawaban dalam bentuk **Mind Map** atau **Catatan Naratif** yang rapi.

| No | Pertanyaan Wajib | Panduan Menjawab (Apa yang harus muncul di jawaban) |
| :--- | :--- | :--- |
| **1** | **Apa itu EDA dan siapa pencetusnya?** | Tulis definisi dari John Tukey. Tekankan bahwa EDA adalah **filosofi** untuk 'menginterogasi' data sebelum menghakiminya dengan model. |
| **2** | **Gambarkan posisi EDA dalam alur proyek Machine Learning!** | Buat diagram alir dari *Problem Definition* → *Data Gathering* → **EDA (di sini!) → *Preprocessing* → *Modeling* → *Deployment*. Jelaskan mengapa EDA HARUS dilakukan sebelum membersihkan data. |
| **3** | **Apa bedanya EDA, Data Cleaning, dan Preprocessing?** | Buat tabel perbandingan. <br> - **Cleaning**: Membuang sampah (null, duplikat). <br> - **EDA**: Memahami pola sampahnya. <br> - **Preprocessing**: Mengubah sampah jadi emas (scaling/encoding). |
| **4** | **Bagaimana EDA bisa menentukan pilihan algoritma ML?** | Berikan 3 contoh konkret: <br> - Jika data **linear** → pakai Regresi. <br> - Jika banyak **outlier** → hindari K-NN, pilih Tree-Based. <br> - Jika target **tidak seimbang (imbalance)** → wajib pakai teknik SMOTE atau algoritma dengan class_weight. |
| **5** | **Sebutkan 3 "Tanda Bahaya" (Red Flag) yang harus dicari saat EDA!** | Jelaskan: <br> a. **Multikolinearitas** (korelasi antarfitur > 0.8) → bikin model regresi rawan. <br> b. **Skewness ekstrem** pada target → bikin prediksi meleset di nilai ekstrem. <br> c. **Missing Value Tidak Acak (MNAR)** → kalau dihapus, hasil modeling jadi bias. |

---

### TAHAP 2: EKSPLORASI FUNGSI / COMMAND (09.30 – 11.30)

**Instruksi:** Sekarang saatnya **berburu fungsi**. Buka Google, dokumentasi Pandas, Seaborn, Matplotlib, atau Scipy. Temukan **SEBANYAK MUNGKIN** fungsi yang berkaitan dengan EDA.

> **Catatan Penting:** Tidak ada target minimal, tapi semakin banyak dan beragam, semakin tinggi nilai eksplorasi Anda. Mahasiswa yang baik bisa menemukan 25–40 fungsi. Mahasiswa kreatif bahkan akan menemukan fungsi dari library seperti `Plotly` atau `Scipy.stats`.

**WAJIB** untuk setiap fungsi yang ditemukan, tuliskan dalam format tabel berikut di buku catatan atau dokumen digital:

| Kolom | Isi yang Harus Ditulis |
| :--- | :--- |
| **Nama Fungsi** | Contoh: `df.info()`, `sns.boxplot()`, `scipy.stats.skewtest()` |
| **Asal Library** | Pandas / Seaborn / Matplotlib / Scipy / dll. |
| **Tujuan / Kegunaan** | Untuk apa fungsi ini dipakai dalam proses EDA? |
| **Parameter Kunci** | Tulis minimal 2 parameter yang sering dipakai (misal: `axis=0`, `dropna=True`). |
| **Contoh Output** | Tulis seperti apa hasil yang keluar (misal: range nilai, bentuk grafik). |
| **Hubungan dengan Teori** | Kaitkan dengan pertanyaan di Tahap 1. (Misal: fungsi `.skew()` membantu mendeteksi **Red Flag Skewness**). |

**Jangan batasi diri pada 3 library itu saja.** Sarankan mereka mencari:

- **Fungsi Statistik**: `mode()`, `median()`, `var()`, `std()`, `quantile()`, `skew()`, `kurtosis()`.
- **Fungsi Pengecekan Data**: `shape`, `columns`, `dtypes`, `memory_usage()`, `select_dtypes()`.
- **Fungsi Visualisasi Lanjutan**: `pairplot()`, `jointplot()`, `violinplot()`, `lmplot()`, `heatmap()`.
- **Fungsi Preprocessing (untuk EDA)**: `pd.cut()` (untuk binning), `pd.get_dummies()` (encoding untuk eksplorasi).
- **Fungsi Uji Statistik (Nilai Tambah)**: `pearsonr()`, `spearmanr()`, `chi2_contingency()`, `ttest_ind()`.

---

### TAHAP 3: REFLEKSI DIRI (11.30 – 12.00)

**Instruksi:** Di akhir dokumen, tuliskan **3 Paragraf Refleksi** yang menjawab:

1. Fungsi mana yang menurut Anda paling "ajaib" dan paling sering akan Anda pakai? Mengapa?
2. Setelah membaca teori Red Flag dan mencoba fungsi, kira-kira kesalahan apa yang paling sering dilakukan pemula saat EDA?
3. Tuliskan 1 rencana kecil: "Sore nanti, saat kerja kelompok, saya akan fokus menggunakan fungsi _______ untuk menemukan _______."

---

**Output yang Dikumpulkan (Pukul 12.00 WIB):**

- 1 File PDF/Doc berisi:
  1. Jawaban 5 pertanyaan konsep + mind map.
  2. **Tabel daftar fungsi** (terserah mau berapa banyak, yang penting orisinil hasil temuan sendiri).
  3. 3 Paragraf Refleksi.

---

## BAGIAN 2: TUGAS KELOMPOK (SORE) – “MENGOTAK-ATIK DATA KOTOR”

*(Waktu: 13.00 – 17.00 | Output: 1 Notebook + 5 Slide Presentasi)*

**Tujuan:** Menerapkan semua fungsi yang sudah dikumpulkan di pagi hari untuk **membersihkan dan menganalisis dataset** sampai benar-benar siap masuk ke tahap pemodelan Machine Learning.

---

### A. KONDISI WAJIB DATASET (Disediakan oleh Dosen/Asesor)

Agar mahasiswa belajar banyak hal, dataset yang diberikan **HARUS** memiliki 5 "penyakit" berikut:

| No | Kondisi Data | Contoh Kasus |
| :--- | :--- | :--- |
| 1 | **Ada Missing Value** | Minimal di 3 kolom (campuran numerik & kategorikal). |
| 2 | **Ada Outlier** | Di kolom numerik (misal: ada gaji Rp 99 Miliar di tengah gaji rata-rata 5 Juta). |
| 3 | **Ada Data Duplikat** | Minimal 5–10 baris yang sama persis. |
| 4 | **Tipe Data Salah** | Kolom tanggal terbaca sebagai `object`, atau angka terbaca sebagai `string`. |
| 5 | **Kategorikal Ber-kardinalitas Tinggi** | Misal: kolom 'Provinsi' dengan 30+ nilai unik, atau kolom teks bebas. |

> **Rekomendasi Dataset Praktis:** Dosen bisa pakai dataset *Titanic* (Kaggle) lalu sengaja merusaknya (tambah outlier di Fare, ubah tipe Age jadi string), atau pakai *Customer Churn* dengan banyak NA.

---

### B. INSTRUKSI PENGERJAAN (Kerjakan Urut!)

#### Tahap 1: Deteksi & Pembersihan Awal (Estimasi: 60 menit)

- **Load** data dan tampilkan 5 baris pertama (`head()`).
- **Deteksi:** Gunakan fungsi-fungsi dari pagi untuk cek duplikat, null, dan tipe data.
- **Tindakan:**
  - Hapus duplikat.
  - Perbaiki tipe data (ubah string tanggal ke `datetime`, ubah kategorikal ke `category`).
  - **Tangani Missing Value**: Untuk numerik, bandingkan 2 metode (misal: median vs mean) dan pilih salah satu dengan alasan tertulis. Untuk kategorikal, isi dengan *Modus* atau buat kategori baru *"Tidak Diketahui"*.

#### Tahap 2: Analisis Univariat (Menganalisis 1 per 1 Kolom) (Estimasi: 45 menit)

- **Kolom Numerik**: Tampilkan `.describe()`, buat **Histogram** dan **Boxplot**. Catat: "Apakah ada outlier? Berapa skewness-nya?"
- **Kolom Kategorikal**: Tampilkan `.value_counts()`, buat **Bar Chart**. Catat: "Apakah ada ketidakseimbangan (imbalance) yang ekstrem?"

#### Tahap 3: Analisis Bivariat & Multivariat (Hubungan) (Estimasi: 60 menit)

- **Numerik vs Numerik**: Buat **Heatmap Korelasi** (`.corr()` + `heatmap`). Tandai 2 pasang fitur dengan korelasi tertinggi (bisa positif atau negatif).
- **Kategorikal vs Numerik**: Buat **Boxplot** atau **Violinplot** yang mengelompokkan data berdasarkan kategori (misal: Boxplot Harga per Kota). Tulis insight-nya.
- **Wajib Feature Engineering**: Ciptakan **1 fitur baru** dari data lama! (Contoh: dari kolom tanggal lahir, buat kolom `usia`. Dari panjang x lebar, buat `luas`). Jelaskan *mengapa* fitur ini berguna.

#### Tahap 4: Final Check & Kesimpulan (Estimasi: 45 menit)

- Jalankan kembali `info()` dan `isnull().sum()` – pastikan **tidak ada null** dan semua tipe data sudah benar.
- Tulis **Kesimpulan Akhir** yang menjawab 3 hal:
  1. **Status Data:** Tulis dengan huruf kapital: **"DATA SIAP UNTUK TAHAP PREPROCESSING / FEATURE SCALING"**.
  2. **Saran Algoritma:** Berdasarkan EDA (lihat distribusi, outlier, dan korelasi), rekomendasikan **1–2 algoritma ML** yang paling cocok. Berikan alasan logis (contoh: "Karena data kita banyak outlier dan hubungannya tidak linear, kita sarankan pakai Random Forest atau XGBoost").
  3. **Catatan Khusus:** Tulis 1 hal yang paling mengejutkan dari data ini selama proses EDA.

---

### C. OUTPUT YANG HARUS DIKUMPULKAN (Pukul 17.00 WIB)

| Output | Format | Keterangan |
| :--- | :--- | :--- |
| **Notebook Praktik** | `.ipynb` (Jupyter/Colab) | Kode harus sudah di-*run all*, tidak ada error. Beri *Markdown* penjelasan di setiap tahap. |
| **Slide Presentasi** | PPT / PDF (maks 5 slide) | Isi: Latar belakang data, 2 grafik paling berbicara, 1 tantangan terbesar, dan kesimpulan rekomendasi algoritma. |

---

### D. RUBRIK PENILAIAN (Skala 100)

| Aspek | Bobot | Kriteria Mahasiswa Unggul (Dapat Nilai A) |
| :--- | :--- | :--- |
| **Eksplorasi Mandiri (Pagi)** | 35% | Jawaban konsep mendalam, daftar fungsi > 25 item (bervariasi antar library), refleksi menghubungkan teori dan praktik. |
| **Pembersihan Data (Sore)** | 25% | Semua missing value, duplikat, outlier, dan tipe data berhasil ditangani dengan alasan statistik yang kuat. |
| **Visualisasi & Insight** | 20% | Grafik rapi, berlabel, dan setiap grafik disertai 1 kalimat kesimpulan (bukan cuma gambar). |
| **Feature Engineering & Kesiapan ML** | 10% | Fitur baru relevan dan rekomendasi algoritma didasarkan pada bukti EDA (misal: melihat distribusi atau korelasi). |
| **Kerapian & Presentasi** | 10% | Notebook terstruktur rapi, tidak error, dan presentasi kilat disampaikan dengan percaya diri. |

---

### E. PESAN TERAKHIR UNTUK MAHASISWA
>
> **"Jangan takut error! Error adalah bagian dari eksplorasi. Jika kode Anda error, copy-paste pesan errornya ke Google, itu adalah proses belajar yang paling efektif. Selamat menjadi Data Detective!"**
