# MODUL AJAR TEORI PENUNJANG
## Unsupervised Learning: Clustering untuk Data Industri Manufaktur

**Mata Kuliah:** Machine Learning
**Program Studi:** D4 Teknologi Rekayasa Informatika dan Komputer (TRIN)
**Institusi:** Politeknik Manufaktur Bandung (Polman Bandung)
**Pertemuan:** Minggu ke-2, Pertemuan ke-2 dari 4
**Topik:** Unsupervised Learning — Clustering dan Segmentasi Data Manufaktur
**Durasi:** 150 menit (teori + diskusi)
**Prasyarat:** Pertemuan 1 (Supervised Learning — Klasifikasi Defect Produk)


## DAFTAR ISI

1. [Pendahuluan](#1-pendahuluan)
2. [Unsupervised Learning: Fondasi Teori](#2-unsupervised-learning-fondasi-teori)
3. [Supervised vs Unsupervised: Perbandingan Mendalam](#3-supervised-vs-unsupervised-perbandingan-mendalam)
4. [Clustering: Definisi, Tujuan, dan Jenis](#4-clustering-definisi-tujuan-dan-jenis)
5. [Metrik Jarak dan Kemiripan](#5-metrik-jarak-dan-kemiripan)
6. [Algoritma K-Means Clustering](#6-algoritma-k-means-clustering)
7. [Algoritma Agglomerative Clustering (Hierarchical)](#7-algoritma-agglomerative-clustering-hierarchical)
8. [Algoritma DBSCAN](#8-algoritma-dbscan)
9. [Menentukan Jumlah Cluster Optimal](#9-menentukan-jumlah-cluster-optimal)
10. [Evaluasi Kualitas Clustering](#10-evaluasi-kualitas-clustering)
11. [Principal Component Analysis (PCA) untuk Visualisasi](#11-principal-component-analysis-pca-untuk-visualisasi)
12. [Preprocessing Data Manufaktur untuk Clustering](#12-preprocessing-data-manufaktur-untuk-clustering)
13. [Feature Engineering untuk Data Sensor Manufaktur](#13-feature-engineering-untuk-data-sensor-manufaktur)
14. [Interpretasi Hasil Clustering dalam Konteks Industri](#14-interpretasi-hasil-clustering-dalam-konteks-industri)
15. [Studi Kasus Industri Nyata](#15-studi-kasus-industri-nyata)
16. [Kesalahan Umum dan Cara Mengatasinya](#16-kesalahan-umum-dan-cara-mengatasinya)
17. [Latihan dan Soal Refleksi](#17-latihan-dan-soal-refleksi)
18. [Glosarium](#18-glosarium)
19. [Referensi](#19-referensi)


## 1. PENDAHULUAN

### 1.1 Konteks Minggu ke-2

Minggu ke-2 merupakan minggu tematik Machine Learning yang terdiri dari 4 pertemuan:

| Pertemuan | Topik | Fokus |
|---|---|---|
| Pertemuan 1 | Supervised Learning — Prediksi Defect Manufaktur | Klasifikasi dengan label |
| **Pertemuan 2** | **Unsupervised Learning — Clustering** | **Menemukan struktur tanpa label** |
| Pertemuan 3 | Reinforcement Learning | Belajar dari reward |
| Pertemuan 4 | Evaluasi Mingguan | Presentasi + Teori |

### 1.2 Mengapa Unsupervised Learning Penting dalam Manufaktur?

Dalam industri manufaktur modern, **data sensor mengalir terus-menerus** dari berbagai sumber: sensor suhu, tekanan, getaran, kecepatan, dan masih banyak lagi. Setiap detik, ribuan titik data dihasilkan dari lini produksi. Namun, **sebagian besar data ini tidak memiliki label**.

Bayangkan sebuah pabrik dengan puluhan mesin yang beroperasi 24 jam sehari. Setiap mesin dilengkapi dengan sensor yang mengukur berbagai parameter. Data mengalir deras, tetapi **tidak ada yang memberi tahu kita** apakah mesin sedang beroperasi dalam kondisi normal, transient, atau abnormal. Tidak ada label "normal" atau "fault" yang tertempel pada setiap baris data.

Di sinilah **unsupervised learning** berperan. Dengan teknik clustering, kita dapat menemukan **state operasional tersembunyi** dalam data sensor — mengelompokkan kondisi mesin yang serupa tanpa memerlukan label manual yang mahal dan memakan waktu.

**Unsupervised learning sangat penting dalam manufaktur karena:**

1. **Label mahal dan langka.** Proses pelabelan data sensor membutuhkan ahli domain, waktu, dan biaya besar.
2. **Data mengalir tanpa henti.** Volume data sensor terlalu besar untuk dianalisis secara manual.
3. **Anomali jarang terjadi.** Dalam data imbalanced, supervised learning sulit bekerja optimal.
4. **State operasional tidak diketahui sebelumnya.** Kita tidak tahu berapa banyak state yang ada — kita harus menemukannya.

Penelitian menunjukkan bahwa unsupervised time-series clustering dan anomaly detection sangat kritis dalam manufaktur, di mana volume besar data streaming dikumpulkan tetapi informasi berlabel sangat langka. Tugas-tugas ini mendukung penemuan pola operasional yang mendasari serta deteksi dini perilaku abnormal — kemampuan esensial untuk monitoring proses yang andal.

### 1.3 Peta Konsep Modul

```
Unsupervised Learning untuk Manufaktur
│
├── Fondasi Teori
│   ├── Unsupervised Learning
│   ├── Perbandingan dengan Supervised
│   └── Aplikasi di Manufaktur
│
├── Clustering
│   ├── K-Means (Partition-based)
│   ├── Agglomerative (Hierarchical)
│   ├── DBSCAN (Density-based)
│   └── Gaussian Mixture Models
│
├── Evaluasi
│   ├── Silhouette Score
│   ├── Davies-Bouldin Index
│   └── Calinski-Harabasz Index
│
├── Reduksi Dimensi
│   ├── PCA
│   └── Visualisasi 2D/3D
│
└── Aplikasi Manufaktur
    ├── State Detection
    ├── Anomaly Detection
    ├── Predictive Maintenance
    └── Process Optimization
```


## 2. UNSUPERVISED LEARNING: FONDASI TEORI

### 2.1 Definisi Formal

**Unsupervised Learning** adalah paradigma Machine Learning di mana model belajar dari **data tanpa label**. Kita hanya memiliki himpunan fitur $D = \{x_1, x_2, ..., x_n\}$ tanpa pasangan output $y$.

**Tujuan:** Menemukan **struktur tersembunyi** dalam data, seperti:
- **Kelompok (cluster)** yang serupa
- **Dimensi yang lebih rendah** yang menangkap esensi data
- **Aturan asosiasi** antar item

### 2.2 Karakteristik Unsupervised Learning

| Karakteristik | Penjelasan |
|---|---|
| **Tanpa Label** | Tidak ada ground truth untuk evaluasi |
| **Eksploratif** | Tujuan adalah menemukan pola, bukan memprediksi |
| **Subjektif** | Interpretasi hasil tergantung konteks |
| **Sensitif Preprocessing** | Scaling dan handling outlier sangat penting |
| **Tidak Ada Evaluasi Mutlak** | Evaluasi menggunakan metrik internal |

### 2.3 Jenis-Jenis Unsupervised Learning

```
Unsupervised Learning
│
├── Clustering (Pengelompokan)
│   ├── Partition-based: K-Means, K-Medoids
│   ├── Hierarchical: Agglomerative, Divisive
│   ├── Density-based: DBSCAN, OPTICS, HDBSCAN
│   ├── Model-based: Gaussian Mixture Models
│   └── Deep: Deep Embedded Clustering, Autoencoder + K-Means
│
├── Dimensionality Reduction (Reduksi Dimensi)
│   ├── Linear: PCA, SVD, LDA
│   └── Non-linear: t-SNE, UMAP, Autoencoder
│
└── Association Rules (Aturan Asosiasi)
    └── Apriori, FP-Growth
```

### 2.4 Kapan Menggunakan Unsupervised Learning?

Gunakan unsupervised learning ketika:

1. **Tidak ada label** yang tersedia atau label terlalu mahal untuk didapatkan.
2. **Ingin menemukan pola** yang tidak diketahui sebelumnya.
3. **Ingin mereduksi dimensi** data untuk visualisasi atau efisiensi.
4. **Ingin segmentasi** mesin, produk, atau pelanggan.
5. **Ingin deteksi anomali** dalam data sensor.

### 2.5 Tantangan Unsupervised Learning

| Tantangan | Deskripsi | Solusi |
|---|---|---|
| **Tidak Ada Ground Truth** | Sulit mengevaluasi apakah hasilnya "benar" | Gunakan metrik internal + validasi domain |
| **Subjektivitas** | Interpretasi tergantung konteks | Diskusikan dengan ahli domain |
| **Sensitif Preprocessing** | Hasil berbeda jika data tidak di-scale | StandardScaler wajib |
| **Penentuan Jumlah Cluster** | Sulit menentukan K yang tepat | Elbow + Silhouette |
| **Ketidakstabilan** | Hasil berbeda dengan random seed berbeda | Set random_state, validasi stabilitas |


## 3. SUPERVISED VS UNSUPERVISED: PERBANDINGAN MENDALAM

### 3.1 Tabel Perbandingan Lengkap

| Aspek | Supervised Learning | Unsupervised Learning |
|---|---|---|
| **Data** | $(x, y)$ — ada label | $(x)$ — tanpa label |
| **Tujuan** | Prediksi output $y$ dari $x$ | Temukan struktur dalam $x$ |
| **Contoh Task** | Klasifikasi, Regresi | Clustering, Dimensionality Reduction |
| **Evaluasi** | Accuracy, Precision, Recall, F1 | Silhouette, Davies-Bouldin, Calinski-Harabasz |
| **Ground Truth** | Ada | Tidak ada |
| **Contoh Algoritma** | Decision Tree, KNN, Random Forest | K-Means, DBSCAN, PCA |
| **Analogi** | Belajar dengan guru | Menjelajah sendiri |
| **Aplikasi Manufaktur** | Prediksi Defect | State Detection, Anomaly Detection |

### 3.2 Ilustrasi Visual

```
SUPERVISED LEARNING (Pertemuan 1):
┌─────────────────────────────────────────────────┐
│  Data Berlabel (Manufacturing Defects)          │
│  ┌──────────────┬──────────────┬─────────────┐  │
│  │ Production   │ QualityScore │ DefectStatus│  │
│  │ Volume       │              │             │  │
│  ├──────────────┼──────────────┼─────────────┤  │
│  │ 1000         │ 85           │ 0 (Low)     │  │
│  │ 2000         │ 45           │ 1 (High)    │  │
│  └──────────────┴──────────────┴─────────────┘  │
│                                  ↑              │
│                            LABEL (y)            │
│  Model belajar: f(Volume, QualityScore) → Defect│
└─────────────────────────────────────────────────┘

UNSUPERVISED LEARNING (Pertemuan 2):
┌─────────────────────────────────────────────────┐
│  Data Tanpa Label (Sensor Readings)             │
│  ┌──────────────┬──────────────┬─────────────┐  │
│  │ Temperature  │ Pressure     │ Vibration   │  │
│  ├──────────────┼──────────────┼─────────────┤  │
│  │ 75.2         │ 3.4          │ 0.12        │  │
│  │ 82.1         │ 4.1          │ 0.45        │  │
│  └──────────────┴──────────────┴─────────────┘  │
│         ↑                                       │
│    TANPA LABEL                                  │
│  Model mencari: kelompok dalam data             │
│  → Cluster 0: Steady State                      │
│  → Cluster 1: Transient State                   │
│  → Cluster 2: Abnormal State                    │
└─────────────────────────────────────────────────┘
```

### 3.3 Kapan Memilih yang Mana?

```
Apakah Anda memiliki label?
│
├── YA → Supervised Learning
│   ├── Output kategorikal? → Klasifikasi (Prediksi Defect)
│   └── Output numerik? → Regresi (Prediksi Kualitas)
│
└── TIDAK → Unsupervised Learning
    ├── Ingin mengelompokkan? → Clustering (State Detection)
    ├── Ingin mereduksi dimensi? → PCA/t-SNE
    └── Ingin menemukan aturan? → Association Rules
```

### 3.4 Hubungan dengan Pertemuan Sebelumnya

Pada **Pertemuan 1**, kita memprediksi defect produk menggunakan **supervised learning** — model belajar dari data berlabel untuk mengklasifikasikan produk ke dalam "High Defect" atau "Low Defect".

Pada **Pertemuan 2**, kita beralih ke **unsupervised learning** — model belajar dari data sensor tanpa label untuk menemukan **state operasional** mesin. Kedua pendekatan ini **saling melengkapi** dalam ekosistem quality control manufaktur:

- **Supervised:** Prediksi defect berdasarkan pola historis.
- **Unsupervised:** Deteksi anomali berdasarkan penyimpangan dari pola normal.

Dalam praktiknya, kedua pendekatan sering **dikombinasikan**: clustering untuk segmentasi awal, lalu klasifikasi per cluster untuk prediksi yang lebih akurat.


## 4. CLUSTERING: DEFINISI, TUJUAN, DAN JENIS

### 4.1 Definisi Clustering

**Clustering** adalah proses mengelompokkan objek-objek ke dalam kelompok (cluster) sedemikian rupa sehingga:

- Objek dalam **cluster yang sama** memiliki **kemiripan tinggi** (high intra-cluster similarity).
- Objek dalam **cluster berbeda** memiliki **kemiripan rendah** (low inter-cluster similarity).

### 4.2 Tujuan Clustering dalam Manufaktur

| Tujuan | Deskripsi | Contoh |
|---|---|---|
| **State Detection** | Mengidentifikasi state operasional mesin | Normal, Transient, Abnormal |
| **Segmentasi Produk** | Mengelompokkan produk berdasarkan karakteristik | Produk A, B, C |
| **Anomaly Detection** | Mengidentifikasi titik yang tidak termasuk cluster manapun | Outlier, fault |
| **Process Optimization** | Mengoptimalkan parameter proses | Setpoint optimal per cluster |
| **Predictive Maintenance** | Prediksi kapan mesin perlu maintenance | Clustering kondisi mesin |

### 4.3 Ilustrasi Clustering

```
Sebelum Clustering:          Sesudah Clustering:

    ●   ●                       ●   ●
  ●   ●   ●                   ●   ●   ●    ← Cluster A (Steady State)
    ●   ●                       ●   ●

        ●   ●                       ●   ●
      ●   ●   ●                   ●   ●   ●  ← Cluster B (Transient)
        ●   ●                       ●   ●

              ▲                         ▲
            ▲   ▲                     ▲   ▲  ← Cluster C (Abnormal)
              ▲                         ▲
```

### 4.4 Jenis-Jenis Clustering

#### A. Hard Clustering vs Soft Clustering

| Aspek | Hard Clustering | Soft Clustering |
|---|---|---|
| **Assign** | Setiap titik → tepat 1 cluster | Setiap titik → probabilitas ke setiap cluster |
| **Contoh** | K-Means, DBSCAN | Gaussian Mixture Models, Fuzzy C-Means |
| **Output** | Label cluster | Probabilitas |

#### B. Partition-based vs Hierarchical vs Density-based

| Jenis | Karakteristik | Contoh | Aplikasi Manufaktur |
|---|---|---|---|
| **Partition-based** | Bagi data menjadi K cluster | K-Means, K-Medoids | State detection, segmentasi |
| **Hierarchical** | Bangun hierarki cluster | Agglomerative, Divisive | Analisis multi-level |
| **Density-based** | Cluster = daerah padat | DBSCAN, HDBSCAN | Anomaly detection |
| **Model-based** | Asumsikan distribusi | Gaussian Mixture Models | Time-series clustering |

### 4.5 Kriteria Clustering yang Baik

1. **Compactness:** Titik dalam cluster yang sama harus berdekatan.
2. **Separation:** Cluster yang berbeda harus berjauhan.
3. **Stability:** Hasil harus konsisten across runs.
4. **Interpretability:** Cluster harus bisa dijelaskan maknanya.
5. **Actionability:** Cluster harus menghasilkan insight yang dapat ditindaklanjuti.


## 5. METRIK JARAK DAN KEMIRIPAN

### 5.1 Mengapa Jarak Penting?

Clustering bergantung pada **jarak** atau **kemiripan** antar titik. Pemilihan metrik jarak sangat mempengaruhi hasil clustering.

### 5.2 Metrik Jarak yang Umum

#### A. Euclidean Distance

**Definisi:** Jarak lurus antara dua titik dalam ruang Euclidean.

$$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

**Contoh:**
- $x = (1, 2)$, $y = (4, 6)$
- $d = \sqrt{(1-4)^2 + (2-6)^2} = \sqrt{9 + 16} = \sqrt{25} = 5$

**Kelebihan:** Intuitif, mudah dihitung.
**Kekurangan:** Sensitif terhadap skala fitur.

**Kapan digunakan:** Data numerik kontinu dengan skala seragam — **paling umum dalam data sensor manufaktur**. Penelitian menunjukkan bahwa metrik Euclidean distance merupakan pilihan paling umum dalam dataset industri, karena interpretabilitas dan robustness-nya.

#### B. Manhattan Distance (City Block)

**Definisi:** Jumlah jarak absolut antar koordinat.

$$d(x, y) = \sum_{i=1}^{n} |x_i - y_i|$$

**Kelebihan:** Lebih robust terhadap outlier.
**Kapan digunakan:** Data dengan outlier, atau data grid-like.

#### C. Cosine Similarity

**Definisi:** Cosinus sudut antara dua vektor.

$$\text{cosine}(x, y) = \frac{x \cdot y}{||x|| \cdot ||y||}$$

**Kapan digunakan:** Data teks (TF-IDF), data sparse.

#### D. Dynamic Time Warping (DTW)

**Definisi:** Ukuran kemiripan antara dua time-series yang mungkin memiliki kecepatan berbeda.

**Kapan digunakan:** Data time-series sensor, vibration analysis, acoustic signals.

### 5.3 Tabel Perbandingan Metrik Jarak

| Metrik | Rumus | Cocok untuk | Sensitif Outlier |
|---|---|---|---|
| Euclidean | $\sqrt{\sum (x_i - y_i)^2}$ | Data numerik sensor | Ya |
| Manhattan | $\sum \|x_i - y_i\|$ | Data dengan outlier | Tidak |
| Cosine | $\frac{x \cdot y}{\|x\| \|y\|}$ | Data teks | Tidak |
| DTW | — | Time-series | Tidak |

### 5.4 Pentingnya Feature Scaling

**Masalah:** Jika fitur memiliki skala berbeda, fitur dengan skala besar akan mendominasi perhitungan jarak.

**Contoh dalam manufaktur:**
- Fitur A: Suhu (0–100°C)
- Fitur B: Tekanan (0–10 bar)
- Fitur C: Getaran (0–0,01 mm)

Tanpa scaling, jarak Euclidean akan didominasi oleh Suhu. Getaran hampir tidak berpengaruh.

**Solusi:** StandardScaler atau MinMaxScaler.

**Dalam konteks manufaktur**, data sensor seringkali memiliki skala yang sangat berbeda (suhu dalam ratusan derajat, getaran dalam pecahan millimeter). Normalisasi fitur untuk menghilangkan efek scaling adalah langkah preprocessing standar yang wajib dilakukan.


## 6. ALGORITMA K-MEANS CLUSTERING

### 6.1 Konsep Dasar

**K-Means** adalah algoritma clustering **partition-based** yang membagi data menjadi **K cluster** dengan meminimalkan **Within-Cluster Sum of Squares (WCSS)**.

**Tujuan Optimasi:**
$$WCSS = \sum_{k=1}^{K} \sum_{x_i \in C_k} ||x_i - \mu_k||^2$$

di mana:
- $C_k$ = cluster ke-k
- $\mu_k$ = centroid (titik pusat) cluster ke-k
- $||x_i - \mu_k||^2$ = jarak kuadrat Euclidean

### 6.2 Cara Kerja K-Means

**Algoritma:**

```
1. Inisialisasi: Pilih K centroid secara acak (k-means++)
2. Assign: Assign setiap titik ke centroid terdekat
3. Update: Hitung ulang centroid sebagai rata-rata titik di cluster
4. Ulangi langkah 2-3 hingga konvergen (centroid tidak berubah)
```

**Ilustrasi:**

```
Iterasi 1:                    Iterasi 2:                    Iterasi 3 (Konvergen):

  ●   ●   ▲                     ●   ●                         ●   ●
●   ●   ●                     ●   ●   ●                     ●   ●   ●
  ●   ●                         ●   ●                         ●   ●

    ●   ●   ■                     ●   ●   ■                   ●   ●   ■
  ●   ●   ●                     ●   ●   ●                   ●   ●   ●
    ●   ●                         ●   ●                       ●   ●

▲ = centroid awal             ▲ = centroid baru             ▲ = centroid final
■ = centroid awal             ■ = centroid baru             ■ = centroid final
```

### 6.3 Contoh Perhitungan Manual

**Data:**
- A = (1, 1)
- B = (2, 1)
- C = (4, 3)
- D = (5, 4)

**Inisialisasi:** K=2, centroid awal: C1 = A (1,1), C2 = C (4,3)

**Iterasi 1 — Assign:**
- A ke C1: $d = 0$
- B ke C1: $d = \sqrt{(2-1)^2 + (1-1)^2} = 1$
- C ke C2: $d = 0$
- D ke C2: $d = \sqrt{(5-4)^2 + (4-3)^2} = \sqrt{2} \approx 1.41$

Cluster 1: {A, B}, Cluster 2: {C, D}

**Iterasi 1 — Update:**
- C1 baru = rata-rata A dan B = ((1+2)/2, (1+1)/2) = (1.5, 1)
- C2 baru = rata-rata C dan D = ((4+5)/2, (3+4)/2) = (4.5, 3.5)

**Iterasi 2 — Assign:**
- A ke C1: $d = \sqrt{(1-1.5)^2 + (1-1)^2} = 0.5$
- B ke C1: $d = \sqrt{(2-1.5)^2 + (1-1)^2} = 0.5$
- C ke C2: $d = \sqrt{(4-4.5)^2 + (3-3.5)^2} = 0.71$
- D ke C2: $d = \sqrt{(5-4.5)^2 + (4-3.5)^2} = 0.71$

Cluster tidak berubah → **Konvergen**.

### 6.4 Hyperparameter K-Means

```python
KMeans(
    n_clusters=3,        # Jumlah cluster (K)
    init='k-means++',    # Metode inisialisasi
    n_init=10,           # Jumlah inisialisasi berbeda
    max_iter=300,        # Iterasi maksimum
    tol=1e-4,            # Toleransi konvergensi
    random_state=42      # Reproducibility
)
```

**Penjelasan:**
- `n_clusters`: Jumlah cluster yang diinginkan.
- `init='k-means++'`: Inisialisasi cerdas — pilih centroid awal yang berjauhan. Penelitian menunjukkan bahwa metode inisialisasi k-means++ diterapkan untuk meningkatkan konvergensi.
- `n_init=10`: Jalankan 10 kali dengan inisialisasi berbeda, pilih yang terbaik.
- `max_iter`: Batas iterasi untuk mencegah infinite loop.
- `tol`: Toleransi perubahan centroid untuk konvergensi.

### 6.5 Kelebihan dan Kekurangan K-Means

| Kelebihan | Kekurangan |
|---|---|
| Cepat dan efisien (O(n × K × d × i)) | Harus menentukan K di awal |
| Sederhana dan mudah diimplementasikan | Sensitif terhadap inisialisasi |
| Mudah diinterpretasikan | Sensitif terhadap outlier |
| Skalabel untuk dataset besar | Hanya cocok untuk cluster berbentuk bulat |
| Konvergen cepat | Tidak cocok untuk cluster dengan kepadatan berbeda |

### 6.6 Kapan Menggunakan K-Means dalam Manufaktur?

✅ **Gunakan K-Means ketika:**
- Anda sudah memiliki perkiraan jumlah state operasional.
- Data sensor berbentuk cluster bulat/kompak.
- Dataset besar dan butuh algoritma cepat.
- Anda butuh hasil yang mudah diinterpretasikan.

**Contoh aplikasi:**
- **Condition monitoring** motor: K-Means berhasil mengelompokkan data menjadi **maintenance mode, normal mode, dan high risk mode** pada motor separator, main, dan fan dengan nilai Davies-Bouldin Index terbaik 0,548.
- **Deteksi anomali** pada laser sheet metal cutting machines: K-Means dan DBSCAN digunakan untuk mengidentifikasi pola anomali dari data piston via OPC UA.
- **Energy-efficient factory machines:** K-Means dengan automatic cluster detection digunakan untuk mendeteksi operating modes dan meningkatkan anomaly detection.

❌ **Hindari K-Means ketika:**
- Jumlah cluster tidak diketahui.
- Data memiliki outlier signifikan.
- Cluster berbentuk arbitrer.

### 6.7 Varian K-Means

| Varian | Karakteristik | Aplikasi Manufaktur |
|---|---|---|
| **K-Means++** | Inisialisasi cerdas (default) | Semua aplikasi |
| **K-Medoids** | Centroid = titik data aktual | Data dengan outlier |
| **Mini-Batch K-Means** | Untuk dataset sangat besar | Data sensor streaming |
| **Fuzzy C-Means** | Soft clustering | Analisis multi-state |
| **TimeSeriesKMeans** | Untuk data time-series | Sensor temporal |


## 7. ALGORITMA AGGLOMERATIVE CLUSTERING (HIERARCHICAL)

### 7.1 Konsep Dasar

**Agglomerative Clustering** adalah algoritma **hierarchical** yang membangun hierarki cluster secara **bottom-up**:

1. Mulai dengan setiap titik sebagai cluster sendiri.
2. Gabungkan dua cluster terdekat secara iteratif.
3. Ulangi hingga semua titik menjadi satu cluster.
4. Potong dendrogram pada level tertentu untuk mendapatkan K cluster.

### 7.2 Cara Kerja

```
Langkah 1: Setiap titik = cluster sendiri
   {A} {B} {C} {D} {E}

Langkah 2: Gabungkan yang terdekat (misal A dan B)
   {A,B} {C} {D} {E}

Langkah 3: Gabungkan lagi (misal D dan E)
   {A,B} {C} {D,E}

Langkah 4: Gabungkan lagi (misal {A,B} dan C)
   {A,B,C} {D,E}

Langkah 5: Gabungkan semua
   {A,B,C,D,E}
```

### 7.3 Dendrogram

**Dendrogram** adalah visualisasi hierarki cluster berbentuk pohon.

```
Distance
   │
 5 │─────────────────┐
   │                 │
 4 │         ┌───────┴───────┐
   │         │               │
 3 │     ┌───┴───┐       ┌───┴───┐
   │     │       │       │       │
 2 │ ┌───┴───┐   │   ┌───┴───┐   │
   │ │       │   │   │       │   │
 1 │ A       B   C   D       E   F
   └─────────────────────────────
```

**Cara Membaca:**
- Sumbu Y = jarak (semakin tinggi = semakin berbeda).
- Garis horizontal = penggabungan cluster.
- Potong dendrogram pada ketinggian tertentu untuk mendapatkan K cluster.

### 7.4 Linkage Criteria

**Linkage** menentukan bagaimana jarak antar cluster dihitung.

#### A. Single Linkage (Minimum)

Jarak = jarak **minimum** antar titik di dua cluster.

$$d(A, B) = \min_{a \in A, b \in B} d(a, b)$$

**Kelebihan:** Bisa menangani cluster non-eliptik.
**Kekurangan:** Sensitif terhadap noise, cenderung "chaining".

#### B. Complete Linkage (Maximum)

Jarak = jarak **maksimum** antar titik di dua cluster.

$$d(A, B) = \max_{a \in A, b \in B} d(a, b)$$

**Kelebihan:** Menghasilkan cluster kompak.
**Kekurangan:** Sensitif terhadap outlier.

#### C. Average Linkage

Jarak = **rata-rata** jarak antar semua pasangan titik.

$$d(A, B) = \frac{1}{|A| \cdot |B|} \sum_{a \in A} \sum_{b \in B} d(a, b)$$

**Kelebihan:** Kompromis antara single dan complete.
**Kekurangan:** Komputasi lebih mahal.

#### D. Ward Linkage

Meminimalkan **variance** dalam cluster setelah penggabungan.

$$d(A, B) = \sum_{x \in A \cup B} ||x - \mu_{A \cup B}||^2 - \sum_{x \in A} ||x - \mu_A||^2 - \sum_{x \in B} ||x - \mu_B||^2$$

**Kelebihan:** Menghasilkan cluster kompak dan seimbang.
**Kekurangan:** Cenderung menghasilkan cluster berukuran sama.

### 7.5 Hyperparameter Agglomerative

```python
AgglomerativeClustering(
    n_clusters=3,           # Jumlah cluster
    linkage='ward',         # 'ward', 'complete', 'average', 'single'
    metric='euclidean',     # Metrik jarak
    distance_threshold=None # Alternatif n_clusters
)
```

### 7.6 Kelebihan dan Kekurangan

| Kelebihan | Kekurangan |
|---|---|
| Tidak perlu menentukan K di awal | Kompleksitas tinggi O(n³) |
| Menghasilkan dendrogram informatif | Tidak cocok untuk dataset besar |
| Bisa menangkap hierarki | Sensitif terhadap noise |
| Deterministik (tidak ada random) | Sulit untuk data berdimensi tinggi |

### 7.7 Aplikasi dalam Manufaktur

Penelitian menunjukkan bahwa **Agglomerative Hierarchical Clustering (AHC)** digunakan bersama DBSCAN dan Self-Organizing Feature Map (SOFM) untuk **predictive maintenance** pada nutrunners di industri otomotif. Metrik yang digunakan adalah **Silhouette Coefficient Score (SC)** dan **Variation Rate Criterion (VRC)**.

**Kapan menggunakan Agglomerative dalam manufaktur:**
- Ingin memahami hierarki state operasional.
- Dataset kecil hingga sedang (< 10.000 sampel).
- Butuh dendrogram untuk visualisasi.
- Tidak yakin berapa jumlah state yang tepat.


## 8. ALGORITMA DBSCAN

### 8.1 Konsep Dasar

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) adalah algoritma clustering berbasis **kepadatan** (density).

**Ide Utama:** Cluster adalah daerah dengan **kepadatan tinggi**, dipisahkan oleh daerah dengan **kepadatan rendah**.

### 8.2 Terminologi DBSCAN

| Istilah | Definisi |
|---|---|
| **eps (ε)** | Radius maksimum untuk mencari tetangga |
| **min_samples** | Jumlah minimum titik dalam radius ε |
| **Core Point** | Titik dengan ≥ min_samples tetangga dalam radius ε |
| **Border Point** | Titik dalam radius ε dari core point, tapi bukan core point |
| **Noise Point** | Titik yang bukan core point maupun border point |
| **Density-Reachable** | Titik yang terhubung melalui rantai core points |

### 8.3 Ilustrasi

```
         ○ = Core Point (≥ min_samples tetangga)
         ● = Border Point (dalam radius core point)
         × = Noise Point (tidak terhubung)

              ○   ○
           ○   ○   ○
            ○   ○         ●
                          (border)
              ○   ○
                ○
                              × (noise)
```

### 8.4 Cara Kerja DBSCAN

```
1. Untuk setiap titik, hitung jumlah tetangga dalam radius eps
2. Identifikasi core points (≥ min_samples tetangga)
3. Bentuk cluster dari core points yang saling terhubung
4. Assign border points ke cluster terdekat
5. Titik yang tidak terhubung = noise (-1)
```

### 8.5 Hyperparameter DBSCAN

```python
DBSCAN(
    eps=0.5,             # Radius
    min_samples=5,       # Minimum titik dalam radius
    metric='euclidean',  # Metrik jarak
    algorithm='auto',    # 'auto', 'ball_tree', 'kd_tree', 'brute'
    n_jobs=-1            # Parallel processing
)
```

### 8.6 Menentukan eps dan min_samples

#### A. Metode k-distance Graph

1. Hitung jarak setiap titik ke tetangga ke-k (misal k=5).
2. Urutkan jarak tersebut.
3. Plot jarak terhadap indeks.
4. Cari titik "knee" (siku) — itulah eps optimal.

```python
from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances[:, 4], axis=0)

plt.plot(distances)
plt.ylabel('5th Nearest Neighbor Distance')
plt.xlabel('Data Points sorted by distance')
plt.show()
```

#### B. Aturan Praktis

- **min_samples:** Minimal = jumlah dimensi + 1. Umumnya 4–10.
- **eps:** Cari di k-distance graph. Jika tidak ada, coba beberapa nilai.

### 8.7 Kelebihan dan Kekurangan

| Kelebihan | Kekurangan |
|---|---|
| Tidak perlu menentukan K | Sensitif terhadap parameter eps dan min_samples |
| Bisa menemukan cluster berbentuk arbitrer | Sulit untuk data dengan kepadatan bervariasi |
| Menangani noise secara eksplisit | Tidak cocok untuk data berdimensi tinggi |
| Deterministik | Hasil bisa berbeda dengan eps berbeda |

### 8.8 Aplikasi DBSCAN dalam Manufaktur

Penelitian menunjukkan bahwa DBSCAN sangat efektif untuk **anomaly detection** dalam manufaktur:

- **Lubrication data:** Ketika DBSCAN diterapkan pada data lubrication, dua cluster berbeda muncul ketika PCA digunakan untuk memproyeksikan data ke ruang 2D. Cluster pertama ("normal cluster") terdiri dari 706 part yang menggambarkan pola yang diharapkan, sedangkan cluster kedua ("anomaly cluster") hanya berisi 4 part yang menyimpang secara signifikan — mengindikasikan potensi masalah dalam proses lubrication yang dapat menyebabkan defect.

- **PCB manufacturing:** K-means dan DBSCAN digunakan untuk menganalisis log data dari proses manufaktur PCB, mengidentifikasi area dengan tingkat defect tinggi dan memvisualisasikannya pada gambar PCB aktual.

- **Laser sheet metal cutting:** DBSCAN dan K-Means dibandingkan untuk mendeteksi pola anomali dari data piston via OPC UA, dengan fokus pada akumulasi slag yang menyebabkan gerakan tidak teratur.

- **Korelasi dengan Autoencoder:** Studi menunjukkan bahwa sampel yang terdeteksi sebagai anomali oleh DBSCAN **berkorelasi tinggi** dengan sampel yang memiliki reconstruction loss tertinggi dari autoencoder — mengonfirmasi efektivitas kedua metode dalam mendeteksi anomali.

### 8.9 Perbandingan K-Means vs DBSCAN

| Aspek | K-Means | DBSCAN |
|---|---|---|
| **Jumlah Cluster** | Harus ditentukan | Otomatis |
| **Bentuk Cluster** | Bulat/kompak | Arbitrer |
| **Noise** | Tidak ditangani | Ditangani (label -1) |
| **Outlier** | Sensitif | Robust |
| **Kecepatan** | Cepat | Sedang |
| **Parameter** | K | eps, min_samples |
| **Aplikasi Manufaktur** | State detection | Anomaly detection |


## 9. MENENTUKAN JUMLAH CLUSTER OPTIMAL

### 9.1 Tantangan

Menentukan jumlah cluster optimal adalah salah satu tantangan terbesar dalam clustering. Tidak ada jawaban mutlak — tergantung konteks dan tujuan.

### 9.2 Elbow Method

**Konsep:** Plot WCSS terhadap K. Cari "siku" (elbow) di mana penurunan WCSS mulai melambat.

```python
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.plot(K_range, inertias, 'bo-')
plt.xlabel('K')
plt.ylabel('WCSS (Inertia)')
plt.title('Elbow Method')
plt.show()
```

**Interpretasi:**
- WCSS selalu menurun dengan bertambahnya K.
- Cari titik di mana penurunan mulai "mendatar".
- Titik itu adalah kandidat K optimal.

**Kelebihan:** Intuitif, mudah.
**Kekurangan:** Kadang "siku" tidak jelas.

### 9.3 Silhouette Score

**Konsep:** Ukur seberapa baik setiap titik berada di cluster-nya.

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

- $a(i)$ = jarak rata-rata ke titik lain dalam cluster yang sama.
- $b(i)$ = jarak rata-rata ke titik di cluster terdekat lainnya.

**Range:** -1 hingga 1.

| Nilai | Interpretasi |
|---|---|
| > 0.70 | Struktur kuat |
| 0.50–0.70 | Struktur wajar |
| 0.25–0.50 | Struktur lemah |
| < 0.25 | Tidak ada struktur substansial |

```python
from sklearn.metrics import silhouette_score

silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

plt.plot(range(2, 11), silhouette_scores, 'ro-')
plt.xlabel('K')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs K')
plt.show()
```

### 9.4 Davies-Bouldin Index

**Konsep:** Rasio antara **within-cluster scatter** dan **between-cluster separation**.

$$DB = \frac{1}{K} \sum_{i=1}^{K} \max_{j \neq i} \left( \frac{\sigma_i + \sigma_j}{d(c_i, c_j)} \right)$$

**Semakin rendah semakin baik** (min 0).

```python
from sklearn.metrics import davies_bouldin_score

db_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = davies_bouldin_score(X_scaled, labels)
    db_scores.append(score)
```

**Dalam konteks manufaktur**, Davies-Bouldin Index rendah mengindikasikan jarak yang jelas antar cluster. Penelitian pada motor condition monitoring menunjukkan nilai Davies-Bouldin Index terbaik 0,548 untuk clustering maintenance mode, normal mode, dan high risk mode.

### 9.5 Calinski-Harabasz Index

**Konsep:** Rasio antara **between-cluster dispersion** dan **within-cluster dispersion**.

$$CH = \frac{\text{trace}(B_K) / (K-1)}{\text{trace}(W_K) / (n-K)}$$

**Semakin tinggi semakin baik.**

### 9.6 Tabel Perbandingan Metrik

| Metrik | Range | Semakin Baik | Kapan Digunakan |
|---|---|---|---|
| **Elbow (WCSS)** | 0–∞ | Rendah | Penentuan K awal |
| **Silhouette** | -1–1 | Tinggi | Evaluasi umum |
| **Davies-Bouldin** | 0–∞ | Rendah | Evaluasi umum |
| **Calinski-Harabasz** | 0–∞ | Tinggi | Evaluasi umum |

### 9.7 Strategi Kombinasi

**Best Practice:** Gunakan **minimal 2 metrik** untuk konfirmasi.

```
1. Elbow Method → kandidat K
2. Silhouette Score → konfirmasi K
3. Davies-Bouldin → konfirmasi K
4. Jika ketiganya setuju → K optimal
5. Jika tidak → pertimbangkan konteks bisnis
```

**Dalam praktik manufaktur**, evaluasi clustering sering menggunakan kombinasi **Silhouette Score** dan **Davies-Bouldin Index**. Hasil menunjukkan bahwa Silhouette Score tinggi mengindikasikan uniformitas data yang baik dalam cluster, sementara Davies-Bouldin Index rendah mengindikasikan jarak yang jelas antar cluster.


## 10. EVALUASI KUALITAS CLUSTERING

### 10.1 Internal Evaluation

Evaluasi berdasarkan **struktur internal** data (tanpa label).

| Metrik | Deskripsi | Interpretasi |
|---|---|---|
| **Silhouette Score** | Seberapa baik titik berada di cluster-nya | -1 hingga 1 (tinggi = baik) |
| **Davies-Bouldin Index** | Rasio within/between cluster | 0 hingga ∞ (rendah = baik) |
| **Calinski-Harabasz** | Rasio dispersi | 0 hingga ∞ (tinggi = baik) |
| **Dunn Index** | Rasio min inter-cluster / max intra-cluster | Tinggi = baik |

### 10.2 External Evaluation

Evaluasi berdasarkan **label ground truth** (jika tersedia).

| Metrik | Deskripsi |
|---|---|
| **Adjusted Rand Index (ARI)** | Kesamaan dengan label asli |
| **Normalized Mutual Information (NMI)** | Informasi bersama |
| **Fowlkes-Mallows Index** | Kesamaan pasangan |
| **Homogeneity, Completeness, V-Measure** | Kualitas cluster vs label |

**Catatan:** External evaluation jarang digunakan karena clustering biasanya diterapkan pada data tanpa label. Namun, dalam beberapa kasus manufaktur, label operasional mungkin tersedia untuk validasi.

### 10.3 Visual Evaluation

- **Scatter plot 2D/3D** dengan PCA atau t-SNE.
- **Dendrogram** untuk hierarchical clustering.
- **Heatmap** untuk melihat profil cluster.

### 10.4 Interpretasi Bisnis

Evaluasi terbaik adalah **interpretasi domain**:
- Apakah cluster bermakna secara operasional?
- Apakah cluster dapat ditindaklanjuti?
- Apakah cluster stabil across runs?

**Dalam konteks manufaktur**, cluster yang bermakna harus bisa dipetakan ke **state operasional nyata** — misalnya "steady state", "transient state", atau "abnormal state". Penelitian pada woodworking factory menunjukkan bahwa cluster yang dihasilkan **berkorelasi erat dengan mode operasional nyata mesin**, memudahkan integrasi mesin lama dengan smart factory.


## 11. PRINCIPAL COMPONENT ANALYSIS (PCA) UNTUK VISUALISASI

### 11.1 Konsep Dasar

**PCA** adalah teknik **dimensionality reduction** yang mengubah fitur asli menjadi **principal components** — kombinasi linear yang saling ortogonal dan menangkap varians maksimum.

**Tujuan:**
1. Mereduksi dimensi untuk visualisasi.
2. Mengurangi noise.
3. Mengatasi multikolinearitas.
4. Mempercepat komputasi.

### 11.2 Cara Kerja PCA

```
1. Standardisasi data (mean=0, std=1)
2. Hitung matriks kovarians
3. Hitung eigenvalue dan eigenvector
4. Urutkan eigenvector berdasarkan eigenvalue (descending)
5. Pilih top-k eigenvector sebagai principal components
6. Transformasi data ke ruang baru
```

### 11.3 Ilustrasi

```
Data Asli (2D):                  Setelah PCA:

    ●   ●                            ●
  ●   ●   ●                        ●   ●
    ●   ●                            ●
        ●   ●                          ●   ●
      ●   ●   ●                      ●   ●
        ●   ●                          ●

PC1 = arah varians maksimum
PC2 = arah ortogonal ke PC1
```

### 11.4 Explained Variance Ratio

**Explained Variance Ratio** menunjukkan berapa persen varians data yang ditangkap oleh setiap principal component.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total:", sum(pca.explained_variance_ratio_))
```

**Interpretasi:**
- PC1: 45% → menangkap 45% varians.
- PC2: 25% → menangkap 25% varians.
- Total 2 PC: 70% → cukup untuk visualisasi.

**Dalam studi manufaktur**, PCA berhasil mengompresi dataset 11-dimensi menjadi ruang 2-dimensi, di mana dua principal component pertama menjelaskan **>80% varians** — memungkinkan visualisasi pola dalam perilaku mesin yang efektif.

### 11.5 Memilih Jumlah Komponen

**Aturan Praktis:**
1. **Kaiser Criterion:** Pilih PC dengan eigenvalue > 1.
2. **Scree Plot:** Cari "elbow" pada plot explained variance.
3. **Threshold:** Pilih PC hingga total explained variance > 80%.

### 11.6 PCA untuk Visualisasi Clustering

**Best Practice:**
1. Standardisasi data.
2. Terapkan PCA ke 2D.
3. Plot scatter dengan warna cluster.
4. Interpretasikan pemisahan cluster.

```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('Visualisasi Cluster dengan PCA')
plt.colorbar(label='Cluster')
plt.show()
```

### 11.7 Hybrid Methods: PCA + Clustering

**Hybrid methods combining PCA with clustering have proven effective in enhancing fault detection capabilities.** Contohnya, sistem monitoring kesehatan railcar yang menggunakan DBSCAN clustering dengan PCA mencapai akurasi deteksi fault sebesar 96,4%.

**Kapan menggunakan PCA sebelum clustering:**
- Data berdimensi tinggi (> 10 fitur).
- Ada multikolinearitas antar fitur.
- Ingin visualisasi 2D/3D.
- Ingin mengurangi noise.

**Kapan TIDAK menggunakan PCA sebelum clustering:**
- Data sudah berdimensi rendah.
- Interpretabilitas fitur penting.
- PCA menghilangkan informasi penting.


## 12. PREPROCESSING DATA MANUFAKTUR UNTUK CLUSTERING

### 12.1 Mengapa Preprocessing Penting?

Clustering sangat sensitif terhadap:
- **Skala fitur** → StandardScaler wajib.
- **Missing value** → Harus di-handle.
- **Outlier** → Bisa mendistorsi cluster.
- **Multikolinearitas** → Bisa dideteksi dengan PCA.

### 12.2 Langkah Preprocessing

#### A. Handling Missing Value

```python
# Cek missing
df.isnull().sum()

# Opsi 1: Hapus baris dengan missing value
# Metode ini digunakan ketika jumlah data yang hilang relatif kecil
data = data.dropna()

# Opsi 2: Imputasi mean/median
df['col'].fillna(df['col'].median(), inplace=True)
```

**Dalam praktik manufaktur**, penghapusan baris dengan missing value digunakan ketika jumlah data yang hilang relatif kecil dan dianggap tidak akan mempengaruhi hasil analisis secara signifikan.

#### B. Handling Outlier

```python
# Metode IQR
Q1 = df['col'].quantile(0.25)
Q3 = df['col'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Capping
df['col'] = df['col'].clip(lower, upper)
```

#### C. Encoding Kategorikal

```python
# Label Encoding untuk ordinal
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['col_encoded'] = le.fit_transform(df['col'])

# One-Hot Encoding untuk nominal
df = pd.get_dummies(df, columns=['col_nominal'])

# Atau gunakan Unique Integer Encoding
for column in ['On Site', 'QR Number', 'Style', 'Process In']:
    data[column] = data[column].astype('category').cat.codes
```

**Dalam konteks manufaktur**, pengkodean fitur kategorikal menggunakan **Unique Integer** umum digunakan, di mana setiap kategori dalam fitur diubah menjadi nilai numerik untuk mempermudah proses pengelompokan.

#### D. Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Penting:** Untuk clustering, **selalu gunakan StandardScaler** karena algoritma berbasis jarak. Normalisasi fitur untuk menghilangkan efek scaling adalah langkah preprocessing standar dalam analisis data sensor manufaktur.

#### E. Feature Selection

Pilih fitur yang relevan untuk clustering. Hindari:
- **Identifier unik** (ID, NIM, nama).
- **Fitur dengan varians rendah** (hampir konstan).
- **Fitur yang redundan** (korelasi tinggi).

### 12.3 Pipeline Preprocessing yang Direkomendasikan

```
1. Load data
2. Hapus kolom tidak relevan (ID, nama)
3. Handle missing value (dropna atau imputasi)
4. Handle outlier (opsional, capping)
5. Encoding kategorikal (Unique Integer / One-Hot)
6. Feature engineering (opsional)
7. Feature scaling (StandardScaler)
8. Feature selection (opsional, VarianceThreshold)
9. Clustering
```


## 13. FEATURE ENGINEERING UNTUK DATA SENSOR MANUFAKTUR

### 13.1 Mengapa Feature Engineering Penting?

Data sensor mentah seringkali terlalu kompleks dan berisik. Feature engineering membantu mengekstrak **informasi yang bermakna** dari data mentah.

### 13.2 Teknik Feature Engineering untuk Data Manufaktur

#### A. Agregasi Statistik

```python
# Rata-rata, std, min, max dari window waktu
df['temp_mean'] = df['temperature'].rolling(window=10).mean()
df['temp_std'] = df['temperature'].rolling(window=10).std()
df['temp_max'] = df['temperature'].rolling(window=10).max()
df['temp_min'] = df['temperature'].rolling(window=10).min()
```

#### B. Fitur Turunan (Derived Features)

```python
# Rasio
df['efficiency_ratio'] = df['output'] / (df['energy_consumption'] + 1)

# Perbedaan
df['temp_diff'] = df['process_temp'] - df['air_temp']

# Perubahan
df['vibration_change'] = df['vibration'].diff()
```

#### C. Fitur Domain-Spesifik

```python
# Tool wear rate
df['tool_wear_rate'] = df['tool_wear'] / df['cycle_time']

# Power factor
df['power_factor'] = df['active_power'] / (df['voltage'] * df['current'] + 1)
```

#### D. STL Decomposition untuk Time-Series

**STL (Seasonal-Trend decomposition using LOESS)** memisahkan time-series menjadi komponen **trend**, **seasonal**, dan **residual**. Metode ini menghilangkan noise frekuensi tinggi dan efek musiman, yang sangat berguna untuk data sensor manufaktur.

```python
from statsmodels.tsa.seasonal import STL

stl = STL(df['sensor_reading'], period=24)
result = stl.fit()

df['trend'] = result.trend
df['seasonal'] = result.seasonal
df['residual'] = result.resid
```

### 13.3 Fitur Time-Series untuk Clustering

Untuk data sensor time-series, fitur yang berguna meliputi:

| Fitur | Deskripsi | Aplikasi |
|---|---|---|
| **Mean** | Rata-rata nilai | Level operasi |
| **Std** | Standar deviasi | Stabilitas |
| **Skewness** | Kemiringan distribusi | Asimetri |
| **Kurtosis** | Keruncingan distribusi | Outlier |
| **Autocorrelation** | Korelasi dengan lag | Periodisitas |
| **FFT Features** | Fitur frekuensi | Getaran, akustik |

### 13.4 Dimensionality Reduction untuk Data Manufaktur

**Modern machinery often incorporates numerous sensors to monitor system health, leading to a profusion of potentially redundant data.** Clustering dapat membantu mengidentifikasi dan mengelompokkan sensor readings yang berkorelasi, memungkinkan reduksi dimensi. Ini tidak hanya menghemat penyimpanan tetapi juga meningkatkan efisiensi komputasi.

**Teknik reduksi dimensi:**
- **PCA:** Linear, cepat, interpretable.
- **t-SNE:** Non-linear, untuk visualisasi.
- **Autoencoder:** Deep learning, untuk representasi kompleks.
- **Feature Selection:** Pilih fitur paling informatif.

**Perbandingan:**
Studi pada **tool wear detection** menunjukkan bahwa **PCA** berhasil mengompresi data 11-dimensi menjadi 2-dimensi dengan explained variance >80%, memungkinkan visualisasi pola perilaku mesin yang efektif. PCA menunjukkan stabilitas dan interpretabilitas superior dibandingkan t-SNE dan Isomap dalam konteks ini.


## 14. INTERPRETASI HASIL CLUSTERING DALAM KONTEKS INDUSTRI

### 14.1 Mengapa Interpretasi Penting?

Menghasilkan cluster itu mudah. **Memahami makna cluster** itu tantangan. Cluster tanpa interpretasi tidak berguna untuk pengambilan keputusan.

### 14.2 Langkah Interpretasi

#### A. Analisis Profil Cluster

Hitung **rata-rata setiap fitur** untuk setiap cluster.

```python
cluster_profile = df.groupby('cluster')[features].mean()
print(cluster_profile)
```

#### B. Visualisasi Profil

```python
# Bar chart
cluster_profile.T.plot(kind='bar', figsize=(12, 6))
plt.title('Profil Cluster')
plt.ylabel('Rata-rata Nilai')
plt.show()

# Heatmap
sns.heatmap(cluster_profile.T, annot=True, cmap='coolwarm')
plt.title('Heatmap Profil Cluster')
plt.show()
```

#### C. Pemberian Label Interpretatif

Berdasarkan profil, beri label yang bermakna:

| Cluster | Karakteristik | Label Manufaktur | Tindakan |
|---|---|---|---|
| 0 | Parameter stabil, variasi rendah | **Steady State** | Operasi normal |
| 1 | Parameter fluktuatif, variasi sedang | **Transient State** | Monitoring ketat |
| 2 | Parameter ekstrem, variasi tinggi | **Abnormal State** | Maintenance segera |

### 14.3 Contoh Interpretasi Konteks Manufaktur

**Cluster 0: "Steady State" (40% data)**
- Suhu stabil: 75°C ± 2°C
- Tekanan stabil: 3.5 bar ± 0.2 bar
- Getaran rendah: 0.05 mm ± 0.01 mm
- **Rekomendasi:** Operasi normal, tidak perlu intervensi.

**Cluster 1: "Transient State" (35% data)**
- Suhu fluktuatif: 82°C ± 8°C
- Tekanan fluktuatif: 4.2 bar ± 0.8 bar
- Getaran sedang: 0.15 mm ± 0.05 mm
- **Rekomendasi:** Monitoring lebih ketat, periksa setpoint.

**Cluster 2: "Abnormal State" (25% data)**
- Suhu tinggi: 95°C ± 15°C
- Tekanan tidak stabil: 5.5 bar ± 1.5 bar
- Getaran tinggi: 0.45 mm ± 0.2 mm
- **Rekomendasi:** Segera lakukan inspeksi, cek sistem pendingin dan pelumasan.

### 14.4 Actionable Insights

Cluster harus menghasilkan **tindakan**:

| Cluster | Tindakan | Dampak |
|---|---|---|
| **Steady State** | Pertahankan parameter | Efisiensi terjaga |
| **Transient State** | Optimasi setpoint | Mengurangi variasi |
| **Abnormal State** | Maintenance preventif | Mencegah kerusakan |

### 14.5 Validasi dengan Domain Expert

**Penting:** Diskusikan hasil dengan ahli domain (teknisi, engineer, operator) untuk memastikan interpretasi valid. Penelitian menunjukkan bahwa validasi oleh subject matter experts sangat penting untuk mengonfirmasi hasil clustering.


## 15. STUDI KASUS INDUSTRI NYATA

### 15.1 Case Study 1: Anomaly Detection pada Laser Sheet Metal Cutting Machines

**Konteks:** Piston pada laser sheet metal cutting machines mengalami akumulasi slag yang menyebabkan gerakan tidak teratur — ini adalah kondisi anomali yang perlu dideteksi.

**Data:** Data tekanan dan forward motion time dari piston, dikumpulkan via OPC UA.

**Pendekatan:** K-Means dan DBSCAN digunakan untuk mengidentifikasi pola anomali.

**Hasil:** Monitoring perilaku piston yang bergantung waktu dan deteksi anomali dapat menjadi langkah awal menuju preventive maintenance activities, meningkatkan efisiensi operasional dan mengurangi biaya maintenance.

### 15.2 Case Study 2: Lubrication Process Anomaly Detection

**Konteks:** Proses lubrication pada stamping production line.

**Pendekatan:** DBSCAN dengan PCA untuk reduksi dimensi.

**Hasil:** Dua cluster berbeda muncul — "normal cluster" (706 part) dan "anomaly cluster" (4 part). Keempat part anomali ini juga memiliki reconstruction loss tertinggi dari autoencoder, mengonfirmasi korelasi antar metode.

**Pelajaran:** DBSCAN sangat efektif untuk mendeteksi anomali dalam jumlah kecil yang menyimpang dari norma. Korelasi dengan metode lain (autoencoder) meningkatkan kepercayaan pada hasil.

### 15.3 Case Study 3: Energy-Efficient Factory Machines

**Konteks:** Deteksi mode operasional mesin di woodworking factory menggunakan sensor energi dan lingkungan.

**Pendekatan:** Enhanced K-Means dengan automatic cluster detection dan Self-Organizing Map (SOM).

**Hasil:** Cluster yang dihasilkan berkorelasi erat dengan mode operasional nyata mesin. Anomaly detection per cluster mencapai recall 1,0 dalam beberapa konfigurasi. Metode ini memungkinkan integrasi mesin lama (tanpa IoT) dengan smart factory.

### 15.4 Case Study 4: Predictive Maintenance Nutrunners (Automotive)

**Konteks:** Nutrunners di high-volume manufacturing environment dengan failure rate tinggi pada satu unit.

**Pendekatan:** Agglomerative Hierarchical Clustering (AHC), DBSCAN, dan Self-Organizing Feature Map (SOFM).

**Metrik:** Silhouette Coefficient Score (SC) dan Variation Rate Criterion (VRC).

**Hasil:** Feasible menggunakan clustering untuk meningkatkan strategi maintenance nutrunners di industri otomotif.

### 15.5 Case Study 5: PCB Manufacturing Defect Detection

**Konteks:** Log data dari proses manufaktur PCB.

**Pendekatan:** K-Means dan DBSCAN untuk mengidentifikasi area dengan tingkat defect tinggi.

**Hasil:** Sistem MVC dikembangkan untuk memvisualisasikan cluster defect pada gambar PCB aktual, memudahkan quality control.

### 15.6 Case Study 6: Anomaly Detection dengan Autoencoder + DBSCAN

**Konteks:** Stamping production line.

**Pendekatan:** Pipeline combining DBSCAN dan autoencoder sebagai fully unsupervised approach — DBSCAN digunakan untuk filter noise dan outlier sebelum training autoencoder.

**Pelajaran:** Kombinasi metode unsupervised dapat memberikan hasil yang lebih robust daripada metode tunggal.


## 16. KESALAHAN UMUM DAN CARA MENGATASINYA

### 16.1 Tidak Melakukan Scaling

**Kesalahan:**
```python
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)  # Tanpa scaling
```

**Akibat:** Fitur dengan skala besar mendominasi.

**Perbaikan:**
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans.fit(X_scaled)
```

### 16.2 Memilih K Tanpa Analisis

**Kesalahan:** Langsung `n_clusters=3` tanpa cek.

**Perbaikan:** Gunakan Elbow Method + Silhouette Score.

### 16.3 Mengabaikan Outlier

**Kesalahan:** Tidak cek outlier sebelum clustering.

**Perbaikan:**
```python
sns.boxplot(data=df)
# Handle dengan capping atau removal
```

### 16.4 Menggunakan Fitur ID

**Kesalahan:** Menyertakan kolom `id`, `nim`, `name`.

**Akibat:** Clustering berdasarkan ID, bukan pola.

**Perbaikan:** Hapus kolom identifier.

### 16.5 Tidak Interpretasi Hasil

**Kesalahan:** Hanya menghasilkan label cluster tanpa interpretasi.

**Perbaikan:** Analisis profil cluster, beri label bermakna.

### 16.6 Menggunakan Metrik yang Salah

**Kesalahan:** Menggunakan accuracy untuk clustering.

**Perbaikan:** Gunakan Silhouette, Davies-Bouldin, Calinski-Harabasz.

### 16.7 Tidak Reproducible

**Kesalahan:** Tidak set `random_state`.

**Perbaikan:** Selalu set `random_state=42`.

### 16.8 Overfitting pada Data Kecil

**Kesalahan:** Terlalu banyak cluster untuk data kecil.

**Perbaikan:** Gunakan cross-validation atau bootstrap.

### 16.9 Mengabaikan Domain Knowledge

**Kesalahan:** Hanya mengandalkan algoritma tanpa konteks.

**Perbaikan:** Diskusikan dengan ahli domain.

### 16.10 Tidak Memvalidasi Stabilitas

**Kesalahan:** Hanya menjalankan sekali.

**Perbaikan:** Jalankan dengan random seed berbeda, cek konsistensi.


## 17. LATIHAN DAN SOAL REFLEKSI

### 17.1 Latihan Coding

**Latihan 1:** Implementasi K-Means dari Scratch

```python
def kmeans_from_scratch(X, k, max_iter=100):
    # 1. Inisialisasi centroid secara acak
    # 2. Assign titik ke centroid terdekat
    # 3. Update centroid
    # 4. Ulangi hingga konvergen
    pass
```

**Latihan 2:** Hitung Silhouette Score Manual

```python
def silhouette_manual(X, labels):
    # Untuk setiap titik:
    #   a(i) = jarak rata-rata ke titik di cluster yang sama
    #   b(i) = jarak rata-rata ke titik di cluster terdekat
    #   s(i) = (b - a) / max(a, b)
    pass
```

**Latihan 3:** Visualisasi Dendrogram

```python
from scipy.cluster.hierarchy import dendrogram, linkage

linked = linkage(X_scaled, method='ward')
dendrogram(linked)
plt.show()
```

### 17.2 Soal Refleksi

1. **Konsep:** Apa perbedaan fundamental antara supervised dan unsupervised learning dalam konteks manufaktur?

2. **Aplikasi:** Sebutkan 3 contoh aplikasi clustering di industri manufaktur.

3. **Algoritma:** Kapan Anda memilih DBSCAN daripada K-Means untuk data sensor manufaktur? Jelaskan.

4. **Evaluasi:** Mengapa accuracy tidak cocok untuk clustering?

5. **Preprocessing:** Mengapa StandardScaler wajib untuk K-Means pada data sensor?

6. **Interpretasi:** Bagaimana Anda memberi label pada cluster yang dihasilkan dari data sensor?

7. **Etika:** Apa risiko menggunakan clustering untuk keputusan maintenance?

8. **Praktik:** Apa yang terjadi jika K terlalu besar? Terlalu kecil?

9. **PCA:** Mengapa PCA berguna untuk visualisasi clustering pada data manufaktur?

10. **Refleksi:** Bagaimana Anda akan menjelaskan hasil clustering ke stakeholder non-teknis?

### 17.3 Studi Kasus Mini

**Skenario:** Sebuah pabrik otomotif ingin mengelompokkan mesin berdasarkan pola sensor (suhu, getaran, tekanan) untuk predictive maintenance.

**Tugas:**
1. Apakah ini clustering? Mengapa?
2. Fitur apa yang akan digunakan?
3. Algoritma apa yang cocok? Mengapa?
4. Bagaimana mengevaluasi hasil?
5. Bagaimana menginterpretasikan cluster?


## 18. GLOSARIUM

| Istilah | Definisi |
|---|---|
| **Agglomerative** | Hierarchical clustering bottom-up |
| **Anomaly Detection** | Deteksi titik yang menyimpang dari norma |
| **Calinski-Harabasz** | Metrik evaluasi clustering (semakin tinggi semakin baik) |
| **Centroid** | Titik pusat cluster |
| **Cluster** | Kelompok objek yang serupa |
| **Cosine Similarity** | Ukuran kemiripan berbasis sudut |
| **Davies-Bouldin** | Metrik evaluasi clustering (semakin rendah semakin baik) |
| **DBSCAN** | Density-based clustering |
| **Dendrogram** | Visualisasi hierarki cluster |
| **Density** | Kepadatan titik dalam ruang |
| **DTW** | Dynamic Time Warping |
| **Elbow Method** | Metode penentuan K dengan plot WCSS |
| **Euclidean Distance** | Jarak lurus antar titik |
| **Explained Variance** | Varians yang ditangkap oleh principal component |
| **Hard Clustering** | Setiap titik → 1 cluster |
| **Hierarchical** | Clustering berbasis hierarki |
| **Inertia** | WCSS dalam K-Means |
| **K-Means** | Partition-based clustering |
| **Linkage** | Metode penggabungan cluster |
| **Manhattan Distance** | Jarak city-block |
| **Noise** | Titik yang tidak termasuk cluster |
| **Outlier** | Titik yang jauh dari cluster |
| **PCA** | Principal Component Analysis |
| **Predictive Maintenance** | Maintenance berbasis prediksi |
| **Silhouette Score** | Metrik evaluasi clustering (-1 hingga 1) |
| **Soft Clustering** | Setiap titik → probabilitas cluster |
| **State Detection** | Identifikasi state operasional |
| **STL** | Seasonal-Trend decomposition using LOESS |
| **Unsupervised Learning** | ML tanpa label |
| **WCSS** | Within-Cluster Sum of Squares |


## 19. REFERENSI

### 19.1 Buku

1. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media. — Bab 9: Unsupervised Learning Techniques.
2. Raschka, S., & Mirjalili, V. (2019). *Python Machine Learning* (3rd ed.). Packt. — Bab 10: Working with Unlabeled Data.
3. James, G., et al. (2021). *An Introduction to Statistical Learning*. Springer. — Bab 12.
4. Aggarwal, C. C. (2015). *Data Mining: The Textbook*. Springer.
5. Han, J., Kamber, M., & Pei, J. (2011). *Data Mining: Concepts and Techniques* (3rd ed.). Morgan Kaufmann.

### 19.2 Artikel dan Jurnal

1. Kim, H., et al. (2025). An Encoder-Agnostic Gaussian Mixture Framework for Unified Time-Series Analysis in Manufacturing. *ETRI Journal*.
2. West, N., et al. (2026). Label-free monitoring of screw driving using anomaly detection and fault clustering. *CIRP Annals*.
3. *Clustering-based anomaly detection for piston behaviour in laser sheet metal cutting machines*. (2026). *Uludağ University Journal of the Faculty of Engineering*.
4. *Boosting anomaly detection with unsupervised K-Means and SOM for energy-efficient factory machines*. (2025). *Journal of Intelligent Manufacturing*.
5. *Detection of the Defected Regions in Manufacturing Process Data using DBSCAN*. (2024). *KCI*.
6. *Predictive Maintenance using Clustering Methods for the use-case of Bolted Connections in the Automotive Industry*. (2025). *South African Journal of Industrial Engineering*.
7. Ramesh, K., et al. (2025). Comparison and assessment of machine learning approaches in manufacturing applications. *Industrial Artificial Intelligence*.
8. *DBSCAN and autoencoder for anomaly detection in stamping production line*. (2024). *IOP Conference Series: Materials Science and Engineering*.

### 19.3 Dokumentasi Online

1. **Scikit-learn Clustering:** https://scikit-learn.org/stable/modules/clustering.html
2. **Scikit-learn PCA:** https://scikit-learn.org/stable/modules/decomposition.html
3. **Scipy Hierarchical Clustering:** https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html

### 19.4 Dataset

1. Kaggle. (2022). *Tabular Playground Series - Jul 2022*. https://www.kaggle.com/competitions/tabular-playground-series-jul-2022/data
2. Meruva Kodanda Suraj. (2025). *Semiconductor Wafer Defect Classification Dataset*. Kaggle.
3. Feil, M. (2022). *Bosch CNC Machining Dataset*. UCI Machine Learning Repository.


## PENUTUP

Modul ajar ini disusun sebagai **penunjang teori** untuk praktikum pertemuan ke-2 minggu ke-2 tentang **Unsupervised Learning — Clustering untuk Data Industri Manufaktur**. Materi ini mencakup:

1. **Fondasi unsupervised learning** dan perbedaannya dengan supervised.
2. **Tiga algoritma clustering** utama: K-Means, Agglomerative, DBSCAN.
3. **Metrik evaluasi** clustering: Silhouette, Davies-Bouldin, Calinski-Harabasz.
4. **PCA** untuk visualisasi cluster.
5. **Preprocessing dan feature engineering** untuk data sensor manufaktur.
6. **Interpretasi hasil** clustering untuk pengambilan keputusan.
7. **Studi kasus industri nyata** dari berbagai sektor manufaktur.

**Pesan untuk Mahasiswa:**
> "Clustering adalah seni menemukan pola tersembunyi dalam data sensor. Dalam industri manufaktur, data mengalir tanpa henti tanpa label — kemampuan untuk menemukan state operasional dan anomali secara otomatis adalah keterampilan yang sangat berharga. Selalu kombinasikan algoritma, metrik evaluasi, dan domain knowledge."

**Pesan untuk Dosen:**
> "Modul ini bisa disesuaikan dengan tingkat pemahaman mahasiswa. Untuk pemula, fokus pada K-Means dan PCA terlebih dahulu. Untuk mahasiswa yang lebih mahir, tambahkan DBSCAN, hierarchical clustering, dan eksperimen dengan dataset sensor yang lebih kompleks."

---

**Dokumen ini disusun untuk keperluan edukasi di Politeknik Manufaktur Bandung (Polman Bandung).**
**Versi:** 1.0 | **Minggu ke-2, Pertemuan ke-2 dari 4**
**Terakhir Diperbarui:** 2026
