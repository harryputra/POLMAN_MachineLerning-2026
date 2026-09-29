# STUDI KASUS MACHINE LEARNING — PERTEMUAN KE-2 MINGGU KE-2 (REVISI)
## Unsupervised Learning: Segmentasi State Kontrol Produksi Manufaktur dengan Clustering

**Mata Kuliah:** Machine Learning
**Program Studi:** D4 Teknologi Rekayasa Informatika dan Komputer (TRIN)
**Institusi:** Politeknik Manufaktur Bandung (Polman Bandung)
**Pertemuan:** Minggu ke-2, Pertemuan ke-2 dari 4
**Topik:** Unsupervised Learning — Clustering untuk Segmentasi Produksi
**Tools:** Anaconda, Jupyter Notebook, Python 3.11
**Durasi:** 150 menit

> **Catatan Revisi:** Dataset diganti dari *Tabular Playground Series - Jul 2022* (500.000 baris, 47,55 MB) menjadi *Semiconductor Wafer Defect Classification Dataset* (5.000 baris, ~500 KB) agar dapat diolah dengan lancar pada laptop mahasiswa dengan RAM 8GB atau 16GB. Struktur studi kasus, algoritma, metrik evaluasi, dan tugas tetap sama — hanya dataset dan detail teknis yang disesuaikan.


## DAFTAR ISI

1. [Latar Belakang Studi Kasus](#1-latar-belakang-studi-kasus)
2. [Tujuan Pembelajaran](#2-tujuan-pembelajaran)
3. [Dataset yang Digunakan](#3-dataset-yang-digunakan)
4. [Landasan Teori Singkat](#4-landasan-teori-singkat)
5. [Step-by-Step Praktikum](#5-step-by-step-praktikum)
6. [Tugas Mandiri (Dikerjakan Hari Ini)](#6-tugas-mandiri-dikerjakan-hari-ini)
7. [Tugas Kelompok (Dikerjakan di Rumah)](#7-tugas-kelompok-dikerjakan-di-rumah)
8. [Rubrik Penilaian](#8-rubrik-penilaian)
9. [Referensi](#9-referensi)


## 1. LATAR BELAKANG STUDI KASUS

Pada pertemuan sebelumnya, kita telah mempelajari **supervised learning** untuk memprediksi defect produk manufaktur menggunakan data berlabel. Model yang dibangun mampu mengklasifikasikan produk ke dalam kategori "High Defect" atau "Low Defect".

Namun, dalam dunia industri manufaktur, tidak semua data memiliki label. Seringkali kita dihadapkan pada data sensor dari lini produksi yang **tidak memiliki anotasi** — kita tidak tahu state kontrol mana yang sedang aktif, atau apakah mesin beroperasi dalam kondisi normal atau abnormal. Di sinilah **unsupervised learning** berperan penting.

**Clustering** — salah satu teknik unsupervised learning — memungkinkan kita menemukan **state kontrol tersembunyi** dalam data produksi. Dalam konteks manufaktur, setiap state kontrol merepresentasikan kondisi operasional tertentu: startup, steady-state, transient, atau bahkan kondisi abnormal yang mengarah ke kegagalan.

Bayangkan sebuah pabrik semikonduktor yang memiliki lini produksi dengan puluhan sensor. Setiap wafer yang diproduksi melewati berbagai proses — oxidation, lithography, etching, deposition, dan CMP. Setiap proses memiliki parameter sensor seperti suhu, tekanan, gas flow, voltage, current, dan etch rate. Tanpa label, sulit untuk mengetahui apakah proses berjalan normal atau ada anomali yang mengindikasikan potensi defect. Dengan clustering, kita dapat mengelompokkan data sensor ke dalam **state-state proses yang bermakna** — memungkinkan **monitoring kondisi real-time** dan **deteksi dini anomali**.

Studi kasus ini sangat relevan dengan **Polman Bandung** sebagai institusi pendidikan vokasi yang fokus pada manufaktur dan otomasi industri. Mahasiswa D4 TRIN perlu memahami bagaimana data dari lini produksi dapat diolah menjadi **insight yang dapat ditindaklanjuti** tanpa memerlukan label manual yang mahal dan memakan waktu.

**Referensi Industri:**
- Dataset *Semiconductor Wafer Defect Classification Dataset* dirancang khusus untuk mendukung penelitian dan praktik machine learning industri, mencakup pembacaan sensor multivariat dari berbagai langkah proses fabrikasi, serta label defect biner yang menunjukkan apakah wafer lulus atau gagal inspeksi kualitas. Dataset ini cocok untuk tugas supervised dan unsupervised, termasuk **dimensionality reduction dengan PCA** dan **unsupervised anomaly detection**.
- Penelitian pada **injection molding machines** menunjukkan bahwa clustering dapat mengidentifikasi state operasional yang berbeda berdasarkan data sensor, memungkinkan optimasi proses dan deteksi anomali.
- Studi pada **Bosch CNC Machining Dataset** menggunakan clustering untuk mengelompokkan kondisi normal dan abnormal dari data accelerometer pada mesin milling.
- Pada **laser sheet metal cutting machines**, clustering dengan K-Means dan DBSCAN berhasil mengidentifikasi cluster berdasarkan pressure value dan forward movement time, memisahkan kondisi operasi yang berbeda.


## 2. TUJUAN PEMBELAJARAN

Setelah mengikuti praktikum ini, mahasiswa diharapkan mampu:

1. **Memahami** perbedaan fundamental antara supervised dan unsupervised learning dalam konteks manufaktur.
2. **Menjelaskan** konsep dasar clustering dan metrik kemiripan.
3. **Melakukan** preprocessing data sensor untuk clustering (scaling, handling missing value).
4. **Menentukan** jumlah cluster optimal menggunakan **Elbow Method** dan **Silhouette Score**.
5. **Melatih** tiga algoritma clustering: **K-Means**, **Agglomerative Clustering**, dan **DBSCAN**.
6. **Mengevaluasi** kualitas cluster dengan metrik yang tepat.
7. **Menginterpretasikan** hasil clustering dalam konteks state kontrol produksi.
8. **Memvisualisasikan** cluster menggunakan PCA untuk reduksi dimensi.


## 3. DATASET YANG DIGUNAKAN

### 3.1 Dataset Utama (Revisi)

| Informasi | Detail |
|---|---|
| **Nama Dataset** | Semiconductor Wafer Defect Classification Dataset |
| **Sumber** | Kaggle |
| **Link** | https://www.kaggle.com/datasets/meruvakodandasuraj/semiconductor-wafer-defect-classification-dataset |
| **Jumlah Data** | 5.000 baris |
| **Jumlah Fitur** | 10 kolom |
| **Ukuran File** | ~500 KB (627,59 kB) |
| **Tipe Data** | CSV (`semiconductor_wafer_defect_dataset.csv`) |
| **Lisensi** | Released for public use for research, education, and non-commercial experimentation |
| **Sifat Data** | Synthetic — dibuat untuk keperluan edukasi dan penelitian |

**Deskripsi Dataset:**
Dataset ini adalah dataset manufaktur wafer semikonduktor yang diinspirasi industri, dirancang untuk mendukung penelitian dan praktik machine learning industri. Dataset ini menangkap pembacaan sensor multivariat yang dikumpulkan dari berbagai langkah proses fabrikasi, bersama dengan label defect biner yang menunjukkan apakah wafer lulus atau gagal inspeksi kualitas.

Dataset ini **sangat ringan** — hanya 5.000 baris dan 10 kolom dengan ukuran file ~500 KB — sehingga dapat diolah dengan lancar pada laptop dengan RAM 8GB atau 16GB tanpa memerlukan sampling.

### 3.2 Deskripsi Kolom

| Kolom | Tipe | Deskripsi |
|---|---|---|
| `wafer_id` | Integer | Identifier unik untuk setiap wafer |
| `temperature_c` | Float | Suhu ruang proses (°C) |
| `pressure_torr` | Float | Tekanan ruang selama proses (Torr) |
| `gas_flow_sccm` | Float | Laju aliran gas proses (sccm) |
| `etch_rate_nm_min` | Float | Laju etsa material dari permukaan wafer (nm/menit) |
| `voltage_v` | Float | Tegangan peralatan selama proses |
| `current_ma` | Float | Arus peralatan selama proses |
| `process_step` | Kategorikal | Tahap manufaktur (Oxidation, Lithography, Etching, Deposition, CMP) |
| **`defect_label`** | **Integer** | **Target:** 0 = Non-defective, 1 = Defective |

**Catatan Penting:**
- Dataset ini **tidak memiliki missing value** — tidak perlu handling missing value.
- Fitur **belum di-scale** — perlu StandardScaler sebelum clustering.
- Kolom `process_step` adalah **kategorikal** dan memerlukan encoding.
- Kolom `wafer_id` adalah **identifier unik** dan harus dihapus sebelum clustering.
- Kolom `defect_label` adalah **label target** — dalam konteks unsupervised murni, kolom ini **tidak digunakan** untuk clustering, tetapi dapat digunakan untuk **validasi eksternal** (external evaluation) jika diperlukan.

### 3.3 Mengapa Dataset Ini Relevan?

1. **Konteks Industri:** Dataset ini mensimulasikan data manufaktur wafer semikonduktor dengan berbagai parameter proses dan state operasional.
2. **Ukuran Ringan:** 5.000 baris × 10 kolom, ~500 KB — dapat diolah pada laptop RAM 8GB tanpa masalah.
3. **Multi-Fitur:** 8 fitur mencakup berbagai aspek proses fabrikasi (suhu, tekanan, gas flow, etch rate, voltage, current).
4. **Cocok untuk Unsupervised:** Dataset dirancang untuk mendukung PCA, anomaly detection, dan clustering.
5. **Tantangan Realistis:** Data kontinu + kategorikal, tanpa missing value, dan perlu scaling — tantangan preprocessing yang edukatif.
6. **Referensi Luas:** Dataset ini digunakan untuk pembelajaran dan portofolio ML, dengan dokumentasi yang jelas.


## 4. LANDASAN TEORI SINGKAT

### 4.1 Supervised vs Unsupervised Learning

| Aspek | Supervised Learning | Unsupervised Learning |
|---|---|---|
| **Data** | $(x, y)$ berlabel | $(x)$ tanpa label |
| **Tujuan** | Prediksi output | Temukan struktur tersembunyi |
| **Contoh Task** | Klasifikasi, Regresi | Clustering, Dimensionality Reduction |
| **Evaluasi** | Accuracy, Precision, Recall | Silhouette Score, Davies-Bouldin |
| **Contoh Algoritma** | Decision Tree, KNN, Random Forest | K-Means, DBSCAN, PCA |

### 4.2 Clustering: Konsep Dasar

**Clustering** adalah teknik unsupervised learning yang mengelompokkan objek-objek yang **serupa** ke dalam kelompok yang sama (cluster), dan objek yang **berbeda** ke dalam kelompok yang berbeda.

**Tujuan:** Memaksimalkan **kemiripan intra-cluster** dan **meminimalkan kemiripan inter-cluster**.

**Metrik Kemiripan yang Umum:**
- **Euclidean Distance:** $\sqrt{\sum (x_i - y_i)^2}$
- **Manhattan Distance:** $\sum |x_i - y_i|$
- **Cosine Similarity:** $\frac{x \cdot y}{||x|| \cdot ||y||}$

### 4.3 Algoritma Clustering yang Digunakan

#### A. K-Means Clustering

**Konsep:** Membagi data menjadi K cluster dengan meminimalkan **Within-Cluster Sum of Squares (WCSS)**.

$$WCSS = \sum_{k=1}^{K} \sum_{x_i \in C_k} ||x_i - \mu_k||^2$$

**Cara Kerja:**
1. Inisialisasi K centroid secara acak.
2. Assign setiap titik ke centroid terdekat.
3. Update centroid berdasarkan rata-rata titik di cluster.
4. Ulangi langkah 2–3 hingga konvergen.

**Kelebihan:** Cepat, sederhana, mudah diinterpretasikan.
**Kekurangan:** Harus menentukan K terlebih dahulu; sensitif terhadap outlier.

#### B. Agglomerative Clustering (Hierarchical)

**Konsep:** Membangun hierarki cluster secara *bottom-up*. Setiap titik dimulai sebagai cluster sendiri, lalu cluster yang paling mirip digabungkan secara iteratif.

**Linkage Criteria:**
- **Ward:** Meminimalkan variance dalam cluster (default).
- **Complete:** Jarak maksimum antar cluster.
- **Average:** Jarak rata-rata antar cluster.

**Kelebihan:** Tidak perlu menentukan K di awal; menghasilkan dendrogram yang informatif.
**Kekurangan:** Kompleksitas komputasi tinggi (O(n³)).

#### C. DBSCAN (Density-Based Spatial Clustering)

**Konsep:** Mengelompokkan titik berdasarkan **kepadatan** (density). Titik yang berada di daerah padat membentuk cluster; titik di daerah jarang dianggap **noise**.

**Parameter:**
- **eps (ε):** Radius maksimum untuk mencari tetangga.
- **min_samples:** Jumlah minimum titik dalam radius ε.

**Kelebihan:** Tidak perlu menentukan K; bisa menemukan cluster berbentuk arbitrer; menangani noise.
**Kekurangan:** Sensitif terhadap parameter eps dan min_samples.

### 4.4 Menentukan Jumlah Cluster Optimal

#### A. Elbow Method

Plot **WCSS** terhadap jumlah cluster K. Cari titik di mana penurunan WCSS mulai melambat (membentuk "siku"/elbow).

#### B. Silhouette Score

Mengukur seberapa baik setiap titik berada di cluster-nya dibandingkan cluster lain.

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

di mana $a(i)$ = jarak rata-rata ke titik lain dalam cluster yang sama, $b(i)$ = jarak rata-rata ke titik di cluster terdekat lainnya.

**Interpretasi:**
- > 0,70: Struktur kuat
- 0,50–0,70: Struktur wajar
- 0,25–0,50: Struktur lemah
- < 0,25: Tidak ada struktur substansial

### 4.5 PCA untuk Visualisasi

**Principal Component Analysis (PCA)** adalah teknik reduksi dimensi yang mengubah fitur asli menjadi **principal components** — kombinasi linear yang saling ortogonal dan menangkap varians maksimum. Untuk visualisasi cluster, kita biasanya mereduksi data ke **2 dimensi** (PC1 dan PC2) sehingga dapat diplot dalam scatter plot 2D.


## 5. STEP-BY-STEP PRAKTIKUM

### 5.1 Setup Environment dan Import Library

Buka **Anaconda Prompt**, aktifkan environment, dan jalankan Jupyter Notebook:

```bash
conda activate ml-beasiswa
jupyter notebook
```

Buat notebook baru dengan nama `clustering_state_kontrol_manufaktur.ipynb`.

**Cell 1: Import Library**

```python
# Import library dasar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import library scikit-learn untuk clustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# Setting tampilan
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')
%matplotlib inline

print("Semua library berhasil diimport!")
```

**Penjelasan:**
- `KMeans`, `AgglomerativeClustering`, `DBSCAN` — tiga algoritma clustering.
- `silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score` — metrik evaluasi.
- `PCA` — reduksi dimensi untuk visualisasi.
- `dendrogram`, `linkage` — untuk visualisasi hierarchical clustering.


### 5.2 Load Dataset

**Cell 2: Load Dataset**

```python
# Load dataset
df = pd.read_csv('semiconductor_wafer_defect_dataset.csv')

# Tampilkan informasi awal
print("Ukuran dataset:", df.shape)
print("\n5 Baris Pertama:")
df.head()
```

**Penjelasan:**
- Dataset ini **hanya 5.000 baris dan 10 kolom** — tidak perlu sampling.
- `pd.read_csv()` membaca file CSV menjadi DataFrame.
- `df.shape` menampilkan jumlah baris dan kolom.
- `df.head()` menampilkan 5 baris pertama.

**Output yang diharapkan:** Tabel dengan 5.000 baris dan 10 kolom.


### 5.3 Eksplorasi Data (EDA)

**Cell 3: Informasi Umum Dataset**

```python
# Informasi tipe data
print("Informasi Dataset:")
df.info()

# Statistik deskriptif
print("\nStatistik Deskriptif:")
df.describe()
```

**Penjelasan:**
- `df.info()` menampilkan tipe data dan jumlah nilai non-null per kolom.
- Dataset ini **tidak memiliki missing value** — semua kolom memiliki 5.000 nilai non-null.
- `df.describe()` memberikan statistik ringkasan untuk kolom numerik.

**Cell 4: Cek Missing Value dan Duplikat**

```python
# Cek missing value
print("Jumlah Missing Value per Kolom:")
print(df.isnull().sum())

# Cek duplikat
print(f"\nJumlah Data Duplikat: {df.duplicated().sum()}")
```

**Penjelasan:**
- Dataset ini **tidak memiliki missing value**.
- Jika ada duplikat, perlu dipertimbangkan untuk dihapus.

**Cell 5: Identifikasi Tipe Kolom**

```python
# Pisahkan kolom numerik dan kategorikal
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Hapus wafer_id dan defect_label dari daftar fitur
numerical_cols = [col for col in numerical_cols if col not in ['wafer_id', 'defect_label']]

print(f"Jumlah kolom numerik (fitur): {len(numerical_cols)}")
print(f"Jumlah kolom kategorikal: {len(categorical_cols)}")
print(f"\nKolom numerik: {numerical_cols}")
print(f"\nKolom kategorikal: {categorical_cols}")
```

**Penjelasan:**
- Dataset memiliki 6 kolom numerik (fitur sensor) + 1 kolom kategorikal (`process_step`).
- Kolom `wafer_id` adalah identifier unik — harus dihapus.
- Kolom `defect_label` adalah label target — dalam konteks unsupervised murni, kolom ini tidak digunakan untuk clustering.

**Cell 6: Visualisasi Distribusi Fitur Numerik**

```python
# Histogram distribusi fitur numerik
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], color='steelblue')
    axes[i].set_title(f'Distribusi {col}', fontsize=11)
    axes[i].set_xlabel('')

plt.tight_layout()
plt.suptitle('Distribusi Fitur Numerik Sensor', fontsize=14, y=1.02)
plt.show()
```

**Penjelasan:**
- Setiap histogram menunjukkan sebaran nilai untuk fitur numerik.
- Perhatikan bentuk distribusi: normal, miring ke kiri, atau miring ke kanan.
- Fitur seperti `etch_rate_nm_min`, `voltage_v`, dan `current_ma` mungkin memiliki distribusi yang berbeda.

**Cell 7: Heatmap Korelasi**

```python
# Heatmap korelasi fitur numerik
plt.figure(figsize=(10, 8))
correlation = df[numerical_cols].corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, mask=mask, annot=True, cmap='coolwarm',
            fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
plt.title('Matriks Korelasi Fitur Numerik', fontsize=14)
plt.tight_layout()
plt.show()
```

**Penjelasan:**
- Heatmap menunjukkan korelasi antar fitur numerik.
- Fitur yang berkorelasi tinggi mungkin redundan dan bisa dipertimbangkan untuk di-drop.
- Perhatikan apakah ada **multikolinearitas** yang signifikan.

**Cell 8: Analisis Distribusi Process Step**

```python
# Distribusi process_step
print("Distribusi Process Step:")
print(df['process_step'].value_counts())
print(f"\nPersentase:")
print(df['process_step'].value_counts(normalize=True) * 100)

# Visualisasi
plt.figure(figsize=(8, 5))
sns.countplot(x='process_step', data=df, palette='Set2')
plt.title('Distribusi Process Step')
plt.xlabel('Process Step')
plt.ylabel('Jumlah')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

**Penjelasan:**
- `process_step` memiliki 5 kategori: Oxidation, Lithography, Etching, Deposition, CMP.
- Distribusi kategori perlu diperiksa untuk memastikan tidak ada kategori yang terlalu dominan.


### 5.4 Preprocessing Data

**Cell 9: Hapus Kolom Tidak Diperlukan**

```python
# Hapus kolom identifier dan target (untuk unsupervised murni)
df_clean = df.drop(['wafer_id', 'defect_label'], axis=1).copy()

print("Kolom setelah dihapus:")
print(df_clean.columns.tolist())
print(f"\nShape df_clean: {df_clean.shape}")
```

**Penjelasan:**
- `wafer_id` adalah **identifier unik** — tidak memiliki nilai prediktif untuk clustering.
- `defect_label` adalah **label target** — dalam konteks unsupervised murni, kita tidak menggunakan label untuk clustering. Namun, label ini bisa disimpan terpisah jika ingin melakukan **validasi eksternal** di akhir.

**Cell 10: Encoding Fitur Kategorikal**

```python
# Label Encoding untuk process_step
le = LabelEncoder()
df_clean['process_step_encoded'] = le.fit_transform(df_clean['process_step'])

print("Mapping Process Step:")
for i, cls in enumerate(le.classes_):
    print(f"  {cls} → {i}")

print(f"\n5 Baris setelah encoding:")
print(df_clean[['process_step', 'process_step_encoded']].head())
```

**Penjelasan:**
- `process_step` diubah menjadi angka menggunakan `LabelEncoder`.
- Setiap kategori proses (Oxidation, Lithography, Etching, Deposition, CMP) diberi nilai integer unik.

**Cell 11: Pilih Fitur untuk Clustering**

```python
# Fitur yang akan digunakan untuk clustering
clustering_features = numerical_cols + ['process_step_encoded']

X = df_clean[clustering_features].copy()

print(f"Jumlah fitur untuk clustering: {len(clustering_features)}")
print(f"Shape X: {X.shape}")
print(f"\nFitur yang digunakan: {clustering_features}")
```

**Penjelasan:**
- Kita menggunakan **7 fitur** untuk clustering: 6 fitur sensor numerik + 1 fitur kategorikal yang sudah di-encoding.
- `X` adalah matriks fitur yang akan di-cluster.

**Cell 12: Feature Scaling (WAJIB untuk Clustering)**

```python
# StandardScaler — WAJIB untuk clustering berbasis jarak
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Konversi kembali ke DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=clustering_features)

print("Setelah StandardScaler:")
print(f"Mean: {X_scaled_df.mean().round(4).tolist()}")
print(f"Std:  {X_scaled_df.std().round(4).tolist()}")
print(f"\nShape X_scaled: {X_scaled.shape}")
```

**Penjelasan:**
- **StandardScaler sangat penting untuk clustering** karena algoritma seperti K-Means dan DBSCAN menggunakan **jarak Euclidean**.
- Tanpa scaling, fitur dengan skala besar (misalnya `etch_rate_nm_min` dengan rentang ratusan) akan mendominasi fitur dengan skala kecil (misalnya `pressure_torr` dengan rentang satuan).
- Setelah scaling, semua fitur memiliki mean ≈ 0 dan std ≈ 1.


### 5.5 Menentukan Jumlah Cluster Optimal

**Cell 13: Elbow Method untuk K-Means**

```python
# Elbow Method: Plot WCSS terhadap jumlah cluster K
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-', markersize=10, linewidth=2)
plt.xlabel('Jumlah Cluster (K)', fontsize=12)
plt.ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12)
plt.title('Elbow Method — Menentukan Jumlah Cluster Optimal', fontsize=14)
plt.xticks(K_range)
plt.grid(True, alpha=0.3)

# Annotate elbow point
plt.annotate('Elbow Point\n(Kandidat Optimal)', xy=(3, inertias[2]),
             xytext=(5, inertias[2] + 20),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=11, color='red')

plt.tight_layout()
plt.show()

# Tampilkan nilai WCSS
print("WCSS per K:")
for k, wcss in zip(K_range, inertias):
    print(f"  K={k}: {wcss:.2f}")
```

**Penjelasan:**
- **WCSS** mengukur total jarak kuadrat antara setiap titik dan centroid cluster-nya.
- Semakin kecil WCSS, semakin rapat cluster.
- Kita mencari titik di mana penurunan WCSS **mulai melambat** — itulah "elbow".

**Cell 14: Silhouette Score untuk Konfirmasi**

```python
# Silhouette Score untuk K=2 hingga K=10
silhouette_scores = []
K_range_sil = range(2, 11)

for k in K_range_sil:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

# Plot Silhouette Score
plt.figure(figsize=(10, 6))
plt.plot(K_range_sil, silhouette_scores, 'ro-', markersize=10, linewidth=2)
plt.xlabel('Jumlah Cluster (K)', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.title('Silhouette Score — Menentukan Jumlah Cluster Optimal', fontsize=14)
plt.xticks(K_range_sil)
plt.grid(True, alpha=0.3)

# Highlight nilai tertinggi
best_k = K_range_sil[np.argmax(silhouette_scores)]
best_score = max(silhouette_scores)
plt.axvline(x=best_k, color='green', linestyle='--', alpha=0.7)
plt.annotate(f'Best K={best_k}\nScore={best_score:.3f}',
             xy=(best_k, best_score), xytext=(best_k+1, best_score-0.02),
             arrowprops=dict(arrowstyle='->', color='green', lw=2),
             fontsize=11, color='green')

plt.tight_layout()
plt.show()

print("Silhouette Score per K:")
for k, score in zip(K_range_sil, silhouette_scores):
    marker = " ← BEST" if k == best_k else ""
    print(f"  K={k}: {score:.4f}{marker}")
```

**Penjelasan:**
- **Silhouette Score** mengukur seberapa baik setiap titik berada di cluster-nya.
- Nilai mendekati 1 = cluster sangat baik.
- Kita memilih K dengan **Silhouette Score tertinggi**.
- Jika Elbow Method dan Silhouette Score **setuju** pada K yang sama, itu pertanda kuat bahwa K tersebut optimal.

**Keputusan:** Berdasarkan kedua metode, kita akan menggunakan **K = 3** (atau sesuai hasil aktual) untuk analisis selanjutnya.


### 5.6 Modeling: K-Means Clustering

**Cell 15: Latih K-Means dengan K Optimal**

```python
# Tentukan K optimal (sesuaikan dengan hasil Elbow + Silhouette)
OPTIMAL_K = 3  # Ganti jika hasil analisis berbeda

# Latih K-Means
kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Tambahkan label cluster ke DataFrame
df_clean['cluster_kmeans'] = kmeans_labels

# Evaluasi
sil_kmeans = silhouette_score(X_scaled, kmeans_labels)
db_kmeans = davies_bouldin_score(X_scaled, kmeans_labels)
ch_kmeans = calinski_harabasz_score(X_scaled, kmeans_labels)

print("=" * 50)
print(f"K-MEANS CLUSTERING (K={OPTIMAL_K})")
print("=" * 50)
print(f"Silhouette Score:        {sil_kmeans:.4f}")
print(f"Davies-Bouldin Index:    {db_kmeans:.4f}")
print(f"Calinski-Harabasz Index: {ch_kmeans:.2f}")
print(f"\nDistribusi Cluster:")
print(df_clean['cluster_kmeans'].value_counts().sort_index())
```

**Penjelasan:**
- `n_init=10` berarti algoritma dijalankan 10 kali dengan inisialisasi berbeda, dan yang terbaik (WCSS terkecil) dipilih.
- **Silhouette Score:** Semakin tinggi semakin baik (max 1).
- **Davies-Bouldin Index:** Semakin rendah semakin baik (min 0).
- **Calinski-Harabasz Index:** Semakin tinggi semakin baik.

**Cell 16: Profil Setiap Cluster (K-Means)**

```python
# Analisis profil cluster
cluster_profile = df_clean.groupby('cluster_kmeans')[clustering_features].mean().round(2)

print("Profil Rata-rata Setiap Cluster:")
print("=" * 80)
print(cluster_profile.T.to_string())

# Visualisasi profil cluster
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, col in enumerate(clustering_features):
    sns.boxplot(x='cluster_kmeans', y=col, data=df_clean, ax=axes[i], palette='Set2')
    axes[i].set_title(f'{col} per Cluster', fontsize=10)
    axes[i].set_xlabel('Cluster')

plt.tight_layout()
plt.suptitle('Profil Fitur per Cluster (K-Means)', fontsize=14, y=1.02)
plt.show()
```

**Penjelasan:**
- Tabel profil menunjukkan **rata-rata setiap fitur** untuk masing-masing cluster.
- Boxplot membantu memvisualisasikan distribusi setiap fitur per cluster.
- Dari sini kita bisa menginterpretasikan karakteristik setiap cluster.


### 5.7 Modeling: Agglomerative Clustering

**Cell 17: Dendrogram untuk Hierarchical Clustering**

```python
# Dendrogram (gunakan subsample untuk efisiensi)
sample_size = min(1000, len(X_scaled))
indices = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sample = X_scaled[indices]

plt.figure(figsize=(14, 8))
linked = linkage(X_sample, method='ward')

dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True,
           truncate_mode='lastp',
           p=20,
           leaf_rotation=90,
           leaf_font_size=10)

plt.title('Dendrogram — Agglomerative Clustering (Ward Linkage)', fontsize=14)
plt.xlabel('Sample Index / Cluster Size', fontsize=12)
plt.ylabel('Euclidean Distance', fontsize=12)
plt.tight_layout()
plt.show()
```

**Penjelasan:**
- **Dendrogram** menunjukkan hierarki penggabungan cluster.
- Sumbu Y adalah jarak (Euclidean). Semakin tinggi penggabungan, semakin berbeda cluster tersebut.
- Kita memilih K dengan melihat "gap" terbesar pada dendrogram.

**Cell 18: Latih Agglomerative Clustering**

```python
# Latih Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=OPTIMAL_K, linkage='ward', metric='euclidean')
agg_labels = agg.fit_predict(X_scaled)

# Tambahkan label
df_clean['cluster_agg'] = agg_labels

# Evaluasi
sil_agg = silhouette_score(X_scaled, agg_labels)
db_agg = davies_bouldin_score(X_scaled, agg_labels)
ch_agg = calinski_harabasz_score(X_scaled, agg_labels)

print("=" * 50)
print(f"AGGLOMERATIVE CLUSTERING (K={OPTIMAL_K}, Ward)")
print("=" * 50)
print(f"Silhouette Score:        {sil_agg:.4f}")
print(f"Davies-Bouldin Index:    {db_agg:.4f}")
print(f"Calinski-Harabasz Index: {ch_agg:.2f}")
print(f"\nDistribusi Cluster:")
print(df_clean['cluster_agg'].value_counts().sort_index())
```

**Penjelasan:**
- `linkage='ward'` meminimalkan variance dalam cluster.
- Hasil Agglomerative biasanya mirip dengan K-Means jika menggunakan Ward linkage.
- Bandingkan metrik evaluasinya dengan K-Means.


### 5.8 Modeling: DBSCAN

**Cell 19: DBSCAN — Mencari Parameter Optimal**

```python
from sklearn.neighbors import NearestNeighbors

# Metode k-distance untuk menentukan eps
k = 5  # min_samples kandidat
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)

# Urutkan jarak ke tetangga ke-k
distances = np.sort(distances[:, k-1], axis=0)

# Plot k-distance graph
plt.figure(figsize=(10, 6))
plt.plot(distances, linewidth=2)
plt.xlabel('Data Points (sorted by distance)', fontsize=12)
plt.ylabel(f'{k}-th Nearest Neighbor Distance', fontsize=12)
plt.title(f'k-Distance Graph (k={k}) — Menentukan eps Optimal', fontsize=14)
plt.axhline(y=1.5, color='red', linestyle='--', label='Contoh eps = 1.5')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Interpretasi: Titik 'knee' pada kurva menunjukkan eps yang optimal.")
print("Cari titik di mana kurva mulai melonjak tajam.")
```

**Penjelasan:**
- **k-distance graph** membantu menentukan `eps` yang optimal.
- Titik "knee" (siku) pada kurva adalah kandidat `eps` yang baik.

**Cell 20: Latih DBSCAN**

```python
# Coba beberapa kombinasi eps dan min_samples
eps_values = [0.8, 1.0, 1.2, 1.5, 1.8]
min_samples_values = [3, 5, 7]

dbscan_results = []

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        # Hitung silhouette hanya jika ada lebih dari 1 cluster
        if n_clusters > 1:
            sil = silhouette_score(X_scaled, labels)
        else:
            sil = -1

        dbscan_results.append({
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_pct': round(n_noise/len(labels)*100, 2),
            'silhouette': round(sil, 4) if sil != -1 else 'N/A'
        })

# Tampilkan hasil
dbscan_df = pd.DataFrame(dbscan_results)
print("Hasil Eksperimen DBSCAN:")
print(dbscan_df.to_string(index=False))
```

**Penjelasan:**
- Kita mencoba **kombinasi eps dan min_samples** yang berbeda.
- Untuk setiap kombinasi, kita catat jumlah cluster, jumlah noise, dan silhouette score.
- Kita mencari kombinasi yang menghasilkan **cluster bermakna** (jumlah cluster > 1) dan **noise rendah**.

**Cell 21: Latih DBSCAN dengan Parameter Terbaik**

```python
# Pilih parameter terbaik berdasarkan eksperimen
BEST_EPS = 1.5
BEST_MIN_SAMPLES = 5

dbscan = DBSCAN(eps=BEST_EPS, min_samples=BEST_MIN_SAMPLES)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Tambahkan label
df_clean['cluster_dbscan'] = dbscan_labels

# Evaluasi
n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise_db = list(dbscan_labels).count(-1)

print("=" * 50)
print(f"DBSCAN CLUSTERING (eps={BEST_EPS}, min_samples={BEST_MIN_SAMPLES})")
print("=" * 50)
print(f"Jumlah Cluster:  {n_clusters_db}")
print(f"Jumlah Noise:    {n_noise_db} ({n_noise_db/len(dbscan_labels)*100:.1f}%)")

if n_clusters_db > 1:
    sil_db = silhouette_score(X_scaled, dbscan_labels)
    print(f"Silhouette Score: {sil_db:.4f}")
    print(f"\nDistribusi Cluster:")
    print(pd.Series(dbscan_labels).value_counts().sort_index())
else:
    print("\nDBSCAN hanya menemukan 1 cluster atau semua titik adalah noise.")
    print("Coba sesuaikan parameter eps dan min_samples.")
```

**Penjelasan:**
- DBSCAN **tidak memerlukan K** — jumlah cluster ditentukan oleh kepadatan data.
- Label `-1` menunjukkan **noise** (titik yang tidak termasuk cluster manapun).
- Jika DBSCAN hanya menemukan 1 cluster, coba naikkan `eps` atau turunkan `min_samples`.


### 5.9 Visualisasi Cluster dengan PCA

**Cell 22: Reduksi Dimensi dengan PCA**

```python
# PCA untuk visualisasi 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Explained variance
print("Explained Variance Ratio per Komponen:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
print(f"Total: {sum(pca.explained_variance_ratio_)*100:.2f}%")

# Tambahkan ke DataFrame
df_clean['PC1'] = X_pca[:, 0]
df_clean['PC2'] = X_pca[:, 1]
```

**Penjelasan:**
- PCA mereduksi 7 fitur menjadi 2 principal components.
- **Explained Variance Ratio** menunjukkan berapa persen varians data yang ditangkap oleh setiap PC.
- Total explained variance > 60% biasanya cukup baik untuk visualisasi.

**Cell 23: Visualisasi Hasil Clustering (3 Algoritma)**

```python
# Plot 3 algoritma berdampingan
fig, axes = plt.subplots(1, 3, figsize=(22, 7))

algorithms = [
    ('K-Means', 'cluster_kmeans', axes[0]),
    ('Agglomerative', 'cluster_agg', axes[1]),
    ('DBSCAN', 'cluster_dbscan', axes[2])
]

for name, col, ax in algorithms:
    scatter = ax.scatter(df_clean['PC1'], df_clean['PC2'],
                         c=df_clean[col], cmap='viridis',
                         alpha=0.5, s=15, edgecolors='none')
    ax.set_title(f'{name} Clustering', fontsize=13)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.colorbar(scatter, ax=ax, label='Cluster')

plt.tight_layout()
plt.suptitle('Perbandingan Hasil Clustering (Visualisasi PCA 2D)', fontsize=15, y=1.02)
plt.show()
```

**Penjelasan:**
- Visualisasi 2D dengan PCA memungkinkan kita melihat sebaran cluster.
- Bandingkan bentuk dan pemisahan cluster antar algoritma.
- K-Means cenderung menghasilkan cluster berbentuk bulat.
- DBSCAN dapat menemukan cluster berbentuk arbitrer dan menandai noise.


### 5.10 Evaluasi dan Perbandingan Model

**Cell 24: Tabel Perbandingan Metrik**

```python
# Kumpulkan metrik untuk semua algoritma
comparison_data = []

# K-Means
comparison_data.append({
    'Algoritma': 'K-Means',
    'Jumlah Cluster': OPTIMAL_K,
    'Silhouette Score': round(sil_kmeans, 4),
    'Davies-Bouldin': round(db_kmeans, 4),
    'Calinski-Harabasz': round(ch_kmeans, 2),
    'Noise (%)': 0
})

# Agglomerative
comparison_data.append({
    'Algoritma': 'Agglomerative',
    'Jumlah Cluster': OPTIMAL_K,
    'Silhouette Score': round(sil_agg, 4),
    'Davies-Bouldin': round(db_agg, 4),
    'Calinski-Harabasz': round(ch_agg, 2),
    'Noise (%)': 0
})

# DBSCAN
comparison_data.append({
    'Algoritma': 'DBSCAN',
    'Jumlah Cluster': n_clusters_db,
    'Silhouette Score': round(sil_db, 4) if n_clusters_db > 1 else 'N/A',
    'Davies-Bouldin': 'N/A' if n_clusters_db <= 1 else round(davies_bouldin_score(X_scaled, dbscan_labels), 4),
    'Calinski-Harabasz': 'N/A' if n_clusters_db <= 1 else round(calinski_harabasz_score(X_scaled, dbscan_labels), 2),
    'Noise (%)': round(n_noise_db/len(dbscan_labels)*100, 1)
})

comparison_df = pd.DataFrame(comparison_data)
print("Perbandingan Metrik Evaluasi:")
print("=" * 90)
print(comparison_df.to_string(index=False))
```

**Penjelasan Metrik:**

| Metrik | Semakin | Interpretasi |
|---|---|---|
| **Silhouette Score** | Tinggi (max 1) | Cluster rapat dan terpisah dengan baik |
| **Davies-Bouldin Index** | Rendah (min 0) | Cluster kompak dan jauh dari cluster lain |
| **Calinski-Harabasz Index** | Tinggi | Rasio dispersi antar-cluster vs intra-cluster |
| **Noise (%)** | — | Persentase titik yang tidak masuk cluster (khusus DBSCAN) |

**Cell 25: Visualisasi Perbandingan**

```python
# Bar chart perbandingan
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

metrics = ['Silhouette Score', 'Davies-Bouldin', 'Calinski-Harabasz']
colors = ['#3498db', '#e74c3c', '#2ecc71']

for i, (metric, color) in enumerate(zip(metrics, colors)):
    values = []
    for algo in ['K-Means', 'Agglomerative', 'DBSCAN']:
        val = comparison_df[comparison_df['Algoritma'] == algo][metric].values[0]
        if val != 'N/A':
            values.append(float(val))
        else:
            values.append(0)

    bars = axes[i].bar(['K-Means', 'Agglomerative', 'DBSCAN'], values, color=color, alpha=0.7)
    axes[i].set_title(metric, fontsize=12)
    axes[i].set_ylabel(metric)

    for bar in bars:
        height = bar.get_height()
        axes[i].annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

plt.tight_layout()
plt.suptitle('Perbandingan Metrik Evaluasi Clustering', fontsize=14, y=1.02)
plt.show()
```


### 5.11 Interpretasi Cluster dan Insight

**Cell 26: Analisis Profil Cluster (K-Means)**

```python
# Profil cluster dengan label interpretatif
cluster_profile = df_clean.groupby('cluster_kmeans')[clustering_features].mean().round(2)
cluster_counts = df_clean['cluster_kmeans'].value_counts().sort_index()

# Tambahkan jumlah anggota
cluster_profile['Jumlah'] = cluster_counts.values

print("PROFIL CLUSTER (K-MEANS):")
print("=" * 100)
print(cluster_profile.T.to_string())

# Identifikasi karakteristik tiap cluster
print("\n\nINTERPRETASI CLUSTER:")
print("=" * 100)

for cluster in sorted(df_clean['cluster_kmeans'].unique()):
    subset = df_clean[df_clean['cluster_kmeans'] == cluster]
    n_members = len(subset)
    pct = n_members / len(df_clean) * 100

    print(f"\nCluster {cluster} ({n_members} observasi, {pct:.1f}%):")
    print(f"  → Rata-rata fitur utama:")
    for col in clustering_features:
        print(f"     {col}: {subset[col].mean():.2f}")
```

**Interpretasi Contoh (K=3):**

| Cluster | Jumlah | Karakteristik | Label Interpretatif |
|---|---|---|---|
| **0** | ~35% | Suhu stabil, tekanan normal, etch rate sedang | **Steady State** — Operasi normal |
| **1** | ~40% | Suhu fluktuatif, tekanan sedang, etch rate tinggi | **Transient State** — Peralihan |
| **2** | ~25% | Suhu ekstrem, tekanan tinggi, etch rate rendah | **Abnormal State** — Perlu perhatian |

**Insight untuk Konteks Manufaktur:**

1. **Cluster Steady State** — Proses beroperasi dalam kondisi normal. Parameter sensor stabil, tidak ada anomali. Ini adalah state yang paling diinginkan.
2. **Cluster Transient State** — Proses dalam kondisi peralihan (startup, shutdown, atau perubahan setpoint). Perlu monitoring lebih ketat.
3. **Cluster Abnormal State** — Proses menunjukkan gejala abnormal. Perlu intervensi segera — maintenance, penyesuaian parameter, atau penghentian sementara.

**Cell 27: Ringkasan Hasil**

```python
print("=" * 70)
print("RINGKASAN HASIL PRAKTIKUM CLUSTERING STATE KONTROL PRODUKSI")
print("=" * 70)

print(f"""
1. DATASET
   - Sumber: Kaggle — Semiconductor Wafer Defect Classification Dataset
   - Jumlah data: {len(df)} observasi
   - Jumlah fitur: {len(clustering_features)} fitur
   - Tidak ada missing value
   - Tidak ada label yang digunakan untuk clustering (unsupervised murni)

2. PREPROCESSING
   - Kolom tidak relevan dihapus: wafer_id, defect_label
   - Encoding: Label Encoding untuk process_step
   - Scaling: StandardScaler (mean=0, std=1)

3. PENENTUAN K OPTIMAL
   - Elbow Method: K = {OPTIMAL_K}
   - Silhouette Score: K = {best_k}

4. HASIL CLUSTERING (K-Means, K={OPTIMAL_K})
   - Silhouette Score: {sil_kmeans:.4f}
   - Jumlah anggota cluster: {dict(cluster_counts)}

5. INTERPRETASI
   - Cluster 0: Steady State (operasi normal)
   - Cluster 1: Transient State (peralihan)
   - Cluster 2: Abnormal State (perlu perhatian)

6. REKOMENDASI
   - Gunakan hasil clustering untuk monitoring kondisi real-time
   - Prioritaskan Cluster Abnormal untuk inspeksi
   - Kembangkan sistem peringatan dini berbasis state
""")
```


## 6. TUGAS MANDIRI (Dikerjakan Hari Ini)

**Tujuan:** Memastikan setiap mahasiswa memahami alur clustering untuk data manufaktur secara hands-on.

**Instruksi:**

1. **Eksperimen K-Means dengan K berbeda:**
   - Latih K-Means dengan K=2, K=3, K=4, dan K=5.
   - Catat Silhouette Score, Davies-Bouldin Index, dan Calinski-Harabasz Index untuk setiap K.
   - Buat tabel perbandingan dan tentukan K mana yang terbaik.
   - Jelaskan mengapa K tersebut yang paling optimal.

2. **Eksperimen DBSCAN:**
   - Coba 5 kombinasi `eps` dan `min_samples` yang berbeda.
   - Catat jumlah cluster, jumlah noise, dan silhouette score untuk setiap kombinasi.
   - Tentukan kombinasi parameter terbaik dan jelaskan alasannya.

3. **Visualisasi Tambahan:**
   - Buat scatter plot 2D menggunakan PCA untuk hasil K-Means dengan K=3.
   - Warnai setiap cluster dengan warna berbeda.
   - Beri label sumbu dengan explained variance ratio.
   - Interpretasikan visualisasi tersebut.

4. **Pertanyaan Konsep (Jawab di Markdown Cell):**
   - Apa perbedaan utama antara supervised dan unsupervised learning?
   - Mengapa StandardScaler wajib digunakan untuk K-Means? Apa yang terjadi jika tidak?
   - Apa kelebihan DBSCAN dibandingkan K-Means? Kapan sebaiknya menggunakan DBSCAN?
   - Bagaimana cara menentukan jumlah cluster optimal? Jelaskan dua metode.
   - Apa itu dendrogram dan bagaimana cara membacanya?

5. **Refleksi Pribadi (Minimal 200 kata):**
   - Apa yang Anda pelajari dari praktikum clustering ini?
   - Bagian mana yang paling sulit? Bagaimana Anda mengatasinya?
   - Bagaimana Anda akan menerapkan clustering di industri manufaktur?

**Pengumpulan:** Simpan notebook dengan nama `TugasMandiri2_NamaAnda_NIM.ipynb` dan kumpulkan di Google Drive sebelum pertemuan ke-4.


## 7. TUGAS KELOMPOK (Dikerjakan di Rumah)

**Tujuan:** Memperdalam pemahaman clustering melalui eksplorasi dataset alternatif dan perbandingan algoritma.

**Pembagian Kelompok:** 4–5 mahasiswa per kelompok.

### Bagian A: Eksplorasi Dataset Alternatif (Bobot 25%)

Pilih **salah satu** dataset berikut:

1. **Bosch CNC Machining Dataset** — https://www.archive.ics.uci.edu/dataset/752/bosch+cnc+machining+dataset
   - 2.700 baris × 3 fitur
   - Data accelerometer dari CNC milling machine
   - Cocok untuk clustering kondisi normal vs abnormal

2. **Reconfigurable_manufacturing.csv** — Tersedia via Kaggle
   - 1.000 baris × 12 variabel
   - Dataset operasional dari lingkungan manufaktur reconfigurable
   - Mencakup downtime, reconfiguration time, material usage, production capacity, energy consumption, carbon emissions, waste generated

3. **Industrial Machine Condition Monitoring Dataset** — Tersedia via WindLab
   - Sensor data dari mesin industri
   - Cocok untuk condition monitoring dan clustering

**Tugas:**
- Lakukan EDA singkat pada dataset yang dipilih.
- Lakukan preprocessing (handle missing value, encoding, scaling).
- Tentukan jumlah cluster optimal dengan Elbow Method dan Silhouette Score.
- Latih K-Means dengan K optimal.
- Interpretasikan hasil cluster — apa karakteristik setiap cluster?
- Buat minimal **3 visualisasi** yang informatif.

### Bagian B: Perbandingan Algoritma (Bobot 35%)

1. Latih **3 algoritma clustering** pada dataset alternatif yang dipilih:
   - K-Means
   - Agglomerative Clustering (Ward linkage)
   - DBSCAN

2. Bandingkan hasilnya dalam satu tabel:

| Algoritma | Jumlah Cluster | Silhouette Score | Davies-Bouldin | Calinski-Harabasz | Noise (%) |
|---|---|---|---|---|---|
| K-Means | ... | ... | ... | ... | 0 |
| Agglomerative | ... | ... | ... | ... | 0 |
| DBSCAN | ... | ... | ... | ... | ... |

3. **Analisis:** Algoritma mana yang paling baik? Mengapa? Apakah ada algoritma yang menghasilkan cluster yang tidak bermakna? Jelaskan.

### Bagian C: PCA dan Visualisasi (Bobot 20%)

1. Terapkan PCA pada dataset alternatif.
2. Tentukan berapa banyak principal component yang diperlukan untuk menangkap > 80% varians.
3. Buat visualisasi 2D untuk setiap algoritma clustering.
4. Jelaskan insight yang diperoleh dari visualisasi tersebut.

### Bagian D: Rekomendasi dan Presentasi (Bobot 20%)

1. Berdasarkan hasil clustering, buat **rekomendasi** untuk institusi manufaktur:
   - Bagaimana hasil clustering dapat digunakan untuk monitoring kondisi mesin?
   - Bagaimana hasil clustering dapat digunakan untuk predictive maintenance?
   - Apa keterbatasan analisis ini?

2. Buat **slide presentasi** (6–8 slide) yang mencakup:
   - Judul dan anggota kelompok
   - Dataset yang digunakan
   - Metodologi (preprocessing, algoritma, evaluasi)
   - Hasil perbandingan (tabel + grafik PCA)
   - Interpretasi cluster
   - Rekomendasi
   - Pembagian tugas anggota

**Pengumpulan:** Kumpulkan notebook (`TugasKelompok2_NamaKelompok.ipynb`) dan slide dalam satu folder ZIP di Google Drive sebelum pertemuan ke-4.


## 8. RUBRIK PENILAIAN

### 8.1 Rubrik Tugas Mandiri (Bobot 20%)

| Komponen | Bobot | Kriteria | Skor |
|---|---|---|---|
| **Eksperimen K-Means** | 25% | 4 nilai K diuji, tabel lengkap, analisis tajam | 0–100 |
| **Eksperimen DBSCAN** | 20% | 5 kombinasi parameter, analisis mendalam | 0–100 |
| **Visualisasi PCA** | 15% | Plot jelas, label lengkap, interpretasi benar | 0–100 |
| **Pertanyaan Konsep** | 25% | Jawaban tepat, ada penjelasan | 0–100 |
| **Refleksi Pribadi** | 15% | Refleksi mendalam, autentik | 0–100 |

### 8.2 Rubrik Tugas Kelompok (Bobot 30%)

| Komponen | Bobot | Kriteria | Skor |
|---|---|---|---|
| **Bagian A: EDA Dataset Alternatif** | 25% | EDA mendalam, visualisasi informatif | 0–100 |
| **Bagian B: Perbandingan 3 Algoritma** | 35% | Tabel lengkap, analisis tajam, kesimpulan jelas | 0–100 |
| **Bagian C: PCA & Visualisasi** | 20% | Implementasi benar, interpretasi tepat | 0–100 |
| **Bagian D: Rekomendasi & Presentasi** | 20% | Rekomendasi logis, slide menarik | 0–100 |

### 8.3 Konversi Nilai

| Nilai Angka | Huruf | Keterangan |
|---|---|---|
| 85–100 | A | Sangat Baik |
| 75–84 | B | Baik |
| 65–74 | C | Cukup |
| 55–64 | D | Kurang |
| < 55 | E | Sangat Kurang |


## 9. REFERENSI

### 9.1 Dataset

1. Meruva Kodanda Suraj. (2025). *Semiconductor Wafer Defect Classification Dataset*. Kaggle. https://www.kaggle.com/datasets/meruvakodandasuraj/semiconductor-wafer-defect-classification-dataset
2. Feil, M. (2022). *Bosch CNC Machining Dataset*. UCI Machine Learning Repository. https://doi.org/10.1016/j.procir.2022.04.022
3. *Reconfigurable_manufacturing.csv*. Tersedia via Kaggle (CC0 Public Domain).

### 9.2 Buku dan Artikel

1. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media. — Bab 9: Unsupervised Learning Techniques.
2. Raschka, S., & Mirjalili, V. (2019). *Python Machine Learning* (3rd ed.). Packt. — Bab 10: Working with Unlabeled Data — Clustering Analysis.
3. Ester, M., et al. (1996). A Density-Based Algorithm for Discovering Clusters. *KDD-96*.
4. Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*.

### 9.3 Dokumentasi Online

1. **Scikit-learn Clustering:** https://scikit-learn.org/stable/modules/clustering.html
2. **Scikit-learn PCA:** https://scikit-learn.org/stable/modules/decomposition.html#pca
3. **Scipy Hierarchical Clustering:** https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html

### 9.4 Video Tutorial

1. StatQuest — K-Means Clustering: https://www.youtube.com/watch?v=4b5d3muPQmA
2. StatQuest — Hierarchical Clustering: https://www.youtube.com/watch?v=7xHsRkOdVwo
3. StatQuest — DBSCAN: https://www.youtube.com/watch?v=RDZUdRSDOok


## PENUTUP

Praktikum ini memberikan pengalaman langsung dalam menerapkan **unsupervised learning** — khususnya **clustering** — untuk menganalisis data kontrol manufaktur wafer semikonduktor. Berbeda dengan pertemuan sebelumnya yang menggunakan supervised learning, di sini kita tidak memiliki label dan harus menemukan **state kontrol tersembunyi** dalam data sensor.

**Poin-poin kunci:**

1. **Clustering** mengelompokkan data berdasarkan kemiripan tanpa memerlukan label.
2. **K-Means** cepat dan sederhana, tetapi memerlukan K yang ditentukan di awal.
3. **Agglomerative Clustering** menghasilkan hierarki yang informatif melalui dendrogram.
4. **DBSCAN** dapat menemukan cluster berbentuk arbitrer dan menangani noise.
5. **PCA** membantu memvisualisasikan cluster dalam dimensi yang lebih rendah.
6. **Interpretasi cluster** sangat penting — dalam konteks manufaktur, cluster merepresentasikan state operasional yang berbeda.

**Pesan untuk Mahasiswa:**
> "Dalam industri manufaktur, data sensor mengalir terus-menerus tanpa label. Clustering memungkinkan Anda menemukan pola tersembunyi — state kontrol yang mungkin tidak terlihat secara manual. Kemampuan ini sangat berharga untuk monitoring kondisi real-time dan predictive maintenance. Dataset yang digunakan dalam praktikum ini ringan (5.000 baris × 10 kolom, ~500 KB) sehingga dapat diolah pada laptop dengan RAM 8GB atau 16GB tanpa kendala."

**Pesan untuk Dosen:**
> "Dataset *Semiconductor Wafer Defect Classification Dataset* dipilih karena ukurannya yang ringan namun tetap relevan dengan konteks manufaktur. Dataset ini dirancang untuk mendukung pembelajaran unsupervised, termasuk PCA dan anomaly detection. Untuk kelas dengan waktu terbatas, fokus pada K-Means dan PCA terlebih dahulu. Untuk kelas yang lebih mahir, tambahkan DBSCAN, Agglomerative Clustering, dan eksperimen dengan dataset alternatif yang lebih kecil seperti Bosch CNC Machining Dataset (2.700 baris)."

---

**Dokumen ini disusun untuk keperluan edukasi di Politeknik Manufaktur Bandung (Polman Bandung).**
**Versi:** 1.1 (Revisi Dataset) | **Minggu ke-2, Pertemuan ke-2 dari 4**
**Terakhir Diperbarui:** 2026
