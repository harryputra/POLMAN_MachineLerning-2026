# STUDI KASUS MACHINE LEARNING — PERTEMUAN KE-2 MINGGU KE-2
## Unsupervised Learning: Segmentasi State Kontrol Produksi Manufaktur dengan Clustering

**Mata Kuliah:** Machine Learning
**Program Studi:** D4 Teknologi Rekayasa Informatika dan Komputer (TRIN)
**Institusi:** Politeknik Manufaktur Bandung (Polman Bandung)
**Pertemuan:** Minggu ke-2, Pertemuan ke-2 dari 4
**Topik:** Unsupervised Learning — Clustering untuk Segmentasi Produksi
**Tools:** Anaconda, Jupyter Notebook, Python 3.11
**Durasi:** 150 menit


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

Bayangkan sebuah pabrik yang memiliki lini produksi dengan puluhan sensor. Setiap detik, sensor-sensor ini mengirimkan data suhu, tekanan, getaran, dan kecepatan. Tanpa label, sulit untuk mengetahui apakah mesin sedang beroperasi normal atau mulai menunjukkan gejala kerusakan. Dengan clustering, kita dapat mengelompokkan data sensor ke dalam state-state yang bermakna — memungkinkan **monitoring kondisi real-time** dan **deteksi dini anomali**.

Studi kasus ini sangat relevan dengan **Polman Bandung** sebagai institusi pendidikan vokasi yang fokus pada manufaktur dan otomasi industri. Mahasiswa D4 TRIN perlu memahami bagaimana data dari lini produksi dapat diolah menjadi **insight yang dapat ditindaklanjuti** tanpa memerlukan label manual yang mahal dan memakan waktu.

**Referensi Industri:**
- Pada **Tabular Playground Series - Jul 2022 (Kaggle)**, peserta diberikan data kontrol manufaktur yang disimulasikan dan diminta untuk mengelompokkannya ke dalam state kontrol yang berbeda. Ini adalah masalah unsupervised murni yang mencerminkan skenario nyata di industri.
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

### 3.1 Dataset Utama

| Informasi | Detail |
|---|---|
| **Nama Dataset** | Tabular Playground Series - Jul 2022 (Manufacturing Control Data) |
| **Sumber** | Kaggle Playground Competition |
| **Link** | https://www.kaggle.com/competitions/tabular-playground-series-jul-2022/data |
| **Jumlah Data** | 500.000 baris (data.csv) |
| **Jumlah Fitur** | 32 kolom |
| **Tipe Data** | CSV (`data.csv`) |
| **Lisensi** | Kaggle Competition Rules — untuk edukasi dan penelitian |

**Deskripsi Dataset:**
Dataset ini berisi data kontrol manufaktur yang disimulasikan dan dapat dikelompokkan ke dalam **state kontrol yang berbeda**. Tugas kita adalah mengelompokkan baris data ke dalam state-state tersebut **tanpa mengetahui berapa jumlah state yang ada** — ini adalah masalah unsupervised murni yang mencerminkan skenario nyata di industri.

Dataset mencakup **data kontinu dan kategorikal**, dengan 32 kolom yang merepresentasikan berbagai parameter proses manufaktur. Setiap baris merepresentasikan satu observasi dari proses produksi, dan kita harus menentukan state kontrol mana yang paling sesuai.

### 3.2 Struktur Dataset

Berdasarkan informasi dari notebook peserta kompetisi, dataset ini memiliki karakteristik:

- **Fitur kontinu:** Sensor readings (suhu, tekanan, getaran, kecepatan, dll.)
- **Fitur kategorikal:** Parameter proses yang bersifat diskrit
- **Missing values:** Beberapa kolom memiliki nilai yang hilang
- **Tidak ada label target:** Dataset ini murni unsupervised — tidak ada training data yang diberikan

**Catatan Penting:**
- Dataset ini berukuran besar (44,46 MB), sehingga perlu strategi sampling untuk praktikum yang efisien.
- Untuk praktikum ini, kita akan menggunakan **subsample 50.000 baris** agar komputasi lebih cepat namun tetap representatif.
- Karena tidak ada label, kita tidak bisa menggunakan metrik seperti accuracy — kita akan menggunakan metrik **internal clustering** seperti Silhouette Score, Davies-Bouldin Index, dan Calinski-Harabasz Index.

### 3.3 Mengapa Dataset Ini Relevan?

1. **Konteks Industri:** Dataset ini mensimulasikan data kontrol manufaktur nyata dengan berbagai state operasional.
2. **Unsupervised Murni:** Tidak ada label sama sekali — mahasiswa harus menemukan struktur sendiri.
3. **Multi-Fitur:** 32 fitur mencakup berbagai aspek proses produksi.
4. **Challenge Realistis:** Data kontinu + kategorikal, missing values, dan skala besar adalah tantangan nyata di industri.
5. **Referensi Kompetisi:** Dataset ini digunakan dalam kompetisi Kaggle resmi, sehingga ada banyak referensi solusi yang bisa dipelajari.


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

**Cell 2: Load Dataset (dengan Sampling)**

```python
# Load dataset (gunakan sampling untuk efisiensi)
df_full = pd.read_csv('data.csv')

# Ambil sampel 50.000 baris untuk praktikum
np.random.seed(42)
df = df_full.sample(n=50000, random_state=42).reset_index(drop=True)

print("Ukuran dataset lengkap:", df_full.shape)
print("Ukuran dataset sampel:", df.shape)
print("\n5 Baris Pertama:")
df.head()
```

**Penjelasan:**
- Dataset lengkap memiliki 500.000 baris dan 32 kolom. Untuk praktikum yang efisien, kita menggunakan **subsample 50.000 baris**.
- `random_state=42` memastikan sampling reproducible.
- `df.head()` menampilkan 5 baris pertama untuk inspeksi awal.

**Output yang diharapkan:** Tabel dengan 50.000 baris dan 32 kolom.


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
- Perhatikan kolom yang memiliki missing value.
- `df.describe()` memberikan statistik ringkasan untuk kolom numerik.

**Cell 4: Cek Missing Value dan Duplikat**

```python
# Cek missing value
print("Jumlah Missing Value per Kolom:")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct.round(2)
})
print(missing_df[missing_df['Missing Count'] > 0])

# Cek duplikat
print(f"\nJumlah Data Duplikat: {df.duplicated().sum()}")
```

**Penjelasan:**
- Identifikasi kolom yang memiliki missing value dan persentasenya.
- Tentukan strategi penanganan: hapus baris, imputasi mean/median, atau biarkan (jika algoritma bisa menangani).

**Cell 5: Identifikasi Tipe Kolom**

```python
# Pisahkan kolom numerik dan kategorikal
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"Jumlah kolom numerik: {len(numerical_cols)}")
print(f"Jumlah kolom kategorikal: {len(categorical_cols)}")
print(f"\nKolom numerik: {numerical_cols[:10]}...")
print(f"\nKolom kategorikal: {categorical_cols}")
```

**Penjelasan:**
- Dataset memiliki campuran kolom numerik dan kategorikal.
- Kolom kategorikal perlu di-encoding sebelum clustering.

**Cell 6: Visualisasi Distribusi Fitur Numerik**

```python
# Histogram distribusi fitur numerik (ambil 8 fitur pertama)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, col in enumerate(numerical_cols[:8]):
    sns.histplot(df[col], kde=True, ax=axes[i], color='steelblue')
    axes[i].set_title(f'Distribusi {col}', fontsize=10)
    axes[i].set_xlabel('')

plt.tight_layout()
plt.suptitle('Distribusi Fitur Numerik (8 Fitur Pertama)', fontsize=14, y=1.02)
plt.show()
```

**Penjelasan:**
- Setiap histogram menunjukkan sebaran nilai untuk fitur numerik.
- Perhatikan bentuk distribusi: normal, miring ke kiri, atau miring ke kanan.
- Fitur dengan distribusi miring mungkin memerlukan transformasi.

**Cell 7: Heatmap Korelasi**

```python
# Heatmap korelasi fitur numerik
plt.figure(figsize=(14, 12))
correlation = df[numerical_cols].corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, mask=mask, annot=False, cmap='coolwarm',
            linewidths=0.5, vmin=-1, vmax=1)
plt.title('Matriks Korelasi Fitur Numerik', fontsize=14)
plt.tight_layout()
plt.show()
```

**Penjelasan:**
- Heatmap menunjukkan korelasi antar fitur numerik.
- Fitur yang berkorelasi tinggi mungkin redundan dan bisa dipertimbangkan untuk di-drop.
- Perhatikan apakah ada **multikolinearitas** yang signifikan.


### 5.4 Preprocessing Data

**Cell 8: Handling Missing Value**

```python
# Strategi: imputasi dengan median untuk numerik, mode untuk kategorikal
df_clean = df.copy()

# Imputasi kolom numerik dengan median
for col in numerical_cols:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)

# Imputasi kolom kategorikal dengan mode
for col in categorical_cols:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

print("Missing value setelah imputasi:")
print(df_clean.isnull().sum().sum())
```

**Penjelasan:**
- **Median** dipilih untuk numerik karena lebih robust terhadap outlier dibanding mean.
- **Mode** (nilai paling sering) dipilih untuk kategorikal.

**Cell 9: Encoding Fitur Kategorikal**

```python
# Label Encoding untuk kolom kategorikal
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col + '_encoded'] = le.fit_transform(df_clean[col].astype(str))
    le_dict[col] = le

print("Kolom kategorikal yang di-encoding:")
for col in categorical_cols:
    print(f"  {col} → {col}_encoded")
```

**Penjelasan:**
- Setiap kolom kategorikal diubah menjadi angka.
- `LabelEncoder` memberikan nilai integer unik untuk setiap kategori.

**Cell 10: Pilih Fitur untuk Clustering**

```python
# Fitur yang akan digunakan untuk clustering
# Gunakan semua kolom numerik + kolom kategorikal yang sudah di-encoding
clustering_features = numerical_cols + [col + '_encoded' for col in categorical_cols]

X = df_clean[clustering_features].copy()

print(f"Jumlah fitur untuk clustering: {len(clustering_features)}")
print(f"Shape X: {X.shape}")
print(f"\nFitur yang digunakan: {clustering_features}")
```

**Penjelasan:**
- Kita menggunakan **semua fitur** yang tersedia untuk clustering.
- Pastikan tidak ada kolom identifier yang ikut masuk.

**Cell 11: Feature Scaling (WAJIB untuk Clustering)**

```python
# StandardScaler — WAJIB untuk clustering berbasis jarak
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Konversi kembali ke DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=clustering_features)

print("Setelah StandardScaler:")
print(f"Mean: {X_scaled_df.mean().round(4).tolist()[:5]}...")
print(f"Std:  {X_scaled_df.std().round(4).tolist()[:5]}...")
print(f"\nShape X_scaled: {X_scaled.shape}")
```

**Penjelasan:**
- **StandardScaler sangat penting untuk clustering** karena algoritma seperti K-Means dan DBSCAN menggunakan **jarak Euclidean**.
- Tanpa scaling, fitur dengan skala besar akan mendominasi perhitungan jarak.
- Setelah scaling, semua fitur memiliki mean ≈ 0 dan std ≈ 1.


### 5.5 Menentukan Jumlah Cluster Optimal

**Cell 12: Elbow Method untuk K-Means**

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

**Cell 13: Silhouette Score untuk Konfirmasi**

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

**Cell 14: Latih K-Means dengan K Optimal**

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

**Cell 15: Profil Setiap Cluster (K-Means)**

```python
# Analisis profil cluster
cluster_profile = df_clean.groupby('cluster_kmeans')[clustering_features[:10]].mean().round(2)

print("Profil Rata-rata Setiap Cluster (10 Fitur Pertama):")
print("=" * 80)
print(cluster_profile.T.to_string())

# Visualisasi profil cluster
fig, axes = plt.subplots(2, 5, figsize=(22, 10))
axes = axes.flatten()

for i, col in enumerate(clustering_features[:10]):
    sns.boxplot(x='cluster_kmeans', y=col, data=df_clean, ax=axes[i], palette='Set2')
    axes[i].set_title(f'{col} per Cluster', fontsize=9)
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

**Cell 16: Dendrogram untuk Hierarchical Clustering**

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

**Cell 17: Latih Agglomerative Clustering**

```python
# Latih Agglomerative Clustering (gunakan subsample untuk efisiensi)
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

**Cell 18: DBSCAN — Mencari Parameter Optimal**

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

**Cell 19: Latih DBSCAN**

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

**Cell 20: Latih DBSCAN dengan Parameter Terbaik**

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

**Cell 21: Reduksi Dimensi dengan PCA**

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
- PCA mereduksi 32 fitur menjadi 2 principal components.
- **Explained Variance Ratio** menunjukkan berapa persen varians data yang ditangkap oleh setiap PC.
- Total explained variance > 60% biasanya cukup baik untuk visualisasi.

**Cell 22: Visualisasi Hasil Clustering (3 Algoritma)**

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
                         alpha=0.5, s=10, edgecolors='none')
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

**Cell 23: Tabel Perbandingan Metrik**

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

**Cell 24: Visualisasi Perbandingan**

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

**Cell 25: Analisis Profil Cluster (K-Means)**

```python
# Profil cluster dengan label interpretatif
cluster_profile = df_clean.groupby('cluster_kmeans')[clustering_features[:10]].mean().round(2)
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
    for col in clustering_features[:5]:
        print(f"     {col}: {subset[col].mean():.2f}")
```

**Interpretasi Contoh (K=3):**

| Cluster | Jumlah | Karakteristik | Label Interpretatif |
|---|---|---|---|
| **0** | ~40% | Parameter stabil, variasi rendah | **Steady State** — Operasi normal |
| **1** | ~35% | Parameter fluktuatif, variasi sedang | **Transient State** — Peralihan |
| **2** | ~25% | Parameter ekstrem, variasi tinggi | **Abnormal State** — Perlu perhatian |

**Insight untuk Konteks Manufaktur:**

1. **Cluster Steady State** — Mesin beroperasi dalam kondisi normal. Parameter stabil, tidak ada anomali. Ini adalah state yang paling diinginkan.
2. **Cluster Transient State** — Mesin dalam kondisi peralihan (startup, shutdown, atau perubahan setpoint). Perlu monitoring lebih ketat.
3. **Cluster Abnormal State** — Mesin menunjukkan gejala abnormal. Perlu intervensi segera — maintenance, penyesuaian parameter, atau penghentian sementara.

**Cell 26: Ringkasan Hasil**

```python
print("=" * 70)
print("RINGKASAN HASIL PRAKTIKUM CLUSTERING STATE KONTROL PRODUKSI")
print("=" * 70)

print(f"""
1. DATASET
   - Sumber: Kaggle — Tabular Playground Series Jul 2022
   - Jumlah data: 50.000 observasi (sampel dari 500.000)
   - Jumlah fitur: {len(clustering_features)} fitur
   - Tidak ada label target (unsupervised murni)

2. PREPROCESSING
   - Missing value: diimputasi dengan median/mode
   - Encoding: Label Encoding untuk kategorikal
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

1. **Semiconductor Wafer Defect Classification Dataset** — https://www.kaggle.com/datasets/meruvakodandasuraj/semiconductor-wafer-defect-classification-dataset
   - 5.000 baris × 10 kolom
   - Sensor readings: temperature, pressure, gas flow, voltage, current, etch rate
   - Process step: Oxidation, Lithography, Etching, Deposition, CMP
   - Cocok untuk PCA, anomaly detection, dan clustering

2. **Bosch CNC Machining Dataset** — https://github.com/boschresearch/CNC_Machining
   - 2.700 baris × 3 fitur
   - Data accelerometer dari CNC milling machine
   - Cocok untuk clustering kondisi normal vs abnormal

3. **Smart Manufacturing IoT-Cloud Monitoring Dataset** — https://www.kaggle.com/datasets/ziya07/smart-manufacturing-iot-cloud-monitoring-dataset
   - Sensor data: temperature, vibration, pressure, acoustic
   - Machine operational status: Normal, Warning, Fault, Critical
   - Cocok untuk unsupervised anomaly detection

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

1. Kaggle. (2022). *Tabular Playground Series - Jul 2022: Manufacturing Control Data Clustering*. https://www.kaggle.com/competitions/tabular-playground-series-jul-2022/data
2. Meruva Kodanda Suraj. (2025). *Semiconductor Wafer Defect Classification Dataset*. Kaggle. https://www.kaggle.com/datasets/meruvakodandasuraj/semiconductor-wafer-defect-classification-dataset
3. Feil, M. (2022). *Bosch CNC Machining Dataset*. UCI Machine Learning Repository. https://doi.org/10.1016/j.procir.2022.04.022
4. Ziya07. (2025). *Smart Manufacturing IoT-Cloud Monitoring Dataset*. Kaggle. https://www.kaggle.com/datasets/ziya07/smart-manufacturing-iot-cloud-monitoring-dataset

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

Praktikum ini memberikan pengalaman langsung dalam menerapkan **unsupervised learning** — khususnya **clustering** — untuk menganalisis data kontrol manufaktur. Berbeda dengan pertemuan sebelumnya yang menggunakan supervised learning, di sini kita tidak memiliki label dan harus menemukan **state kontrol tersembunyi** dalam data sensor.

**Poin-poin kunci:**

1. **Clustering** mengelompokkan data berdasarkan kemiripan tanpa memerlukan label.
2. **K-Means** cepat dan sederhana, tetapi memerlukan K yang ditentukan di awal.
3. **Agglomerative Clustering** menghasilkan hierarki yang informatif melalui dendrogram.
4. **DBSCAN** dapat menemukan cluster berbentuk arbitrer dan menangani noise.
5. **PCA** membantu memvisualisasikan cluster dalam dimensi yang lebih rendah.
6. **Interpretasi cluster** sangat penting — dalam konteks manufaktur, cluster merepresentasikan state operasional yang berbeda.

**Pesan untuk Mahasiswa:**
> "Dalam industri manufaktur, data sensor mengalir terus-menerus tanpa label. Clustering memungkinkan Anda menemukan pola tersembunyi — state kontrol yang mungkin tidak terlihat secara manual. Kemampuan ini sangat berharga untuk monitoring kondisi real-time dan predictive maintenance."

**Pesan untuk Dosen:**
> "Studi kasus ini menggunakan dataset kompetisi Kaggle yang menantang namun tetap dapat diakses. Untuk kelas dengan waktu terbatas, fokus pada K-Means dan PCA terlebih dahulu. Untuk kelas yang lebih mahir, tambahkan DBSCAN, Agglomerative Clustering, dan eksperimen dengan dataset alternatif."

---

**Dokumen ini disusun untuk keperluan edukasi di Politeknik Manufaktur Bandung (Polman Bandung).**
**Versi:** 1.0 | **Minggu ke-2, Pertemuan ke-2 dari 4**
**Terakhir Diperbarui:** 2026
