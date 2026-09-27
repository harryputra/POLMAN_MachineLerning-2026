# MODUL AJAR TEORI PENUNJANG

## Supervised Learning untuk Klasifikasi Defect Produk Manufaktur

**Mata Kuliah:** Machine Learning
**Program Studi:** D4 Teknologi Rekayasa Informatika dan Komputer (TRIN)
**Institusi:** Politeknik Manufaktur Bandung (Polman Bandung)
**Pertemuan:** Minggu ke-2, Pertemuan ke-1 dari 4
**Topik:** Supervised Learning — Klasifikasi Kualitas Produk Manufaktur
**Durasi:** 150 menit (teori + diskusi)
**Prasyarat:** Dasar Python, Statistik Deskriptif

## DAFTAR ISI

1. [Pendahuluan](#1-pendahuluan)
2. [Supervised Learning: Fondasi Teori](#2-supervised-learning-fondasi-teori)
3. [Klasifikasi untuk Quality Control Manufaktur](#3-klasifikasi-untuk-quality-control-manufaktur)
4. [Alur Kerja Machine Learning Pipeline](#4-alur-kerja-machine-learning-pipeline)
5. [Eksplorasi Data (EDA) untuk Data Manufaktur](#5-eksplorasi-data-eda-untuk-data-manufaktur)
6. [Preprocessing Data Manufaktur](#6-preprocessing-data-manufaktur)
7. [Class Imbalance: Tantangan Utama dalam Quality Control](#7-class-imbalance-tantangan-utama-dalam-quality-control)
8. [Algoritma Klasifikasi untuk Defect Prediction](#8-algoritma-klasifikasi-untuk-defect-prediction)
9. [Evaluasi Model untuk Data Imbalanced](#9-evaluasi-model-untuk-data-imbalanced)
10. [Feature Importance dan Interpretability](#10-feature-importance-dan-interpretability)
11. [Cross-Validation dan Hyperparameter Tuning](#11-cross-validation-dan-hyperparameter-tuning)
12. [Studi Kasus Industri Nyata](#12-studi-kasus-industri-nyata)
13. [Deployment dan Monitoring](#13-deployment-dan-monitoring)
14. [Etika dan Tantangan dalam Predictive Quality](#14-etika-dan-tantangan-dalam-predictive-quality)
15. [Latihan dan Soal Refleksi](#15-latihan-dan-soal-refleksi)
16. [Glosarium](#16-glosarium)
17. [Referensi](#17-referensi)

## 1. PENDAHULUAN

### 1.1 Konteks Minggu ke-2

Minggu ke-2 merupakan **minggu tematik Machine Learning** yang terdiri dari 4 pertemuan:

| Pertemuan | Topik | Fokus |
| --- | --- | --- |
| **Pertemuan 1** | **Supervised Learning — Prediksi Defect Manufaktur** | **Klasifikasi dengan label** |
| Pertemuan 2 | Unsupervised Learning | Clustering tanpa label |
| Pertemuan 3 | Reinforcement Learning | Belajar dari reward |
| Pertemuan 4 | Evaluasi Mingguan | Presentasi + Teori |

### 1.2 Mengapa Predictive Quality Penting?

Dalam industri manufaktur modern, kualitas produk adalah faktor krusial yang menentukan kepuasan pelanggan, efisiensi biaya, dan daya saing perusahaan. Produksi cacat (defect) yang lolos ke pasar dapat menyebabkan kerugian finansial yang signifikan, kerusakan reputasi, bahkan risiko keselamatan.

**Pendekatan tradisional** dalam quality control bersifat **reaktif** — produk diperiksa setelah selesai diproduksi, dan defect ditemukan setelah terjadi. Pendekatan ini memiliki keterbatasan:

- Inspektur manusia bisa lelah dan tidak konsisten.
- Tidak semua unit dapat diperiksa secara menyeluruh.
- Defect baru ditemukan setelah biaya produksi sudah dikeluarkan.

**Pendekatan predictive quality** menggunakan Machine Learning untuk **memprediksi probabilitas defect sebelum produk selesai diproduksi**. Alih-alih memeriksa di akhir lini produksi, sistem dapat mengidentifikasi risiko cacat sejak dini berdasarkan parameter proses, kondisi mesin, dan faktor lingkungan.

Transisi dari quality control reaktif ke prediktif ini merupakan inti dari **Quality 4.0** — paradigma di mana data dan kecerdasan buatan digunakan untuk mengoptimalkan kualitas secara proaktif. Penelitian terbaru menunjukkan bahwa integrasi predictive quality management dengan Machine Learning memungkinkan organisasi untuk mengantisipasi defect secara lebih efektif dan beralih dari memperbaiki masalah menjadi mencegah masalah.

### 1.3 Relevansi dengan Polman Bandung

Sebagai institusi pendidikan vokasi yang fokus pada manufaktur dan otomasi industri, Polman Bandung perlu membekali mahasiswa dengan kemampuan mengolah data dari lini produksi menjadi insight yang dapat ditindaklanjuti. Studi kasus ini dirancang untuk menjembatani teori Machine Learning dengan praktik nyata di industri manufaktur.

### 1.4 Peta Konsep Modul

```
Supervised Learning untuk Defect Prediction
│
├── Fondasi Teori
│   ├── Supervised Learning
│   ├── Klasifikasi Binary
│   └── Alur Kerja ML Pipeline
│
├── Data Understanding
│   ├── EDA untuk Data Manufaktur
│   ├── Distribusi Fitur
│   └── Analisis Korelasi
│
├── Data Preparation
│   ├── Handling Missing Value
│   ├── Feature Engineering
│   ├── Encoding & Scaling
│   └── Class Imbalance (SMOTE)
│
├── Modeling
│   ├── Decision Tree
│   ├── KNN
│   ├── Naive Bayes
│   ├── Logistic Regression
│   └── Ensemble (Random Forest, XGBoost)
│
├── Evaluation
│   ├── Confusion Matrix
│   ├── Precision, Recall, F1
│   ├── AUC-ROC
│   └── Cross-Validation
│
└── Interpretation
    ├── Feature Importance
    ├── SHAP Values
    └── Actionable Insights
```

## 2. SUPERVISED LEARNING: FONDASI TEORI

### 2.1 Definisi Formal

**Supervised Learning** adalah paradigma Machine Learning di mana model belajar dari **data berlabel** — pasangan input-output $(x, y)$.

$$D = \{(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)\}$$

- $x_i$ = **fitur** (input) — parameter proses, kondisi mesin, kondisi lingkungan
- $y_i$ = **label** (output) — status defect (0 = Low Defect, 1 = High Defect)
- $n$ = jumlah sampel

**Tujuan:** Menemukan fungsi $f$ sehingga $f(x) \approx y$ untuk data baru yang belum pernah dilihat.

### 2.2 Klasifikasi vs Regresi

| Aspek | Klasifikasi | Regresi |
| --- | --- | --- |
| **Output** | Kategori/kelas diskrit | Angka kontinu |
| **Contoh** | Defect/No Defect, Pass/Fail | Suhu, tekanan, dimensi |
| **Algoritma** | Decision Tree, KNN, NB, LR | Linear Regression, Ridge, Lasso |
| **Metrik** | Accuracy, Precision, Recall, F1, AUC | MAE, MSE, RMSE, R² |

Studi kasus kita adalah **klasifikasi binary** karena outputnya adalah kategori: High Defect (1) atau Low Defect (0).

### 2.3 Terminologi Penting

| Istilah | Arti | Contoh Manufaktur |
| --- | --- | --- |
| **Fitur** | Variabel input | ProductionVolume, QualityScore, MaintenanceHours |
| **Label/Target** | Variabel output | DefectStatus |
| **Sampel** | Satu baris data | Satu batch produksi |
| **Training Set** | Data untuk melatih | 80% dari dataset |
| **Testing Set** | Data untuk menguji | 20% dari dataset |
| **Model** | Hasil pembelajaran | Random Forest yang sudah dilatih |

### 2.4 Tiga Fase Supervised Learning

```
FASE 1: TRAINING
┌─────────────────┐
│  Training Data  │ ──► [Algoritma ML] ──► [Model]
│  (X_train, y_train)                     (parameter terlatih)
└─────────────────┘

FASE 2: TESTING/VALIDASI
┌─────────────────┐
│  Testing Data   │ ──► [Model] ──► [Prediksi] ──► [Evaluasi]
│  (X_test, y_test)                   vs y_test
└─────────────────┘

FASE 3: DEPLOYMENT
┌─────────────────┐
│  Data Baru      │ ──► [Model] ──► [Prediksi Defect]
│  (tanpa label)  │
└─────────────────┘
```

**Prinsip Emas:** Model **hanya boleh dilatih pada training data**. Testing data disimpan sebagai "ujian" untuk mengukur seberapa baik model bekerja pada data yang belum pernah dilihat.

## 3. KLASIFIKASI UNTUK QUALITY CONTROL MANUFAKTUR

### 3.1 Jenis-Jenis Defect dalam Manufaktur

| Jenis Defect | Deskripsi | Contoh |
| --- | --- | --- |
| **Dimensional** | Ukuran tidak sesuai spesifikasi | Diameter terlalu besar/kecil |
| **Surface** | Cacat permukaan | Goresan, pori-pori, warna tidak rata |
| **Structural** | Cacat struktural | Retak, void, delaminasi |
| **Functional** | Produk tidak berfungsi | Rangkaian tidak menyala |
| **Assembly** | Kesalahan perakitan | Komponen salah posisi |

### 3.2 Sumber Data untuk Defect Prediction

| Sumber | Jenis Data | Contoh Fitur |
| --- | --- | --- |
| **Sensor Mesin** | Time-series | Suhu, tekanan, getaran, torsi |
| **Parameter Proses** | Numerik | Kecepatan, feed rate, cycle time |
| **Data Supply Chain** | Numerik/Kategorikal | Kualitas supplier, lead time |
| **Data Lingkungan** | Numerik | Suhu ruangan, kelembaban |
| **Data Operator** | Kategorikal | Shift, pengalaman, training |
| **Data Maintenance** | Numerik | Jam maintenance, downtime |

### 3.3 Tantangan Khas Data Manufaktur

1. **Class Imbalance:** Defect biasanya jarang terjadi (1–5% dari total produksi).
2. **Data Tidak Seimbang:** Distribusi fitur bisa sangat skewed.
3. **Missing Value:** Sensor bisa gagal, data bisa tidak lengkap.
4. **Multikolinearitas:** Banyak fitur yang saling berkorelasi.
5. **Data Drift:** Kondisi mesin berubah seiring waktu.
6. **Label Noise:** Terkadang label defect tidak akurat.

### 3.4 Quality 4.0: Paradigma Baru

Quality 4.0 adalah evolusi dari quality management tradisional yang mengintegrasikan teknologi digital dan AI. Transisi dari traditional quality management systems ke sistem yang menggabungkan Machine Learning menawarkan kemampuan prediktif yang lebih baik, memungkinkan organisasi untuk mengantisipasi defect secara lebih efektif.

Penelitian menunjukkan bahwa Machine Learning memungkinkan perusahaan untuk beralih dari **reaktif** (memperbaiki masalah setelah terjadi) menjadi **prediktif** (mencegah masalah sebelum terjadi), yang merupakan perubahan kunci bagi perusahaan yang ingin tetap kompetitif.

## 4. ALUR KERJA MACHINE LEARNING PIPELINE

### 4.1 Overview Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                    ML PIPELINE UNTUK MANUFAKTUR              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. PROBLEM DEFINITION                                       │
│     └─ Apa yang ingin diprediksi? Defect atau kualitas?      │
│                                                              │
│  2. DATA COLLECTION                                          │
│     └─ Kumpulkan data dari sensor, MES, database             │
│                                                              │
│  3. EXPLORATORY DATA ANALYSIS (EDA)                          │
│     └─ Pahami distribusi, missing value, outlier             │
│                                                              │
│  4. DATA PREPROCESSING                                       │
│     └─ Cleaning, encoding, scaling, imbalance handling       │
│                                                              │
│  5. FEATURE ENGINEERING                                      │
│     └─ Buat fitur baru yang informatif                       │
│                                                              │
│  6. MODELING                                                 │
│     └─ Pilih algoritma, latih model                          │
│                                                              │
│  7. EVALUATION                                               │
│     └─ Ukur performa dengan metrik yang tepat                │
│                                                              │
│  8. HYPERPARAMETER TUNING                                    │
│     └─ Optimasi parameter model                              │
│                                                              │
│  9. INTERPRETATION                                           │
│     └─ Feature importance, SHAP, actionable insights         │
│                                                              │
│  10. DEPLOYMENT & MONITORING                                 │
│      └─ Terapkan ke produksi, pantau performa                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 Iterasi dan Feedback Loop

ML pipeline bukan proses linear satu arah. Seringkali kita perlu kembali ke tahap sebelumnya:

- Jika model overfitting → kembali ke preprocessing (tambah regularisasi, kurangi fitur).
- Jika akurasi rendah → kembali ke EDA (cari fitur baru).
- Jika data drift → kumpulkan data baru, latih ulang.

**Referensi Industri:**
Penelitian terbaru menunjukkan bahwa feature engineering yang komprehensif — termasuk derivasi agregat station, line, dan path — sangat penting untuk performa model dalam prediksi defect manufaktur. Pada dataset Bosch Production Line, derivasi 178 agregat dari 969 kolom numerik dan kompresi 2.139 kolom nominal menjadi 16 descriptor risiko berhasil mencapai AUC-ROC 0,966 dengan XGBoost.

## 5. EKSPLORASI DATA (EDA) UNTUK DATA MANUFAKTUR

### 5.1 Mengapa EDA Penting?

**"Garbage in, garbage out."** Jika data yang masuk buruk, model yang dihasilkan juga buruk. EDA membantu kita:

- Memahami karakteristik data manufaktur.
- Mendeteksi masalah (missing value, outlier, duplikat).
- Menemukan pola awal yang relevan.
- Memilih fitur yang relevan untuk modeling.

### 5.2 Tahapan EDA

#### A. Cek Struktur Data

```python
df.shape          # Ukuran data
df.info()         # Tipe data dan missing value
df.head()         # 5 baris pertama
df.describe()     # Statistik deskriptif
df.dtypes         # Tipe data per kolom
```

**Yang Dicari:**

- Berapa baris dan kolom?
- Kolom mana yang numerik vs kategorikal?
- Apakah ada tipe data yang salah?

#### B. Cek Missing Value

```python
df.isnull().sum()           # Jumlah missing per kolom
df.isnull().sum() / len(df) * 100  # Persentase missing
```

**Strategi Penanganan Missing Value:**

| Persentase Missing | Strategi |
| --- | --- |
| < 5% | Hapus baris atau isi dengan mean/median/mode |
| 5–30% | Imputasi (mean/median/mode atau prediksi) |
| > 30% | Pertimbangkan untuk hapus kolom |

#### C. Cek Duplikat

```python
df.duplicated().sum()
df[df.duplicated(keep=False)]  # Tampilkan baris duplikat
```

#### D. Cek Outlier

```python
# Metode IQR
Q1 = df['QualityScore'].quantile(0.25)
Q3 = df['QualityScore'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['QualityScore'] < lower_bound) | (df['QualityScore'] > upper_bound)]
print(f"Jumlah outlier: {len(outliers)}")
```

#### E. Analisis Univariat

```python
# Numerik: histogram + KDE
sns.histplot(df['DefectRate'], kde=True)

# Kategorikal: countplot
sns.countplot(x='DefectStatus', data=df)
```

#### F. Analisis Bivariat

```python
# Numerik vs Kategorikal
sns.boxplot(x='DefectStatus', y='QualityScore', data=df)

# Korelasi
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
```

### 5.3 Insight yang Diharapkan dari EDA Manufaktur

Dari dataset manufaktur, insight yang biasanya muncul:

1. Fitur dengan **korelasi tinggi** dengan DefectStatus adalah kandidat prediktor kuat.
2. Distribusi fitur mungkin **skewed** (tidak normal) karena proses produksi.
3. **Multikolinearitas** antar fitur perlu diidentifikasi (misal: EnergyConsumption vs EnergyEfficiency).
4. **Class imbalance** hampir pasti ada — defect biasanya minoritas.

## 6. PREPROCESSING DATA MANUFAKTUR

### 6.1 Mengapa Preprocessing Penting?

Algoritma ML tidak bisa bekerja dengan data mentah yang "kotor". Preprocessing memastikan data dalam format yang bisa diproses algoritma.

### 6.2 Encoding Data Kategorikal

Algoritma ML hanya mengerti **angka**. Data kategorikal harus diubah menjadi angka.

#### A. Label Encoding (Ordinal)

Digunakan untuk data kategorikal yang **memiliki urutan**.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['ProductType_encoded'] = le.fit_transform(df['ProductType'])
# Hasil: "L" → 0, "M" → 1, "H" → 2
```

**Kapan digunakan:** Target (label) atau fitur ordinal seperti ProductType (Low < Medium < High).

#### B. One-Hot Encoding (Nominal)

Digunakan untuk data kategorikal **tanpa urutan**.

```python
df_encoded = pd.get_dummies(df, columns=['Shift'], prefix='Shift')
```

**Kapan digunakan:** Fitur nominal seperti shift, warna, kota.

**Perbandingan:**

| Aspek | Label Encoding | One-Hot Encoding |
| --- | --- | --- |
| Cocok untuk | Data ordinal | Data nominal |
| Jumlah kolom | Tetap 1 | Bertambah sesuai kategori |
| Masalah | Model bisa salah interpretasi urutan | Dimensionality bertambah |

### 6.3 Feature Scaling

#### A. StandardScaler (Z-score Normalization)

Mengubah data sehingga **mean = 0** dan **std = 1**.

$$z = \frac{x - \mu}{\sigma}$$

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # PENTING: transform, bukan fit_transform
```

#### B. MinMaxScaler

Mengubah data ke rentang [0, 1].

$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

#### Kapan Scaling Dibutuhkan?

| Algoritma | Butuh Scaling? | Alasan |
| --- | --- | --- |
| KNN | ✅ Ya | Berbasis jarak Euclidean |
| SVM | ✅ Ya | Berbasis jarak |
| Logistic Regression | ✅ Ya | Sensitif terhadap skala |
| Naive Bayes | ❌ Tidak | Berbasis probabilitas |
| Decision Tree | ❌ Tidak | Berbasis threshold |
| Random Forest | ❌ Tidak | Berbasis threshold |

### 6.4 Feature Engineering untuk Data Manufaktur

Feature engineering adalah seni menciptakan fitur baru yang lebih informatif dari fitur yang ada.

**Contoh untuk data manufaktur:**

```python
# Total cost
df['Total_Cost'] = df['ProductionVolume'] * df['ProductionCost']

# Efficiency ratio
df['Efficiency_Ratio'] = df['WorkerProductivity'] / (df['EnergyConsumption'] + 1)

# Quality index
df['Quality_Index'] = df['QualityScore'] / (df['DefectRate'] + 1)

# Maintenance intensity
df['Maintenance_Intensity'] = df['MaintenanceHours'] / (df['DowntimePercentage'] + 1)

# Energy per unit
df['Energy_Per_Unit'] = df['EnergyConsumption'] / (df['ProductionVolume'] + 1)
```

**Prinsip Feature Engineering:**

1. **Domain Knowledge:** Pahami proses manufaktur.
2. **Interaksi:** Kombinasikan fitur yang relevan.
3. **Rasio:** Rasio sering lebih informatif daripada nilai absolut.
4. **Agregasi:** Rata-rata, max, min dari time-series.

### 6.5 Data Leakage: Musuh Terbesar ML

**Data leakage** terjadi ketika informasi dari testing data "bocor" ke proses training.

```python
# ❌ SALAH — data leakage!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # fit pada SEMUA data
X_train, X_test = train_test_split(X_scaled, ...)

# ✅ BENAR
X_train, X_test = train_test_split(X, ...)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit HANYA training
X_test_scaled = scaler.transform(X_test)        # transform testing
```

**Akibat data leakage:** Akurasi testing terlihat sangat tinggi, tapi model gagal di dunia nyata.

### 6.6 Split Data: Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% untuk testing
    random_state=42,     # Reproducible
    stratify=y           # Jaga proporsi kelas
)
```

#### Mengapa `stratify=y` Penting?

Dengan `stratify=y`, proporsi kelas di training dan testing **sama** dengan dataset asli. Ini sangat penting untuk data imbalanced seperti data defect.

## 7. CLASS IMBALANCE: TANTANGAN UTAMA DALAM QUALITY CONTROL

### 7.1 Apa Itu Class Imbalance?

**Class imbalance** terjadi ketika distribusi kelas dalam dataset tidak seimbang — satu kelas jauh lebih banyak daripada kelas lainnya.

**Dalam konteks manufaktur:**

- Defect biasanya jarang terjadi (1–5% dari total produksi).
- Contoh: 84,04% High Defects vs 15,96% Low Defects (pada dataset Predicting Manufacturing Defects).
- Contoh ekstrem: 0,58% labeled failures pada Bosch Production Line dataset.

### 7.2 Mengapa Class Imbalance Berbahaya?

**Masalah utama:** Algoritma ML cenderung memprediksi kelas mayoritas karena memberikan akurasi tinggi secara keseluruhan.

**Contoh:**

- Dataset: 95% produk baik, 5% defect.
- Model yang selalu memprediksi "baik" akan punya akurasi 95%.
- Tapi model ini **tidak berguna** karena tidak pernah mendeteksi defect!

**Dampak dalam manufaktur:**

- **False Negative (FN):** Defect lolos ke pasar → risiko besar.
- **False Positive (FP):** Produk baik dinyatakan defect → pemborosan.
- FN biasanya lebih berbahaya, sehingga **Recall** menjadi metrik prioritas.

### 7.3 Strategi Penanganan Class Imbalance

#### A. Resampling Techniques

**1. Random Over-Sampling:** Duplikasi sampel kelas minoritas.

```python
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
```

**Kelebihan:** Sederhana.
**Kekurangan:** Bisa menyebabkan overfitting (duplikasi).

**2. Random Under-Sampling:** Hapus sampel kelas mayoritas.

```python
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
```

**Kelebihan:** Cepat.
**Kekurangan:** Kehilangan informasi.

**3. SMOTE (Synthetic Minority Over-sampling Technique):** Buat sampel sintetis kelas minoritas.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print("Sebelum SMOTE:", y_train.value_counts().to_dict())
print("Setelah SMOTE:", pd.Series(y_train_resampled).value_counts().to_dict())
```

**Cara Kerja SMOTE:**

1. Pilih sampel dari kelas minoritas.
2. Cari K tetangga terdekat dari kelas yang sama.
3. Buat sampel baru di antara sampel asli dan tetangga (interpolasi).

**Kelebihan:** Menghindari duplikasi, menghasilkan sampel yang lebih beragam.
**Kekurangan:** Bisa menghasilkan sampel yang tidak realistis.

**4. ADASYN (Adaptive Synthetic Sampling):** Varian SMOTE yang fokus pada sampel yang sulit diklasifikasi.

```python
from imblearn.over_sampling import ADASYN
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
```

#### B. Cost-Sensitive Learning

Memberikan **bobot lebih tinggi** pada kelas minoritas.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced', random_state=42)
```

**Kelebihan:** Tidak mengubah distribusi data.
**Kekurangan:** Perlu tuning bobot.

#### C. Threshold Moving

Mengubah threshold klasifikasi dari 0,5 menjadi nilai lain.

```python
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba > 0.3).astype(int)  # Threshold 0,3
```

### 7.4 Studi Kasus: SMOTE dalam Manufaktur

Penelitian menunjukkan bahwa SMOTE secara efektif mengatasi class imbalance dalam klasifikasi kualitas manufaktur:

- Pada klasifikasi **welding quality** di manufaktur battery pack, SMOTE digunakan untuk augmentasi kelas minoritas sebelum melatih Random Forest classifier.

- Pada prediksi **defect status produksi** dengan dataset Predicting Manufacturing Defects, class imbalance (84,04% High Defects vs 15,96% Low Defects) diatasi menggunakan SMOTE.

- Pada **injection molding products**, SMOTE dan ADASYN diterapkan bersama cost-sensitive learning untuk mengatasi class imbalance.

### 7.5 Kapan Menggunakan Strategi Apa?

| Situasi | Strategi yang Direkomendasikan |
| --- | --- |
| Imbalance ringan (< 3:1) | Class weight, threshold moving |
| Imbalance sedang (3:1 – 10:1) | SMOTE, Random Over-Sampling |
| Imbalance berat (10:1 – 100:1) | SMOTE + Tomek Links, ADASYN |
| Imbalance ekstrem (> 100:1) | Kombinasi resampling + cost-sensitive |

## 8. ALGORITMA KLASIFIKASI UNTUK DEFECT PREDICTION

### 8.1 Decision Tree (Pohon Keputusan)

#### Konsep

Decision Tree memecah data berdasarkan **pertanyaan bersyarat** secara rekursif, membentuk struktur pohon.

#### Cara Kerja

1. Pilih fitur terbaik untuk memecah data berdasarkan **Information Gain** atau **Gini Impurity**.
2. Ulangi proses pada setiap cabang hingga kriteria berhenti terpenuhi.

**Gini Impurity:**
$$Gini = 1 - \sum_{i=1}^{n} p_i^2$$

**Information Gain (Entropy):**
$$Entropy = -\sum_{i=1}^{n} p_i \log_2(p_i)$$

#### Kelebihan & Kekurangan

| Kelebihan | Kekurangan |
| --- | --- |
| Mudah dipahami dan divisualisasikan | Mudah overfitting |
| Tidak butuh scaling | Sensitif terhadap data kecil |
| Bisa menangani data kategorikal | Tidak stabil |

#### Hyperparameter Penting

```python
DecisionTreeClassifier(
    max_depth=5,              # Kedalaman maksimum
    min_samples_split=2,      # Min sampel untuk split
    min_samples_leaf=1,       # Min sampel di leaf
    criterion='gini',         # atau 'entropy'
    class_weight='balanced',  # Untuk imbalance
    random_state=42
)
```

### 8.2 K-Nearest Neighbors (KNN)

#### Konsep

KNN mengklasifikasikan data baru berdasarkan **mayoritas kelas dari K tetangga terdekat**.

**Jarak Euclidean:**
$$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

#### Kelebihan & Kekurangan

| Kelebihan | Kekurangan |
| --- | --- |
| Sederhana | Lambat untuk dataset besar |
| Tidak ada proses training | Sensitif terhadap scaling |
| Bisa menangani multi-kelas | Sensitif terhadap outlier |

#### Tips Memilih K

- K kecil (1–3): sensitif terhadap noise, overfitting.
- K besar (>10): terlalu generalisasi, underfitting.
- K ganjil: hindari seri dalam voting.

### 8.3 Naive Bayes

#### Konsep

Naive Bayes menggunakan **Teorema Bayes** dengan asumsi **independensi antar fitur**.

$$P(y|X) = \frac{P(X|y) \cdot P(y)}{P(X)}$$

#### Kelebihan & Kekurangan

| Kelebihan | Kekurangan |
| --- | --- |
| Cepat dan efisien | Asumsi independensi sering salah |
| Bekerja baik untuk dataset kecil | Kurang akurat untuk data kompleks |
| Tidak butuh scaling | Tidak bisa menangkap interaksi fitur |

### 8.4 Logistic Regression

#### Konsep

Logistic Regression memodelkan **probabilitas** kelas menggunakan fungsi sigmoid.

$$P(y=1|X) = \frac{1}{1 + e^{-(w_0 + w_1 x_1 + ... + w_n x_n)}}$$

#### Kelebihan & Kekurangan

| Kelebihan | Kekurangan |
| --- | --- |
| Probabilistik (output berupa probabilitas) | Hanya untuk hubungan linear |
| Interpretable (koefisien menunjukkan pengaruh) | Butuh scaling |
| Cepat dan stabil | Kurang baik untuk hubungan non-linear |

### 8.5 Random Forest

#### Konsep

Random Forest adalah **ensemble** dari banyak Decision Tree. Setiap tree dilatih pada **bootstrap sample** yang berbeda dan menggunakan **subset fitur** yang random. Prediksi akhir adalah **voting** dari semua tree.

#### Cara Kerja

1. Buat N bootstrap sample dari training data.
2. Latih Decision Tree pada setiap sample (dengan subset fitur random).
3. Prediksi dengan voting (klasifikasi) atau rata-rata (regresi).

#### Kelebihan & Kekurangan

| Kelebihan | Kekurangan |
| --- | --- |
| Akurasi tinggi | Kurang interpretable |
| Robust terhadap outlier | Lebih lambat dari single tree |
| Bisa menangani data imbalanced | Memori lebih besar |
| Tidak mudah overfitting | — |

#### Hyperparameter Penting

```python
RandomForestClassifier(
    n_estimators=100,         # Jumlah tree
    max_depth=10,             # Kedalaman maksimum
    min_samples_split=2,      # Min sampel untuk split
    max_features='sqrt',      # Jumlah fitur per split
    class_weight='balanced',  # Untuk imbalance
    random_state=42,
    n_jobs=-1                 # Parallel processing
)
```

### 8.6 XGBoost (Extreme Gradient Boosting)

#### Konsep

XGBoost adalah algoritma **gradient boosting** yang membangun tree secara **sequential** — setiap tree baru memperbaiki kesalahan tree sebelumnya.

#### Kelebihan & Kekurangan

| Kelebihan | Kekurangan |
| --- | --- |
| Akurasi sangat tinggi | Lebih kompleks |
| Menangani imbalance dengan baik | Butuh tuning lebih banyak |
| Regularisasi built-in | Lebih lambat dari Random Forest |
| Feature importance tersedia | — |

### 8.7 Perbandingan Algoritma

| Algoritma | Kecepatan | Akurasi | Interpretability | Butuh Scaling | Imbalance Handling |
| --- | --- | --- | --- | --- | --- |
| Decision Tree | Cepat | Sedang | Tinggi | Tidak | class_weight |
| KNN | Lambat (prediksi) | Sedang | Rendah | Ya | SMOTE |
| Naive Bayes | Sangat cepat | Sedang | Sedang | Tidak | class_prior |
| Logistic Regression | Cepat | Sedang-Baik | Tinggi | Ya | class_weight |
| Random Forest | Sedang | Tinggi | Sedang | Tidak | class_weight |
| XGBoost | Sedang | Sangat Tinggi | Sedang | Tidak | scale_pos_weight |

## 9. EVALUASI MODEL UNTUK DATA IMBALANCED

### 9.1 Confusion Matrix

Confusion Matrix adalah tabel yang menunjukkan **perbandingan prediksi vs aktual**.

|  | Prediksi: Negatif | Prediksi: Positif |
|---|---|---|
| **Aktual: Negatif** | True Negative (TN) | False Positive (FP) |
| **Aktual: Positif** | False Negative (FN) | True Positive (TP) |

**Dalam Konteks Manufaktur:**

- **Positif** = "High Defect"
- **Negatif** = "Low Defect"
- **TP** = Benar diprediksi High Defect
- **TN** = Benar diprediksi Low Defect
- **FP** = Salah diprediksi High Defect (padahal Low) — **Error Tipe I**
- **FN** = Salah diprediksi Low Defect (padahal High) — **Error Tipe II**

**Mana yang Lebih Berbahaya?**

- **FP:** Produk baik dinyatakan defect → pemborosan material.
- **FN:** Defect lolos ke pasar → risiko besar.
- Dalam manufaktur, **FN biasanya lebih berbahaya**, sehingga **Recall** menjadi metrik prioritas.

### 9.2 Metrik Evaluasi

#### A. Accuracy

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

**Masalah:** Tidak cocok untuk data imbalanced. Jika 95% data adalah "Low Defect", model yang selalu memprediksi "Low Defect" akan punya akurasi 95%, tapi tidak berguna.

#### B. Precision

$$Precision = \frac{TP}{TP + FP}$$

**Arti:** Dari semua yang diprediksi "High Defect", berapa persen yang benar-benar defect?

**Kapan diprioritaskan:** Ketika **FP mahal**. Contoh: mengurangi pemborosan.

#### C. Recall (Sensitivity)

$$Recall = \frac{TP}{TP + FN}$$

**Arti:** Dari semua yang sebenarnya "High Defect", berapa persen yang berhasil diprediksi?

**Kapan diprioritaskan:** Ketika **FN mahal**. Contoh: **defect detection di manufaktur**.

#### D. F1-Score

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

**Arti:** Rata-rata harmonik Precision dan Recall. Berguna saat data tidak seimbang.

#### E. AUC-ROC

**Area Under the ROC Curve** mengukur kemampuan model membedakan kelas.

- **AUC = 0,5:** Model tidak lebih baik dari random.
- **AUC > 0,8:** Model baik.
- **AUC > 0,9:** Model sangat baik.

**Kelebihan AUC-ROC:** Tidak terpengaruh threshold dan class imbalance.

### 9.3 Memilih Metrik yang Tepat

| Situasi | Metrik Utama |
| --- | --- |
| Dataset seimbang | Accuracy |
| Dataset tidak seimbang | F1-Score, AUC-ROC |
| FP lebih mahal | Precision |
| FN lebih mahal | Recall |
| Butuh keseimbangan | F1-Score |

### 9.4 Contoh Hasil Evaluasi

**Logistic Boosting untuk anomaly detection di IoT-driven factories:**

- AUC: 0,992
- Accuracy: 96,6%
- Precision: 93,5%
- Recall: 94,8%
- F1-Score: 0,941
- Model ini menunjukkan superior handling of imbalanced data dengan hanya 134 false positives dan 117 false negatives.

**Random Forest untuk prediksi defect produksi:**

- Accuracy: 94,75%
- F1-Score: 94,49%
- SHAP analysis mengungkapkan bahwa MaintenanceHours, DefectRate, dan QualityScore adalah tiga faktor paling dominan dalam menentukan status defect produksi.

## 10. FEATURE IMPORTANCE DAN INTERPRETABILITY

### 10.1 Mengapa Feature Importance Penting?

Feature importance membantu kita memahami **fitur mana yang paling berpengaruh** terhadap prediksi. Dalam konteks manufaktur, ini sangat berharga karena:

1. **Prioritas Monitoring:** Fokuskan sensor dan monitoring pada fitur penting.
2. **Root Cause Analysis:** Identifikasi penyebab utama defect.
3. **Actionable Insights:** Berikan rekomendasi konkret ke tim produksi.
4. **Transparansi:** Jelaskan keputusan model ke stakeholder.

### 10.2 Metode Feature Importance

#### A. Built-in Feature Importance (Tree-based)

```python
# Feature importance dari Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    'Fitur': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance)
```

#### B. Permutation Importance

```python
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(
    rf_model, X_test, y_test, n_repeats=10, random_state=42
)

perm_df = pd.DataFrame({
    'Fitur': feature_cols,
    'Importance': perm_importance.importances_mean
}).sort_values('Importance', ascending=False)
```

**Kelebihan:** Model-agnostic, mengukur dampak aktual pada performa.

#### C. SHAP (SHapley Additive exPlanations)

SHAP adalah metode **game-theoretic** untuk menjelaskan prediksi model.

```python
import shap

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, feature_names=feature_cols)
```

**Kelebihan:** Memberikan penjelasan per-prediksi, arah pengaruh fitur.

### 10.3 Contoh Feature Importance dalam Manufaktur

**Dataset Predicting Manufacturing Defects:**
SHAP analysis mengungkapkan bahwa **MaintenanceHours**, **DefectRate**, dan **QualityScore** adalah tiga faktor paling dominan dalam menentukan status defect produksi.

**Dataset Tire Manufacturing (PT Multistrada):**
Feature importance menunjukkan **Cure_Temperature** (21,87%) dan **Uniformity_Index** (17,54%) sebagai faktor dominan. Integrasi Random Forest meningkatkan recall 20,79 poin persentase dibanding pendekatan konvensional.

**Dataset High-Pressure Die Casting:**
Block temperature sensors dan force measurements memiliki korelasi tinggi dengan semua kelas defect dan diberi bobot tinggi oleh model Random Forest.

### 10.4 Dari Feature Importance ke Actionable Insights

| Fitur Penting | Insight | Tindakan |
| --- | --- | --- |
| MaintenanceHours | Maintenance berkorelasi dengan defect | Jadwalkan maintenance preventif |
| DefectRate | Defect rate historis prediktif | Monitor trend defect rate |
| QualityScore | Skor kualitas rendah → risiko defect | Tingkatkan quality control |
| Cure_Temperature | Suhu cure mempengaruhi kualitas | Optimasi parameter suhu |
| SupplierQuality | Kualitas supplier mempengaruhi | Evaluasi supplier |

## 11. CROSS-VALIDATION DAN HYPERPARAMETER TUNING

### 11.1 Cross-Validation

**Masalah dengan single train-test split:** Hasil bisa bervariasi tergantung "keberuntungan" split.

**Solusi:** K-Fold Cross-Validation — bagi data menjadi K bagian, latih K kali, rata-ratakan hasilnya.

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X_train_scaled, y_train,
                         cv=cv, scoring='f1')
print(f"F1 per fold: {scores}")
print(f"Rata-rata: {scores.mean():.4f} ± {scores.std():.4f}")
```

**Ilustrasi K=5:**

```
Fold 1: [Test] [Train] [Train] [Train] [Train]
Fold 2: [Train] [Test] [Train] [Train] [Train]
Fold 3: [Train] [Train] [Test] [Train] [Train]
Fold 4: [Train] [Train] [Train] [Test] [Train]
Fold 5: [Train] [Train] [Train] [Train] [Test]
```

**Kelebihan Stratified K-Fold:** Menjaga proporsi kelas di setiap fold, penting untuk data imbalanced.

### 11.2 Hyperparameter Tuning

#### A. Grid Search

Mencoba semua kombinasi hyperparameter.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [50, 100, 200]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best F1: {grid_search.best_score_:.4f}")
```

#### B. Random Search

Mencoba kombinasi random — lebih efisien untuk ruang parameter besar.

```python
from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    scoring='f1',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_scaled, y_train)
```

### 11.3 Best Practices

1. **Selalu gunakan cross-validation** untuk evaluasi yang robust.
2. **Gunakan scoring metric yang sesuai** — F1 untuk imbalance, bukan accuracy.
3. **Hindari overfitting** dengan tidak tuning terlalu banyak.
4. **Simpan model terbaik** dengan `joblib` atau `pickle`.

## 12. STUDI KASUS INDUSTRI NYATA

### 12.1 Case Study 1: Predicting Manufacturing Defects (Kaggle)

**Dataset:** 3.240 record, 16 fitur, target DefectStatus (0/1).

**Pendekatan:** Ensemble Learning (XGBoost, LightGBM, Random Forest) vs baseline (SVM, KNN).

**Hasil:**

- Random Forest: Accuracy 94,75%, F1-Score 94,49%.
- LightGBM: Mean F1-Score 95,92% ± 1,07% (5-fold CV).
- SHAP analysis: MaintenanceHours, DefectRate, QualityScore adalah fitur paling dominan.

**Pelajaran:** Ensemble methods mengungguli model baseline untuk prediksi defect produksi.

### 12.2 Case Study 2: AI4I 2020 Predictive Maintenance

**Dataset:** 10.000 data points, 14 fitur (suhu udara, suhu proses, kecepatan rotasi, torsi, tool wear).

**Target:** Machine failure (binary) dan 5 failure modes (TWF, HDF, PWF, OSF, RNF).

**Pendekatan:** Klasifikasi binary dan multi-label.

**Pelajaran:**

- Dataset sintetis yang merefleksikan kondisi nyata industri.
- Fitur fisik seperti torsi dan tool wear sangat prediktif.
- Cocok untuk latihan predictive maintenance.

### 12.3 Case Study 3: Bosch Production Line Performance

**Dataset:** 1.183.165 part, 4.264 station features, hanya 0,58% labeled failures.

**Tantangan:** Extreme class imbalance, sparse high-cardinality nominal codes.

**Pendekatan:** Weight of Evidence (WoE) compression, XGBoost.

**Hasil:** AUC-ROC 0,966, F1 0,929.

**Pelajaran:**

- Feature engineering yang komprehensif sangat penting.
- Extreme imbalance membutuhkan teknik khusus.
- Distributed computing (Spark) diperlukan untuk dataset besar.

### 12.4 Case Study 4: Tire Manufacturing (PT Multistrada)

**Dataset:** 15 atribut karakteristik kualitas, target status defect.

**Distribusi:** Non-Defect 76,6%, Defect 23,4% (imbalance ringan, rasio 3,27:1).

**Pendekatan:** Six Sigma DMAIC + Random Forest, threshold adjustment 0,45.

**Hasil:** Akurasi 96,70%, Recall 95,09%, F1-Score 93,10%, AUC-ROC 0,971.

**Feature importance:** Cure_Temperature (21,87%), Uniformity_Index (17,54%).

**Pelajaran:**

- Integrasi metodologi kualitas (Six Sigma) dengan ML memberikan hasil superior.
- Recall meningkat 20,79 poin persentase dibanding pendekatan konvensional.

### 12.5 Case Study 5: High-Pressure Die Casting

**Dataset:** 7.982 casting cycles, target 5 kelas defect (cold flow, shrinkage cavity, blister, soldering point, scrap).

**Pendekatan:** SVM, Random Forest, AutoML.

**Hasil:** Random Forest terbaik, terutama untuk kelas "soldering point" (Balanced Accuracy > 80%).

**Pelajaran:**

- Data dari casting machine saja sudah cukup untuk baseline.
- Penambahan die sensor data meningkatkan performa secara signifikan.

## 13. DEPLOYMENT DAN MONITORING

### 13.1 Dari Notebook ke Produksi

Setelah model dilatih dan dievaluasi, langkah selanjutnya adalah deployment. Dalam konteks manufaktur, model dapat di-deploy sebagai:

1. **Batch Prediction:** Prediksi dilakukan secara periodik (misal setiap shift).
2. **Real-time Prediction:** Prediksi dilakukan secara real-time saat produksi berjalan.
3. **Edge Deployment:** Model dijalankan langsung di mesin/PLC.

### 13.2 Model Serialization

```python
import joblib

# Simpan model
joblib.dump(model, 'defect_prediction_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Load model
model = joblib.load('defect_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')
```

### 13.3 Monitoring Model

Setelah deployment, model perlu dipantau secara berkala:

| Aspek | Metrik | Frekuensi |
| --- | --- | --- |
| **Performance** | F1-Score, Recall, AUC | Harian/Mingguan |
| **Data Drift** | Distribusi fitur | Mingguan |
| **Concept Drift** | Hubungan fitur-target | Bulanan |
| **Latency** | Waktu prediksi | Real-time |

### 13.4 Retraining Strategy

Model perlu dilatih ulang ketika:

- Performa menurun di bawah threshold.
- Data drift terdeteksi.
- Ada perubahan proses produksi.
- Ada data baru yang signifikan.

### 13.5 Tantangan Deployment di Manufaktur

1. **Integrasi dengan MES/SCADA:** Model harus terintegrasi dengan sistem yang ada.
2. **Latency:** Prediksi harus cukup cepat untuk real-time.
3. **Reliability:** Model harus tersedia 24/7.
4. **Explainability:** Operator perlu memahami mengapa model memprediksi defect.
5. **Human-in-the-Loop:** Keputusan akhir tetap pada manusia.

## 14. ETIKA DAN TANTANGAN DALAM PREDICTIVE QUALITY

### 14.1 Tantangan Teknis

| Tantangan | Deskripsi | Solusi |
| --- | --- | --- |
| **Class Imbalance** | Defect jarang terjadi | SMOTE, cost-sensitive |
| **Data Drift** | Kondisi mesin berubah | Monitoring, retraining |
| **Label Noise** | Label defect tidak akurat | Data cleaning, human verification |
| **Multikolinearitas** | Fitur saling berkorelasi | Feature selection, PCA |
| **Interpretability** | Sulit menjelaskan prediksi | SHAP, LIME |

### 14.2 Tantangan Non-Teknis

| Tantangan | Deskripsi |
| --- | --- |
| **Adopsi Operator** | Operator mungkin resisten terhadap AI |
| **Kejelasan Data** | Data dari berbagai sumber, format berbeda |
| **Biaya Implementasi** | Investasi sensor, infrastruktur |
| **Regulasi** | Standar kualitas dan keselamatan |
| **Etika** | Keputusan otonom vs human oversight |

### 14.3 Prinsip Etika dalam Predictive Quality

1. **Transparansi:** Model harus bisa dijelaskan.
2. **Akuntabilitas:** Ada pihak yang bertanggung jawab.
3. **Keadilan:** Model tidak diskriminatif.
4. **Keamanan:** Model tidak membahayakan.
5. **Human Oversight:** Manusia tetap dalam kendali.

### 14.4 Human-in-the-Loop

Dalam banyak aplikasi manufaktur, keputusan akhir tetap pada manusia. Model ML berfungsi sebagai **decision support system** — memberikan rekomendasi, bukan keputusan final.

Studi menunjukkan bahwa beberapa peneliti memperingatkan terhadap **over-reliance on automated systems**, menekankan perlunya **human oversight** dalam menginterpretasikan output Machine Learning untuk menghindari potensi bias.

## 15. LATIHAN DAN SOAL REFLEKSI

### 15.1 Latihan Coding

**Latihan 1:** Implementasi Confusion Matrix Manual

```python
def confusion_matrix_manual(y_true, y_pred):
    """
    Hitung TP, TN, FP, FN secara manual.
    """
    tp = sum((y_true == 1) & (y_pred == 1))
    tn = sum((y_true == 0) & (y_pred == 0))
    fp = sum((y_true == 0) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))

    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
```

**Latihan 2:** Implementasi SMOTE Sederhana

```python
def simple_smote(X_minority, k_neighbors=5, n_synthetic=100):
    """
    Buat sampel sintetis menggunakan interpolasi.
    """
    from sklearn.neighbors import NearestNeighbors
    import numpy as np

    nn = NearestNeighbors(n_neighbors=k_neighbors)
    nn.fit(X_minority)

    synthetic = []
    for _ in range(n_synthetic):
        idx = np.random.randint(len(X_minority))
        neighbors = nn.kneighbors([X_minority[idx]], return_distance=False)[0]
        neighbor_idx = np.random.choice(neighbors[1:])

        # Interpolasi
        diff = X_minority[neighbor_idx] - X_minority[idx]
        gap = np.random.random()
        new_sample = X_minority[idx] + gap * diff
        synthetic.append(new_sample)

    return np.array(synthetic)
```

**Latihan 3:** Visualisasi Feature Importance

```python
def plot_feature_importance(model, feature_names, top_n=15):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]

    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importance[indices], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
```

### 15.2 Soal Refleksi

1. **Konsep:** Mengapa accuracy tidak cukup untuk data imbalanced? Berikan contoh konkret.

2. **Aplikasi:** Jika Anda diminta membuat sistem prediksi defect untuk lini produksi Polman, fitur apa saja yang akan Anda gunakan? Mengapa?

3. **Etika:** Apakah model ML boleh digunakan sebagai satu-satunya penentu keputusan kualitas produk? Diskusikan.

4. **Teknis:** Mengapa `stratify=y` penting? Apa yang terjadi jika dihilangkan pada dataset yang tidak seimbang?

5. **Analisis:** Jika model Anda memiliki Train Accuracy 0,99 dan Test Accuracy 0,70, apa yang terjadi? Bagaimana cara mengatasinya?

6. **Metrik:** Dalam konteks defect detection, mana yang lebih penting: Precision atau Recall? Jelaskan.

7. **Preprocessing:** Mengapa kita perlu melakukan SMOTE? Apa risikonya?

8. **Interpretasi:** Apa arti feature importance? Bagaimana menggunakannya untuk perbaikan proses?

### 15.3 Studi Kasus Mini

**Skenario:** Sebuah pabrik komponen otomotif ingin memprediksi defect pada proses stamping.

**Tugas:**

1. Tentukan apakah ini klasifikasi atau regresi. Mengapa?
2. Fitur apa yang akan Anda gunakan? (Suhu, tekanan, kecepatan, material, dll.)
3. Metrik apa yang paling penting? Mengapa?
4. Algoritma apa yang akan Anda pilih? Mengapa?
5. Bagaimana Anda menangani data yang tidak seimbang?

## 16. GLOSARIUM

| Istilah | Definisi |
| --- | --- |
| **Accuracy** | Proporsi prediksi benar dari total prediksi |
| **AUC-ROC** | Area Under ROC Curve — mengukur kemampuan diskriminasi |
| **Class Imbalance** | Distribusi kelas yang tidak seimbang |
| **Confusion Matrix** | Tabel TP, TN, FP, FN |
| **Cross-Validation** | Teknik evaluasi dengan membagi data menjadi K fold |
| **Data Drift** | Perubahan distribusi data seiring waktu |
| **Data Leakage** | Informasi testing bocor ke training |
| **Decision Tree** | Algoritma klasifikasi berbasis pohon keputusan |
| **EDA** | Exploratory Data Analysis |
| **Encoding** | Konversi data kategorikal ke numerik |
| **F1-Score** | Rata-rata harmonik Precision dan Recall |
| **Feature** | Variabel input dalam model ML |
| **Feature Engineering** | Menciptakan fitur baru yang informatif |
| **Feature Importance** | Ukuran kontribusi fitur terhadap prediksi |
| **Feature Scaling** | Normalisasi/standardisasi fitur |
| **Gini Impurity** | Ukuran ketidakmurnian dalam Decision Tree |
| **Hyperparameter** | Parameter yang diatur sebelum training |
| **Imbalanced Data** | Dataset dengan distribusi kelas tidak seimbang |
| **KNN** | K-Nearest Neighbors |
| **Label** | Variabel output yang diprediksi |
| **Logistic Regression** | Algoritma klasifikasi dengan fungsi sigmoid |
| **Naive Bayes** | Algoritma klasifikasi berbasis Teorema Bayes |
| **Overfitting** | Model terlalu menghafal data training |
| **Precision** | TP / (TP + FP) |
| **Random Forest** | Ensemble dari banyak Decision Tree |
| **Recall** | TP / (TP + FN) |
| **Regularisasi** | Teknik untuk mencegah overfitting |
| **SHAP** | Metode interpretability berbasis game theory |
| **SMOTE** | Synthetic Minority Over-sampling Technique |
| **Supervised Learning** | ML dengan data berlabel |
| **Test Set** | Data untuk menguji model |
| **Train Set** | Data untuk melatih model |
| **Underfitting** | Model terlalu sederhana |
| **XGBoost** | Extreme Gradient Boosting |

## 17. REFERENSI

### 17.1 Buku

1. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.
2. Raschka, S., & Mirjalili, V. (2019). *Python Machine Learning* (3rd ed.). Packt.
3. James, G., et al. (2021). *An Introduction to Statistical Learning*. Springer.
4. Montgomery, D. C. (2019). *Introduction to Statistical Quality Control*. Wiley.

### 17.2 Artikel dan Jurnal

1. Studi Ensemble Learning untuk klasifikasi defect produksi. *Garuda Kemdiktisaintek* (2026). — Dataset Predicting Manufacturing Defects, SMOTE, SHAP analysis.
2. Zdravevski, E., et al. (2026). Real-time IIoT-driven machine failure forecasting. *Scientific Reports*. — Bosch Production Line, XGBoost, WoE compression.
3. Studi Six Sigma DMAIC + Random Forest untuk prediksi defect ban. *Garuda Kemdiktisaintek* (2026). — PT Multistrada, feature importance.
4. Pachandrin, S., et al. (2025). Data-Driven Prediction of Casting Defects. *International Journal of Metalcasting*. — High-pressure die casting, Random Forest.
5. Enhancing anomaly detection in IoT-driven factories. *Scientific Reports* (2025). — Logistic Boosting, Random Forest, SVM.
6. Matzka, S. (2020). Explainable AI for Predictive Maintenance Applications. *IEEE International Conference on AI for Industries*.
7. Tercan, H., & Meisen, T. (2022). Machine learning and deep learning based predictive quality in manufacturing: A systematic review. *Journal of Intelligent Manufacturing*.

### 17.3 Dataset

1. El Kharoua, R. (2023). *Predicting Manufacturing Defects Dataset*. Kaggle. <https://www.kaggle.com/datasets/rabieelkharoua/predicting-manufacturing-defects-dataset>
2. Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset*. UCI Machine Learning Repository. <https://archive.ics.uci.edu/dataset/601/predictive+maintenance+dataset>
3. *Bosch Production Line Performance*. Kaggle. <https://www.kaggle.com/c/bosch-production-line-performance>

### 17.4 Dokumentasi Online

1. **Scikit-learn Classification:** <https://scikit-learn.org/stable/modules/classification.html>
2. **Imbalanced-learn (SMOTE):** <https://imbalanced-learn.org/stable/>
3. **SHAP Documentation:** <https://shap.readthedocs.io/>
4. **XGBoost Documentation:** <https://xgboost.readthedocs.io/>

### 17.5 Video Tutorial

1. StatQuest — Decision Trees: <https://www.youtube.com/watch?v=7VeUPuFGJHk>
2. StatQuest — Random Forests: <https://www.youtube.com/watch?v=J4Wdy0Wc_xQ>
3. StatQuest — Logistic Regression: <https://www.youtube.com/watch?v=yIYKR4sgzI8>
4. StatQuest — ROC and AUC: <https://www.youtube.com/watch?v=4jRBRDbJemM>

## PENUTUP

Modul ajar ini disusun sebagai **penunjang teori** untuk praktikum pertemuan ke-1 minggu ke-2 tentang **Supervised Learning untuk Klasifikasi Defect Produk Manufaktur**. Materi ini mencakup:

1. **Fondasi supervised learning** dan alur kerja ML pipeline.
2. **EDA untuk data manufaktur** — memahami karakteristik data sensor dan proses.
3. **Preprocessing data manufaktur** — encoding, scaling, feature engineering.
4. **Class imbalance** — tantangan utama dalam quality control dan strategi penanganannya.
5. **Algoritma klasifikasi** — dari Decision Tree hingga XGBoost.
6. **Evaluasi model** — metrik yang tepat untuk data imbalanced.
7. **Feature importance** — dari model ke actionable insights.
8. **Studi kasus industri nyata** — dari Kaggle, UCI, dan penelitian terbaru.
9. **Deployment dan etika** — tantangan menerapkan ML di lantai produksi.

**Pesan untuk Mahasiswa:**
> "Predictive quality bukan hanya tentang membangun model akurat, tetapi tentang **memahami proses manufaktur** dan **mengomunikasikan insight** ke tim produksi. Model terbaik adalah model yang dapat ditindaklanjuti — yang memberikan rekomendasi konkret untuk meningkatkan kualitas produk."

**Pesan untuk Dosen:**
> "Modul ini dirancang untuk menjembatani teori Machine Learning dengan praktik industri manufaktur. Untuk mahasiswa pemula, fokus pada Decision Tree, Logistic Regression, dan Random Forest terlebih dahulu. Untuk mahasiswa mahir, tambahkan XGBoost, SHAP, dan deployment."

---

**Dokumen ini disusun untuk keperluan edukasi di Politeknik Manufaktur Bandung (Polman Bandung).**
**Versi:** 1.0 | **Minggu ke-2, Pertemuan ke-1 dari 4**
**Terakhir Diperbarui:** 2026
