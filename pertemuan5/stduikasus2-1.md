# STUDI KASUS MACHINE LEARNING — PERTEMUAN KE-1 MINGGU KE-2

## Supervised Learning: Prediksi Defect Produk pada Industri Manufaktur

**Mata Kuliah:** Machine Learning
**Program Studi:** D4 Teknologi Rekayasa Informatika dan Komputer (TRIN)
**Institusi:** Politeknik Manufaktur Bandung (Polman Bandung)
**Pertemuan:** Minggu ke-2, Pertemuan ke-1 dari 4
**Topik:** Supervised Learning — Klasifikasi Kualitas Produk Manufaktur
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

Dalam dunia industri manufaktur modern, **kualitas produk** adalah faktor krusial yang menentukan kepuasan pelanggan, efisiensi biaya, dan daya saing perusahaan. Setiap cacat produk (defect) yang lolos ke pasar dapat menyebabkan kerugian finansial yang signifikan, kerusakan reputasi, bahkan risiko keselamatan bagi pengguna.

Namun, mendeteksi cacat produk secara manual seringkali tidak efisien dan tidak konsisten. Inspektur manusia bisa lelah, subjektif, dan tidak mampu memeriksa setiap unit dengan standar yang sama. Di sinilah **Machine Learning (ML)** berperan: dengan menganalisis data historis dari lini produksi, kita dapat membangun model yang mampu **memprediksi probabilitas cacat produk** sebelum produk tersebut selesai diproduksi.

Pendekatan ini dikenal sebagai **predictive quality** atau **defect prediction**. Alih-alih memeriksa produk di akhir lini produksi, sistem dapat mengidentifikasi **risiko cacat sejak dini** berdasarkan parameter proses, kondisi mesin, dan faktor lingkungan. Dengan demikian, perusahaan dapat melakukan **tindakan preventif** — menyesuaikan parameter mesin, menjadwalkan maintenance, atau menghentikan produksi sementara — sebelum cacat terjadi.

Studi ini sangat relevan dengan **Polman Bandung** sebagai institusi pendidikan vokasi yang fokus pada **manufaktur dan otomasi industri**. Mahasiswa D4 TRIN perlu memahami bagaimana data dari lini produksi dapat diolah menjadi **insight yang dapat ditindaklanjuti** untuk meningkatkan kualitas produk.

**Referensi Industri:**

- Penelitian menunjukkan bahwa pendekatan Ensemble Learning (XGBoost, LightGBM, Random Forest) dapat mencapai akurasi hingga **94,75%** dalam memprediksi status defect produksi menggunakan dataset *Predicting Manufacturing Defects* dari Kaggle.
- Dataset AI4I 2020 Predictive Maintenance dari UCI Machine Learning Repository menyediakan 10.000 data poin dengan 14 fitur untuk prediksi kegagalan mesin, termasuk tool wear, torque, dan rotational speed.
- Bosch Production Line Performance (Kaggle) mencatat 1.184.687 produk selama proses manufaktur, menjadi salah satu dataset terbesar untuk prediksi cacat produksi.

## 2. TUJUAN PEMBELAJARAN

Setelah mengikuti praktikum ini, mahasiswa diharapkan mampu:

1. **Memahami** konsep supervised learning untuk klasifikasi kualitas produk manufaktur.
2. **Melakukan** eksplorasi data (EDA) pada dataset defect produksi.
3. **Melakukan** preprocessing data: encoding, scaling, handling class imbalance.
4. **Melatih** minimal **4 algoritma klasifikasi**: Decision Tree, KNN, Naive Bayes, dan Logistic Regression.
5. **Mengevaluasi** performa model dengan metrik yang tepat (Accuracy, Precision, Recall, F1-Score, AUC-ROC).
6. **Menginterpretasikan** hasil dan mengidentifikasi fitur paling berpengaruh terhadap defect.
7. **Mengaitkan** hasil ML dengan konteks nyata industri manufaktur.

## 3. DATASET YANG DIGUNAKAN

### 3.1 Dataset Utama

| Informasi | Detail |
| --- | --- |
| **Nama Dataset** | Predicting Manufacturing Defects Dataset |
| **Sumber** | Kaggle |
| **Link** | <https://www.kaggle.com/datasets/rabieelkharoua/predicting-manufacturing-defects-dataset> |
| **Jumlah Data** | 3.240 entri |
| **Jumlah Fitur** | 16 kolom |
| **Tipe Data** | CSV (`manufacturing_defect_dataset.csv`) |
| **Lisensi** | CC BY 4.0 — Free to use, share, and adapt with proper attribution |
| **Sifat Data** | Synthetic — dibuat untuk keperluan edukasi dan penelitian |

**Deskripsi Dataset:**
Dataset ini menyediakan wawasan tentang faktor-faktor yang mempengaruhi tingkat defect dalam lingkungan manufaktur. Setiap record merepresentasikan berbagai metrik krusial untuk memprediksi **High Defects** vs **Low Defects** dalam proses produksi. Dataset mencakup volume produksi, kualitas supply chain, penilaian quality control, jadwal maintenance, manajemen inventori, produktivitas tenaga kerja, konsumsi energi, dan spesifikasi additive manufacturing.

### 3.2 Deskripsi Kolom

| Kolom | Tipe | Deskripsi |
| --- | --- | --- |
| `ProductionVolume` | int64 | Jumlah unit yang diproduksi |
| `ProductionCost` | float64 | Biaya produksi per unit |
| `SupplierQuality` | float64 | Skor kualitas supplier (0–100) |
| `DeliveryDelay` | int64 | Jumlah hari keterlambatan pengiriman |
| `DefectRate` | float64 | Tingkat cacat saat ini |
| `QualityScore` | float64 | Skor kualitas keseluruhan (0–100) |
| `MaintenanceHours` | int64 | Jam maintenance yang dilakukan |
| `DowntimePercentage` | float64 | Persentase downtime mesin |
| `InventoryTurnover` | float64 | Rasio perputaran inventori |
| `StockoutRate` | float64 | Tingkat kehabisan stok |
| `WorkerProductivity` | float64 | Produktivitas pekerja |
| `SafetyIncidents` | int64 | Jumlah insiden keselamatan |
| `EnergyConsumption` | float64 | Konsumsi energi (kWh) |
| `EnergyEfficiency` | float64 | Efisiensi energi |
| `AdditiveProcessTime` | float64 | Waktu proses additive manufacturing |
| `AdditiveMaterialCost` | float64 | Biaya material additive manufacturing |
| **`DefectStatus`** | **int64** | **Target:** 1 = High Defects, 0 = Low Defects |

**Catatan Penting:**

- Dataset ini **imbalanced**: sekitar **84,04%** High Defects dan **15,96%** Low Defects. Strategi penanganan class imbalance perlu diterapkan.
- Beberapa fitur seperti `DefectRate`, `QualityScore`, dan `MaintenanceHours` terbukti menjadi **faktor paling dominan** dalam menentukan status defect berdasarkan analisis SHAP.

### 3.3 Mengapa Dataset Ini Relevan?

1. **Konteks Industri:** Dataset ini mencerminkan permasalahan nyata di lini produksi manufaktur.
2. **Supervised Learning:** Target `DefectStatus` bersifat kategorikal (High/Low Defects), cocok untuk klasifikasi.
3. **Ukuran Memadai:** 3.240 record cukup untuk latihan tanpa terlalu berat secara komputasi.
4. **Multi-Fitur:** 16 fitur mencakup aspek produksi, supply chain, quality control, dan energi.
5. **Challenge Realistis:** Class imbalance dan multikolinearitas membuat praktikum lebih menantang dan mendidik.

## 4. LANDASAN TEORI SINGKAT

### 4.1 Supervised Learning: Klasifikasi

**Supervised Learning** adalah paradigma ML di mana model belajar dari **data berlabel** $(x, y)$. Untuk **klasifikasi**, $y$ adalah kategori diskrit. Dalam studi kasus ini, $y$ adalah `DefectStatus` (0 atau 1).

### 4.2 Alur Kerja ML Pipeline

```
Load Data → EDA → Preprocessing → Split → Training → Testing → Evaluasi → Kesimpulan
```

### 4.3 Algoritma yang Digunakan

| Algoritma | Cara Kerja Singkat | Kelebihan | Kekurangan |
| --- | --- | --- | --- |
| **Decision Tree** | Memecah data berdasarkan threshold fitur | Mudah dipahami, interpretable | Mudah overfitting |
| **KNN** | Voting dari K tetangga terdekat | Sederhana | Sensitif scaling, lambat |
| **Naive Bayes** | Teorema Bayes + asumsi independensi | Cepat, baik untuk dataset kecil | Asumsi independensi sering salah |
| **Logistic Regression** | Fungsi sigmoid untuk probabilitas | Probabilistik, interpretable | Hanya untuk hubungan linear |

### 4.4 Metrik Evaluasi untuk Data Imbalanced

Karena dataset memiliki **class imbalance** (84% vs 16%), **accuracy saja tidak cukup**. Kita perlu:

- **Precision:** Dari yang diprediksi High Defect, berapa persen yang benar?
- **Recall:** Dari semua High Defect aktual, berapa persen yang berhasil diprediksi?
- **F1-Score:** Rata-rata harmonik Precision dan Recall.
- **AUC-ROC:** Area Under ROC Curve — mengukur kemampuan model membedakan kelas.
- **Confusion Matrix:** Tabel TP, TN, FP, FN.

**Dalam konteks manufaktur:**

- **False Negative (FN):** Defect lolos ke pasar → **risiko besar**.
- **False Positive (FP):** Produk baik dinyatakan defect → **pemborosan**.
- **FN biasanya lebih berbahaya**, sehingga **Recall** menjadi metrik prioritas.

## 5. STEP-BY-STEP PRAKTIKUM

### 5.1 Setup Environment dan Import Library

Buka **Anaconda Prompt**, aktifkan environment, dan jalankan Jupyter Notebook:

```bash
conda activate ml-beasiswa
jupyter notebook
```

Buat notebook baru dengan nama `prediksi_defect_manufaktur.ipynb`.

**Cell 1: Import Library**

```python
# Import library dasar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import library scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score
)

# Setting tampilan
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')
%matplotlib inline

print("Semua library berhasil diimport!")
```

**Penjelasan:**

- `pandas` dan `numpy` untuk manipulasi data.
- `matplotlib` dan `seaborn` untuk visualisasi.
- `scikit-learn` untuk preprocessing, modeling, dan evaluasi.
- Kita akan menggunakan **4 algoritma klasifikasi** dasar dan **1 ensemble** (Random Forest) untuk perbandingan.

### 5.2 Load Dataset

**Cell 2: Load Dataset**

```python
# Load dataset
df = pd.read_csv('manufacturing_defect_dataset.csv')

# Tampilkan informasi awal
print("Ukuran dataset:", df.shape)
print("\n5 Baris Pertama:")
df.head()
```

**Penjelasan:**

- `pd.read_csv()` membaca file CSV menjadi DataFrame.
- `df.shape` menampilkan jumlah baris dan kolom.
- `df.head()` menampilkan 5 baris pertama.

**Output yang diharapkan:** Tabel dengan 3.240 baris dan 16 kolom.

### 5.3 Eksplorasi Data (EDA)

**Cell 3: Informasi Umum Dataset**

```python
# Informasi tipe data dan missing value
print("Informasi Dataset:")
df.info()

# Statistik deskriptif
print("\nStatistik Deskriptif:")
df.describe()
```

**Penjelasan:**

- `df.info()` menampilkan tipe data dan jumlah nilai non-null per kolom.
- `df.describe()` memberikan statistik ringkasan (mean, std, min, max, quartile) untuk kolom numerik.

**Cell 4: Cek Missing Value dan Duplikat**

```python
# Cek missing value
print("Jumlah Missing Value per Kolom:")
missing = df.isnull().sum()
print(missing[missing > 0] if any(missing > 0) else "Tidak ada missing value!")

# Cek duplikat
print(f"\nJumlah Data Duplikat: {df.duplicated().sum()}")
```

**Penjelasan:**

- `df.isnull().sum()` menghitung berapa banyak nilai kosong di setiap kolom.
- `df.duplicated().sum()` menghitung baris yang persis sama (duplikat).

**Cell 5: Distribusi Target (Class Balance)**

```python
# Distribusi target
print("Distribusi Target (DefectStatus):")
print(df['DefectStatus'].value_counts())
print(f"\nPersentase:")
print(df['DefectStatus'].value_counts(normalize=True) * 100)

# Visualisasi
plt.figure(figsize=(6, 4))
sns.countplot(x='DefectStatus', data=df, palette='Set2')
plt.title('Distribusi Target DefectStatus')
plt.xlabel('Defect Status (0=Low, 1=High)')
plt.ylabel('Jumlah')
for i, v in enumerate(df['DefectStatus'].value_counts().sort_index()):
    plt.text(i, v + 50, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.show()
```

**Penjelasan:**

- Dataset ini **imbalanced**: sekitar 84% High Defects dan 16% Low Defects.
- Ini akan mempengaruhi pemilihan metrik evaluasi dan strategi modeling.
- Kita perlu mempertimbangkan **SMOTE** atau **class_weight** untuk menangani imbalance.

**Cell 6: Visualisasi Distribusi Fitur Numerik**

```python
# Pilih fitur numerik (semua kecuali target)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('DefectStatus')

# Histogram distribusi
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], color='steelblue')
    axes[i].set_title(f'Distribusi {col}', fontsize=10)
    axes[i].set_xlabel('')

# Hapus subplot kosong
for j in range(len(numeric_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle('Distribusi Fitur Numerik', fontsize=14, y=1.02)
plt.show()
```

**Penjelasan:**

- Setiap histogram menunjukkan sebaran nilai untuk fitur numerik.
- Perhatikan fitur dengan distribusi normal, miring, atau bimodal.
- Fitur seperti `ProductionVolume`, `ProductionCost`, dan `EnergyConsumption` mungkin memiliki outlier.

**Cell 7: Heatmap Korelasi**

```python
# Heatmap korelasi fitur numerik
plt.figure(figsize=(14, 12))
correlation = df[numeric_cols + ['DefectStatus']].corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, mask=mask, annot=True, cmap='coolwarm',
            fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
plt.title('Matriks Korelasi Fitur Numerik', fontsize=14)
plt.tight_layout()
plt.show()
```

**Penjelasan:**

- Heatmap menunjukkan korelasi antar fitur dan dengan target.
- Perhatikan fitur yang berkorelasi tinggi dengan `DefectStatus`.
- Perhatikan juga **multikolinearitas** antar fitur (misal `EnergyConsumption` vs `EnergyEfficiency`).

**Cell 8: Analisis Fitur vs Target**

```python
# Boxplot fitur vs target
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.boxplot(x='DefectStatus', y=col, data=df, ax=axes[i], palette='Set2')
    axes[i].set_title(f'{col} vs DefectStatus', fontsize=10)
    axes[i].set_xlabel('Defect Status')

for j in range(len(numeric_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle('Perbandingan Fitur berdasarkan Defect Status', fontsize=14, y=1.02)
plt.show()
```

**Penjelasan:**

- Boxplot menunjukkan perbedaan distribusi setiap fitur antara High dan Low Defects.
- Fitur dengan perbedaan signifikan adalah **kandidat prediktor kuat**.
- Berdasarkan penelitian, `MaintenanceHours`, `DefectRate`, dan `QualityScore` adalah fitur paling dominan.

### 5.4 Preprocessing Data

**Cell 9: Feature Engineering**

```python
# Fitur baru: total cost
df['Total_Cost'] = df['ProductionVolume'] * df['ProductionCost']

# Fitur baru: efficiency ratio
df['Efficiency_Ratio'] = df['WorkerProductivity'] / (df['EnergyConsumption'] + 1)

# Fitur baru: quality index
df['Quality_Index'] = df['QualityScore'] / (df['DefectRate'] + 1)

# Fitur baru: maintenance intensity
df['Maintenance_Intensity'] = df['MaintenanceHours'] / (df['DowntimePercentage'] + 1)

print("Fitur baru yang dibuat:")
print(df[['Total_Cost', 'Efficiency_Ratio', 'Quality_Index', 'Maintenance_Intensity']].describe())
```

**Penjelasan:**

- **Total_Cost:** Total biaya produksi keseluruhan.
- **Efficiency_Ratio:** Rasio produktivitas terhadap konsumsi energi.
- **Quality_Index:** Indeks kualitas yang mempertimbangkan defect rate.
- **Maintenance_Intensity:** Intensitas maintenance relatif terhadap downtime.

**Cell 10: Hapus Kolom Tidak Diperlukan**

```python
# Cek kolom yang tersedia
print("Kolom setelah feature engineering:")
print(df.columns.tolist())

# Hapus kolom jika ada identifier unik (tidak ada dalam dataset ini)
# df_clean = df.drop(['ID'], axis=1)  # Jika ada

df_clean = df.copy()
print(f"\nShape df_clean: {df_clean.shape}")
```

**Cell 11: Pisahkan Fitur (X) dan Target (y)**

```python
# Tentukan fitur dan target
feature_cols = [col for col in df_clean.columns if col != 'DefectStatus']
X = df_clean[feature_cols]
y = df_clean['DefectStatus']

print("Shape X:", X.shape)
print("Shape y:", y.shape)
print(f"\nJumlah fitur: {len(feature_cols)}")
print("\nDistribusi Target:")
print(y.value_counts())
```

**Penjelasan:**

- `X` adalah matriks fitur (input).
- `y` adalah vektor target (output).
- `X.shape` menghasilkan (jumlah_baris, jumlah_fitur).

**Cell 12: Split Data Train dan Test**

```python
# Bagi data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Jumlah data training:", X_train.shape[0])
print("Jumlah data testing:", X_test.shape[0])
print("\nDistribusi target di training:")
print(y_train.value_counts(normalize=True))
print("\nDistribusi target di testing:")
print(y_test.value_counts(normalize=True))
```

**Penjelasan:**

- `test_size=0.2` berarti 20% data untuk testing.
- `random_state=42` memastikan hasil reproducible.
- `stratify=y` memastikan proporsi kelas sama di training dan testing.

**Cell 13: Feature Scaling**

```python
# Standardisasi fitur numerik
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Mean training setelah scaling (harus mendekati 0):")
print(X_train_scaled.mean(axis=0).round(4)[:5], "...")
print("\nStd training setelah scaling (harus mendekati 1):")
print(X_train_scaled.std(axis=0).round(4)[:5], "...")
```

**Penjelasan:**

- **StandardScaler** mengubah data sehingga memiliki mean = 0 dan std = 1.
- **PENTING:** `fit_transform` hanya pada training data. Testing data hanya di-`transform`.

### 5.5 Modeling: Melatih 5 Algoritma Klasifikasi

**Cell 14: Inisialisasi dan Training Model**

```python
# Inisialisasi model
models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
}

# Dictionary untuk menyimpan hasil
results = {}

# Latih dan evaluasi setiap model
for name, model in models.items():
    # Latih model
    model.fit(X_train_scaled, y_train)

    # Prediksi
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Hitung metrik
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_test_proba)

    # Simpan hasil
    results[name] = {
        'model': model,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba
    }

    print(f"{'='*60}")
    print(f"Model: {name}")
    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Test Accuracy  : {test_acc:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1-Score       : {f1:.4f}")
    print(f"AUC-ROC        : {auc:.4f}")
    print()
```

**Penjelasan Setiap Algoritma:**

| Algoritma | Cara Kerja | Kelebihan | Kekurangan |
| --- | --- | --- | --- |
| **Decision Tree** | Memecah data berdasarkan threshold fitur | Mudah dipahami, interpretable | Mudah overfitting |
| **KNN** | Voting dari K tetangga terdekat | Sederhana, tidak ada training | Sensitif scaling, lambat |
| **Naive Bayes** | Teorema Bayes + asumsi independensi | Cepat, baik untuk dataset kecil | Asumsi independensi sering salah |
| **Logistic Regression** | Fungsi sigmoid untuk probabilitas | Probabilistik, interpretable | Hanya untuk hubungan linear |
| **Random Forest** | Ensemble dari banyak Decision Tree | Akurasi tinggi, robust | Kurang interpretable |

### 5.6 Evaluasi Model

**Cell 15: Tabel Perbandingan Metrik**

```python
# Buat DataFrame perbandingan
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train Acc': [results[m]['train_accuracy'] for m in results],
    'Test Acc': [results[m]['test_accuracy'] for m in results],
    'Precision': [results[m]['precision'] for m in results],
    'Recall': [results[m]['recall'] for m in results],
    'F1-Score': [results[m]['f1_score'] for m in results],
    'AUC-ROC': [results[m]['auc_roc'] for m in results]
}).round(4)

# Hitung selisih (indikasi overfitting)
comparison_df['Overfit Gap'] = (
    comparison_df['Train Acc'] - comparison_df['Test Acc']
).round(4)

comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
print("Perbandingan Model:")
print(comparison_df.to_string(index=False))
```

**Cara Membaca Tabel:**

- **Test Acc** tinggi menunjukkan model baik.
- **Overfit Gap** kecil menunjukkan model tidak overfitting.
- **F1-Score** penting untuk data imbalanced.
- **AUC-ROC** mengukur kemampuan diskriminasi model.

**Cell 16: Visualisasi Perbandingan**

```python
# Bar chart perbandingan
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(comparison_df))
width = 0.15

metrics = ['Train Acc', 'Test Acc', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

for i, (metric, color) in enumerate(zip(metrics, colors)):
    bars = ax.bar(x + i*width - 2.5*width, comparison_df[metric],
                  width, label=metric, color=color)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', fontsize=7, rotation=90)

ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('Perbandingan Metrik Evaluasi Model', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(comparison_df['Model'], rotation=15)
ax.legend(loc='lower right')
ax.set_ylim(0, 1.15)
plt.tight_layout()
plt.show()
```

**Cell 17: Confusion Matrix untuk Model Terbaik**

```python
# Ambil model terbaik berdasarkan F1-Score
best_model_name = comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
best_y_pred = results[best_model_name]['y_test_pred']

print(f"Model Terbaik: {best_model_name}\n")

# Confusion Matrix
cm = confusion_matrix(y_test, best_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Low Defect', 'High Defect'])
disp.plot(cmap='Blues')
plt.title(f'Confusion Matrix — {best_model_name}')
plt.show()

# Classification Report
print(f"Classification Report — {best_model_name}:\n")
print(classification_report(y_test, best_y_pred,
                            target_names=['Low Defect', 'High Defect']))
```

**Penjelasan Confusion Matrix:**

| | Prediksi: Low | Prediksi: High |
| --- | --- | --- |
| **Aktual: Low** | True Negative (TN) | False Positive (FP) |
| **Aktual: High** | False Negative (FN) | True Positive (TP) |

- **TN:** Benar diprediksi Low Defect.
- **TP:** Benar diprediksi High Defect.
- **FP:** Salah diprediksi High Defect (padahal Low).
- **FN:** Salah diprediksi Low Defect (padahal High) — **ini yang berbahaya!**

**Dalam konteks manufaktur:**

- **FN:** Defect lolos ke pasar → risiko besar.
- **FP:** Produk baik dinyatakan defect → pemborosan.
- **FN lebih berbahaya**, sehingga **Recall** menjadi metrik prioritas.

**Cell 18: ROC Curve**

```python
# ROC Curve untuk semua model
plt.figure(figsize=(10, 8))

for name in results:
    y_proba = results[name]['y_test_proba']
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = results[name]['auc_roc']
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)
plt.title('ROC Curve — Perbandingan Model', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Penjelasan ROC Curve:**

- **Sumbu X:** False Positive Rate (1 - Specificity).
- **Sumbu Y:** True Positive Rate (Recall).
- **Semakin dekat ke kiri atas**, semakin baik model.
- **AUC = 0.5** berarti model tidak lebih baik dari random.
- **AUC > 0.8** berarti model baik.

**Cell 19: Feature Importance**

```python
# Feature importance dari Random Forest
rf_model = results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'Fitur': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Feature Importance (Random Forest):")
print(feature_importance.to_string(index=False))

# Visualisasi
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Fitur', data=feature_importance.head(15),
            palette='viridis')
plt.title('Top 15 Fitur Paling Berpengaruh (Random Forest)', fontsize=14)
plt.xlabel('Importance', fontsize=12)
plt.tight_layout()
plt.show()
```

**Penjelasan:**

- Feature importance menunjukkan seberapa besar kontribusi setiap fitur terhadap prediksi.
- Berdasarkan penelitian, `MaintenanceHours`, `DefectRate`, dan `QualityScore` adalah fitur paling dominan.
- Fitur dengan importance tinggi adalah **prioritas untuk monitoring** di lini produksi.

### 5.7 Penanganan Class Imbalance (Bonus)

**Cell 20: Eksperimen dengan Class Weight**

```python
# Latih ulang model dengan class_weight='balanced'
models_balanced = {
    'Decision Tree (Balanced)': DecisionTreeClassifier(
        max_depth=5, class_weight='balanced', random_state=42),
    'Logistic Regression (Balanced)': LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42),
    'Random Forest (Balanced)': RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
}

print("Perbandingan dengan Class Weight Balancing:")
print("=" * 80)

for name, model in models_balanced.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n{name}:")
    print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | "
          f"Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
```

**Penjelasan:**

- `class_weight='balanced'` secara otomatis menyesuaikan bobot kelas berdasarkan frekuensinya.
- Ini membantu model lebih memperhatikan kelas minoritas (Low Defect).
- Bandingkan hasilnya dengan model tanpa balancing.

**Cell 21: Cross-Validation**

```python
# 5-Fold Stratified Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("5-Fold Cross-Validation (F1-Score):")
print("=" * 60)

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train,
                             cv=cv, scoring='f1')
    print(f"{name}: F1 = {scores.mean():.4f} ± {scores.std():.4f}")
```

**Penjelasan:**

- **Cross-validation** memberikan estimasi performa yang lebih robust.
- **Stratified** memastikan proporsi kelas terjaga di setiap fold.
- **F1-Score** dipilih karena data imbalanced.

### 5.8 Kesimpulan Praktikum

**Cell 22: Ringkasan Hasil**

```python
print("=" * 70)
print("RINGKASAN HASIL PRAKTIKUM — PREDIKSI DEFECT PRODUK MANUFAKTUR")
print("=" * 70)

print(f"""
1. DATASET
   - Sumber: Kaggle — Predicting Manufacturing Defects Dataset
   - Jumlah data: {len(df)} record
   - Jumlah fitur: {len(feature_cols)} fitur
   - Target: DefectStatus (0=Low, 1=High)
   - Class imbalance: {y.value_counts(normalize=True).round(4).to_dict()}

2. PREPROCESSING
   - Missing value: {df.isnull().sum().sum()}
   - Duplikat: {df.duplicated().sum()}
   - Fitur baru dibuat: Total_Cost, Efficiency_Ratio, Quality_Index, Maintenance_Intensity
   - Scaling: StandardScaler
   - Split: 80% train, 20% test (stratified)

3. HASIL PERBANDINGAN MODEL (Test Set)
""")

for _, row in comparison_df.iterrows():
    print(f"   {row['Model']:30s} | F1={row['F1-Score']:.4f} | "
          f"AUC={row['AUC-ROC']:.4f} | Recall={row['Recall']:.4f}")

print(f"""
4. MODEL TERBAIK: {best_model_name}
   - F1-Score: {results[best_model_name]['f1_score']:.4f}
   - AUC-ROC: {results[best_model_name]['auc_roc']:.4f}
   - Recall: {results[best_model_name]['recall']:.4f}

5. FITUR PALING BERPENGARUH (Top 5):
""")

for _, row in feature_importance.head(5).iterrows():
    print(f"   - {row['Fitur']}: {row['Importance']:.4f}")

print(f"""
6. REKOMENDASI
   - Model {best_model_name} direkomendasikan untuk deployment awal.
   - Prioritaskan monitoring pada fitur: {feature_importance.iloc[0]['Fitur']},
     {feature_importance.iloc[1]['Fitur']}, {feature_importance.iloc[2]['Fitur']}.
   - Pertimbangkan SMOTE untuk meningkatkan Recall kelas minoritas.
""")
```

## 6. TUGAS MANDIRI (Dikerjakan Hari Ini)

**Tujuan:** Memastikan setiap mahasiswa memahami alur supervised learning untuk klasifikasi defect manufaktur secara hands-on.

**Instruksi:**

1. **Eksperimen Hyperparameter Decision Tree:**
   - Ubah `max_depth` menjadi 3, 5, 7, 10, dan 15.
   - Catat F1-Score dan AUC-ROC untuk setiap nilai.
   - Buat tabel dan grafik.
   - Jelaskan nilai mana yang optimal dan mengapa.

2. **Eksperimen Nilai K pada KNN:**
   - Ubah `n_neighbors` menjadi 3, 5, 7, 9, 11.
   - Catat F1-Score dan Recall untuk setiap nilai K.
   - Buat tabel dan grafik.
   - Jelaskan nilai K optimal.

3. **Visualisasi Tambahan:**
   - Buat boxplot `DefectRate` berdasarkan `DefectStatus`.
   - Buat scatter plot antara `ProductionVolume` dan `DefectRate` dengan warna berdasarkan `DefectStatus`.
   - Interpretasikan hasilnya.

4. **Pertanyaan Konsep (Jawab di Markdown Cell):**
   - Mengapa accuracy tidak cukup untuk data imbalanced? Berikan contoh.
   - Apa perbedaan Precision dan Recall? Dalam konteks defect produk, mana yang lebih penting?
   - Jelaskan mengapa `stratify=y` penting pada `train_test_split`.
   - Apa itu overfitting? Bagaimana mendeteksinya dari tabel perbandingan?
   - Mengapa Random Forest sering lebih baik dari Decision Tree tunggal?

5. **Refleksi Pribadi (Minimal 200 kata):**
   - Apa yang Anda pelajari dari praktikum ini?
   - Bagian mana yang paling sulit?
   - Bagaimana Anda akan menerapkan ML di industri manufaktur?

**Pengumpulan:** Simpan notebook dengan nama `TugasMandiri1_NamaAnda_NIM.ipynb` dan kumpulkan di Google Drive sebelum pertemuan ke-4.

## 7. TUGAS KELOMPOK (Dikerjakan di Rumah)

**Tujuan:** Memperdalam pemahaman supervised learning melalui eksplorasi dataset alternatif dan perbandingan algoritma.

**Pembagian Kelompok:** 4–5 mahasiswa per kelompok.

### Bagian A: Eksplorasi Dataset Alternatif (Bobot 25%)

Pilih **salah satu** dataset berikut dari Kaggle/UCI:

1. **AI4I 2020 Predictive Maintenance Dataset** — <https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020>
   - 10.000 record, 14 fitur
   - Prediksi kegagalan mesin (binary classification)
   - Fitur: air temperature, process temperature, rotational speed, torque, tool wear

2. **Steel Plates Faults Dataset** — <https://www.kaggle.com/datasets/uciml/faulty-steel-plates>
   - 1.941 record, 27 fitur
   - Klasifikasi 7 jenis defect pelat baja
   - Fitur: geometric dan luminosity features

3. **Tool Wear Detection in CNC Mill** — <https://www.kaggle.com/datasets/shasun/tool-wear-detection-in-cnc-mill>
   - Data sensor dari CNC milling machine
   - Klasifikasi kondisi tool (worn/unworn)
   - Fitur: feed rate, clamp pressure, sensor readings

**Tugas:**

- Lakukan EDA singkat pada dataset yang dipilih.
- Lakukan preprocessing (handle missing value, encoding, scaling).
- Latih minimal **3 algoritma klasifikasi**.
- Bandingkan performa model.
- Interpretasikan hasil.

### Bagian B: Perbandingan Algoritma (Bobot 35%)

1. Latih **5 algoritma** pada dataset alternatif:
   - Decision Tree
   - KNN
   - Naive Bayes
   - Logistic Regression
   - Random Forest

2. Bandingkan hasilnya dalam satu tabel:

| Model | Train Acc | Test Acc | Precision | Recall | F1-Score | AUC-ROC |
| --- | --- | --- | --- | --- | --- | --- |
| Decision Tree | ... | ... | ... | ... | ... | ... |
| KNN | ... | ... | ... | ... | ... | ... |
| Naive Bayes | ... | ... | ... | ... | ... | ... |
| Logistic Regression | ... | ... | ... | ... | ... | ... |
| Random Forest | ... | ... | ... | ... | ... | ... |

1. **Analisis:** Model mana yang terbaik? Mengapa? Apakah ada yang overfitting?

### Bagian C: Feature Importance (Bobot 20%)

1. Analisis feature importance dari model terbaik.
2. Identifikasi **5 fitur paling berpengaruh**.
3. Jelaskan mengapa fitur-fitur tersebut penting dalam konteks manufaktur.
4. Buat rekomendasi untuk monitoring di lini produksi.

### Bagian D: Rekomendasi dan Presentasi (Bobot 20%)

1. Berdasarkan hasil analisis, buat **rekomendasi** untuk perusahaan manufaktur:
   - Bagaimana model ML dapat digunakan untuk meningkatkan kualitas produk?
   - Fitur apa yang harus diprioritaskan untuk monitoring?
   - Apa keterbatasan analisis ini?

2. Buat **slide presentasi** (6–8 slide):
   - Judul dan anggota kelompok
   - Dataset yang digunakan
   - Metodologi
   - Hasil perbandingan
   - Feature importance
   - Rekomendasi

**Pengumpulan:** Kumpulkan notebook (`TugasKelompok1_NamaKelompok.ipynb`) dan slide dalam satu folder ZIP di Google Drive sebelum pertemuan ke-4.

## 8. RUBRIK PENILAIAN

### 8.1 Rubrik Tugas Mandiri (Bobot 20%)

| Komponen | Bobot | Kriteria | Skor |
| --- | --- | --- | --- |
| **Eksperimen Decision Tree** | 20% | 5 nilai max_depth diuji, analisis tajam | 0–100 |
| **Eksperimen KNN** | 20% | 5 nilai K diuji, plot lengkap | 0–100 |
| **Visualisasi Tambahan** | 15% | 2 visualisasi informatif, interpretasi benar | 0–100 |
| **Pertanyaan Konsep** | 25% | Jawaban tepat, ada penjelasan | 0–100 |
| **Refleksi Pribadi** | 20% | Refleksi mendalam, autentik | 0–100 |

### 8.2 Rubrik Tugas Kelompok (Bobot 30%)

| Komponen | Bobot | Kriteria | Skor |
| --- | --- | --- | --- |
| **Bagian A: EDA Dataset Alternatif** | 25% | EDA mendalam, visualisasi informatif | 0–100 |
| **Bagian B: Perbandingan 5 Algoritma** | 35% | Tabel lengkap, analisis tajam | 0–100 |
| **Bagian C: Feature Importance** | 20% | Analisis tepat, rekomendasi logis | 0–100 |
| **Bagian D: Rekomendasi & Presentasi** | 20% | Rekomendasi relevan, slide menarik | 0–100 |

### 8.3 Konversi Nilai

| Nilai Angka | Huruf | Keterangan |
| --- | --- | --- |
| 85–100 | A | Sangat Baik |
| 75–84 | B | Baik |
| 65–74 | C | Cukup |
| 55–64 | D | Kurang |
| < 55 | E | Sangat Kurang |

## 9. REFERENSI

### 9.1 Dataset

1. El Kharoua, R. (2023). *Predicting Manufacturing Defects Dataset*. Kaggle. <https://www.kaggle.com/datasets/rabieelkharoua/predicting-manufacturing-defects-dataset>
2. Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset*. UCI Machine Learning Repository. <https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset>
3. *Steel Plates Faults Dataset*. UCI Machine Learning Repository. <https://archive.ics.uci.edu/dataset/198/steel+plates+faults>
4. *Tool Wear Detection in CNC Mill*. Kaggle. <https://www.kaggle.com/datasets/shasun/tool-wear-detection-in-cnc-mill>

### 9.2 Buku dan Artikel

1. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.
2. Raschka, S., & Mirjalili, V. (2019). *Python Machine Learning* (3rd ed.). Packt.
3. Studi tentang Ensemble Learning untuk prediksi defect manufaktur. *Garuda Kemdiktisaintek* (2026).
4. Matzka, S. (2020). Explainable Artificial Intelligence for Predictive Maintenance Applications. *Semantic Scholar*.

### 9.3 Dokumentasi Online

1. **Scikit-learn Classification:** <https://scikit-learn.org/stable/modules/classification.html>
2. **Pandas:** <https://pandas.pydata.org/docs/>
3. **Seaborn:** <https://seaborn.pydata.org/>

### 9.4 Video Tutorial

1. StatQuest — Decision Trees: <https://www.youtube.com/watch?v=7VeUPuFGJHk>
2. StatQuest — Random Forests: <https://www.youtube.com/watch?v=J4Wdy0Wc_xQ>
3. StatQuest — Logistic Regression: <https://www.youtube.com/watch?v=yIYKR4sgzI8>

## PENUTUP

Praktikum ini memberikan pengalaman langsung dalam menerapkan **supervised learning** untuk **prediksi defect produk** di industri manufaktur. Berbeda dengan konteks beasiswa, studi kasus ini lebih dekat dengan **dunia industri** yang menjadi fokus Polman Bandung.

**Poin-poin kunci:**

1. **Supervised learning** membutuhkan data berlabel untuk klasifikasi.
2. **EDA** membantu memahami karakteristik data dan mendeteksi masalah.
3. **Preprocessing** (scaling, handling imbalance) sangat penting untuk performa model.
4. **Perbandingan algoritma** membantu memilih model terbaik.
5. **Metrik evaluasi** harus disesuaikan dengan konteks bisnis (Recall untuk defect).
6. **Feature importance** memberikan insight untuk monitoring lini produksi.

**Pesan untuk Mahasiswa:**
> "Dalam industri manufaktur, setiap defect yang lolos adalah kerugian. Machine Learning memberi kita kemampuan untuk **memprediksi dan mencegah** defect sebelum terjadi. Kuasai tools ini, karena industri 4.0 membutuhkan talenta yang mampu mengubah data menjadi keputusan."

**Pesan untuk Dosen:**
> "Studi kasus ini dapat disesuaikan dengan dataset yang lebih besar atau lebih kecil. Untuk kelas dengan waktu terbatas, fokus pada Decision Tree, Logistic Regression, dan Random Forest. Untuk kelas yang lebih mahir, tambahkan SMOTE, hyperparameter tuning, dan eksperimen dengan dataset AI4I 2020."

---

**Dokumen ini disusun untuk keperluan edukasi di Politeknik Manufaktur Bandung (Polman Bandung).**
**Versi:** 1.0 | **Minggu ke-2, Pertemuan ke-1 dari 4**
**Terakhir Diperbarui:** 2026
