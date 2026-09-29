# STUDI KASUS IMPLEMENTASI TERPADU — PROYEK CAPSTONE
## Smart Manufacturing AI: Aplikasi Web Terintegrasi Supervised, Unsupervised, dan Reinforcement Learning

**Mata Kuliah:** Machine Learning
**Program Studi:** D4 Teknologi Rekayasa Informatika dan Komputer (TRIN)
**Institusi:** Politeknik Manufaktur Bandung (Polman Bandung)
**Pertemuan:** Minggu ke-2, Integrasi Pertemuan 1–3
**Topik:** Proyek Capstone — Implementasi Web Terintegrasi
**Tools:** Anaconda, Python 3.11, Flask, HTML/CSS/JS, Joblib, Bootstrap
**Durasi:** 150 menit (demo + praktik) + Tugas Proyek Akhir


## DAFTAR ISI

1. [Latar Belakang Proyek](#1-latar-belakang-proyek)
2. [Tujuan Pembelajaran](#2-tujuan-pembelajaran)
3. [Arsitektur Sistem](#3-arsitektur-sistem)
4. [Struktur Folder Proyek](#4-struktur-folder-proyek)
5. [Fase 1: Setup Environment](#5-fase-1-setup-environment)
6. [Fase 2: Training dan Export Model](#6-fase-2-training-dan-export-model)
7. [Fase 3: Backend Flask](#7-fase-3-backend-flask)
8. [Fase 4: Frontend Web](#8-fase-4-frontend-web)
9. [Fase 5: Integrasi dan Testing](#9-fase-5-integrasi-dan-testing)
10. [Fase 6: Deployment](#10-fase-6-deployment)
11. [Tugas Proyek Akhir](#11-tugas-proyek-akhir)
12. [Rubrik Penilaian](#12-rubrik-penilaian)
13. [Referensi](#13-referensi)


## 1. LATAR BELAKANG PROYEK

### 1.1 Konteks Integrasi

Selama tiga pertemuan di minggu ke-2, kita telah mempelajari tiga paradigma Machine Learning secara terpisah:

| Pertemuan | Topik | Dataset | Output |
|---|---|---|---|
| **Pertemuan 1** | Supervised Learning | Manufacturing Defects | Klasifikasi Defect |
| **Pertemuan 2** | Unsupervised Learning | Semiconductor Wafer | Clustering State |
| **Pertemuan 3** | Reinforcement Learning | Batch Reactor Anomaly | Kontrol Optimal |

Namun, dalam dunia industri nyata, **ketiga paradigma ini tidak berdiri sendiri** — mereka saling melengkapi dalam satu **sistem cerdas terintegrasi**. Proyek capstone ini bertujuan untuk **mengintegrasikan ketiga model** ke dalam satu **aplikasi web** yang dapat digunakan oleh operator, engineer, dan manajer produksi.

### 1.2 Skenario Industri

**PT. Nusantara Precision Manufacturing** adalah perusahaan manufaktur komponen otomotif presisi tinggi. Perusahaan menghadapi tiga tantangan utama:

1. **Tingkat defect produk yang tinggi** — perlu prediksi dini untuk mencegah defect lolos ke pelanggan.
2. **Monitoring kondisi mesin secara manual** — perlu segmentasi otomatis state operasional mesin.
3. **Kontrol proses reaktor batch yang belum optimal** — perlu sistem kontrol cerdas adaptif.

Untuk mengatasi ketiga tantangan ini, perusahaan mengembangkan **Smart Manufacturing AI** — sebuah aplikasi web yang mengintegrasikan:

- **Modul Prediksi Defect** (Supervised Learning) — memprediksi probabilitas defect produk berdasarkan parameter produksi.
- **Modul Segmentasi State** (Unsupervised Learning) — mengelompokkan kondisi operasional mesin ke dalam state bermakna.
- **Modul Kontrol Reaktor** (Reinforcement Learning) — merekomendasikan action optimal untuk kontrol suhu reaktor batch.

### 1.3 Nilai Bisnis

| Modul | Nilai Bisnis |
|---|---|
| **Prediksi Defect** | Mengurangi defect rate hingga 40%, menghemat biaya produksi |
| **Segmentasi State** | Monitoring 24/7 tanpa operator, deteksi anomali dini |
| **Kontrol Reaktor** | Meningkatkan yield 15%, mencegah overheating |

### 1.4 Mengapa Web?

Aplikasi web dipilih karena:

1. **Aksesibilitas:** Dapat diakses dari mana saja melalui browser.
2. **Cross-platform:** Berjalan di Windows, Mac, Linux, bahkan tablet.
3. **Kolaborasi:** Operator, engineer, dan manajer dapat mengakses sistem yang sama.
4. **Skalabilitas:** Mudah dikembangkan dan diintegrasikan dengan sistem lain (MES, ERP).
5. **Relevan dengan Industri 4.0:** Web-based monitoring adalah standar di smart factory.


## 2. TUJUAN PEMBELAJARAN

Setelah menyelesaikan proyek ini, mahasiswa diharapkan mampu:

1. **Mengintegrasikan** tiga paradigma Machine Learning (Supervised, Unsupervised, Reinforcement) dalam satu sistem.
2. **Mengekspor** model ML yang sudah dilatih ke format yang dapat digunakan di production (joblib/pickle).
3. **Membangun** backend REST API menggunakan Flask untuk melayani prediksi ML.
4. **Membangun** frontend web interaktif menggunakan HTML, CSS, dan JavaScript.
5. **Melakukan** testing end-to-end pada sistem terintegrasi.
6. **Mendeploy** aplikasi web ke environment production (local server).
7. **Mendokumentasikan** proyek secara profesional.


## 3. ARSITEKTUR SISTEM

### 3.1 Arsitektur High-Level

```
┌─────────────────────────────────────────────────────────────┐
│                     SMARTER MANUFACTURING AI                │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   FRONTEND (Browser)                 │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐      │   │
│  │  │ Supervised │  │Unsupervised│  │Reinforcement│     │   │
│  │  │    Page    │  │    Page    │  │    Page     │     │   │
│  │  └─────┬──────┘  └─────┬──────┘  └─────┬───────┘     │   │
│  │        │               │               │             │   │
│  │        └───────────────┼───────────────┘             │   │
│  │                        │                             │   │
│  │                   HTTP Request                       │   │
│  └────────────────────────┼─────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                 BACKEND (Flask API)                  │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  /api/predict-defect    (Supervised)           │  │   │
│  │  │  /api/cluster-state     (Unsupervised)         │  │   │
│  │  │  /api/reactor-control   (Reinforcement)        │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  │                        │                             │   │
│  │                        ▼                             │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │             ML MODELS (Joblib)                 │  │   │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐     │  │   │
│  │  │  │RandomFor.│  │  K-Means │  │ Q-Table  │     │  │   │
│  │  │  │  .pkl    │  │   .pkl   │  │   .pkl   │     │  │   │
│  │  │  └──────────┘  └──────────┘  └──────────┘     │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Alur Data

**Modul Supervised (Prediksi Defect):**
```
User Input (Parameter Produksi)
    ↓
Flask API /api/predict-defect
    ↓
Load Supervised Model (Random Forest)
    ↓
Preprocessing (StandardScaler)
    ↓
Prediction (0 = Low Defect, 1 = High Defect)
    ↓
Response JSON → Frontend Display
```

**Modul Unsupervised (Segmentasi State):**
```
User Input (Sensor Readings)
    ↓
Flask API /api/cluster-state
    ↓
Load Unsupervised Model (K-Means)
    ↓
Preprocessing (StandardScaler)
    ↓
Cluster Prediction (0, 1, 2)
    ↓
Interpretasi State → Response JSON
```

**Modul Reinforcement (Kontrol Reaktor):**
```
User Input (State Reaktor: Suhu, Tekanan, dll)
    ↓
Flask API /api/reactor-control
    ↓
Load RL Model (Q-Table)
    ↓
Discretize State
    ↓
Lookup Q-Table → Best Action
    ↓
Response JSON → Frontend Display
```

### 3.3 Teknologi Stack

| Layer | Teknologi | Fungsi |
|---|---|---|
| **Frontend** | HTML5, CSS3, JavaScript | UI/UX |
| **CSS Framework** | Bootstrap 5 | Responsive design |
| **Visualisasi** | Chart.js | Grafik interaktif |
| **Backend** | Flask (Python) | REST API |
| **ML Library** | scikit-learn, numpy | Model ML |
| **Model Storage** | Joblib | Serialisasi model |
| **Data Processing** | Pandas | Manipulasi data |
| **Web Server** | Werkzeug (dev), Gunicorn (prod) | HTTP server |


## 4. STRUKTUR FOLDER PROYEK

```
smart-manufacturing-ai/
│
├── app.py                              # Flask main application
├── requirements.txt                    # Python dependencies
├── README.md                           # Dokumentasi proyek
├── .gitignore                          # Git ignore file
│
├── models/                             # Trained ML models
│   ├── supervised_defect_model.pkl
│   ├── supervised_scaler.pkl
│   ├── unsupervised_kmeans.pkl
│   ├── unsupervised_scaler.pkl
│   ├── rl_qtable.pkl
│   └── rl_metadata.pkl
│
├── data/                               # Datasets
│   ├── manufacturing_defect_dataset.csv
│   ├── semiconductor_wafer_defect_dataset.csv
│   └── reactor_sample_5k.csv
│
├── training/                           # Training scripts
│   ├── 01_train_supervised.py
│   ├── 02_train_unsupervised.py
│   └── 03_train_reinforcement.py
│
├── static/                             # Static files
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   ├── main.js
│   │   ├── supervised.js
│   │   ├── unsupervised.js
│   │   └── reinforcement.js
│   └── img/
│       ├── logo.png
│       └── favicon.ico
│
├── templates/                          # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── supervised.html
│   ├── unsupervised.html
│   ├── reinforcement.html
│   └── about.html
│
└── tests/                              # Testing
    ├── test_api.py
    └── test_models.py
```


## 5. FASE 1: SETUP ENVIRONMENT

### 5.1 Buat Folder Proyek

Buka **Anaconda Prompt** dan jalankan:

```bash
# Buat folder proyek
mkdir smart-manufacturing-ai
cd smart-manufacturing-ai

# Buat subfolder
mkdir models data training static static\css static\js static\img templates tests

# Aktifkan environment
conda activate ml-beasiswa

# Install dependencies tambahan
pip install flask joblib
```

### 5.2 Buat File `requirements.txt`

```txt
flask==3.0.0
joblib==1.3.2
numpy==1.26.0
pandas==2.1.3
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
gymnasium==0.29.1
```

Install semua dependencies:

```bash
pip install -r requirements.txt
```

### 5.3 Siapkan Dataset

Unduh ketiga dataset dan letakkan di folder `data/`:

1. **manufacturing_defect_dataset.csv** — dari [Kaggle Predicting Manufacturing Defects](https://www.kaggle.com/datasets/rabieelkharoua/predicting-manufacturing-defects-dataset)
2. **semiconductor_wafer_defect_dataset.csv** — dari [Kaggle Semiconductor Wafer Defect](https://www.kaggle.com/datasets/meruvakodandasuraj/semiconductor-wafer-defect-classification-dataset)
3. **reactor_sample_5k.csv** — dari [Kaggle Batch Reactor Anomaly](https://www.kaggle.com/datasets/aimindteams/batch-reactor-anomaly-data-sample)


## 6. FASE 2: TRAINING DAN EXPORT MODEL

### 6.1 Training Script Supervised Learning

Buat file `training/01_train_supervised.py`:

```python
"""
Training Script — Supervised Learning
Prediksi Defect Produk Manufaktur
"""
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

# ============================================================
# 1. LOAD DATASET
# ============================================================
print("=" * 60)
print("TRAINING SUPERVISED MODEL — PREDIKSI DEFECT PRODUK")
print("=" * 60)

df = pd.read_csv('../data/manufacturing_defect_dataset.csv')
print(f"Dataset shape: {df.shape}")

# ============================================================
# 2. PREPROCESSING
# ============================================================
# Feature engineering
df['Total_Cost'] = df['ProductionVolume'] * df['ProductionCost']
df['Efficiency_Ratio'] = df['WorkerProductivity'] / (df['EnergyConsumption'] + 1)
df['Quality_Index'] = df['QualityScore'] / (df['DefectRate'] + 1)
df['Maintenance_Intensity'] = df['MaintenanceHours'] / (df['DowntimePercentage'] + 1)

# Pisahkan fitur dan target
feature_cols = [col for col in df.columns if col != 'DefectStatus']
X = df[feature_cols]
y = df['DefectStatus']

print(f"Jumlah fitur: {len(feature_cols)}")
print(f"Distribusi target: {dict(y.value_counts())}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 3. TRAINING
# ============================================================
print("\nTraining Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# ============================================================
# 4. EVALUATION
# ============================================================
y_pred = model.predict(X_test_scaled)
print(f"\nAccuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ============================================================
# 5. EXPORT MODEL
# ============================================================
os.makedirs('../models', exist_ok=True)

joblib.dump(model, '../models/supervised_defect_model.pkl')
joblib.dump(scaler, '../models/supervised_scaler.pkl')
joblib.dump(feature_cols, '../models/supervised_features.pkl')

print("\n✓ Model berhasil disimpan:")
print("  - models/supervised_defect_model.pkl")
print("  - models/supervised_scaler.pkl")
print("  - models/supervised_features.pkl")
```

### 6.2 Training Script Unsupervised Learning

Buat file `training/02_train_unsupervised.py`:

```python
"""
Training Script — Unsupervised Learning
Segmentasi State Kontrol Produksi
"""
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ============================================================
# 1. LOAD DATASET
# ============================================================
print("=" * 60)
print("TRAINING UNSUPERVISED MODEL — SEGMENTASI STATE")
print("=" * 60)

df = pd.read_csv('../data/semiconductor_wafer_defect_dataset.csv')
print(f"Dataset shape: {df.shape}")

# ============================================================
# 2. PREPROCESSING
# ============================================================
# Hapus kolom tidak relevan
if 'wafer_id' in df.columns:
    df = df.drop('wafer_id', axis=1)
if 'defect_label' in df.columns:
    df = df.drop('defect_label', axis=1)

# Encoding kategorikal
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le
    df = df.drop(col, axis=1)

# Pilih fitur numerik
feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X = df[feature_cols]

print(f"Jumlah fitur: {len(feature_cols)}")

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 3. TENTUKAN K OPTIMAL
# ============================================================
print("\nMenentukan K optimal...")
sil_scores = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))
    print(f"  K={k}: Silhouette = {sil_scores[-1]:.4f}")

best_k = range(2, 8)[np.argmax(sil_scores)]
print(f"\nK optimal: {best_k}")

# ============================================================
# 4. TRAINING FINAL
# ============================================================
print(f"\nTraining K-Means dengan K={best_k}...")
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# ============================================================
# 5. EXPORT MODEL
# ============================================================
os.makedirs('../models', exist_ok=True)

joblib.dump(kmeans, '../models/unsupervised_kmeans.pkl')
joblib.dump(scaler, '../models/unsupervised_scaler.pkl')
joblib.dump(feature_cols, '../models/unsupervised_features.pkl')
joblib.dump(le_dict, '../models/unsupervised_encoders.pkl')
joblib.dump(best_k, '../models/unsupervised_n_clusters.pkl')

# Simpan profil cluster
cluster_profile = pd.DataFrame(X_scaled, columns=feature_cols)
cluster_profile['cluster'] = kmeans.labels_
cluster_means = cluster_profile.groupby('cluster').mean()
cluster_means.to_csv('../models/cluster_profile.csv')

print("\n✓ Model berhasil disimpan:")
print("  - models/unsupervised_kmeans.pkl")
print("  - models/unsupervised_scaler.pkl")
print("  - models/unsupervised_features.pkl")
print("  - models/cluster_profile.csv")
```

### 6.3 Training Script Reinforcement Learning

Buat file `training/03_train_reinforcement.py`:

```python
"""
Training Script — Reinforcement Learning
Kontrol Optimal Reaktor Batch
"""
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# 1. LOAD DATASET
# ============================================================
print("=" * 60)
print("TRAINING RL MODEL — KONTROL REAKTOR BATCH")
print("=" * 60)

df = pd.read_csv('../data/reactor_sample_5k.csv')
print(f"Dataset shape: {df.shape}")
print(f"Jumlah episode: {df['Reactor_Run_ID'].nunique()}")

# ============================================================
# 2. DISKRETISASI STATE
# ============================================================
# Batas untuk setiap dimensi state
STATE_BOUNDS = [
    (0, 200),    # Suhu
    (0, 10),     # Tekanan
    (0, 5),      # Reaktan
    (0, 5),      # Produk
    (0, 50)      # Coolant
]
N_BINS = [20, 5, 5, 5, 10]

def discretize_state(state, n_bins, bounds):
    """Diskretisasi state kontinu menjadi flat index."""
    indices = []
    for val, (low, high), n_bin in zip(state, bounds, n_bins):
        normalized = np.clip((val - low) / (high - low + 1e-8), 0, 1)
        bin_idx = int(normalized * (n_bin - 1))
        indices.append(bin_idx)
    return np.ravel_multi_index(indices, n_bins)

# ============================================================
# 3. ENVIRONMENT SEDERHANA
# ============================================================
class SimpleReactorEnv:
    def __init__(self, data, max_steps=50):
        self.data = data.reset_index(drop=True)
        self.max_steps = min(max_steps, len(self.data) - 1)
        self.current_step = 0
        self.state = None

    def reset(self):
        self.current_step = 0
        row = self.data.iloc[0]
        self.state = np.array([
            row['Reactor_Temp_C'],
            row['Pressure_atm'],
            row['Reactant_A_Conc_mol_L'],
            row['Product_B_Conc_mol_L'],
            row['Jacket_Flow_Rate_L_min']
        ], dtype=np.float32)
        return self.state

    def step(self, action):
        action_effect = {0: -2.0, 1: 0.0, 2: +2.0}
        self.current_step += 1

        if self.current_step < len(self.data):
            row = self.data.iloc[self.current_step]
            next_temp = row['Reactor_Temp_C'] + action_effect[action]
            next_product = row['Product_B_Conc_mol_L']
            next_coolant = row['Jacket_Flow_Rate_L_min'] + action_effect[action]
            next_pressure = row['Pressure_atm']
            next_reactant = row['Reactant_A_Conc_mol_L']
        else:
            next_temp = self.state[0] + action_effect[action]
            next_pressure = self.state[1]
            next_reactant = self.state[2]
            next_product = self.state[3]
            next_coolant = self.state[4] + action_effect[action]

        next_temp = np.clip(next_temp, 0, 200)
        next_coolant = np.clip(next_coolant, 0, 50)

        next_state = np.array([
            next_temp, next_pressure, next_reactant,
            next_product, next_coolant
        ], dtype=np.float32)

        reward = 0
        if next_temp < 100:
            reward += 1.0
        else:
            reward -= 10.0
        reward += 2.0 if next_product > 0.5 else 0.5
        if next_coolant > 30:
            reward -= 1.0

        terminated = (next_temp > 120) or (self.current_step >= self.max_steps)
        self.state = next_state
        return next_state, reward, terminated

    def action_space_sample(self):
        return np.random.randint(0, 3)

# ============================================================
# 4. TRAINING Q-LEARNING
# ============================================================
N_ACTIONS = 3
Q_table = np.zeros((np.prod(N_BINS), N_ACTIONS))

learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005
n_episodes = 500

# Ambil satu episode untuk training
episode_data = df[df['Reactor_Run_ID'] == df['Reactor_Run_ID'].unique()[0]]
env = SimpleReactorEnv(episode_data, max_steps=50)

print(f"\nTraining Q-Learning selama {n_episodes} episode...")

for episode in range(n_episodes):
    state = env.reset()
    done = False

    while not done:
        state_idx = discretize_state(state, N_BINS, STATE_BOUNDS)

        # Epsilon-greedy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space_sample()
        else:
            action = np.argmax(Q_table[state_idx, :])

        next_state, reward, done = env.step(action)
        next_state_idx = discretize_state(next_state, N_BINS, STATE_BOUNDS)

        # Update Q-table
        Q_table[state_idx, action] += learning_rate * (
            reward + discount_factor * np.max(Q_table[next_state_idx, :])
            - Q_table[state_idx, action]
        )

        state = next_state

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    if (episode + 1) % 100 == 0:
        print(f"  Episode {episode+1}/{n_episodes} — Epsilon: {epsilon:.4f}")

# ============================================================
# 5. EXPORT MODEL
# ============================================================
os.makedirs('../models', exist_ok=True)

joblib.dump(Q_table, '../models/rl_qtable.pkl')
joblib.dump({
    'state_bounds': STATE_BOUNDS,
    'n_bins': N_BINS,
    'n_actions': N_ACTIONS,
    'action_names': ['Turunkan Coolant', 'Pertahankan', 'Naikkan Coolant']
}, '../models/rl_metadata.pkl')

print("\n✓ Model berhasil disimpan:")
print("  - models/rl_qtable.pkl")
print("  - models/rl_metadata.pkl")
```

### 6.4 Jalankan Training

```bash
cd training
python 01_train_supervised.py
python 02_train_unsupervised.py
python 03_train_reinforcement.py
```

Setelah selesai, folder `models/` akan berisi:

```
models/
├── supervised_defect_model.pkl
├── supervised_scaler.pkl
├── supervised_features.pkl
├── unsupervised_kmeans.pkl
├── unsupervised_scaler.pkl
├── unsupervised_features.pkl
├── unsupervised_encoders.pkl
├── unsupervised_n_clusters.pkl
├── cluster_profile.csv
├── rl_qtable.pkl
└── rl_metadata.pkl
```


## 7. FASE 3: BACKEND FLASK

### 7.1 File Utama `app.py`

```python
"""
Smart Manufacturing AI — Flask Backend
Mengintegrasikan Supervised, Unsupervised, dan Reinforcement Learning
"""
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

# ============================================================
# LOAD MODELS
# ============================================================
print("Loading ML models...")

MODELS = {}

# Supervised
MODELS['supervised_model'] = joblib.load('models/supervised_defect_model.pkl')
MODELS['supervised_scaler'] = joblib.load('models/supervised_scaler.pkl')
MODELS['supervised_features'] = joblib.load('models/supervised_features.pkl')

# Unsupervised
MODELS['unsup_kmeans'] = joblib.load('models/unsupervised_kmeans.pkl')
MODELS['unsup_scaler'] = joblib.load('models/unsupervised_scaler.pkl')
MODELS['unsup_features'] = joblib.load('models/unsupervised_features.pkl')
MODELS['unsup_n_clusters'] = joblib.load('models/unsupervised_n_clusters.pkl')

# Reinforcement
MODELS['rl_qtable'] = joblib.load('models/rl_qtable.pkl')
MODELS['rl_metadata'] = joblib.load('models/rl_metadata.pkl')

print("✓ Semua model berhasil diload!")

# ============================================================
# ROUTES — PAGES
# ============================================================
@app.route('/')
def index():
    """Halaman utama."""
    return render_template('index.html')

@app.route('/supervised')
def supervised_page():
    """Halaman prediksi defect (Supervised Learning)."""
    features = MODELS['supervised_features']
    return render_template('supervised.html', features=features)

@app.route('/unsupervised')
def unsupervised_page():
    """Halaman segmentasi state (Unsupervised Learning)."""
    features = MODELS['unsup_features']
    n_clusters = MODELS['unsup_n_clusters']
    return render_template('unsupervised.html',
                          features=features,
                          n_clusters=n_clusters)

@app.route('/reinforcement')
def reinforcement_page():
    """Halaman kontrol reaktor (Reinforcement Learning)."""
    metadata = MODELS['rl_metadata']
    return render_template('reinforcement.html', metadata=metadata)

@app.route('/about')
def about_page():
    """Halaman tentang proyek."""
    return render_template('about.html')

# ============================================================
# API — SUPERVISED (Prediksi Defect)
# ============================================================
@app.route('/api/predict-defect', methods=['POST'])
def predict_defect():
    """
    Prediksi defect produk berdasarkan parameter produksi.

    Input JSON:
    {
        "ProductionVolume": 1000,
        "ProductionCost": 50.5,
        "SupplierQuality": 85.0,
        ...
    }
    """
    try:
        data = request.get_json()

        # Buat DataFrame dari input
        df = pd.DataFrame([data])

        # Feature engineering (sama seperti training)
        df['Total_Cost'] = df['ProductionVolume'] * df['ProductionCost']
        df['Efficiency_Ratio'] = df['WorkerProductivity'] / (df['EnergyConsumption'] + 1)
        df['Quality_Index'] = df['QualityScore'] / (df['DefectRate'] + 1)
        df['Maintenance_Intensity'] = df['MaintenanceHours'] / (df['DowntimePercentage'] + 1)

        # Pastikan urutan fitur sesuai
        features = MODELS['supervised_features']
        X = df[features]

        # Scaling
        X_scaled = MODELS['supervised_scaler'].transform(X)

        # Prediksi
        prediction = MODELS['supervised_model'].predict(X_scaled)[0]
        probability = MODELS['supervised_model'].predict_proba(X_scaled)[0]

        result = {
            'success': True,
            'prediction': int(prediction),
            'label': 'High Defect' if prediction == 1 else 'Low Defect',
            'probability_high': float(probability[1]),
            'probability_low': float(probability[0]),
            'recommendation': get_defect_recommendation(prediction, probability[1])
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

def get_defect_recommendation(prediction, prob_high):
    """Berikan rekomendasi berdasarkan hasil prediksi."""
    if prediction == 1 and prob_high > 0.8:
        return "⚠️ RISIKO TINGGI: Segera periksa parameter proses. Pertimbangkan untuk menghentikan produksi sementara."
    elif prediction == 1:
        return "⚠️ RISIKO SEDANG: Tingkatkan monitoring. Periksa kualitas supplier dan maintenance."
    else:
        return "✓ RISIKO RENDAH: Produksi dapat dilanjutkan dengan monitoring normal."

# ============================================================
# API — UNSUPERVISED (Segmentasi State)
# ============================================================
@app.route('/api/cluster-state', methods=['POST'])
def cluster_state():
    """
    Segmentasi state operasional mesin.

    Input JSON:
    {
        "temperature_c": 75.5,
        "pressure_torr": 3.2,
        ...
    }
    """
    try:
        data = request.get_json()

        df = pd.DataFrame([data])
        features = MODELS['unsup_features']

        # Pastikan semua fitur ada
        for f in features:
            if f not in df.columns:
                df[f] = 0

        X = df[features]
        X_scaled = MODELS['unsup_scaler'].transform(X)

        # Prediksi cluster
        cluster = MODELS['unsup_kmeans'].predict(X_scaled)[0]

        # Interpretasi cluster
        cluster_info = get_cluster_info(cluster)

        result = {
            'success': True,
            'cluster': int(cluster),
            'cluster_name': cluster_info['name'],
            'cluster_description': cluster_info['description'],
            'cluster_color': cluster_info['color'],
            'recommendation': cluster_info['recommendation']
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

def get_cluster_info(cluster):
    """Interpretasi cluster."""
    info_map = {
        0: {
            'name': 'Steady State',
            'description': 'Mesin beroperasi dalam kondisi normal. Parameter stabil.',
            'color': '#28a745',
            'recommendation': '✓ Operasi normal. Lanjutkan monitoring rutin.'
        },
        1: {
            'name': 'Transient State',
            'description': 'Mesin dalam kondisi peralihan. Parameter fluktuatif.',
            'color': '#ffc107',
            'recommendation': '⚠️ Monitoring lebih ketat. Periksa setpoint parameter.'
        },
        2: {
            'name': 'Abnormal State',
            'description': 'Mesin menunjukkan gejala abnormal. Perlu intervensi.',
            'color': '#dc3545',
            'recommendation': '🚨 Segera lakukan inspeksi! Cek sistem pendingin dan pelumasan.'
        }
    }
    return info_map.get(cluster, info_map[0])

# ============================================================
# API — REINFORCEMENT (Kontrol Reaktor)
# ============================================================
@app.route('/api/reactor-control', methods=['POST'])
def reactor_control():
    """
    Rekomendasi action kontrol reaktor.

    Input JSON:
    {
        "temperature_c": 85.0,
        "pressure_atm": 3.5,
        "reactant_conc": 2.0,
        "product_conc": 1.5,
        "coolant_flow": 15.0
    }
    """
    try:
        data = request.get_json()

        # Buat state vector
        state = np.array([
            data['temperature_c'],
            data['pressure_atm'],
            data['reactant_conc'],
            data['product_conc'],
            data['coolant_flow']
        ], dtype=np.float32)

        # Discretize state
        metadata = MODELS['rl_metadata']
        state_idx = discretize_state(
            state,
            metadata['n_bins'],
            metadata['state_bounds']
        )

        # Lookup Q-table
        q_values = MODELS['rl_qtable'][state_idx, :]
        best_action = int(np.argmax(q_values))

        # Evaluasi safety
        safety_status = evaluate_safety(state[0], state[1])

        result = {
            'success': True,
            'recommended_action': best_action,
            'action_name': metadata['action_names'][best_action],
            'q_values': q_values.tolist(),
            'confidence': float(np.max(q_values) - np.min(q_values)),
            'safety_status': safety_status['status'],
            'safety_message': safety_status['message'],
            'safety_color': safety_status['color']
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

def discretize_state(state, n_bins, bounds):
    """Diskretisasi state (sama seperti training)."""
    indices = []
    for val, (low, high), n_bin in zip(state, bounds, n_bins):
        normalized = np.clip((val - low) / (high - low + 1e-8), 0, 1)
        bin_idx = int(normalized * (n_bin - 1))
        indices.append(bin_idx)
    return np.ravel_multi_index(indices, n_bins)

def evaluate_safety(temp, pressure):
    """Evaluasi status keselamatan reaktor."""
    if temp > 110 or pressure > 8:
        return {
            'status': 'DANGER',
            'message': '🚨 BAHAYA! Suhu/tekanan melebihi batas aman. Segera turunkan coolant dan pertimbangkan shutdown.',
            'color': '#dc3545'
        }
    elif temp > 95 or pressure > 6:
        return {
            'status': 'WARNING',
            'message': '⚠️ PERINGATAN! Suhu/tekanan mendekati batas. Tingkatkan aliran coolant.',
            'color': '#ffc107'
        }
    else:
        return {
            'status': 'SAFE',
            'message': '✓ Kondisi aman. Reaktor beroperasi normal.',
            'color': '#28a745'
        }

# ============================================================
# API — SYSTEM INFO
# ============================================================
@app.route('/api/system-info', methods=['GET'])
def system_info():
    """Informasi sistem dan model."""
    return jsonify({
        'supervised': {
            'model': 'Random Forest Classifier',
            'features': len(MODELS['supervised_features']),
            'task': 'Binary Classification (Defect Prediction)'
        },
        'unsupervised': {
            'model': 'K-Means Clustering',
            'n_clusters': int(MODELS['unsup_n_clusters']),
            'features': len(MODELS['unsup_features']),
            'task': 'State Segmentation'
        },
        'reinforcement': {
            'model': 'Q-Learning',
            'n_actions': MODELS['rl_metadata']['n_actions'],
            'actions': MODELS['rl_metadata']['action_names'],
            'task': 'Reactor Control Optimization'
        }
    })

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("SMART MANUFACTURING AI — FLASK SERVER")
    print("=" * 60)
    print("Server berjalan di: http://127.0.0.1:5000")
    print("Tekan CTRL+C untuk menghentikan server")
    print("=" * 60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
```


## 8. FASE 4: FRONTEND WEB

### 8.1 Base Template `templates/base.html`

```html
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Smart Manufacturing AI{% endblock %}</title>

    <!-- Bootstrap 5 -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Bootstrap Icons -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css" rel="stylesheet">
    <!-- Custom CSS -->
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">

    {% block extra_css %}{% endblock %}
</head>
<body>

<!-- ============================================================ -->
<!-- NAVBAR                                                        -->
<!-- ============================================================ -->
<nav class="navbar navbar-expand-lg navbar-dark bg-dark sticky-top">
    <div class="container">
        <a class="navbar-brand" href="/">
            <i class="bi bi-cpu"></i> <strong>Smart Manufacturing AI</strong>
        </a>
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
            <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav ms-auto">
                <li class="nav-item">
                    <a class="nav-link" href="/">
                        <i class="bi bi-house-door"></i> Home
                    </a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="/supervised">
                        <i class="bi bi-shield-check"></i> Prediksi Defect
                    </a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="/unsupervised">
                        <i class="bi bi-diagram-3"></i> Segmentasi State
                    </a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="/reinforcement">
                        <i class="bi bi-robot"></i> Kontrol Reaktor
                    </a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" href="/about">
                        <i class="bi bi-info-circle"></i> About
                    </a>
                </li>
            </ul>
        </div>
    </div>
</nav>

<!-- ============================================================ -->
<!-- MAIN CONTENT                                                  -->
<!-- ============================================================ -->
<main>
    {% block content %}{% endblock %}
</main>

<!-- ============================================================ -->
<!-- FOOTER                                                        -->
<!-- ============================================================ -->
<footer class="bg-dark text-white text-center py-4 mt-5">
    <div class="container">
        <p class="mb-1">
            <strong>Smart Manufacturing AI</strong> — Proyek Capstone Machine Learning
        </p>
        <p class="mb-0 small">
            D4 Teknologi Rekayasa Informatika dan Komputer<br>
            Politeknik Manufaktur Bandung © 2025
        </p>
    </div>
</footer>

<!-- Bootstrap JS -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<!-- Chart.js -->
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>

{% block extra_js %}{% endblock %}

</body>
</html>
```

### 8.2 Home Page `templates/index.html`

```html
{% extends "base.html" %}

{% block title %}Home — Smart Manufacturing AI{% endblock %}

{% block content %}
<!-- HERO SECTION -->
<div class="hero-section text-white text-center py-5">
    <div class="container">
        <h1 class="display-4 fw-bold">
            <i class="bi bi-cpu"></i> Smart Manufacturing AI
        </h1>
        <p class="lead mt-3">
            Sistem Cerdas Terintegrasi untuk Industri Manufaktur 4.0
        </p>
        <p class="mb-4">
            Menggabungkan <strong>Supervised</strong>, <strong>Unsupervised</strong>,
            dan <strong>Reinforcement Learning</strong> dalam satu platform
        </p>
    </div>
</div>

<!-- MODULE CARDS -->
<div class="container my-5">
    <h2 class="text-center mb-5">Tiga Modul Machine Learning</h2>

    <div class="row g-4">
        <!-- Supervised -->
        <div class="col-md-4">
            <div class="card module-card h-100 shadow-sm">
                <div class="card-body text-center p-4">
                    <div class="module-icon bg-primary text-white">
                        <i class="bi bi-shield-check"></i>
                    </div>
                    <h4 class="mt-3">Prediksi Defect</h4>
                    <p class="text-muted">
                        <span class="badge bg-primary">Supervised Learning</span>
                    </p>
                    <p>
                        Memprediksi probabilitas defect produk berdasarkan
                        parameter produksi seperti volume, biaya, kualitas supplier,
                        dan maintenance.
                    </p>
                    <a href="/supervised" class="btn btn-primary">
                        <i class="bi bi-arrow-right-circle"></i> Coba Modul
                    </a>
                </div>
            </div>
        </div>

        <!-- Unsupervised -->
        <div class="col-md-4">
            <div class="card module-card h-100 shadow-sm">
                <div class="card-body text-center p-4">
                    <div class="module-icon bg-success text-white">
                        <i class="bi bi-diagram-3"></i>
                    </div>
                    <h4 class="mt-3">Segmentasi State</h4>
                    <p class="text-muted">
                        <span class="badge bg-success">Unsupervised Learning</span>
                    </p>
                    <p>
                        Mengelompokkan kondisi operasional mesin ke dalam state
                        bermakna: Steady, Transient, atau Abnormal menggunakan
                        K-Means clustering.
                    </p>
                    <a href="/unsupervised" class="btn btn-success">
                        <i class="bi bi-arrow-right-circle"></i> Coba Modul
                    </a>
                </div>
            </div>
        </div>

        <!-- Reinforcement -->
        <div class="col-md-4">
            <div class="card module-card h-100 shadow-sm">
                <div class="card-body text-center p-4">
                    <div class="module-icon bg-warning text-white">
                        <i class="bi bi-robot"></i>
                    </div>
                    <h4 class="mt-3">Kontrol Reaktor</h4>
                    <p class="text-muted">
                        <span class="badge bg-warning">Reinforcement Learning</span>
                    </p>
                    <p>
                        Merekomendasikan action optimal untuk kontrol reaktor batch
                        menggunakan Q-Learning — mencegah overheating dan
                        memaksimalkan yield.
                    </p>
                    <a href="/reinforcement" class="btn btn-warning">
                        <i class="bi bi-arrow-right-circle"></i> Coba Modul
                    </a>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- ARCHITECTURE -->
<div class="bg-light py-5">
    <div class="container">
        <h2 class="text-center mb-4">Arsitektur Sistem</h2>
        <div class="text-center">
            <p class="lead mb-4">
                Web → Flask API → ML Models → Response
            </p>
            <div class="row text-center">
                <div class="col-md-3">
                    <div class="arch-box bg-white">
                        <i class="bi bi-window text-primary"></i>
                        <h6>Frontend</h6>
                        <small>HTML, CSS, JS</small>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="arch-box bg-white">
                        <i class="bi bi-server text-success"></i>
                        <h6>Backend</h6>
                        <small>Flask REST API</small>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="arch-box bg-white">
                        <i class="bi bi-gear text-warning"></i>
                        <h6>ML Models</h6>
                        <small>RF, K-Means, Q-Table</small>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="arch-box bg-white">
                        <i class="bi bi-database text-danger"></i>
                        <h6>Data</h6>
                        <small>CSV Datasets</small>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
```

### 8.3 Supervised Page `templates/supervised.html`

```html
{% extends "base.html" %}

{% block title %}Prediksi Defect — Smart Manufacturing AI{% endblock %}

{% block content %}
<div class="container my-5">
    <div class="row">
        <div class="col-lg-10 mx-auto">
            <!-- HEADER -->
            <div class="text-center mb-5">
                <h1><i class="bi bi-shield-check text-primary"></i> Prediksi Defect Produk</h1>
                <p class="lead">
                    Masukkan parameter produksi untuk memprediksi probabilitas defect
                </p>
                <span class="badge bg-primary">Supervised Learning — Random Forest</span>
            </div>

            <div class="row">
                <!-- FORM INPUT -->
                <div class="col-md-7">
                    <div class="card shadow-sm">
                        <div class="card-header bg-primary text-white">
                            <i class="bi bi-input-cursor"></i> Parameter Produksi
                        </div>
                        <div class="card-body">
                            <form id="defectForm">
                                <div class="row g-3">
                                    <div class="col-md-6">
                                        <label class="form-label">Production Volume</label>
                                        <input type="number" class="form-control" name="ProductionVolume"
                                               value="1000" step="1" required>
                                    </div>
                                    <div class="col-md-6">
                                        <label class="form-label">Production Cost ($)</label>
                                        <input type="number" class="form-control" name="ProductionCost"
                                               value="50.5" step="0.1" required>
                                    </div>
                                    <div class="col-md-6">
                                        <label class="form-label">Supplier Quality (0-100)</label>
                                        <input type="number" class="form-control" name="SupplierQuality"
                                               value="85" step="1" required>
                                    </div>
                                    <div class="col-md-6">
                                        <label class="form-label">Delivery Delay (hari)</label>
                                        <input type="number" class="form-control" name="DeliveryDelay"
                                               value="2" step="1" required>
                                    </div>
                                    <div class="col-md-6">
                                        <label class="form-label">Defect Rate (%)</label>
                                        <input type="number" class="form-control" name="DefectRate"
                                               value="2.5" step="0.1" required>
                                    </div>
                                    <div class="col-md-6">
                                        <label class="form-label">Quality Score (0-100)</label>
                                        <input type="number" class="form-control" name="QualityScore"
                                               value="90" step="1" required>
                                    </div>
                                    <div class="col-md-6">
                                        <label class="form-label">Maintenance Hours</label>
                                        <input type="number" class="form-control" name="MaintenanceHours"
                                               value="10" step="1" required>
                                    </div>
                                    <div class="col-md-6">
                                        <label class="form-label">Downtime (%)</label>
                                        <input type="number" class="form-control" name="DowntimePercentage"
                                               value="3.5" step="0.1" required>
                                    </div>
                                    <div class="col-md-6">
                                        <label class="form-label">Inventory Turnover</label>
                                        <input type="number" class="form-control" name="InventoryTurnover"
                                               value="5.2" step="0.1" required>
                                    </div>
                                    <div class="col-md-6">
                                        <label class="form-label">Stockout Rate (%)</label>
                                        <input type="number" class="form-control" name="StockoutRate"
                                               value="1.2" step="0.1" required>
                                    </div>
                                    <div class="col-md-6">
                                        <label class="form-label">Worker Productivity</label>
                                        <input type="number" class="form-control" name="WorkerProductivity"
                                               value="85" step="1" required>
                                    </div>
                                    <div class="col-md-6">
                                        <label class="form-label">Safety Incidents</label>
                                        <input type="number" class="form-control" name="SafetyIncidents"
                                               value="0" step="1" required>
                                    </div>
                                    <div class="col-md-6">
                                        <label class="form-label">Energy Consumption (kWh)</label>
                                        <input type="number" class="form-control" name="EnergyConsumption"
                                               value="500" step="10" required>
                                    </div>
                                    <div class="col-md-6">
                                        <label class="form-label">Energy Efficiency</label>
                                        <input type="number" class="form-control" name="EnergyEfficiency"
                                               value="0.85" step="0.01" required>
                                    </div>
                                    <div class="col-md-6">
                                        <label class="form-label">Additive Process Time</label>
                                        <input type="number" class="form-control" name="AdditiveProcessTime"
                                               value="2.5" step="0.1" required>
                                    </div>
                                    <div class="col-md-6">
                                        <label class="form-label">Additive Material Cost</label>
                                        <input type="number" class="form-control" name="AdditiveMaterialCost"
                                               value="25.5" step="0.1" required>
                                    </div>
                                </div>
                                <button type="submit" class="btn btn-primary w-100 mt-4">
                                    <i class="bi bi-search"></i> Prediksi Defect
                                </button>
                            </form>
                        </div>
                    </div>
                </div>

                <!-- RESULT -->
                <div class="col-md-5">
                    <div class="card shadow-sm">
                        <div class="card-header bg-dark text-white">
                            <i class="bi bi-clipboard-data"></i> Hasil Prediksi
                        </div>
                        <div class="card-body" id="resultPanel">
                            <div class="text-center text-muted py-5">
                                <i class="bi bi-hourglass-split" style="font-size: 3rem;"></i>
                                <p class="mt-3">Menunggu input...</p>
                            </div>
                        </div>
                    </div>

                    <!-- CHART -->
                    <div class="card shadow-sm mt-3" id="chartCard" style="display: none;">
                        <div class="card-header bg-dark text-white">
                            <i class="bi bi-bar-chart"></i> Probabilitas
                        </div>
                        <div class="card-body">
                            <canvas id="probChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_js %}
<script src="{{ url_for('static', filename='js/supervised.js') }}"></script>
{% endblock %}
```

### 8.4 JavaScript `static/js/supervised.js`

```javascript
/**
 * Supervised Learning — Frontend Logic
 */

let probChartInstance = null;

document.getElementById('defectForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    // Ambil data form
    const formData = new FormData(this);
    const data = {};
    formData.forEach((value, key) => {
        data[key] = parseFloat(value);
    });

    // Tampilkan loading
    const resultPanel = document.getElementById('resultPanel');
    resultPanel.innerHTML = `
        <div class="text-center py-5">
            <div class="spinner-border text-primary" role="status"></div>
            <p class="mt-3">Memproses prediksi...</p>
        </div>
    `;

    try {
        // Kirim request ke API
        const response = await fetch('/api/predict-defect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (!result.success) {
            throw new Error(result.error);
        }

        // Tampilkan hasil
        displayResult(result);

    } catch (error) {
        resultPanel.innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle"></i>
                Error: ${error.message}
            </div>
        `;
    }
});

function displayResult(result) {
    const resultPanel = document.getElementById('resultPanel');
    const isHigh = result.prediction === 1;
    const alertClass = isHigh ? 'danger' : 'success';
    const icon = isHigh ? 'exclamation-triangle' : 'check-circle';

    resultPanel.innerHTML = `
        <div class="alert alert-${alertClass} text-center">
            <i class="bi bi-${icon}" style="font-size: 3rem;"></i>
            <h3 class="mt-2">${result.label}</h3>
            <p class="mb-0">Probabilitas: ${(result.probability_high * 100).toFixed(2)}%</p>
        </div>
        <div class="mt-3">
            <h6><i class="bi bi-lightbulb"></i> Rekomendasi:</h6>
            <p class="small">${result.recommendation}</p>
        </div>
    `;

    // Tampilkan chart
    document.getElementById('chartCard').style.display = 'block';

    if (probChartInstance) {
        probChartInstance.destroy();
    }

    const ctx = document.getElementById('probChart').getContext('2d');
    probChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Low Defect', 'High Defect'],
            datasets: [{
                data: [result.probability_low * 100, result.probability_high * 100],
                backgroundColor: ['#28a745', '#dc3545']
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'bottom' }
            }
        }
    });
}
```

### 8.5 Unsupervised Page `templates/unsupervised.html`

```html
{% extends "base.html" %}

{% block title %}Segmentasi State — Smart Manufacturing AI{% endblock %}

{% block content %}
<div class="container my-5">
    <div class="row">
        <div class="col-lg-10 mx-auto">
            <div class="text-center mb-5">
                <h1><i class="bi bi-diagram-3 text-success"></i> Segmentasi State Mesin</h1>
                <p class="lead">
                    Masukkan sensor readings untuk mengidentifikasi state operasional
                </p>
                <span class="badge bg-success">Unsupervised Learning — K-Means (K={{ n_clusters }})</span>
            </div>

            <div class="row">
                <div class="col-md-6">
                    <div class="card shadow-sm">
                        <div class="card-header bg-success text-white">
                            <i class="bi bi-input-cursor"></i> Sensor Readings
                        </div>
                        <div class="card-body">
                            <form id="clusterForm">
                                {% for feature in features %}
                                <div class="mb-3">
                                    <label class="form-label">{{ feature }}</label>
                                    <input type="number" class="form-control"
                                           name="{{ feature }}"
                                           value="0" step="0.01" required>
                                </div>
                                {% endfor %}
                                <button type="submit" class="btn btn-success w-100 mt-3">
                                    <i class="bi bi-search"></i> Identifikasi State
                                </button>
                            </form>
                        </div>
                    </div>
                </div>

                <div class="col-md-6">
                    <div class="card shadow-sm">
                        <div class="card-header bg-dark text-white">
                            <i class="bi bi-clipboard-data"></i> Hasil Segmentasi
                        </div>
                        <div class="card-body" id="clusterResult">
                            <div class="text-center text-muted py-5">
                                <i class="bi bi-hourglass-split" style="font-size: 3rem;"></i>
                                <p class="mt-3">Menunggu input...</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_js %}
<script src="{{ url_for('static', filename='js/unsupervised.js') }}"></script>
{% endblock %}
```

### 8.6 JavaScript `static/js/unsupervised.js`

```javascript
document.getElementById('clusterForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    const formData = new FormData(this);
    const data = {};
    formData.forEach((value, key) => {
        data[key] = parseFloat(value);
    });

    const resultPanel = document.getElementById('clusterResult');
    resultPanel.innerHTML = `
        <div class="text-center py-5">
            <div class="spinner-border text-success" role="status"></div>
            <p class="mt-3">Menganalisis state...</p>
        </div>
    `;

    try {
        const response = await fetch('/api/cluster-state', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (!result.success) throw new Error(result.error);

        resultPanel.innerHTML = `
            <div class="alert text-center" style="background-color: ${result.cluster_color}; color: white;">
                <h2 class="mb-0">Cluster ${result.cluster}</h2>
                <h4 class="mt-2">${result.cluster_name}</h4>
            </div>
            <div class="mt-3">
                <h6><i class="bi bi-info-circle"></i> Deskripsi:</h6>
                <p class="small">${result.cluster_description}</p>
                <h6 class="mt-3"><i class="bi bi-lightbulb"></i> Rekomendasi:</h6>
                <p class="small">${result.recommendation}</p>
            </div>
        `;

    } catch (error) {
        resultPanel.innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle"></i> Error: ${error.message}
            </div>
        `;
    }
});
```

### 8.7 Reinforcement Page `templates/reinforcement.html`

```html
{% extends "base.html" %}

{% block title %}Kontrol Reaktor — Smart Manufacturing AI{% endblock %}

{% block content %}
<div class="container my-5">
    <div class="row">
        <div class="col-lg-10 mx-auto">
            <div class="text-center mb-5">
                <h1><i class="bi bi-robot text-warning"></i> Kontrol Reaktor Batch</h1>
                <p class="lead">
                    Masukkan state reaktor untuk mendapatkan rekomendasi action optimal
                </p>
                <span class="badge bg-warning">Reinforcement Learning — Q-Learning</span>
            </div>

            <div class="row">
                <div class="col-md-6">
                    <div class="card shadow-sm">
                        <div class="card-header bg-warning text-white">
                            <i class="bi bi-input-cursor"></i> State Reaktor
                        </div>
                        <div class="card-body">
                            <form id="reactorForm">
                                <div class="mb-3">
                                    <label class="form-label">Suhu Reaktor (°C)</label>
                                    <input type="number" class="form-control" name="temperature_c"
                                           value="85" step="0.1" required>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Tekanan (atm)</label>
                                    <input type="number" class="form-control" name="pressure_atm"
                                           value="3.5" step="0.01" required>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Konsentrasi Reaktan (mol/L)</label>
                                    <input type="number" class="form-control" name="reactant_conc"
                                           value="2.0" step="0.01" required>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Konsentrasi Produk (mol/L)</label>
                                    <input type="number" class="form-control" name="product_conc"
                                           value="1.5" step="0.01" required>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Coolant Flow (L/min)</label>
                                    <input type="number" class="form-control" name="coolant_flow"
                                           value="15" step="0.1" required>
                                </div>
                                <button type="submit" class="btn btn-warning w-100 mt-3">
                                    <i class="bi bi-lightning-charge"></i> Rekomendasi Action
                                </button>
                            </form>
                        </div>
                    </div>
                </div>

                <div class="col-md-6">
                    <div class="card shadow-sm">
                        <div class="card-header bg-dark text-white">
                            <i class="bi bi-clipboard-data"></i> Hasil Rekomendasi
                        </div>
                        <div class="card-body" id="reactorResult">
                            <div class="text-center text-muted py-5">
                                <i class="bi bi-hourglass-split" style="font-size: 3rem;"></i>
                                <p class="mt-3">Menunggu input...</p>
                            </div>
                        </div>
                    </div>

                    <div class="card shadow-sm mt-3" id="qChartCard" style="display: none;">
                        <div class="card-header bg-dark text-white">
                            <i class="bi bi-bar-chart"></i> Q-Values
                        </div>
                        <div class="card-body">
                            <canvas id="qChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_js %}
<script>
let qChartInstance = null;

document.getElementById('reactorForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    const formData = new FormData(this);
    const data = {};
    formData.forEach((value, key) => {
        data[key] = parseFloat(value);
    });

    const resultPanel = document.getElementById('reactorResult');
    resultPanel.innerHTML = `
        <div class="text-center py-5">
            <div class="spinner-border text-warning" role="status"></div>
            <p class="mt-3">Menghitung action optimal...</p>
        </div>
    `;

    try {
        const response = await fetch('/api/reactor-control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (!result.success) throw new Error(result.error);

        resultPanel.innerHTML = `
            <div class="alert text-center" style="background-color: ${result.safety_color}; color: white;">
                <h5 class="mb-0">${result.safety_status}</h5>
                <p class="small mb-0 mt-1">${result.safety_message}</p>
            </div>
            <div class="text-center mt-3">
                <h6>Action yang Direkomendasikan:</h6>
                <h3 class="text-warning">${result.action_name}</h3>
                <p class="small text-muted">Confidence: ${(result.confidence * 100).toFixed(1)}%</p>
            </div>
        `;

        // Tampilkan chart
        document.getElementById('qChartCard').style.display = 'block';

        if (qChartInstance) qChartInstance.destroy();

        const ctx = document.getElementById('qChart').getContext('2d');
        qChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Turunkan', 'Pertahankan', 'Naikkan'],
                datasets: [{
                    label: 'Q-Value',
                    data: result.q_values,
                    backgroundColor: ['#dc3545', '#6c757d', '#28a745']
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: { y: { beginAtZero: true } }
            }
        });

    } catch (error) {
        resultPanel.innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle"></i> Error: ${error.message}
            </div>
        `;
    }
});
</script>
{% endblock %}
```

### 8.8 Custom CSS `static/css/style.css`

```css
/* ============================================================
   GLOBAL STYLES
   ============================================================ */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f8f9fa;
}

/* ============================================================
   HERO SECTION
   ============================================================ */
.hero-section {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    padding: 80px 0;
}

.hero-section h1 {
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

/* ============================================================
   MODULE CARDS
   ============================================================ */
.module-card {
    transition: transform 0.3s, box-shadow 0.3s;
    border: none;
}

.module-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 24px rgba(0,0,0,0.15) !important;
}

.module-icon {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 2.5rem;
    margin: 0 auto;
}

/* ============================================================
   ARCHITECTURE BOXES
   ============================================================ */
.arch-box {
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    margin-bottom: 15px;
}

.arch-box i {
    font-size: 2.5rem;
    margin-bottom: 10px;
}

/* ============================================================
   FORM
   ============================================================ */
.form-control:focus {
    border-color: #0d6efd;
    box-shadow: 0 0 0 0.2rem rgba(13, 110, 253, 0.15);
}

/* ============================================================
   RESULT PANEL
   ============================================================ */
#resultPanel .alert,
#clusterResult .alert,
#reactorResult .alert {
    border-radius: 10px;
}

/* ============================================================
   RESPONSIVE
   ============================================================ */
@media (max-width: 768px) {
    .hero-section {
        padding: 50px 0;
    }

    .hero-section h1 {
        font-size: 2rem;
    }

    .module-icon {
        width: 60px;
        height: 60px;
        font-size: 1.8rem;
    }
}
```


## 9. FASE 5: INTEGRASI DAN TESTING

### 9.1 Jalankan Aplikasi

```bash
# Pastikan di folder utama proyek
cd smart-manufacturing-ai

# Jalankan Flask
python app.py
```

**Output yang diharapkan:**
```
Loading ML models...
✓ Semua model berhasil diload!

============================================================
SMART MANUFACTURING AI — FLASK SERVER
============================================================
Server berjalan di: http://127.0.0.1:5000
Tekan CTRL+C untuk menghentikan server
============================================================
```

### 9.2 Testing Manual di Browser

Buka browser dan akses: **http://127.0.0.1:5000**

**Test Case 1 — Supervised (Prediksi Defect):**

1. Klik menu **"Prediksi Defect"**
2. Isi form dengan nilai default
3. Klik **"Prediksi Defect"**
4. **Expected:** Muncul hasil prediksi dengan label, probabilitas, dan rekomendasi

**Test Case 2 — Unsupervised (Segmentasi State):**

1. Klik menu **"Segmentasi State"**
2. Isi form dengan sensor readings
3. Klik **"Identifikasi State"**
4. **Expected:** Muncul cluster dengan nama, deskripsi, dan rekomendasi

**Test Case 3 — Reinforcement (Kontrol Reaktor):**

1. Klik menu **"Kontrol Reaktor"**
2. Isi form dengan state reaktor
3. Klik **"Rekomendasi Action"**
4. **Expected:** Muncul action yang direkomendasikan, Q-values, dan status safety

### 9.3 Testing dengan Python

Buat file `tests/test_api.py`:

```python
"""
Testing API endpoints menggunakan Python requests.
"""
import requests
import json

BASE_URL = "http://127.0.0.1:5000"

def test_supervised():
    """Test endpoint prediksi defect."""
    print("\n" + "=" * 60)
    print("TEST 1: Supervised — Prediksi Defect")
    print("=" * 60)

    payload = {
        "ProductionVolume": 1000,
        "ProductionCost": 50.5,
        "SupplierQuality": 85.0,
        "DeliveryDelay": 2,
        "DefectRate": 2.5,
        "QualityScore": 90.0,
        "MaintenanceHours": 10,
        "DowntimePercentage": 3.5,
        "InventoryTurnover": 5.2,
        "StockoutRate": 1.2,
        "WorkerProductivity": 85.0,
        "SafetyIncidents": 0,
        "EnergyConsumption": 500.0,
        "EnergyEfficiency": 0.85,
        "AdditiveProcessTime": 2.5,
        "AdditiveMaterialCost": 25.5
    }

    response = requests.post(f"{BASE_URL}/api/predict-defect", json=payload)
    result = response.json()

    print(f"Status Code: {response.status_code}")
    print(f"Result: {json.dumps(result, indent=2)}")

    assert response.status_code == 200
    assert result['success'] == True
    print("✓ Test PASSED")

def test_unsupervised():
    """Test endpoint cluster state."""
    print("\n" + "=" * 60)
    print("TEST 2: Unsupervised — Segmentasi State")
    print("=" * 60)

    payload = {
        "temperature_c": 75.5,
        "pressure_torr": 3.2,
        "gas_flow_sccm": 100.0,
        "etch_rate_nm_min": 50.0,
        "voltage_v": 220.0,
        "current_ma": 1500.0,
        "process_step_encoded": 0
    }

    response = requests.post(f"{BASE_URL}/api/cluster-state", json=payload)
    result = response.json()

    print(f"Status Code: {response.status_code}")
    print(f"Result: {json.dumps(result, indent=2)}")

    assert response.status_code == 200
    assert result['success'] == True
    print("✓ Test PASSED")

def test_reinforcement():
    """Test endpoint reactor control."""
    print("\n" + "=" * 60)
    print("TEST 3: Reinforcement — Kontrol Reaktor")
    print("=" * 60)

    payload = {
        "temperature_c": 85.0,
        "pressure_atm": 3.5,
        "reactant_conc": 2.0,
        "product_conc": 1.5,
        "coolant_flow": 15.0
    }

    response = requests.post(f"{BASE_URL}/api/reactor-control", json=payload)
    result = response.json()

    print(f"Status Code: {response.status_code}")
    print(f"Result: {json.dumps(result, indent=2)}")

    assert response.status_code == 200
    assert result['success'] == True
    print("✓ Test PASSED")

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("TESTING SMART MANUFACTURING AI API")
    print("=" * 60)

    try:
        test_supervised()
        test_unsupervised()
        test_reinforcement()

        print("\n" + "=" * 60)
        print("✓ SEMUA TEST PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
```

Jalankan test:

```bash
# Di terminal terpisah (server harus tetap berjalan)
python tests/test_api.py
```

### 9.4 Troubleshooting

| Masalah | Penyebab | Solusi |
|---|---|---|
| `FileNotFoundError: models/*.pkl` | Model belum ditraining | Jalankan training scripts dulu |
| `KeyError: 'feature'` | Nama kolom tidak sesuai | Cek `supervised_features.pkl` |
| `Connection refused` | Flask server tidak berjalan | Jalankan `python app.py` |
| `ModuleNotFoundError: flask` | Flask belum diinstall | `pip install flask` |
| `ValueError: could not convert` | Input bukan numerik | Pastikan input berupa angka |


## 10. FASE 6: DEPLOYMENT

### 10.1 Deployment Lokal (Development)

```bash
python app.py
```

### 10.2 Deployment dengan Gunicorn (Production)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### 10.3 Deployment dengan Docker

Buat file `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build dan run:

```bash
docker build -t smart-manufacturing-ai .
docker run -p 5000:5000 smart-manufacturing-ai
```

### 10.4 Deployment ke Ngrok (untuk Demo)

```bash
# Install ngrok
pip install pyngrok

# Jalankan Flask
python app.py

# Di terminal lain, buat tunnel
ngrok http 5000
```

Akses melalui URL publik yang diberikan ngrok.


## 11. TUGAS PROYEK AKHIR

### 11.1 Tugas Individu (Bobot 30%)

**Instruksi:**

1. **Modifikasi Modul Supervised:**
   - Tambahkan fitur **upload CSV** untuk batch prediction.
   - Hasil prediksi ditampilkan dalam tabel dan dapat diunduh sebagai CSV.
   - Tambahkan visualisasi **feature importance** dari model Random Forest.

2. **Modifikasi Modul Unsupervised:**
   - Tambahkan visualisasi **PCA 2D** untuk hasil clustering.
   - Tampilkan **profil cluster** dalam bentuk heatmap.
   - Tambahkan fitur **input multiple samples** dan bandingkan hasilnya.

3. **Modifikasi Modul Reinforcement:**
   - Tambahkan **simulasi langkah-demi-langkah** kontrol reaktor.
   - Visualisasikan **trayektori suhu** selama simulasi.
   - Bandingkan performa agen RL dengan baseline (random action).

**Pengumpulan:** Notebook + screenshot + link GitHub repository.

### 11.2 Tugas Kelompok (Bobot 70%)

**Proyek:** Kembangkan **Smart Manufacturing AI** menjadi sistem yang lebih lengkap dengan menambahkan **minimal 2 fitur baru**:

**Pilihan Fitur:**

1. **Dashboard Monitoring Real-time:**
   - Simulasi data streaming sensor.
   - Auto-refresh prediksi setiap 5 detik.
   - Alert otomatis jika state abnormal.

2. **User Authentication:**
   - Login/logout dengan Flask-Login.
   - Role-based access (operator, engineer, admin).
   - Audit log aktivitas.

3. **Database Integration:**
   - Simpan history prediksi ke SQLite/PostgreSQL.
   - Query dan analisis trend.
   - Export laporan PDF.

4. **REST API Documentation:**
   - Dokumentasi API dengan Swagger/OpenAPI.
   - Interactive API testing.
   - Client library (Python SDK).

5. **Deployment ke Cloud:**
   - Deploy ke Heroku/Railway/Render.
   - CI/CD dengan GitHub Actions.
   - Monitoring dengan Sentry.

**Deliverables:**

1. **Aplikasi Web** yang berjalan dan dapat diakses.
2. **Source Code** di GitHub dengan README lengkap.
3. **Laporan Proyek** (20-30 halaman):
   - Pendahuluan
   - Arsitektur sistem
   - Implementasi
   - Testing
   - Deployment
   - Kesimpulan
4. **Video Demo** (5-10 menit).
5. **Presentasi** (15 menit).

### 11.3 Timeline

| Minggu | Aktivitas |
|---|---|
| **Minggu 1** | Setup, training, backend |
| **Minggu 2** | Frontend, integrasi |
| **Minggu 3** | Fitur tambahan, testing |
| **Minggu 4** | Dokumentasi, presentasi |


## 12. RUBRIK PENILAIAN

### 12.1 Rubrik Tugas Individu (30%)

| Komponen | Bobot | Kriteria |
|---|---|---|
| **Modul Supervised** | 30% | Upload CSV, batch prediction, feature importance |
| **Modul Unsupervised** | 30% | PCA 2D, heatmap, multiple samples |
| **Modul Reinforcement** | 30% | Simulasi, visualisasi, perbandingan baseline |
| **Dokumentasi** | 10% | README, komentar kode |

### 12.2 Rubrik Tugas Kelompok (70%)

| Komponen | Bobot | Kriteria |
|---|---|---|
| **Fungsionalitas** | 30% | Semua fitur berjalan dengan baik |
| **Kualitas Kode** | 20% | Bersih, terstruktur, ada komentar |
| **UI/UX** | 15% | Menarik, responsif, mudah digunakan |
| **Fitur Tambahan** | 15% | Minimal 2 fitur baru berfungsi |
| **Dokumentasi** | 10% | Laporan lengkap, video demo |
| **Presentasi** | 10% | Jelas, menjawab pertanyaan |

### 12.3 Konversi Nilai

| Nilai | Huruf | Keterangan |
|---|---|---|
| 85–100 | A | Sangat Baik |
| 75–84 | B | Baik |
| 65–74 | C | Cukup |
| 55–64 | D | Kurang |
| < 55 | E | Sangat Kurang |


## 13. REFERENSI

### 13.1 Dokumentasi Resmi

1. **Flask:** https://flask.palletsprojects.com/
2. **Bootstrap 5:** https://getbootstrap.com/docs/5.3/
3. **Chart.js:** https://www.chartjs.org/docs/
4. **Joblib:** https://joblib.readthedocs.io/
5. **Scikit-learn:** https://scikit-learn.org/

### 13.2 Tutorial

1. **Flask Mega-Tutorial:** https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world
2. **Deploying ML Models with Flask:** https://towardsdatascience.com/deploying-machine-learning-models-with-flask
3. **Full Stack Python:** https://www.fullstackpython.com/

### 13.3 Dataset

1. Kaggle. *Predicting Manufacturing Defects*. https://www.kaggle.com/datasets/rabieelkharoua/predicting-manufacturing-defects-dataset
2. Kaggle. *Semiconductor Wafer Defect Classification Dataset*. https://www.kaggle.com/datasets/meruvakodandasuraj/semiconductor-wafer-defect-classification-dataset
3. Kaggle. *Batch Reactor Anomaly Data (Sample)*. https://www.kaggle.com/datasets/aimindteams/batch-reactor-anomaly-data-sample


## PENUTUP

Proyek capstone ini mengintegrasikan **tiga paradigma Machine Learning** — Supervised, Unsupervised, dan Reinforcement Learning — ke dalam satu **aplikasi web terpadu** yang dapat digunakan di industri manufaktur nyata. Mahasiswa tidak hanya belajar membangun model ML, tetapi juga:

1. **Mengekspor model** ke format production.
2. **Membangun REST API** dengan Flask.
3. **Membangun frontend** interaktif dengan HTML/CSS/JS.
4. **Mengintegrasikan** ketiga model dalam satu sistem.
5. **Mendeploy** aplikasi ke environment production.

**Pesan untuk Mahasiswa:**
> "Proyek ini adalah simulasi dunia nyata — di industri, ML tidak berdiri sendiri. Kemampuan untuk mengintegrasikan berbagai model ke dalam satu sistem yang dapat digunakan oleh user non-teknis adalah keterampilan yang sangat berharga. Kembangkan proyek ini sebagai portofolio Anda!"

**Pesan untuk Dosen:**
> "Proyek ini dapat disesuaikan dengan tingkat kemampuan mahasiswa. Untuk pemula, fokus pada integrasi dasar. Untuk mahasiswa mahir, tantang dengan fitur tambahan seperti real-time streaming, database, atau deployment ke cloud. Proyek ini juga dapat dijadikan sebagai **tugas akhir semester** dengan pengembangan lebih lanjut."

---

**Dokumen ini disusun untuk keperluan edukasi di Politeknik Manufaktur Bandung (Polman Bandung).**
**Versi:** 1.0 | **Integrasi Pertemuan 1–3 Minggu ke-2**
**Terakhir Diperbarui:** 2026
