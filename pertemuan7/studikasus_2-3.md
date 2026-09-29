# STUDI KASUS MACHINE LEARNING — PERTEMUAN KE-3 MINGGU KE-2
## Reinforcement Learning: Agen Cerdas untuk Optimasi Kontrol Reaktor Batch Industri

**Mata Kuliah:** Machine Learning
**Program Studi:** D4 Teknologi Rekayasa Informatika dan Komputer (TRIN)
**Institusi:** Politeknik Manufaktur Bandung (Polman Bandung)
**Pertemuan:** Minggu ke-2, Pertemuan ke-3 dari 4
**Topik:** Reinforcement Learning — Q-Learning untuk Kontrol Proses Industri
**Tools:** Anaconda, Jupyter Notebook, Python 3.11, Gymnasium, NumPy, Matplotlib
**Durasi:** 150 menit

> **Catatan:** Dataset yang digunakan dalam praktikum ini adalah **Batch Reactor Anomaly Data (Sample)** — dataset ringan (5.000 baris, ~500 KB) yang dirancang khusus untuk pelatihan Reinforcement Learning (RL) dan Predictive Maintenance (PdM). Dataset ini dapat diolah dengan lancar pada laptop dengan RAM 8GB atau 16GB tanpa memerlukan GPU atau komputasi berat.


## DAFTAR ISI

1. [Latar Belakang Studi Kasus](#1-latar-belakang-studi-kasus)
2. [Tujuan Pembelajaran](#2-tujuan-pembelajaran)
3. [Dataset yang Digunakan](#3-dataset-yang-digunakan)
4. [Landasan Teori Reinforcement Learning](#4-landasan-teori-reinforcement-learning)
5. [Step-by-Step Praktikum](#5-step-by-step-praktikum)
6. [Tugas Mandiri (Dikerjakan Hari Ini)](#6-tugas-mandiri-dikerjakan-hari-ini)
7. [Tugas Kelompok (Dikerjakan di Rumah)](#7-tugas-kelompok-dikerjakan-di-rumah)
8. [Rubrik Penilaian](#8-rubrik-penilaian)
9. [Referensi](#9-referensi)


## 1. LATAR BELAKANG STUDI KASUS

Pada dua pertemuan sebelumnya, kita telah mempelajari dua paradigma Machine Learning:

- **Pertemuan 1 — Supervised Learning:** Prediksi defect produk manufaktur menggunakan data berlabel (klasifikasi).
- **Pertemuan 2 — Unsupervised Learning:** Segmentasi state kontrol produksi menggunakan data tanpa label (clustering).

Namun, ada satu paradigma lagi yang sangat penting dalam sistem cerdas modern: **Reinforcement Learning (RL)**. Berbeda dengan dua paradigma sebelumnya, RL tidak belajar dari dataset statis, melainkan belajar melalui **interaksi langsung dengan lingkungan** — agen mengambil tindakan, menerima umpan balik (reward), dan secara bertahap mempelajari strategi optimal.

**Mengapa RL Penting dalam Industri Manufaktur?**

Bayangkan sebuah **reaktor batch** di industri kimia atau farmasi. Reaktor ini harus menjaga suhu dan tekanan dalam batas aman selama proses produksi. Jika suhu terlalu tinggi, bisa terjadi **thermal runaway** — reaksi kimia yang tidak terkendali yang dapat menyebabkan ledakan. Jika suhu terlalu rendah, produk yang dihasilkan tidak optimal.

Sistem kontrol tradisional seperti PID (Proportional-Integral-Derivative) bekerja baik untuk sistem linear sederhana, tetapi **tidak cukup untuk sistem kompleks** dengan dinamika non-linear, gangguan, dan ketidakpastian. Di sinilah RL berperan: agen RL dapat belajar **strategi kontrol optimal** melalui trial and error, tanpa memerlukan model matematis yang rumit.

Penelitian terbaru menunjukkan bahwa **Q-Learning berbasis Nonlinear Model Predictive Control (QL-NMPC)** telah berhasil divalidasi untuk kontrol suhu reaktor batch, dengan agen RL belajar menggunakan **coolant flow rate** dan **heater current** sebagai input untuk melacak trajectory suhu yang diinginkan. Pendekatan ini menjembatani teori RL dengan kontrol proses industri nyata.

**Studi Kasus Praktikum Ini:** Kita akan membangun agen RL menggunakan **Q-Learning** untuk mengontrol **reaktor batch eksotermik**. Agen harus belajar **kapan harus meningkatkan aliran coolant** (pendingin) untuk mencegah overheating, sambil tetap memaksimalkan yield produk. Dataset yang digunakan adalah **Batch Reactor Anomaly Data (Sample)** — dataset time-series yang berisi log operasional reaktor dengan berbagai anomali yang disimulasikan, termasuk **kegagalan sistem pendingin** dan **lonjakan tekanan**.

**Relevansi dengan Polman Bandung:** Sebagai institusi pendidikan vokasi yang fokus pada manufaktur dan otomasi industri, pemahaman tentang RL untuk kontrol proses sangat relevan. Industri 4.0 membutuhkan talenta yang mampu mengimplementasikan sistem kontrol cerdas yang adaptif dan otonom.


## 2. TUJUAN PEMBELAJARAN

Setelah mengikuti praktikum ini, mahasiswa diharapkan mampu:

1. **Memahami** konsep fundamental Reinforcement Learning: agen, lingkungan, state, action, reward, dan policy.
2. **Menjelaskan** perbedaan RL dengan supervised dan unsupervised learning.
3. **Memahami** Markov Decision Process (MDP) sebagai kerangka formal RL.
4. **Mengimplementasikan** algoritma Q-Learning dari nol menggunakan NumPy.
5. **Membangun** lingkungan RL kustom berbasis data reaktor batch.
6. **Melatih** agen Q-Learning dan menganalisis kurva pembelajaran.
7. **Mengevaluasi** performa agen dengan metrik yang tepat.
8. **Mengaitkan** konsep RL dengan aplikasi nyata di bidang kontrol proses industri.


## 3. DATASET YANG DIGUNAKAN

### 3.1 Dataset Utama

| Informasi | Detail |
|---|---|
| **Nama Dataset** | Batch Reactor Anomaly Data (Sample) |
| **Sumber** | Kaggle |
| **Link** | https://www.kaggle.com/datasets/aimindteams/batch-reactor-anomaly-data-sample |
| **Jumlah Data** | 5.000 baris |
| **Jumlah Fitur** | 7 kolom |
| **Ukuran File** | ~500 KB (5k rows) |
| **Tipe Data** | CSV (`reactor_sample_5k.csv`) |
| **Lisensi** | CC BY-NC 4.0 — Free for academic and non-commercial research |
| **Sifat Data** | Synthetic — high-fidelity simulation of an industrial liquid-phase exothermic batch reactor |

**Deskripsi Dataset:**
Dataset ini menyediakan simulasi matematis yang konsisten dan high-fidelity dari **reaktor batch eksotermik fase cair industri**. Dataset berisi **log operasional time-series multivariat** yang dirancang khusus untuk melatih **agen Reinforcement Learning (RL)** dan menguji algoritma **Predictive Maintenance (PdM)**. Karena sifat proprietary data kimia dunia nyata, dataset ini mengisi celah dengan menyimulasikan operasi baseline normal bersama dengan **anomali edge-case kritis**, dibatasi oleh **termodinamika, neraca massa, dan kinetika yang ketat**.

### 3.2 Deskripsi Kolom

| Kolom | Tipe | Deskripsi |
|---|---|---|
| `Timestamp_min` | Integer | Langkah waktu sekuensial operasi reaktor (menit) |
| `Reactor_Temp_C` | Float | Suhu internal reaktor (°C) |
| `Jacket_Flow_Rate_L_min` | Float | Laju aliran volumetrik coolant melalui jaket reaktor (L/min) |
| `Pressure_atm` | Float | Tekanan internal vessel (atm) |
| `Reactant_A_Conc_mol_L` | Float | Konsentrasi molar reaktan utama (mol/L) |
| `Product_B_Conc_mol_L` | Float | **Target Variable:** Konsentrasi molar produk jadi (mol/L) |
| `Reactor_Run_ID` | Integer | **Episode ID untuk RL** — identifier unik untuk setiap siklus hidup batch |

**Catatan Penting:**
- Dataset ini **tidak memiliki missing value** pada sampel 5.000 baris.
- Kolom `Reactor_Run_ID` adalah **kunci untuk RL** — setiap ID merepresentasikan satu **episode** (satu siklus hidup reaktor).
- Dataset berisi **anomali yang disimulasikan**, termasuk **Cooling System Failure** (penurunan atau pembekuan tiba-tiba pada `Jacket_Flow_Rate_L_min` yang menyebabkan lonjakan `Reactor_Temp_C` dan `Pressure_atm`) dan **Pressure Spikes** (deviasi termodinamika mendadak yang memerlukan penyesuaian aliran jacket untuk mencegah thermal runaway).
- **Rekomendasi penggunaan RL:** Melatih agen pada episode diskrit (`Reactor_Run_ID`) untuk **mengoptimalkan kontrol coolant** (`Jacket_Flow_Rate_L_min`) agar **memaksimalkan `Product_B_Conc_mol_L`** sambil **membatasi `Reactor_Temp_C`** secara ketat.

### 3.3 Mengapa Dataset Ini Relevan?

1. **RL-Ready:** Dataset secara eksplisit dirancang untuk pelatihan RL, dengan episode, state, dan action yang jelas.
2. **Konteks Industri Nyata:** Reaktor batch adalah peralatan standar di industri kimia, farmasi, dan makanan.
3. **Ringan:** 5.000 baris × 7 kolom, ~500 KB — dapat diolah pada laptop RAM 8GB tanpa sampling.
4. **Tantangan Realistis:** Anomali seperti cooling system failure dan pressure spikes membuat agen harus belajar menghadapi ketidakpastian.
5. **Reward Jelas:** Optimasi coolant flow dengan batasan suhu memberikan sinyal reward yang terdefinisi dengan baik.


## 4. LANDASAN TEORI REINFORCEMENT LEARNING

### 4.1 Definisi Reinforcement Learning

**Reinforcement Learning (RL)** adalah paradigma Machine Learning di mana **agen** (agent) belajar untuk mengambil **tindakan** (action) dalam **lingkungan** (environment) untuk memaksimalkan **reward kumulatif jangka panjang**.

**Analogi:** Bayangkan seorang operator reaktor yang belajar mengontrol suhu. Ia tidak diberi buku panduan — ia mencoba berbagai posisi valve coolant, mengamati respons suhu, dan secara bertahap belajar strategi terbaik. Inilah RL.

### 4.2 Komponen Utama RL

| Komponen | Simbol | Deskripsi | Contoh dalam Reaktor Batch |
|---|---|---|---|
| **Agent** | — | Entitas yang belajar dan mengambil keputusan | Sistem kontrol otomatis |
| **Environment** | — | Dunia tempat agen berinteraksi | Reaktor batch + data sensor |
| **State** | $s$ | Kondisi agen saat ini | Suhu, tekanan, konsentrasi |
| **Action** | $a$ | Tindakan yang bisa diambil agen | Naikkan/turunkan coolant flow |
| **Reward** | $r$ | Umpan balik dari lingkungan | +1 jika produk optimal & aman |
| **Policy** | $\pi$ | Strategi agen memilih action | "Jika suhu > 80°C, buka coolant" |
| **Q-Value** | $Q(s,a)$ | Nilai jangka panjang action di state | Seberapa baik membuka coolant di suhu tinggi |

### 4.3 Perbandingan Tiga Paradigma ML

| Aspek | Supervised | Unsupervised | Reinforcement |
|---|---|---|---|
| **Data** | $(x, y)$ berlabel | $(x)$ tanpa label | Interaksi $(s, a, r, s')$ |
| **Tujuan** | Prediksi output | Temukan struktur | Maksimalkan reward |
| **Feedback** | Label benar/salah | Tidak ada | Reward (bisa delayed) |
| **Contoh Manufaktur** | Prediksi Defect | Segmentasi State | Kontrol Proses |
| **Algoritma** | Decision Tree, RF | K-Means, DBSCAN | Q-Learning, DQN |

### 4.4 Markov Decision Process (MDP)

MDP adalah kerangka matematis formal untuk RL. Didefinisikan oleh tuple $(S, A, P, R, \gamma)$:

- $S$ = himpunan state (suhu, tekanan, konsentrasi)
- $A$ = himpunan action (level coolant flow)
- $P(s'|s, a)$ = probabilitas transisi
- $R(s, a, s')$ = reward
- $\gamma$ = discount factor (0 hingga 1)

**Markov Property:** State saat ini mengandung semua informasi yang diperlukan untuk keputusan optimal.

### 4.5 Q-Learning: Algoritma Dasar RL

**Q-Learning** adalah algoritma RL **off-policy** yang mempelajari **Q-value** — nilai optimal dari mengambil action $a$ di state $s$.

**Update Rule (Bellman Equation):**

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

di mana:
- $\alpha$ = learning rate (0 hingga 1)
- $\gamma$ = discount factor
- $r$ = reward
- $s'$ = state berikutnya

### 4.6 Eksplorasi vs Eksploitasi

**Dilema Fundamental RL:** Agen harus memilih antara:
- **Eksplorasi:** Mencoba action baru (misal, buka coolant lebih lebar).
- **Eksploitasi:** Menggunakan action terbaik yang sudah diketahui.

**Strategi Epsilon-Greedy:**
- Dengan probabilitas $\epsilon$: pilih action random (eksplorasi).
- Dengan probabilitas $1-\epsilon$: pilih action dengan Q-value tertinggi (eksploitasi).

**Epsilon Decay:** Nilai $\epsilon$ diturunkan secara bertahap — mulai dari 1.0 (eksplorasi penuh) hingga 0.01 (eksploitasi dominan).

### 4.7 Hyperparameter Q-Learning

| Parameter | Simbol | Deskripsi | Nilai Umum |
|---|---|---|---|
| **Learning Rate** | $\alpha$ | Seberapa cepat Q-value diperbarui | 0.01 – 0.5 |
| **Discount Factor** | $\gamma$ | Seberapa penting reward masa depan | 0.9 – 0.99 |
| **Epsilon** | $\epsilon$ | Probabilitas eksplorasi | 1.0 → 0.01 |
| **Epsilon Decay** | — | Tingkat penurunan epsilon | 0.99 – 0.999 |
| **Episode** | — | Satu putaran (satu batch reaktor) | 100 – 1000 |


## 5. STEP-BY-STEP PRAKTIKUM

### 5.1 Setup Environment

Buka **Anaconda Prompt**, aktifkan environment, dan jalankan Jupyter Notebook:

```bash
conda activate ml-beasiswa
pip install gymnasium numpy matplotlib pandas seaborn
jupyter notebook
```

Buat notebook baru dengan nama `reinforcement_learning_reactor.ipynb`.

### 5.2 Import Library

**Cell 1: Import Library**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
from gymnasium import spaces
import random
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
random.seed(42)

print("Gymnasium version:", gym.__version__)
print("Semua library berhasil diimport!")
```

**Penjelasan:**
- `gymnasium` untuk membuat lingkungan RL kustom.
- `numpy` untuk operasi numerik (Q-table).
- `pandas` untuk manipulasi data.
- `matplotlib` dan `seaborn` untuk visualisasi.

### 5.3 Load dan Eksplorasi Dataset

**Cell 2: Load Dataset**

```python
# Load dataset
df = pd.read_csv('reactor_sample_5k.csv')

print("Ukuran dataset:", df.shape)
print("\n5 Baris Pertama:")
df.head()
```

**Cell 3: Eksplorasi Data (EDA)**

```python
# Informasi umum
print("Informasi Dataset:")
df.info()

# Statistik deskriptif
print("\nStatistik Deskriptif:")
df.describe()

# Cek missing value
print("\nMissing Value:")
print(df.isnull().sum())

# Jumlah episode (Reactor_Run_ID unik)
print(f"\nJumlah Episode (Reactor_Run_ID unik): {df['Reactor_Run_ID'].nunique()}")
print(f"Rata-rata langkah per episode: {len(df) / df['Reactor_Run_ID'].nunique():.1f}")
```

**Penjelasan:**
- Dataset memiliki 5.000 baris dan 7 kolom.
- `Reactor_Run_ID` adalah episode ID untuk RL.
- Setiap episode merepresentasikan satu siklus hidup reaktor batch.

**Cell 4: Visualisasi Data Time-Series**

```python
# Visualisasi beberapa episode
fig, axes = plt.subplots(3, 2, figsize=(16, 12))

# Ambil 3 episode pertama
sample_runs = df['Reactor_Run_ID'].unique()[:3]

for i, run_id in enumerate(sample_runs):
    subset = df[df['Reactor_Run_ID'] == run_id]

    axes[i, 0].plot(subset['Timestamp_min'], subset['Reactor_Temp_C'],
                    color='red', linewidth=2)
    axes[i, 0].set_title(f'Episode {run_id} — Suhu Reaktor (°C)')
    axes[i, 0].set_xlabel('Timestamp (menit)')
    axes[i, 0].set_ylabel('Suhu (°C)')
    axes[i, 0].axhline(y=100, color='orange', linestyle='--', label='Batas Aman')
    axes[i, 0].legend()

    axes[i, 1].plot(subset['Timestamp_min'], subset['Product_B_Conc_mol_L'],
                    color='green', linewidth=2)
    axes[i, 1].set_title(f'Episode {run_id} — Konsentrasi Produk (mol/L)')
    axes[i, 1].set_xlabel('Timestamp (menit)')
    axes[i, 1].set_ylabel('Product B (mol/L)')

plt.tight_layout()
plt.show()
```

**Penjelasan:**
- Plot suhu menunjukkan apakah reaktor mendekati batas aman (100°C).
- Plot konsentrasi produk menunjukkan yield yang dihasilkan.
- Perhatikan anomali seperti lonjakan suhu akibat cooling system failure.

### 5.4 Membangun Lingkungan RL Kustom

**Cell 5: Definisikan Lingkungan Reaktor**

```python
class BatchReactorEnv(gym.Env):
    """
    Lingkungan RL kustom untuk kontrol reaktor batch.

    State: [Reactor_Temp_C, Pressure_atm, Reactant_A_Conc_mol_L,
            Product_B_Conc_mol_L, Jacket_Flow_Rate_L_min]
    Action: 0 = Turunkan coolant, 1 = Pertahankan, 2 = Naikkan coolant
    Reward: +1 jika suhu < 100°C dan produk > threshold, -10 jika overheating
    """

    def __init__(self, data, max_steps=100):
        super().__init__()

        # Data reaktor (satu episode)
        self.data = data.reset_index(drop=True)
        self.max_steps = min(max_steps, len(self.data) - 1)

        # Definisi state space (5 dimensi: suhu, tekanan, reaktan, produk, coolant)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([200, 10, 5, 5, 50], dtype=np.float32),
            dtype=np.float32
        )

        # Definisi action space (3 action diskrit)
        self.action_space = spaces.Discrete(3)

        # Inisialisasi
        self.current_step = 0
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # State awal dari data
        row = self.data.iloc[0]
        self.state = np.array([
            row['Reactor_Temp_C'],
            row['Pressure_atm'],
            row['Reactant_A_Conc_mol_L'],
            row['Product_B_Conc_mol_L'],
            row['Jacket_Flow_Rate_L_min']
        ], dtype=np.float32)

        return self.state, {}

    def step(self, action):
        # Simulasi efek action pada state
        # Action: 0 = turunkan coolant, 1 = pertahankan, 2 = naikkan coolant
        action_effect = {0: -2.0, 1: 0.0, 2: +2.0}

        self.current_step += 1

        # Ambil data berikutnya (jika tersedia)
        if self.current_step < len(self.data):
            row = self.data.iloc[self.current_step]
            next_temp = row['Reactor_Temp_C'] + action_effect[action]
            next_pressure = row['Pressure_atm']
            next_reactant = row['Reactant_A_Conc_mol_L']
            next_product = row['Product_B_Conc_mol_L']
            next_coolant = row['Jacket_Flow_Rate_L_min'] + action_effect[action]
        else:
            # Jika data habis, pertahankan state
            next_temp = self.state[0] + action_effect[action]
            next_pressure = self.state[1]
            next_reactant = self.state[2]
            next_product = self.state[3]
            next_coolant = self.state[4] + action_effect[action]

        # Batasi nilai
        next_temp = np.clip(next_temp, 0, 200)
        next_coolant = np.clip(next_coolant, 0, 50)

        # State berikutnya
        next_state = np.array([
            next_temp, next_pressure, next_reactant,
            next_product, next_coolant
        ], dtype=np.float32)

        # Hitung reward
        reward = 0

        # Reward untuk suhu aman
        if next_temp < 100:
            reward += 1.0
        else:
            reward -= 10.0  # Penalti overheating

        # Reward untuk yield produk tinggi
        if next_product > 0.5:
            reward += 2.0
        else:
            reward += 0.5

        # Penalti jika coolant berlebihan (pemborosan)
        if next_coolant > 30:
            reward -= 1.0

        # Cek terminal
        terminated = (next_temp > 120) or (self.current_step >= self.max_steps)
        truncated = False

        self.state = next_state

        return next_state, reward, terminated, truncated, {}
```

**Penjelasan:**
- **State:** 5 dimensi — suhu, tekanan, konsentrasi reaktan, konsentrasi produk, dan coolant flow.
- **Action:** 3 action diskrit — turunkan, pertahankan, atau naikkan coolant.
- **Reward:** +1 untuk suhu aman, +2 untuk yield tinggi, -10 untuk overheating.
- **Terminal:** Episode berakhir jika suhu > 120°C atau mencapai max steps.

### 5.5 Melatih Agen Q-Learning

**Cell 6: Inisialisasi Q-Table dan Hyperparameter**

```python
# Ambil satu episode untuk training
episode_data = df[df['Reactor_Run_ID'] == df['Reactor_Run_ID'].unique()[0]]

env = BatchReactorEnv(episode_data, max_steps=50)

# Hyperparameter
n_states = 5  # State direpresentasikan sebagai vektor 5-dimensi
n_actions = env.action_space.n
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005
n_episodes = 500

# Q-table: gunakan diskretisasi state
# Diskretisasi: suhu dibagi 20 bin, tekanan 5 bin, dll.
n_bins = [20, 5, 5, 5, 10]  # bin untuk setiap dimensi state
Q_table = np.zeros((np.prod(n_bins), n_actions))

print("=" * 50)
print("HYPERPARAMETER Q-LEARNING")
print("=" * 50)
print(f"Jumlah Action: {n_actions}")
print(f"Learning Rate (α): {learning_rate}")
print(f"Discount Factor (γ): {discount_factor}")
print(f"Epsilon awal: {epsilon}")
print(f"Epsilon minimum: {min_epsilon}")
print(f"Decay rate: {decay_rate}")
print(f"Jumlah Episode: {n_episodes}")
print(f"Shape Q-table: {Q_table.shape}")
```

**Cell 7: Fungsi Diskretisasi State**

```python
def discretize_state(state, n_bins):
    """
    Diskretisasi state kontinu menjadi index integer.
    """
    # Batas untuk setiap dimensi
    bounds = [
        (0, 200),    # Suhu
        (0, 10),     # Tekanan
        (0, 5),      # Reaktan
        (0, 5),      # Produk
        (0, 50)      # Coolant
    ]

    indices = []
    for i, (val, (low, high), n_bin) in enumerate(zip(state, bounds, n_bins)):
        # Normalisasi ke [0, 1]
        normalized = (val - low) / (high - low + 1e-8)
        normalized = np.clip(normalized, 0, 1)
        # Konversi ke bin index
        bin_idx = int(normalized * (n_bin - 1))
        indices.append(bin_idx)

    # Konversi multi-dimensional index ke flat index
    flat_idx = np.ravel_multi_index(indices, n_bins)
    return flat_idx
```

**Cell 8: Training Loop Q-Learning**

```python
# Training
rewards_per_episode = []
epsilon_history = []

print("Memulai training Q-Learning...")
print("=" * 50)

for episode in range(n_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < 50:
        # Diskretisasi state
        state_idx = discretize_state(state, n_bins)

        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[state_idx, :])

        # Ambil action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Diskretisasi next state
        next_state_idx = discretize_state(next_state, n_bins)

        # Update Q-table
        Q_table[state_idx, action] += learning_rate * (
            reward + discount_factor * np.max(Q_table[next_state_idx, :])
            - Q_table[state_idx, action]
        )

        state = next_state
        total_reward += reward
        steps += 1

    # Simpan metrik
    rewards_per_episode.append(total_reward)
    epsilon_history.append(epsilon)

    # Decay epsilon
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(rewards_per_episode[-100:])
        print(f"Episode {episode+1:4d}/{n_episodes} | "
              f"Avg Reward: {avg_reward:.2f} | Epsilon: {epsilon:.4f}")

env.close()
print("\nTraining selesai!")
```

### 5.6 Visualisasi Hasil Training

**Cell 9: Plot Kurva Pembelajaran**

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Reward per Episode
window = 50
rewards_smooth = np.convolve(rewards_per_episode, np.ones(window)/window, mode='valid')
axes[0].plot(rewards_smooth, color='steelblue', linewidth=2)
axes[0].set_xlabel('Episode', fontsize=12)
axes[0].set_ylabel('Average Reward', fontsize=12)
axes[0].set_title('Kurva Pembelajaran Q-Learning', fontsize=14)
axes[0].grid(True, alpha=0.3)

# Plot 2: Epsilon Decay
axes[1].plot(epsilon_history, color='purple', linewidth=2)
axes[1].set_xlabel('Episode', fontsize=12)
axes[1].set_ylabel('Epsilon', fontsize=12)
axes[1].set_title('Epsilon Decay (Eksplorasi → Eksploitasi)', fontsize=14)
axes[1].grid(True, alpha=0.3)

# Plot 3: Distribusi Reward
axes[2].hist(rewards_per_episode, bins=30, color='coral', edgecolor='black', alpha=0.7)
axes[2].set_xlabel('Total Reward per Episode', fontsize=12)
axes[2].set_ylabel('Frekuensi', fontsize=12)
axes[2].set_title('Distribusi Reward', fontsize=14)

plt.tight_layout()
plt.show()
```

### 5.7 Evaluasi Agen Terlatih

**Cell 10: Uji Agen di Lingkungan**

```python
# Evaluasi 50 episode
n_eval = 50
eval_rewards = []

for _ in range(n_eval):
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < 50:
        state_idx = discretize_state(state, n_bins)
        action = np.argmax(Q_table[state_idx, :])  # Eksploitasi penuh
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
        steps += 1

    eval_rewards.append(total_reward)

print("=" * 50)
print("HASIL EVALUASI AGEN TERLATIH")
print("=" * 50)
print(f"Rata-rata Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
print(f"Reward Minimum: {np.min(eval_rewards):.2f}")
print(f"Reward Maksimum: {np.max(eval_rewards):.2f}")
print(f"Rata-rata Reward per Step: {np.mean(eval_rewards)/50:.4f}")
```

### 5.8 Analisis Policy yang Dipelajari

**Cell 11: Visualisasi Q-Table**

```python
# Visualisasi Q-values untuk beberapa state
# Ambil 10 state acak
sample_states = np.random.choice(len(Q_table), 10, replace=False)

print("Q-Values untuk 10 State Acak:")
print(f"{'State':<10} {'Action 0':<12} {'Action 1':<12} {'Action 2':<12} {'Best':<8}")
print("-" * 60)

action_names = ['Turunkan', 'Pertahankan', 'Naikkan']
for state_idx in sample_states:
    q_vals = Q_table[state_idx, :]
    best_action = action_names[np.argmax(q_vals)]
    print(f"{state_idx:<10} {q_vals[0]:<12.4f} {q_vals[1]:<12.4f} {q_vals[2]:<12.4f} {best_action:<8}")

# Heatmap Q-table
plt.figure(figsize=(10, 6))
plt.imshow(Q_table[:100, :].T, aspect='auto', cmap='viridis')
plt.colorbar(label='Q-value')
plt.xlabel('State Index', fontsize=12)
plt.ylabel('Action', fontsize=12)
plt.title('Heatmap Q-Table (100 State Pertama)', fontsize=14)
plt.yticks([0, 1, 2], ['Turunkan', 'Pertahankan', 'Naikkan'])
plt.tight_layout()
plt.show()
```

### 5.9 Ringkasan Hasil

**Cell 12: Ringkasan**

```python
print("=" * 70)
print("RINGKASAN HASIL PRAKTIKUM REINFORCEMENT LEARNING — REAKTOR BATCH")
print("=" * 70)

print(f"""
1. DATASET
   - Sumber: Kaggle — Batch Reactor Anomaly Data (Sample)
   - Jumlah data: {len(df)} baris
   - Jumlah episode: {df['Reactor_Run_ID'].nunique()}
   - Fitur: Suhu, Tekanan, Reaktan, Produk, Coolant Flow

2. LINGKUNGAN RL KUSTOM
   - State: 5 dimensi (suhu, tekanan, reaktan, produk, coolant)
   - Action: 3 (turunkan, pertahankan, naikkan coolant)
   - Reward: +1 suhu aman, +2 produk tinggi, -10 overheating
   - Terminal: suhu > 120°C atau max steps

3. HASIL TRAINING (Q-Learning)
   - Jumlah episode: {n_episodes}
   - Rata-rata reward 100 episode terakhir: {np.mean(rewards_per_episode[-100:]):.2f}
   - Epsilon final: {epsilon:.4f}

4. HASIL EVALUASI
   - Rata-rata reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}
   - Agen berhasil mempelajari policy untuk menjaga suhu aman
   - Agen memprioritaskan action yang memaksimalkan yield produk

5. REKOMENDASI
   - Gunakan Q-Learning sebagai baseline untuk kontrol reaktor
   - Kembangkan ke Deep Q-Network (DQN) untuk state kontinu
   - Integrasikan dengan sistem SCADA untuk deployment
""")
```


## 6. TUGAS MANDIRI (Dikerjakan Hari Ini)

**Tujuan:** Memastikan setiap mahasiswa memahami alur RL untuk kontrol proses industri secara hands-on.

**Instruksi:**

1. **Eksperimen Learning Rate (α):**
   - Latih Q-Learning dengan α = 0.01, 0.1, 0.5.
   - Catat rata-rata reward evaluasi untuk setiap nilai.
   - Buat tabel dan grafik perbandingan.
   - Jelaskan nilai α mana yang optimal dan mengapa.

2. **Eksperimen Discount Factor (γ):**
   - Latih Q-Learning dengan γ = 0.5, 0.9, 0.99.
   - Catat rata-rata reward evaluasi untuk setiap nilai.
   - Jelaskan mengapa γ mempengaruhi performa agen.

3. **Eksperimen Epsilon Decay:**
   - Ubah `decay_rate` menjadi 0.001, 0.005, 0.01.
   - Plot kurva epsilon untuk ketiga nilai tersebut.
   - Jelaskan efek decay rate terhadap kecepatan pembelajaran.

4. **Modifikasi Lingkungan:**
   - Tambahkan action ke-4: **"Emergency Shutdown"** (matikan reaktor).
   - Berikan reward -5 untuk emergency shutdown (biaya tinggi).
   - Latih ulang agen dan analisis apakah agen mempelajari kapan harus shutdown.

5. **Pertanyaan Konsep:**
   - Apa perbedaan fundamental antara RL dan supervised learning?
   - Jelaskan komponen MDP: State, Action, Reward, Transition, Discount Factor.
   - Mengapa Q-Learning disebut "off-policy"? Apa artinya?
   - Apa itu epsilon-greedy? Mengapa kita perlu eksplorasi?
   - Apa keterbatasan Q-Learning? Kapan kita perlu DQN?

6. **Refleksi Pribadi (Minimal 200 kata):**
   - Apa yang Anda pelajari dari praktikum RL ini?
   - Bagian mana yang paling sulit?
   - Bagaimana Anda akan menerapkan RL di industri manufaktur?

**Pengumpulan:** Simpan notebook dengan nama `TugasMandiri3_NamaAnda_NIM.ipynb` dan kumpulkan di Google Drive sebelum pertemuan ke-4.


## 7. TUGAS KELOMPOK (Dikerjakan di Rumah)

**Tujuan:** Memperdalam pemahaman RL melalui eksplorasi lingkungan alternatif dan perbandingan algoritma.

**Pembagian Kelompok:** 4–5 mahasiswa per kelompok.

### Bagian A: Eksplorasi Lingkungan Alternatif (Bobot 25%)

Pilih **salah satu** lingkungan berikut:

1. **Adaptive PLC Automation Dataset** — https://www.kaggle.com/datasets/colabsss/adaptive-plc-automation-dataset
   - 2,77 MB, ribuan record
   - Fokus pada human-machine integration dalam adaptive manufacturing
   - Cocok untuk RL-based industrial control

2. **6G-Enabled Intelligent Manufacturing Resource Data** — https://www.kaggle.com/datasets/s3programmer/6g-enabled-intelligent-manufacturing-resource-data
   - 60,47 kB, 1.000 task records
   - Task scheduling dan resource allocation
   - Cocok untuk deep reinforcement learning

3. **MfgRL — Manufacturing Reinforcement Learning Environment** (GitHub)
   - https://github.com/torayeff/mfgrl
   - Gymnasium environment untuk dynamic job shop scheduling
   - Cocok untuk RL scheduling optimizer

**Tugas:**
- Jelaskan state space, action space, dan reward function dari lingkungan yang dipilih.
- Latih Q-Learning atau DQN pada lingkungan tersebut.
- Analisis kurva pembelajaran.
- Bandingkan hasil dengan reaktor batch.

### Bagian B: Perbandingan Hyperparameter (Bobot 40%)

1. Latih Q-Learning dengan **3 kombinasi hyperparameter** berbeda pada lingkungan reaktor batch.

| Kombinasi | α | γ | ε_decay | Avg Reward | Avg Steps |
|---|---|---|---|---|---|
| A | 0.01 | 0.9 | 0.001 | ... | ... |
| B | 0.1 | 0.99 | 0.005 | ... | ... |
| C | 0.5 | 0.99 | 0.01 | ... | ... |

2. **Analisis:** Kombinasi mana yang terbaik? Mengapa?

### Bagian C: Koneksi ke Industri Manufaktur (Bobot 35%)

1. Buat **analogi** antara lingkungan RL yang Anda pelajari dengan skenario manufaktur nyata.
2. Jelaskan bagaimana konsep RL (state, action, reward) dapat diterapkan untuk:
   - **Kontrol kualitas** di lini produksi.
   - **Predictive maintenance** mesin.
   - **Optimasi energy consumption** di pabrik.
3. Tuliskan dalam **1 halaman** (markdown di notebook).

### Bagian D: Presentasi (Bonus 10%)

Buat slide presentasi (5–7 slide) yang mencakup:
- Lingkungan yang digunakan.
- Hasil perbandingan hyperparameter.
- Koneksi ke industri manufaktur.
- Kesimpulan.


## 8. RUBRIK PENILAIAN

### 8.1 Rubrik Tugas Mandiri (Bobot 20%)

| Komponen | Bobot | Kriteria | Skor |
|---|---|---|---|
| **Eksperimen α** | 20% | 3 nilai diuji, analisis tajam | 0–100 |
| **Eksperimen γ** | 20% | 3 nilai diuji, plot lengkap | 0–100 |
| **Eksperimen Decay** | 15% | 3 nilai diuji, interpretasi benar | 0–100 |
| **Modifikasi Lingkungan** | 25% | Action tambahan diimplementasikan, analisis | 0–100 |
| **Refleksi Pribadi** | 20% | Refleksi mendalam, autentik | 0–100 |

### 8.2 Rubrik Tugas Kelompok (Bobot 30%)

| Komponen | Bobot | Kriteria | Skor |
|---|---|---|---|
| **Bagian A: Eksplorasi Lingkungan** | 25% | Analisis mendalam, perbandingan jelas | 0–100 |
| **Bagian B: Perbandingan Hyperparameter** | 40% | 3 kombinasi, tabel lengkap, analisis tajam | 0–100 |
| **Bagian C: Koneksi ke Industri** | 35% | Analogi kreatif, relevan | 0–100 |
| **Bagian D: Presentasi** | Bonus 10% | Slide menarik, penyampaian jelas | 0–100 |

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

1. AI Mind Teams. (2026). *Batch Reactor Anomaly Data (Sample)*. Kaggle. https://www.kaggle.com/datasets/aimindteams/batch-reactor-anomaly-data-sample
2. Colabsss. (2026). *Adaptive PLC Automation Dataset*. Kaggle. https://www.kaggle.com/datasets/colabsss/adaptive-plc-automation-dataset
3. S3Programmer. (2024). *6G-Enabled Intelligent Manufacturing Resource Data*. Kaggle. https://www.kaggle.com/datasets/s3programmer/6g-enabled-intelligent-manufacturing-resource-data
4. Torayeff. (2023). *MfgRL: Manufacturing Reinforcement Learning Environment*. GitHub. https://github.com/torayeff/mfgrl

### 9.2 Buku dan Artikel

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
2. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media. — Bab 18: Reinforcement Learning.
3. Q-Learning-Based Multivariate Nonlinear Model Predictive Controller: Experimental Validation on Batch Reactor for Temperature Trajectory Tracking. (2025). *PMC*.
4. Towards Non-defective Injection Molding: Reinforcement Learning with an Anomaly Detection–based Environment. (2026). *Springer*.

### 9.3 Dokumentasi Online

1. **Gymnasium Documentation:** https://gymnasium.farama.org/
2. **Stable-Baselines3:** https://stable-baselines3.readthedocs.io/
3. **PyTorch DQN Tutorial:** https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


## PENUTUP

Praktikum ini memberikan pengalaman langsung dalam menerapkan **Reinforcement Learning** — paradigma ketiga Machine Learning yang melengkapi supervised dan unsupervised learning. Berbeda dengan dua paradigma sebelumnya, RL belajar melalui **interaksi langsung** dengan lingkungan, bukan dari dataset statis.

**Poin-poin kunci:**

1. **RL adalah paradigma trial-and-error** — agen belajar dari pengalaman.
2. **Q-Learning** adalah algoritma RL fundamental yang mempelajari Q-value.
3. **Eksplorasi vs Eksploitasi** adalah dilema fundamental dalam RL.
4. **Lingkungan RL kustom** dapat dibangun dari data time-series industri.
5. **RL memiliki banyak aplikasi** di manufaktur, termasuk kontrol proses, predictive maintenance, dan optimasi energi.

**Pesan untuk Mahasiswa:**
> "Reinforcement Learning adalah tentang belajar dari pengalaman. Sama seperti operator reaktor yang belajar mengontrol suhu melalui trial and error, agen RL menjadi lebih baik melalui latihan. Dataset yang digunakan dalam praktikum ini ringan (5.000 baris, ~500 KB) sehingga dapat diolah pada laptop dengan RAM 8GB atau 16GB tanpa kendala."

**Pesan untuk Dosen:**
> "Dataset *Batch Reactor Anomaly Data (Sample)* dipilih karena dirancang khusus untuk RL dan PdM, dengan episode dan action yang jelas. Untuk kelas dengan waktu terbatas, fokus pada Q-Learning dan lingkungan reaktor batch. Untuk kelas yang lebih mahir, tambahkan DQN, eksperimen dengan lingkungan alternatif, dan deployment ke sistem SCADA."

---

**Dokumen ini disusun untuk keperluan edukasi di Politeknik Manufaktur Bandung (Polman Bandung).**
**Versi:** 1.0 | **Minggu ke-2, Pertemuan ke-3 dari 4**
**Terakhir Diperbarui:** 2025
