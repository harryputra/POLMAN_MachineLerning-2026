# MODUL AJAR TEORI PENUNJANG
## Reinforcement Learning: Fondasi Teori, Algoritma, dan Aplikasi di Industri Manufaktur

**Mata Kuliah:** Machine Learning
**Program Studi:** D4 Teknologi Rekayasa Informatika dan Komputer (TRIN)
**Institusi:** Politeknik Manufaktur Bandung (Polman Bandung)
**Pertemuan:** Minggu ke-2, Pertemuan ke-3 dari 4
**Topik:** Reinforcement Learning — Teori, Algoritma, dan Aplikasi
**Durasi:** 150 menit (teori + diskusi)
**Prasyarat:** Pertemuan 1 (Supervised Learning) & Pertemuan 2 (Unsupervised Learning)


## DAFTAR ISI

1. [Pendahuluan](#1-pendahuluan)
2. [Reinforcement Learning: Paradigma Ketiga](#2-reinforcement-learning-paradigma-ketiga)
3. [Perbandingan Tiga Paradigma ML](#3-perbandingan-tiga-paradigma-ml)
4. [Komponen Utama Reinforcement Learning](#4-komponen-utama-reinforcement-learning)
5. [Markov Decision Process (MDP)](#5-markov-decision-process-mdp)
6. [Value Function dan Bellman Equation](#6-value-function-dan-bellman-equation)
7. [Dynamic Programming](#7-dynamic-programming)
8. [Monte Carlo Methods](#8-monte-carlo-methods)
9. [Temporal Difference Learning](#9-temporal-difference-learning)
10. [Q-Learning: Algoritma Fundamental](#10-q-learning-algoritma-fundamental)
11. [SARSA: On-Policy Alternative](#11-sarsa-on-policy-alternative)
12. [Deep Q-Network (DQN)](#12-deep-q-network-dqn)
13. [Policy Gradient Methods](#13-policy-gradient-methods)
14. [Exploration vs Exploitation](#14-exploration-vs-exploitation)
15. [Multi-Armed Bandit](#15-multi-armed-bandit)
16. [Aplikasi RL di Dunia Nyata](#16-aplikasi-rl-di-dunia-nyata)
17. [Aplikasi RL di Industri Manufaktur](#17-aplikasi-rl-di-industri-manufaktur)
18. [Tantangan dan Keterbatasan RL](#18-tantangan-dan-keterbatasan-rl)
19. [Etika dan Keamanan dalam RL](#19-etika-dan-keamanan-dalam-rl)
20. [Latihan dan Soal Refleksi](#20-latihan-dan-soal-refleksi)
21. [Glosarium](#21-glosarium)
22. [Referensi](#22-referensi)


## 1. PENDAHULUAN

### 1.1 Konteks Minggu ke-2

Minggu ke-2 merupakan **minggu tematik Machine Learning** yang terdiri dari 4 pertemuan:

| Pertemuan | Topik | Fokus |
|---|---|---|
| Pertemuan 1 | Supervised Learning | Belajar dari data berlabel |
| Pertemuan 2 | Unsupervised Learning | Menemukan struktur tanpa label |
| **Pertemuan 3** | **Reinforcement Learning** | **Belajar dari interaksi dan reward** |
| Pertemuan 4 | Evaluasi Mingguan | Presentasi + Teori |

### 1.2 Mengapa Reinforcement Learning Penting?

Selama dua pertemuan sebelumnya, kita telah mempelajari dua paradigma Machine Learning yang belajar dari **dataset statis**:

- **Supervised Learning** — belajar dari pasangan (input, output) yang sudah ada.
- **Unsupervised Learning** — belajar dari data tanpa label untuk menemukan struktur.

Namun, ada banyak masalah di dunia nyata yang **tidak bisa diselesaikan dengan dataset statis**:

- **Robot yang harus belajar berjalan** — tidak ada dataset "cara berjalan yang benar".
- **AI yang harus mengalahkan grandmaster Go** — tidak ada buku panduan strategi optimal.
- **Sistem kontrol reaktor batch** — keputusan hari ini mempengaruhi keselamatan besok.
- **Mobil otonom** — harus mengambil keputusan real-time di lingkungan dinamis.

Semua masalah ini memiliki karakteristik yang sama: **agen harus belajar melalui interaksi langsung dengan lingkungan**, menerima umpan balik, dan secara bertahap mempelajari strategi optimal. Inilah domain **Reinforcement Learning (RL)** .

### 1.3 Sejarah Singkat RL

| Era | Perkembangan |
|---|---|
| **1950-an** | Richard Bellman mengembangkan Dynamic Programming dan Bellman Equation |
| **1959** | Arthur Samuel mengembangkan program checkers yang belajar dari pengalaman |
| **1970-an** | Pengembangan teori MDP dan Temporal Difference Learning |
| **1989** | Chris Watkins mengembangkan Q-Learning |
| **1992** | Gerald Tesauro mengembangkan TD-Gammon (backgammon) |
| **2013** | DeepMind mengembangkan DQN untuk Atari games |
| **2016** | AlphaGo mengalahkan Lee Sedol |
| **2017** | AlphaZero mengalahkan semua engine catur dan Go |
| **2019** | OpenAI Five mengalahkan tim Dota 2 profesional |
| **2022** | ChatGPT menggunakan RLHF (Reinforcement Learning from Human Feedback) |

### 1.4 Peta Konsep Modul Ini

```
Reinforcement Learning
│
├── Konsep Dasar
│   ├── Agent, Environment, State, Action, Reward
│   ├── Policy, Value Function, Q-Value
│   └── Markov Decision Process (MDP)
│
├── Algoritma Klasik
│   ├── Dynamic Programming (Policy Iteration, Value Iteration)
│   ├── Monte Carlo Methods
│   ├── Temporal Difference Learning
│   ├── Q-Learning
│   └── SARSA
│
├── Algoritma Modern
│   ├── Deep Q-Network (DQN)
│   ├── Policy Gradient (REINFORCE)
│   ├── Actor-Critic (A2C, A3C)
│   └── PPO, TRPO, SAC
│
├── Eksplorasi vs Eksploitasi
│   ├── Epsilon-Greedy
│   ├── Softmax
│   ├── UCB (Upper Confidence Bound)
│   └── Thompson Sampling
│
└── Aplikasi
    ├── Game (AlphaGo, Dota 2)
    ├── Robotika
    ├── Manufaktur
    ├── Kesehatan
    └── Keuangan
```


## 2. REINFORCEMENT LEARNING: PARADIGMA KETIGA

### 2.1 Definisi Formal

**Reinforcement Learning (RL)** adalah paradigma Machine Learning di mana **agen** (agent) belajar untuk mengambil **tindakan** (action) dalam **lingkungan** (environment) untuk memaksimalkan **reward kumulatif jangka panjang**.

**Definisi dari Sutton & Barto (2018):**
> "Reinforcement learning is learning what to do—how to map situations to actions—so as to maximize a numerical reward signal. The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them."

**Terjemahan:**
> "Reinforcement learning adalah belajar apa yang harus dilakukan—bagaimana memetakan situasi ke tindakan—untuk memaksimalkan sinyal reward numerik. Pelajar tidak diberitahu tindakan mana yang harus diambil, melainkan harus menemukan sendiri tindakan mana yang menghasilkan reward terbanyak dengan mencobanya."

### 2.2 Karakteristik Unik RL

| Karakteristik | Penjelasan |
|---|---|
| **Trial and Error** | Agen belajar melalui percobaan dan kesalahan |
| **Delayed Reward** | Reward mungkin baru diterima setelah banyak langkah |
| **Sequential Decision** | Keputusan saat ini mempengaruhi keadaan masa depan |
| **No Supervisor** | Tidak ada yang memberi tahu action yang benar |
| **Exploration Needed** | Agen harus mengeksplorasi untuk menemukan strategi optimal |
| **Non-stationary** | Lingkungan bisa berubah seiring waktu |

### 2.3 Contoh Sederhana

**Skenario:** Seorang operator reaktor batch harus mengontrol suhu reaktor.

- **Agent:** Sistem kontrol otomatis
- **Environment:** Reaktor batch dengan sensor suhu, tekanan, dan konsentrasi
- **State:** Suhu, tekanan, konsentrasi reaktan dan produk
- **Action:** Naikkan, pertahankan, atau turunkan aliran coolant
- **Reward:** +1 untuk suhu aman, +2 untuk yield tinggi, -10 untuk overheating
- **Policy:** Strategi "kapan harus menambah coolant berdasarkan kondisi reaktor"

**Tantangan:** Sistem tidak tahu di awal action mana yang optimal. Ia harus mencoba berbagai strategi, menerima feedback (reward), dan secara bertahap mempelajari strategi terbaik.

### 2.4 Perbedaan dengan Supervised Learning

| Aspek | Supervised Learning | Reinforcement Learning |
|---|---|---|
| **Data** | $(x, y)$ berlabel | Interaksi $(s, a, r, s')$ |
| **Feedback** | Langsung dan benar | Delayed dan mungkin noisy |
| **Tujuan** | Prediksi output | Maksimalkan reward kumulatif |
| **Contoh** | Prediksi defect produk | Kontrol reaktor batch |
| **Analogi** | Belajar dengan kunci jawaban | Belajar dari pengalaman |
| **Kesalahan** | Langsung dikoreksi | Mungkin baru terasa nanti |

**Contoh Perbedaan:**

- **Supervised:** "Ini data sensor. Prediksi apakah produk defect?" → "Defect." → "Benar!"
- **RL:** "Suhu reaktor 85°C. Mau naikkan coolant?" → "Ya." → (5 langkah kemudian) "Suhu stabil." → "Bagus, ulangi."

### 2.5 Perbedaan dengan Unsupervised Learning

| Aspek | Unsupervised Learning | Reinforcement Learning |
|---|---|---|
| **Data** | $(x)$ tanpa label | Interaksi $(s, a, r, s')$ |
| **Tujuan** | Temukan struktur | Maksimalkan reward |
| **Feedback** | Tidak ada | Reward |
| **Contoh** | Clustering state mesin | Agen mengontrol reaktor |
| **Output** | Cluster, representasi | Policy, value function |

### 2.6 Kapan Menggunakan RL?

Gunakan RL ketika:

1. **Tidak ada dataset berlabel** dan tidak mungkin mengumpulkannya.
2. **Keputusan bersifat sequential** — keputusan saat ini mempengaruhi masa depan.
3. **Ada lingkungan yang bisa disimulasikan** untuk trial and error.
4. **Reward bisa didefinisikan** secara numerik.
5. **Tujuan adalah memaksimalkan** metrik jangka panjang.

**Contoh Kasus:**
- Kontrol proses industri (reaktor batch)
- Robot navigasi dan manipulator
- Sistem rekomendasi adaptif
- Manajemen energi pabrik
- Predictive maintenance


## 3. PERBANDINGAN TIGA PARADIGMA ML

### 3.1 Tabel Perbandingan Lengkap

| Aspek | Supervised | Unsupervised | Reinforcement |
|---|---|---|---|
| **Input** | $(x, y)$ | $(x)$ | $(s, a, r, s')$ |
| **Supervisor** | Ada (label) | Tidak ada | Tidak ada (hanya reward) |
| **Feedback** | Langsung | Tidak ada | Delayed |
| **Tujuan** | Prediksi | Struktur | Maksimalkan reward |
| **Output** | Fungsi $f: X \to Y$ | Cluster/representasi | Policy $\pi: S \to A$ |
| **Evaluasi** | Accuracy, F1 | Silhouette | Cumulative reward |
| **Contoh Manufaktur** | Prediksi Defect | Segmentasi State | Kontrol Proses |
| **Algoritma** | DT, KNN, RF | K-Means, DBSCAN | Q-Learning, DQN |
| **Data** | Statis | Statis | Dinamis (interaksi) |
| **Analogi** | Belajar dengan guru | Menjelajah sendiri | Belajar dari pengalaman |

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

REINFORCEMENT LEARNING (Pertemuan 3):
┌─────────────────────────────────────────────────┐
│  Interaksi dengan Lingkungan (Batch Reactor)    │
│                                                 │
│  ┌─────────┐  action   ┌─────────────┐         │
│  │  Agent  │ ────────► │ Environment │         │
│  │         │ ◄──────── │  (Reactor)  │         │
│  └─────────┘  state,   └─────────────┘         │
│               reward                            │
│                                                 │
│  Model belajar: π(s) → a                        │
│  Feedback: Reward (mungkin delayed)             │
│  → Agen belajar kapan harus menambah coolant    │
│  → Agen belajar menghindari overheating         │
└─────────────────────────────────────────────────┘
```

### 3.3 Kapan Menggunakan Masing-Masing?

```
Apakah Anda memiliki data berlabel?
│
├── YA → Supervised Learning
│   ├── Output kategorikal? → Klasifikasi (Prediksi Defect)
│   └── Output numerik? → Regresi (Prediksi Kualitas)
│
└── TIDAK → Apakah Anda memiliki lingkungan untuk interaksi?
    │
    ├── YA → Reinforcement Learning
    │   ├── Action diskrit? → Q-Learning, DQN
    │   └── Action kontinu? → DDPG, SAC, PPO
    │
    └── TIDAK → Unsupervised Learning
        ├── Ingin mengelompokkan? → Clustering (State Detection)
        ├── Ingin mereduksi dimensi? → PCA, t-SNE
        └── Ingin menemukan aturan? → Association Rules
```

### 3.4 Kombinasi Paradigma

Dalam praktik nyata, ketiga paradigma sering **dikombinasikan**:

| Kombinasi | Contoh |
|---|---|
| **Supervised + RL** | DQN menggunakan neural network (supervised) untuk memprediksi Q-value |
| **Unsupervised + RL** | Autoencoder untuk representasi state, lalu RL untuk policy |
| **Supervised + Unsupervised** | Clustering untuk segmentasi, lalu klasifikasi per segmen |
| **Ketiganya** | AlphaGo: supervised (belajar dari game manusia) + RL (self-play) + unsupervised (representasi) |


## 4. KOMPONEN UTAMA REINFORCEMENT LEARNING

### 4.1 Agent (Agen)

**Definisi:** Entitas yang belajar dan mengambil keputusan.

**Karakteristik:**
- Memiliki **policy** (strategi) untuk memilih action.
- Memiliki **value function** untuk mengevaluasi state.
- Belajar dari **pengalaman** (interaksi dengan environment).

**Contoh:**
- Sistem kontrol otomatis pada reaktor batch.
- Robot yang belajar berjalan.
- AI yang bermain catur.
- Sistem rekomendasi yang belajar preferensi pengguna.

### 4.2 Environment (Lingkungan)

**Definisi:** Dunia tempat agen berinteraksi.

**Karakteristik:**
- Menerima **action** dari agen.
- Mengembalikan **state** baru dan **reward**.
- Bisa **deterministik** atau **stokastik**.
- Bisa **fully observable** atau **partially observable**.

**Contoh:**
- Papan catur (deterministik, fully observable).
- Jalan raya (stokastik, partially observable).
- Reaktor batch (stokastik, partially observable).

### 4.3 State (Keadaan)

**Definisi:** Representasi kondisi agen dan lingkungan saat ini.

**Jenis State:**

| Jenis | Deskripsi | Contoh |
|---|---|---|
| **Fully Observable** | Agen melihat seluruh state | Catur, Go |
| **Partially Observable** | Agen hanya melihat sebagian | Reaktor batch, driving |
| **Discrete** | State terbatas dan diskrit | FrozenLake (16 state) |
| **Continuous** | State kontinu | Reaktor (suhu, tekanan) |

**Pentingnya State:**
- State harus mengandung **semua informasi** yang diperlukan untuk keputusan optimal (Markov Property).
- Jika tidak, agen mungkin perlu **memori** (history).

### 4.4 Action (Tindakan)

**Definisi:** Pilihan yang bisa diambil agen.

**Jenis Action:**

| Jenis | Deskripsi | Contoh |
|---|---|---|
| **Discrete** | Action terbatas | Turunkan, Pertahankan, Naikkan coolant |
| **Continuous** | Action kontinu | Kecepatan aliran coolant (L/min) |
| **Deterministic** | Hasil action pasti | Catur |
| **Stochastic** | Hasil action probabilistik | Reaktor batch (ada gangguan) |

### 4.5 Reward (Imbalan)

**Definisi:** Sinyal numerik yang menunjukkan seberapa baik action yang diambil.

**Karakteristik:**
- **Scalar:** Berupa angka tunggal.
- **Delayed:** Mungkin baru diterima setelah banyak langkah.
- **Sparse:** Mungkin jarang diberikan.
- **Dense:** Diberikan setiap langkah.

**Jenis Reward:**

| Jenis | Deskripsi | Contoh |
|---|---|---|
| **Positive** | Reward untuk perilaku baik | +1 suhu aman |
| **Negative** | Punishment untuk perilaku buruk | -10 overheating |
| **Zero** | Tidak ada reward | Langkah biasa |
| **Sparse** | Jarang diberikan | Hanya di akhir episode |
| **Dense** | Setiap langkah | Reward shaping |

**Reward Shaping:**
Terkadang kita perlu **mendesain reward** agar agen belajar lebih cepat. Contoh:
- Memberikan reward kecil setiap langkah mendekati tujuan.
- Memberikan punishment untuk setiap langkah (agar efisien).

### 4.6 Policy (Kebijakan)

**Definisi:** Strategi agen untuk memilih action berdasarkan state.

**Notasi:** $\pi(a|s)$ = probabilitas memilih action $a$ di state $s$.

**Jenis Policy:**

| Jenis | Deskripsi | Contoh |
|---|---|---|
| **Deterministic** | Satu action untuk setiap state | $\pi(s) = a$ |
| **Stochastic** | Distribusi probabilitas action | $\pi(a\|s) = P(a\|s)$ |

**Tujuan RL:** Menemukan **optimal policy** $\pi^*$ yang memaksimalkan reward kumulatif.

### 4.7 Value Function

**Definisi:** Fungsi yang mengukur seberapa baik suatu state atau action.

**Dua Jenis Value Function:**

#### A. State Value Function $V^\pi(s)$

Expected return dari state $s$ mengikuti policy $\pi$:

$$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_{t+1} \mid s_0 = s \right]$$

#### B. Action Value Function $Q^\pi(s, a)$

Expected return dari state $s$, mengambil action $a$, lalu mengikuti policy $\pi$:

$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_{t+1} \mid s_0 = s, a_0 = a \right]$$

**Hubungan:**
$$V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s, a)$$

### 4.8 Model (Opsional)

**Definisi:** Representasi lingkungan yang memungkinkan agen **memprediksi** hasil action.

**Model-Based vs Model-Free:**

| Aspek | Model-Based | Model-Free |
|---|---|---|
| **Model** | Ada | Tidak ada |
| **Pendekatan** | Rencanakan menggunakan model | Belajar dari pengalaman |
| **Contoh** | Dynamic Programming, MCTS | Q-Learning, SARSA |
| **Kelebihan** | Sample efficient | Sederhana, general |
| **Kekurangan** | Butuh model akurat | Butuh banyak sample |

### 4.9 Return dan Discount Factor

**Return ($G_t$):** Total reward kumulatif dari waktu $t$:

$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

**Discount Factor ($\gamma$):** Menentukan seberapa penting reward masa depan.

| Nilai $\gamma$ | Interpretasi |
|---|---|
| $\gamma = 0$ | Hanya peduli reward sekarang (myopic) |
| $\gamma = 0.9$ | Cukup peduli reward masa depan |
| $\gamma = 0.99$ | Sangat peduli reward masa depan |
| $\gamma = 1$ | Peduli semua reward sama (bisa infinite) |

**Mengapa Discount Factor Penting?**
1. **Matematis:** Menjaga return tetap finite.
2. **Praktis:** Merefleksikan preferensi waktu (reward sekarang lebih berharga).
3. **Efisiensi:** Mempercepat konvergensi.


## 5. MARKOV DECISION PROCESS (MDP)

### 5.1 Definisi MDP

**Markov Decision Process (MDP)** adalah kerangka matematis formal untuk pemodelan pengambilan keputusan sequential.

**MDP didefinisikan oleh tuple $(S, A, P, R, \gamma)$:**

| Simbol | Nama | Deskripsi |
|---|---|---|
| $S$ | State Space | Himpunan semua state |
| $A$ | Action Space | Himpunan semua action |
| $P$ | Transition Probability | $P(s'\|s, a)$ = probabilitas transisi |
| $R$ | Reward Function | $R(s, a, s')$ = reward |
| $\gamma$ | Discount Factor | Bobot reward masa depan |

### 5.2 Markov Property

**Definisi:** State saat ini mengandung **semua informasi** yang diperlukan untuk memprediksi masa depan.

$$P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, ..., s_t, a_t)$$

**Artinya:** Masa lalu tidak relevan jika state saat ini sudah diketahui.

**Contoh:**
- **Catur:** State = posisi semua bidak. Masa lalu tidak perlu diketahui.
- **Reaktor batch:** State = suhu, tekanan, konsentrasi. Masa lalu mungkin relevan untuk dinamika, tetapi state saat ini sudah cukup untuk keputusan Markov.
- **FrozenLake:** State = posisi pemain. Masa lalu tidak relevan.

### 5.3 Contoh MDP Sederhana

**Reaktor Batch Sederhana:**

- **State:** Suhu reaktor (diskretisasi: Low, Medium, High)
- **Action:** Turunkan coolant, Pertahankan, Naikkan coolant
- **Transition:** Suhu berubah berdasarkan action dan dinamika reaktor
- **Reward:** +1 jika suhu aman, -10 jika overheating
- **Discount:** $\gamma = 0.95$

### 5.4 Policy dan Optimal Policy

**Policy $\pi$:** Mapping dari state ke action (atau distribusi action).

**Optimal Policy $\pi^*$:** Policy yang memaksimalkan expected return dari setiap state.

$$\pi^*(s) = \arg\max_a Q^*(s, a)$$

**Optimal Value Function:**
$$V^*(s) = \max_\pi V^\pi(s)$$
$$Q^*(s, a) = \max_\pi Q^\pi(s, a)$$

### 5.5 Bellman Optimality Equation

**Untuk $V^*$:**
$$V^*(s) = \max_a \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + \gamma V^*(s') \right]$$

**Untuk $Q^*$:**
$$Q^*(s, a) = \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + \gamma \max_{a'} Q^*(s', a') \right]$$

**Intuisi:** Nilai optimal suatu state = reward terbaik yang bisa didapat + nilai optimal state berikutnya (discounted).


## 6. VALUE FUNCTION DAN BELLMAN EQUATION

### 6.1 Bellman Equation untuk Policy Tertentu

**State Value Function:**
$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + \gamma V^\pi(s') \right]$$

**Action Value Function:**
$$Q^\pi(s, a) = \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a') \right]$$

### 6.2 Bellman Optimality Equation

**State Value Function:**
$$V^*(s) = \max_a \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + \gamma V^*(s') \right]$$

**Action Value Function:**
$$Q^*(s, a) = \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + \gamma \max_{a'} Q^*(s', a') \right]$$

### 6.3 Hubungan V dan Q

$$V^*(s) = \max_a Q^*(s, a)$$
$$Q^*(s, a) = \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + \gamma V^*(s') \right]$$

### 6.4 Contoh Perhitungan Manual

**Sederhana: Grid 2×2**

```
S (0)  →  (1)
↓         ↓
(2)   →  G (3)
```

- Reward: +1 jika mencapai G, 0 otherwise
- $\gamma = 0.9$
- Deterministik (action selalu berhasil)

**Q-Values:**
- $Q(3, \text{any}) = 0$ (terminal)
- $Q(2, \text{Kanan}) = R(2, \text{Kanan}, 3) + \gamma V(3) = 1 + 0.9 \times 0 = 1$
- $Q(1, \text{Bawah}) = R(1, \text{Bawah}, 3) + \gamma V(3) = 1 + 0 = 1$
- $Q(0, \text{Kanan}) = R(0, \text{Kanan}, 1) + \gamma V(1) = 0 + 0.9 \times 1 = 0.9$
- $Q(0, \text{Bawah}) = R(0, \text{Bawah}, 2) + \gamma V(2) = 0 + 0.9 \times 1 = 0.9$

**V-Values:**
- $V(3) = 0$ (terminal)
- $V(2) = \max(Q(2, \text{Kanan}), Q(2, \text{other})) = 1$
- $V(1) = \max(Q(1, \text{Bawah}), ...) = 1$
- $V(0) = \max(0.9, 0.9) = 0.9$

**Optimal Policy:**
- $\pi^*(0) = \text{Kanan atau Bawah}$
- $\pi^*(1) = \text{Bawah}$
- $\pi^*(2) = \text{Kanan}$


## 7. DYNAMIC PROGRAMMING

### 7.1 Konsep Dasar

**Dynamic Programming (DP)** adalah metode untuk menyelesaikan MDP **ketika model lingkungan diketahui** (transition probability dan reward function tersedia).

**Asumsi:**
- Model MDP lengkap diketahui.
- State space terbatas dan diskrit.

**Dua Pendekatan Utama:**
1. **Policy Iteration**
2. **Value Iteration**

### 7.2 Policy Evaluation

**Tujuan:** Menghitung $V^\pi(s)$ untuk policy $\pi$ tertentu.

**Algoritma:**
```
1. Inisialisasi V(s) = 0 untuk semua s
2. Ulangi hingga konvergen:
   Untuk setiap state s:
      V(s) = Σ_a π(a|s) Σ_s' P(s'|s,a) [R(s,a,s') + γ V(s')]
```

### 7.3 Policy Improvement

**Tujuan:** Memperbaiki policy berdasarkan value function.

**Teorema Policy Improvement:**
Jika kita memilih action greedy terhadap $V^\pi$:
$$\pi'(s) = \arg\max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^\pi(s')]$$

Maka $\pi'$ lebih baik atau sama dengan $\pi$.

### 7.4 Policy Iteration

**Algoritma:**
```
1. Inisialisasi policy π secara acak
2. Ulangi:
   a. Policy Evaluation: Hitung V^π
   b. Policy Improvement: Update π berdasarkan V^π
   c. Jika policy tidak berubah → konvergen
```

**Ilustrasi:**
```
π₀ → V^π₀ → π₁ → V^π₁ → π₂ → ... → π*
```

### 7.5 Value Iteration

**Algoritma:**
```
1. Inisialisasi V(s) = 0 untuk semua s
2. Ulangi hingga konvergen:
   Untuk setiap state s:
      V(s) = max_a Σ_s' P(s'|s,a) [R(s,a,s') + γ V(s')]
3. Ekstrak policy: π(s) = argmax_a Σ_s' P(s'|s,a) [R(s,a,s') + γ V(s')]
```

**Perbedaan dengan Policy Iteration:**
- Value Iteration menggabungkan evaluation dan improvement.
- Lebih cepat per iterasi, tapi mungkin butuh lebih banyak iterasi.

### 7.6 Kelebihan dan Kekurangan DP

| Kelebihan | Kekurangan |
|---|---|
| Konvergen ke solusi optimal | Butuh model MDP lengkap |
| Efisien untuk state kecil | Tidak skalabel untuk state besar |
| Matematis solid | Butuh state space diskrit |

### 7.7 Keterbatasan DP

1. **Curse of Dimensionality:** Jumlah state tumbuh eksponensial dengan dimensi.
2. **Model Harus Diketahui:** Dalam banyak masalah nyata, model tidak tersedia.
3. **Komputasi Mahal:** Setiap iterasi memerlukan sweep seluruh state space.

**Solusi:** Monte Carlo dan Temporal Difference Learning (tidak butuh model).


## 8. MONTE CARLO METHODS

### 8.1 Konsep Dasar

**Monte Carlo (MC)** methods adalah metode RL yang **tidak membutuhkan model lingkungan**. Agen belajar dari **pengalaman aktual** (episode lengkap).

**Karakteristik:**
- **Model-free:** Tidak butuh transition probability.
- **Episode-based:** Butuh episode yang lengkap.
- **Sample-based:** Belajar dari rata-rata return.

### 8.2 MC Prediction (Policy Evaluation)

**Tujuan:** Menghitung $V^\pi(s)$ menggunakan return aktual.

**Algoritma:**
```
1. Inisialisasi V(s) = 0, N(s) = 0
2. Untuk setiap episode:
   a. Generate episode menggunakan policy π
   b. Untuk setiap state s dalam episode:
      - G = return dari state s
      - N(s) += 1
      - V(s) += (G - V(s)) / N(s)  # Incremental mean
```

**First-Visit vs Every-Visit MC:**

| Jenis | Deskripsi |
|---|---|
| **First-Visit** | Hanya hitung return pertama kali state dikunjungi |
| **Every-Visit** | Hitung return setiap kali state dikunjungi |

### 8.3 MC Control (Finding Optimal Policy)

**Tujuan:** Menemukan policy optimal.

**Algoritma:**
```
1. Inisialisasi Q(s,a) = 0, N(s,a) = 0, π = random
2. Untuk setiap episode:
   a. Generate episode menggunakan π (dengan eksplorasi)
   b. Untuk setiap (s,a) dalam episode:
      - G = return dari (s,a)
      - N(s,a) += 1
      - Q(s,a) += (G - Q(s,a)) / N(s,a)
   c. Update policy: π(s) = argmax_a Q(s,a)
```

**Masalah:** Jika policy deterministik, agen mungkin tidak mengeksplorasi action lain.

**Solusi:** **Exploring Starts** atau **Epsilon-Greedy**.

### 8.4 Kelebihan dan Kekurangan MC

| Kelebihan | Kekurangan |
|---|---|
| Tidak butuh model | Butuh episode lengkap |
| Bisa menangani environment non-Markov | Varians tinggi |
| Sederhana | Tidak bisa belajar online |
| Tidak bias | Konvergensi lambat |

### 8.5 Perbandingan DP vs MC

| Aspek | DP | MC |
|---|---|---|
| **Model** | Butuh | Tidak butuh |
| **Episode** | Tidak butuh | Butuh |
| **Bootstrap** | Ya | Tidak |
| **Varians** | Rendah | Tinggi |
| **Bias** | Tidak | Tidak |
| **Komputasi** | Mahal per iterasi | Ringan per episode |


## 9. TEMPORAL DIFFERENCE LEARNING

### 9.1 Konsep Dasar

**Temporal Difference (TD) Learning** menggabungkan kelebihan DP dan MC:
- **Dari DP:** Bootstrap (update menggunakan estimasi).
- **Dari MC:** Model-free (belajar dari pengalaman).

**Ide Kunci:** Update value function menggunakan **estimasi** dari value function itu sendiri.

### 9.2 TD Prediction (TD(0))

**Update Rule:**
$$V(s_t) \leftarrow V(s_t) + \alpha [r_{t+1} + \gamma V(s_{t+1}) - V(s_t)]$$

**TD Error:**
$$\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$$

**Perbandingan dengan MC:**
- MC: $V(s_t) \leftarrow V(s_t) + \alpha [G_t - V(s_t)]$
- TD: $V(s_t) \leftarrow V(s_t) + \alpha [r_{t+1} + \gamma V(s_{t+1}) - V(s_t)]$

**Perbedaan:** MC menggunakan return aktual $G_t$; TD menggunakan estimasi $r + \gamma V(s')$.

### 9.3 TD Control: Q-Learning dan SARSA

TD Control adalah metode untuk menemukan policy optimal menggunakan TD.

**Dua Pendekatan Utama:**

| Aspek | Q-Learning | SARSA |
|---|---|---|
| **Tipe** | Off-policy | On-policy |
| **Update** | Menggunakan max Q | Menggunakan Q action aktual |
| **Formula** | $Q(s,a) + \alpha[r + \gamma \max Q(s',a') - Q(s,a)]$ | $Q(s,a) + \alpha[r + \gamma Q(s',a') - Q(s,a)]$ |
| **Eksplorasi** | Aman (belajar optimal policy) | Dipengaruhi policy |

### 9.4 TD(λ) dan Eligibility Traces

**TD(λ)** adalah generalisasi antara TD(0) dan MC:
- $\lambda = 0$ → TD(0)
- $\lambda = 1$ → MC

**Eligibility Traces:** Mekanisme untuk melacak state yang "berhak" di-update.

$$E_t(s) = \gamma \lambda E_{t-1}(s) + \mathbb{1}(s_t = s)$$

### 9.5 Kelebihan dan Kekurangan TD

| Kelebihan | Kekurangan |
|---|---|
| Tidak butuh model | Bias (karena bootstrap) |
| Bisa belajar online | Sensitif terhadap learning rate |
| Konvergensi lebih cepat dari MC | Butuh tuning hyperparameter |
| Bisa menangani episode panjang | — |


## 10. Q-LEARNING: ALGORITMA FUNDAMENTAL

### 10.1 Konsep Dasar

**Q-Learning** adalah algoritma **off-policy** yang mempelajari **optimal action value function** $Q^*(s, a)$ secara langsung, terlepas dari policy yang diikuti.

**Dikembangkan oleh:** Chris Watkins (1989).

### 10.2 Update Rule

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]$$

**Komponen:**
- $Q(s_t, a_t)$ = nilai saat ini
- $\alpha$ = learning rate
- $r_{t+1}$ = reward
- $\gamma$ = discount factor
- $\max_{a'} Q(s_{t+1}, a')$ = estimasi nilai optimal state berikutnya

### 10.3 Mengapa Off-Policy?

**Off-policy** berarti Q-Learning belajar tentang **optimal policy** ($\pi^*$) sambil mengikuti **behavior policy** ($\pi_b$, misal epsilon-greedy).

**Keuntungan:**
- Bisa belajar dari pengalaman orang lain (replay buffer).
- Bisa belajar sambil mengeksplorasi.
- Konvergen ke optimal policy meskipun behavior policy suboptimal.

### 10.4 Algoritma Q-Learning

```
1. Inisialisasi Q(s,a) = 0 untuk semua s, a
2. Untuk setiap episode:
   a. Inisialisasi state s
   b. Untuk setiap langkah:
      i.   Pilih action a dari s menggunakan policy (epsilon-greedy)
      ii.  Ambil action a, amati reward r dan state s'
      iii. Update: Q(s,a) += α [r + γ max_a' Q(s',a') - Q(s,a)]
      iv.  s = s'
   c. Ulangi hingga terminal
```

### 10.5 Contoh Perhitungan Manual

**FrozenLake 2×2 (sederhana):**

```
S (0)  →  (1)
↓         ↓
(2)   →  G (3)
```

- $\alpha = 0.1$, $\gamma = 0.9$
- Semua Q awal = 0

**Episode 1:**
- State 0, action Kanan → state 1, reward 0
  - $Q(0, \text{Kanan}) = 0 + 0.1[0 + 0.9 \times 0 - 0] = 0$
- State 1, action Bawah → state 3, reward 1
  - $Q(1, \text{Bawah}) = 0 + 0.1[1 + 0.9 \times 0 - 0] = 0.1$

**Episode 2:**
- State 0, action Kanan → state 1, reward 0
  - $Q(0, \text{Kanan}) = 0 + 0.1[0 + 0.9 \times 0.1 - 0] = 0.009$
- State 1, action Bawah → state 3, reward 1
  - $Q(1, \text{Bawah}) = 0.1 + 0.1[1 + 0.9 \times 0 - 0.1] = 0.19$

...dan seterusnya hingga konvergen.

### 10.6 Hyperparameter Q-Learning

| Parameter | Simbol | Range | Efek |
|---|---|---|---|
| **Learning Rate** | $\alpha$ | 0.01–0.5 | Besar = cepat tapi tidak stabil |
| **Discount Factor** | $\gamma$ | 0.9–0.99 | Besar = peduli masa depan |
| **Epsilon** | $\epsilon$ | 1.0→0.01 | Besar = eksplorasi |
| **Epsilon Decay** | — | 0.001–0.01 | Besar = cepat eksploitasi |
| **Episode** | — | 1000–100000 | Banyak = lebih baik |

### 10.7 Kelebihan dan Kekurangan Q-Learning

| Kelebihan | Kekurangan |
|---|---|
| Model-free | Tidak skalabel untuk state besar |
| Off-policy (bisa pakai replay) | Tidak bisa generalisasi |
| Konvergen ke optimal | Butuh banyak episode |
| Sederhana | Tidak cocok untuk action kontinu |

### 10.8 Kapan Menggunakan Q-Learning?

✅ **Gunakan ketika:**
- State space kecil dan diskrit.
- Action space kecil dan diskrit.
- Butuh baseline sederhana.
- Tidak butuh generalisasi.

❌ **Hindari ketika:**
- State space besar (gunakan DQN).
- Action space kontinu (gunakan DDPG, SAC).
- Butuh generalisasi (gunakan neural network).


## 11. SARSA: ON-POLICY ALTERNATIVE

### 11.1 Konsep Dasar

**SARSA** (State-Action-Reward-State-Action) adalah algoritma **on-policy** yang belajar dari action yang **benar-benar diambil**.

**Nama:** Mengacu pada urutan $(s, a, r, s', a')$.

### 11.2 Update Rule

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]$$

**Perbedaan dengan Q-Learning:**
- Q-Learning: menggunakan $\max_{a'} Q(s', a')$ (action optimal).
- SARSA: menggunakan $Q(s', a')$ (action aktual yang diambil).

### 11.3 Perbandingan Q-Learning vs SARSA

| Aspek | Q-Learning | SARSA |
|---|---|---|
| **Tipe** | Off-policy | On-policy |
| **Update** | $\max_{a'} Q(s', a')$ | $Q(s', a')$ |
| **Behavior** | Belajar optimal policy | Belajar policy yang diikuti |
| **Keamanan** | Kurang aman (bisa eksplorasi berbahaya) | Lebih aman |
| **Contoh** | Cliff Walking: ambil risiko | Cliff Walking: hindari risiko |

### 11.4 Ilustrasi Cliff Walking

**Environment:** Grid dengan tebing (cliff) di bawah.

```
┌─────────────────────────┐
│ S . . . . . . . . . . G │
│ . . . . . . . . . . . . │
│ . . . . . . . . . . . . │
│ C C C C C C C C C C C C │  ← Cliff (reward -100)
└─────────────────────────┘
```

- **Q-Learning:** Akan mengambil jalur **optimal** (dekat tebing) karena belajar optimal policy.
- **SARSA:** Akan mengambil jalur **aman** (jauh dari tebing) karena memperhitungkan eksplorasi.

### 11.5 Kelebihan dan Kekurangan SARSA

| Kelebihan | Kekurangan |
|---|---|
| Lebih aman (memperhitungkan eksplorasi) | Konvergen ke policy yang diikuti |
| Cocok untuk aplikasi kritis | Tidak belajar optimal policy jika eksplorasi tinggi |
| On-policy | — |


## 12. DEEP Q-NETWORK (DQN)

### 12.1 Keterbatasan Q-Learning

Q-Learning menggunakan **Q-table** yang:
- Tidak skalabel untuk state besar (misal gambar 84×84 = 7056 dimensi).
- Tidak bisa generalisasi ke state yang belum dilihat.
- Tidak cocok untuk action kontinu.

**Solusi:** **Deep Q-Network (DQN)** — menggunakan neural network untuk memprediksi Q-value.

### 12.2 Arsitektur DQN

```
Input: State (misal sensor readings)
   │
   ▼
┌─────────────────┐
│  FC Layer 1     │
│  FC Layer 2     │
│  FC Layer 3     │
└─────────────────┘
   │
   ▼
Output: Q-value untuk setiap action
```

**Dikembangkan oleh:** DeepMind (2013, 2015).

### 12.3 Dua Inovasi Kunci DQN

#### A. Experience Replay

**Masalah:** Data berurutan (sequential) melanggar asumsi i.i.d. (independent and identically distributed).

**Solusi:** Simpan transisi $(s, a, r, s')$ dalam **replay buffer** dan sample secara random saat training.

```
Replay Buffer:
┌─────────────────────────────────────┐
│ (s1, a1, r1, s1')                   │
│ (s2, a2, r2, s2')                   │
│ (s3, a3, r3, s3')                   │
│ ...                                 │
│ (sN, aN, rN, sN')                   │
└─────────────────────────────────────┘
         │
         ▼ (sample random)
    Mini-batch untuk training
```

**Keuntungan:**
- Menghilangkan korelasi temporal.
- Efisien (data bisa digunakan berulang).
- Stabil.

#### B. Target Network

**Masalah:** Target $r + \gamma \max Q(s', a')$ berubah setiap kali Q-network di-update → tidak stabil.

**Solusi:** Gunakan **target network** yang di-update secara periodik.

$$\text{Loss} = \left( r + \gamma \max_{a'} Q_{\text{target}}(s', a') - Q(s, a) \right)^2$$

**Update:** Target network di-copy dari main network setiap $C$ langkah.

### 12.4 Algoritma DQN

```
1. Inisialisasi replay buffer D dengan kapasitas N
2. Inisialisasi Q-network dengan bobot random
3. Inisialisasi target network dengan bobot yang sama
4. Untuk setiap episode:
   a. Inisialisasi state s
   b. Untuk setiap langkah:
      i.   Pilih action a dengan epsilon-greedy
      ii.  Ambil action, amati r dan s'
      iii. Simpan (s, a, r, s') di D
      iv.  Sample mini-batch dari D
      v.   Hitung target: y = r + γ max Q_target(s', a')
      vi.  Update Q-network dengan loss (y - Q(s,a))²
      vii. Setiap C langkah: copy Q → Q_target
      viii.s = s'
```

### 12.5 DQN dan Atari Games

**Hasil DQN:**
- DQN mencapai performa setara manusia pada 49 game Atari.
- Belajar hanya dari pixel mentah dan skor.
- Tidak ada feature engineering.

### 12.6 Varian DQN

| Varian | Inovasi |
|---|---|
| **Double DQN** | Mengatasi overestimasi Q-value |
| **Dueling DQN** | Memisahkan value dan advantage |
| **Prioritized Replay** | Sample transisi penting lebih sering |
| **Noisy DQN** | Eksplorasi melalui noise di network |
| **Rainbow** | Kombinasi semua varian |

### 12.7 Kelebihan dan Kekurangan DQN

| Kelebihan | Kekurangan |
|---|---|
| Bisa menangani state besar | Butuh banyak data |
| Generalisasi ke state baru | Training lama |
| Bisa menangani gambar | Hyperparameter sensitif |
| End-to-end learning | Tidak stabil tanpa tricks |


## 13. POLICY GRADIENT METHODS

### 13.1 Konsep Dasar

**Policy Gradient** adalah metode RL yang **langsung mengoptimalkan policy** tanpa melalui value function.

**Ide:** Parameterisasi policy $\pi_\theta(a|s)$ dan optimalkan $\theta$ menggunakan gradient ascent.

**Tujuan:**
$$J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_t \gamma^t r_t \right]$$

**Gradient:**
$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot G_t \right]$$

### 13.2 REINFORCE Algorithm

**Algoritma Monte Carlo Policy Gradient:**

```
1. Inisialisasi parameter θ secara random
2. Untuk setiap episode:
   a. Generate episode menggunakan π_θ
   b. Untuk setiap langkah t:
      - Hitung return G_t
      - Update: θ += α ∇_θ log π_θ(a_t|s_t) · G_t
```

### 13.3 Actor-Critic Methods

**Ide:** Kombinasikan policy gradient (actor) dengan value function (critic).

- **Actor:** Policy $\pi_\theta(a|s)$ — memilih action.
- **Critic:** Value function $V_w(s)$ — mengevaluasi action.

**Keuntungan:**
- Mengurangi varians (dibanding REINFORCE).
- Bisa belajar online.

**Contoh:**
- **A2C** (Advantage Actor-Critic)
- **A3C** (Asynchronous Advantage Actor-Critic)
- **PPO** (Proximal Policy Optimization)
- **SAC** (Soft Actor-Critic)

### 13.4 Perbandingan Value-Based vs Policy-Based

| Aspek | Value-Based | Policy-Based |
|---|---|---|
| **Contoh** | Q-Learning, DQN | REINFORCE, PPO |
| **Output** | Q-value | Policy |
| **Action Space** | Diskrit | Diskrit & Kontinu |
| **Konvergensi** | Bisa tidak stabil | Lebih stabil |
| **Efisiensi Sample** | Lebih efisien | Kurang efisien |
| **Eksplorasi** | Implisit (epsilon) | Eksplisit (stochastic policy) |

### 13.5 PPO (Proximal Policy Optimization)

**PPO** adalah algoritma policy gradient yang **stabil dan efisien**.

**Inovasi:** Membatasi perubahan policy agar tidak terlalu besar.

**Objective:**
$$L^{CLIP}(\theta) = \mathbb{E} \left[ \min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$$

di mana $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$.

**Kelebihan:**
- Stabil.
- Efisien.
- Mudah di-tune.
- Digunakan di banyak aplikasi (ChatGPT RLHF).


## 14. EXPLORATION VS EXPLOITATION

### 14.1 Dilema Fundamental

**Exploration:** Mencoba action baru untuk menemukan strategi yang lebih baik.
**Exploitation:** Menggunakan action terbaik yang sudah diketahui.

**Trade-off:**
- Terlalu banyak eksplorasi → tidak pernah optimal.
- Terlalu sedikit eksplorasi → terjebak di lokal optimal.

### 14.2 Strategi Eksplorasi

#### A. Epsilon-Greedy

Dengan probabilitas $\epsilon$: pilih action random.
Dengan probabilitas $1-\epsilon$: pilih action terbaik.

```python
if np.random.random() < epsilon:
    action = env.action_space.sample()
else:
    action = np.argmax(Q[state, :])
```

**Kelebihan:** Sederhana.
**Kekurangan:** Eksplorasi tidak efisien (semua action sama).

#### B. Softmax (Boltzmann)

Pilih action berdasarkan probabilitas yang proporsional dengan Q-value.

$$P(a|s) = \frac{e^{Q(s,a)/\tau}}{\sum_{a'} e^{Q(s,a')/\tau}}$$

di mana $\tau$ = temperature.

**Kelebihan:** Eksplorasi lebih cerdas.
**Kekurangan:** Butuh tuning $\tau$.

#### C. Upper Confidence Bound (UCB)

Pilih action yang memaksimalkan:
$$a = \arg\max_a \left[ Q(s,a) + c \sqrt{\frac{\ln N(s)}{N(s,a)}} \right]$$

**Kelebihan:** Eksplorasi efisien.
**Kekurangan:** Kompleks.

#### D. Thompson Sampling

Sample dari posterior distribution Q-value dan pilih action terbaik.

**Kelebihan:** Bayesian, efisien.
**Kekurangan:** Kompleks.

### 14.3 Epsilon Decay

**Strategi:** Mulai dengan eksplorasi tinggi, turunkan secara bertahap.

```python
epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
```

**Jadwal Decay:**

| Jadwal | Formula |
|---|---|
| **Linear** | $\epsilon = \max(0, \epsilon_0 - k \cdot t)$ |
| **Exponential** | $\epsilon = \epsilon_0 \cdot e^{-kt}$ |
| **Inverse** | $\epsilon = \frac{\epsilon_0}{1 + kt}$ |

### 14.4 Kapan Berhenti Eksplorasi?

- Ketika Q-value sudah konvergen.
- Ketika performa sudah stabil.
- Ketika budget komputasi habis.


## 15. MULTI-ARMED BANDIT

### 15.1 Konsep Dasar

**Multi-Armed Bandit** adalah kasus khusus RL dengan **satu state** dan **multiple actions**.

**Analogi:** Mesin slot dengan K tuas. Setiap tuas memiliki distribusi reward yang berbeda. Tujuan: memaksimalkan reward total.

### 15.2 Formal

- **Action:** $a \in \{1, 2, ..., K\}$
- **Reward:** $r \sim R_a$ (distribusi unknown)
- **Tujuan:** Maksimalkan $\sum_{t=1}^T r_t$

### 15.3 Strategi

| Strategi | Deskripsi |
|---|---|
| **Epsilon-Greedy** | Eksplorasi dengan probabilitas ε |
| **UCB** | Optimism in the face of uncertainty |
| **Thompson Sampling** | Bayesian approach |
| **Softmax** | Probabilistic action selection |

### 15.4 Aplikasi

- **A/B Testing:** Pilih varian terbaik.
- **Rekomendasi:** Pilih item terbaik untuk user.
- **Klinis:** Pilih treatment terbaik.
- **Iklan:** Pilih iklan terbaik.
- **Manufaktur:** Pilih setpoint optimal untuk mesin.

### 15.5 Regret

**Regret:** Selisih antara reward optimal dan reward aktual.

$$\text{Regret}(T) = T \cdot \mu^* - \sum_{t=1}^T \mu_{a_t}$$

**Tujuan:** Meminimalkan regret.


## 16. APLIKASI RL DI DUNIA NYATA

### 16.1 Game

| Game | Algoritma | Pencapaian |
|---|---|---|
| **Backgammon** | TD-Gammon | Setara grandmaster |
| **Atari** | DQN | Setara manusia |
| **Go** | AlphaGo | Mengalahkan Lee Sedol |
| **Chess** | AlphaZero | Mengalahkan Stockfish |
| **Dota 2** | OpenAI Five | Mengalahkan tim pro |
| **StarCraft II** | AlphaStar | Grandmaster level |

### 16.2 Robotika

| Aplikasi | Deskripsi |
|---|---|
| **Locomotion** | Robot belajar berjalan |
| **Manipulation** | Robot belajar mengambil objek |
| **Navigation** | Robot belajar navigasi |
| **Assembly** | Robot belajar merakit |

### 16.3 Kesehatan

| Aplikasi | Deskripsi |
|---|---|
| **Treatment Optimization** | Optimasi dosis obat |
| **Personalized Medicine** | Pengobatan personal |
| **Surgery** | Robot bedah |
| **Diagnosis** | Sistem diagnosis adaptif |

### 16.4 Keuangan

| Aplikasi | Deskripsi |
|---|---|
| **Trading** | Strategi trading otomatis |
| **Portfolio Management** | Alokasi aset |
| **Risk Management** | Manajemen risiko |
| **Fraud Detection** | Deteksi fraud adaptif |

### 16.5 Energi

| Aplikasi | Deskripsi |
|---|---|
| **Smart Grid** | Optimasi distribusi energi |
| **HVAC** | Optimasi pendinginan/pemanasan |
| **Data Center** | Optimasi cooling |
| **Renewable** | Optimasi energi terbarukan |


## 17. APLIKASI RL DI INDUSTRI MANUFAKTUR

### 17.1 Kontrol Proses Industri

Penelitian terbaru menunjukkan bahwa RL telah berhasil diterapkan untuk kontrol proses industri:

- **Q-Learning-Based Nonlinear Model Predictive Control (QL-NMPC):** Kerangka ini telah divalidasi secara eksperimental untuk kontrol suhu reaktor batch. Agen RL dilatih dalam simulasi untuk mempelajari strategi kontrol optimal menggunakan **coolant flow rate** dan **heater current** sebagai input.

- **Actor-Critic RL untuk Jacketed Reactor:** Metode A2CRL (Actor-Critic Reinforcement Learning) digunakan untuk melacak profil suhu di reaktor batch. Pendekatan ini menggabungkan optimasi policy dan estimasi value function untuk mengatur panas yang dihasilkan oleh reaksi eksotermik.

- **Deep RL untuk Fed-Batch Penicillin Fermentation:** Kontroler RL berbasis adaptive control digunakan untuk menyesuaikan feed rate secara dinamis dalam proses fermentasi penisilin fed-batch, yang secara inheren kompleks dan non-linear karena sensitivitas pertumbuhan mikroba.

### 17.2 Optimasi Produksi dan Penjadwalan

- **Digital Twin-enabled Nested Q-Learning:** Kerangka ini digunakan untuk mengoordinasikan keputusan strategis dan operasional di tingkat bulanan, mingguan, dan harian dalam lingkungan job shop yang fleksibel. Struktur Hierarchical RL (HRL) digunakan untuk mempelajari policy adaptif untuk perencanaan produksi, pengadaan material, dan penjadwalan pekerjaan.

- **Pick-and-Place Optimization:** Q-Learning dalam Digital Twin untuk operasi Pick-and-Place menunjukkan pengurangan waktu siklus sekitar 10% dan penyimpangan posisi di bawah 2% dibandingkan kontrol konvensional.

- **Pallet Loop Control:** Deep RL digunakan untuk masalah kontrol pallet loop skala besar di pabrik perakitan otomotif dengan berbagai tipe bodi yang dirakit di fasilitas yang sama.

### 17.3 Predictive Maintenance

- **Condition Monitoring:** RL digunakan untuk memantau kondisi mesin dan memprediksi kapan maintenance diperlukan. Cluster yang dihasilkan dari data sensor berkorelasi dengan state operasional nyata mesin。

- **Fault Diagnosis:** RL approaches, khususnya Q-learning, diterapkan dalam proses milling untuk perencanaan, penjadwalan, fault diagnosis, dan peningkatan efisiensi sistem。

### 17.4 Optimasi Energi

- **Energy-Efficient Factory Machines:** K-Means dengan automatic cluster detection digunakan untuk mendeteksi operating modes dan meningkatkan anomaly detection, memungkinkan integrasi mesin lama dengan smart factory.

- **Multi-Agent RL untuk Koordinasi:** Multi-Agent RL (MARL) memungkinkan pengambilan keputusan otonom yang menyeimbangkan tujuan yang saling bertentangan seperti produktivitas, kualitas, dan efisiensi energi。

### 17.5 Studi Kasus: Batch Reactor Anomaly Data

Dataset **Batch Reactor Anomaly Data (Sample)** yang digunakan dalam praktikum ini dirancang khusus untuk melatih agen RL dan menguji algoritma Predictive Maintenance. Dataset ini mensimulasikan operasi baseline normal bersama dengan anomali edge-case kritis, termasuk **Cooling System Failure** dan **Pressure Spikes**. Rekomendasi penggunaan RL adalah melatih agen pada episode diskrit (`Reactor_Run_ID`) untuk mengoptimalkan kontrol coolant (`Jacket_Flow_Rate_L_min`) agar memaksimalkan konsentrasi produk (`Product_B_Conc_mol_L`) sambil membatasi suhu reaktor (`Reactor_Temp_C`) secara ketat.


## 18. TANTANGAN DAN KETERBATASAN RL

### 18.1 Tantangan Teknis

| Tantangan | Deskripsi | Solusi |
|---|---|---|
| **Sample Efficiency** | Butuh banyak interaksi | Model-based RL, transfer learning |
| **Credit Assignment** | Sulit menentukan action mana yang berkontribusi | Eligibility traces, reward shaping |
| **Exploration** | Sulit mengeksplorasi efisien | UCB, Thompson Sampling |
| **Stability** | Training tidak stabil | Target network, PPO |
| **Scalability** | Tidak skalabel untuk state besar | DQN, function approximation |
| **Sparse Reward** | Reward jarang | Reward shaping, curiosity |

### 18.2 Tantangan Non-Teknis

| Tantangan | Deskripsi |
|---|---|
| **Simulasi** | Butuh environment untuk training |
| **Reward Design** | Sulit mendesain reward yang tepat |
| **Safety** | Agen bisa mengambil action berbahaya |
| **Interpretability** | Sulit menjelaskan keputusan agen |
| **Etika** | Bisa diskriminatif jika data bias |

### 18.3 Curse of Dimensionality

Jumlah state tumbuh eksponensial dengan dimensi:

| Dimensi | Jumlah State (10 nilai/dimensi) |
|---|---|
| 1 | 10 |
| 2 | 100 |
| 3 | 1.000 |
| 5 | 100.000 |
| 10 | 10.000.000.000 |

**Solusi:** Function approximation (neural network), dimensionality reduction.

### 18.4 Deadly Triad

**Deadly Triad** adalah kombinasi yang bisa menyebabkan divergensi:
1. **Function Approximation**
2. **Bootstrapping**
3. **Off-policy Learning**

**Solusi:** Batch RL, target network, atau on-policy methods.


## 19. ETIKA DAN KEAMANAN DALAM RL

### 19.1 Risiko RL

| Risiko | Deskripsi |
|---|---|
| **Reward Hacking** | Agen menemukan cara curang untuk mendapat reward |
| **Unsafe Exploration** | Agen mengeksplorasi action berbahaya |
| **Bias** | Agen belajar bias dari data |
| **Manipulasi** | Agen memanipulasi environment |
| **Ketidakstabilan** | Keputusan agen tidak konsisten |

### 19.2 Prinsip Etika RL

1. **Transparansi:** Keputusan agen harus bisa dijelaskan.
2. **Akuntabilitas:** Ada pihak yang bertanggung jawab.
3. **Keadilan:** Agen tidak diskriminatif.
4. **Keamanan:** Agen tidak membahayakan.
5. **Privasi:** Data pengguna dilindungi.

### 19.3 Contoh Kasus

**Reward Hacking:**
- **CoastRunners:** Agen boat berputar-putar mengambil power-up alih-alih menyelesaikan balapan.
- **Tetris:** Agen pause game agar tidak kalah.

**Solusi:** Desain reward yang robust, human oversight.

### 19.4 RLHF (Reinforcement Learning from Human Feedback)

**RLHF** adalah teknik untuk melatih AI menggunakan feedback manusia.

**Digunakan di:**
- ChatGPT
- Claude
- Gemini

**Proses:**
1. Pre-training (supervised).
2. Reward model dari feedback manusia.
3. Fine-tuning dengan RL (PPO).


## 20. LATIHAN DAN SOAL REFLEKSI

### 20.1 Latihan Coding

**Latihan 1:** Implementasi Q-Learning dari Scratch

```python
def q_learning(env, n_episodes, alpha, gamma, epsilon, decay_rate):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_r = 0

        while not done:
            # Epsilon-greedy
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Update Q
            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
            )

            state = next_state
            total_r += reward

        rewards.append(total_r)
        epsilon = 0.01 + (1.0 - 0.01) * np.exp(-decay_rate * episode)

    return Q, rewards
```

**Latihan 2:** Implementasi SARSA

```python
def sarsa(env, n_episodes, alpha, gamma, epsilon):
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    for episode in range(n_episodes):
        state, _ = env.reset()

        # Pilih action pertama
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        done = False
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Pilih action berikutnya
            if np.random.random() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state, :])

            # Update Q (SARSA)
            Q[state, action] += alpha * (
                reward + gamma * Q[next_state, next_action] - Q[state, action]
            )

            state = next_state
            action = next_action

    return Q
```

**Latihan 3:** Visualisasi Policy

```python
def visualize_policy(Q, grid_size=4):
    arrow_map = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    policy = np.full((grid_size, grid_size), '', dtype=object)

    for state in range(grid_size * grid_size):
        row = state // grid_size
        col = state % grid_size
        best_action = np.argmax(Q[state, :])
        policy[row, col] = arrow_map[best_action]

    for row in policy:
        print("  ".join(row))
```

### 20.2 Soal Refleksi

1. **Konsep:** Jelaskan perbedaan fundamental antara RL dan supervised learning.

2. **MDP:** Apa itu Markov Property? Mengapa penting dalam RL?

3. **Bellman:** Jelaskan intuisi di balik Bellman Optimality Equation.

4. **Q-Learning:** Mengapa Q-Learning disebut "off-policy"? Apa keuntungannya?

5. **SARSA:** Kapan Anda memilih SARSA daripada Q-Learning?

6. **DQN:** Apa dua inovasi kunci DQN? Mengapa keduanya penting?

7. **Eksplorasi:** Jelaskan dilema exploration vs exploitation dengan contoh.

8. **Reward:** Apa itu reward shaping? Berikan contoh.

9. **Etika:** Apa risiko menggunakan RL untuk keputusan kontrol proses?

10. **Aplikasi:** Bagaimana RL bisa diterapkan untuk kontrol reaktor batch?

### 20.3 Studi Kasus Mini

**Skenario:** Sebuah pabrik kimia ingin mengontrol suhu reaktor batch secara otomatis.

**Tugas:**
1. Definisikan state, action, dan reward.
2. Algoritma RL apa yang cocok? Mengapa?
3. Bagaimana mengevaluasi sistem?
4. Apa tantangan yang mungkin dihadapi?
5. Bagaimana mengatasi cold-start problem?


## 21. GLOSARIUM

| Istilah | Definisi |
|---|---|
| **Action** | Tindakan yang diambil agen |
| **Actor-Critic** | Metode RL yang menggabungkan policy dan value |
| **Agent** | Entitas yang belajar dan mengambil keputusan |
| **Bellman Equation** | Persamaan rekursif untuk value function |
| **Bootstrap** | Update menggunakan estimasi |
| **Credit Assignment** | Menentukan action mana yang berkontribusi |
| **Discount Factor** | Bobot reward masa depan (γ) |
| **DQN** | Deep Q-Network |
| **Dynamic Programming** | Metode optimasi dengan model |
| **Eligibility Trace** | Mekanisme untuk TD(λ) |
| **Environment** | Dunia tempat agen berinteraksi |
| **Episode** | Satu putaran interaksi |
| **Epsilon-Greedy** | Strategi eksplorasi |
| **Exploration** | Mencoba action baru |
| **Exploitation** | Menggunakan action terbaik |
| **Function Approximation** | Representasi value function dengan fungsi |
| **Markov Property** | State mengandung semua informasi |
| **MDP** | Markov Decision Process |
| **Monte Carlo** | Metode RL berbasis episode |
| **Multi-Armed Bandit** | Kasus khusus RL dengan 1 state |
| **Off-Policy** | Belajar policy berbeda dari behavior |
| **On-Policy** | Belajar policy yang diikuti |
| **Policy** | Strategi agen |
| **Policy Gradient** | Metode optimasi policy langsung |
| **PPO** | Proximal Policy Optimization |
| **Q-Learning** | Algoritma RL off-policy |
| **Q-Table** | Tabel Q-value |
| **Q-Value** | Nilai action di state |
| **Regret** | Selisih reward optimal dan aktual |
| **Replay Buffer** | Penyimpanan pengalaman |
| **Return** | Total reward kumulatif |
| **Reward** | Sinyal umpan balik |
| **RLHF** | RL from Human Feedback |
| **SARSA** | Algoritma RL on-policy |
| **State** | Kondisi agen dan environment |
| **Target Network** | Network stabil untuk DQN |
| **TD Learning** | Temporal Difference Learning |
| **Trajectory** | Urutan state-action-reward |
| **Value Function** | Fungsi nilai state/action |


## 22. REFERENSI

### 22.1 Buku

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. — **Buku wajib RL.**
2. Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media. — Bab 18.
3. Lapan, M. (2024). *Deep Reinforcement Learning Hands-On* (3rd ed.). Packt.
4. Szepesvári, C. (2010). *Algorithms for Reinforcement Learning*. Morgan & Claypool.
5. Puterman, M. L. (2014). *Markov Decision Processes*. Wiley.

### 22.2 Artikel Kunci

1. Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8, 279–292.
2. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
3. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. *arXiv*.
4. Silver, D., et al. (2016). Mastering the game of Go with deep neural networks. *Nature*, 529, 484–489.
5. Christiano, P. F., et al. (2017). Deep Reinforcement Learning from Human Preferences. *NeurIPS*.
6. Selvamurugan, A., et al. (2025). Reinforcement Learning-Based Nonlinear Model Predictive Controller for a Jacketed Reactor. *ACS Omega*, 10, 30864-30878.
7. Q-Learning-Based Multivariate Nonlinear Model Predictive Controller: Experimental Validation on Batch Reactor. *ACS Publications*.
8. A Survey and Tutorial of Reinforcement Learning Methods in Process Systems Engineering. *arXiv*.

### 22.3 Dokumentasi Online

1. **Gymnasium:** https://gymnasium.farama.org/
2. **Stable-Baselines3:** https://stable-baselines3.readthedocs.io/
3. **PyTorch RL:** https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
4. **OpenAI Spinning Up:** https://spinningup.openai.com/
5. **Hugging Face Deep RL Course:** https://huggingface.co/learn/deep-rl-course

### 22.4 Video Tutorial

1. David Silver — UCL RL Course: https://www.youtube.com/watch?v=2pWv7GOvuf0
2. StatQuest — Reinforcement Learning: https://www.youtube.com/watch?v=JgvyzIkgxF0
3. DeepMind — AlphaGo Documentary: https://www.youtube.com/watch?v=WXuK6gekU1Y

### 22.5 Dataset

1. AI Mind Teams. (2026). *Batch Reactor Anomaly Data (Sample)*. Kaggle. https://www.kaggle.com/datasets/aimindteams/batch-reactor-anomaly-data-sample
2. Colabsss. (2026). *Adaptive PLC Automation Dataset*. Kaggle. https://www.kaggle.com/datasets/colabsss/adaptive-plc-automation-dataset
3. S3Programmer. (2024). *6G-Enabled Intelligent Manufacturing Resource Data*. Kaggle. https://www.kaggle.com/datasets/s3programmer/6g-enabled-intelligent-manufacturing-resource-data
4. Torayeff. (2023). *MfgRL: Manufacturing Reinforcement Learning Environment*. GitHub. https://github.com/torayeff/mfgrl


## PENUTUP

Modul ajar ini disusun sebagai **penunjang teori** untuk praktikum pertemuan ke-3 minggu ke-2 tentang **Reinforcement Learning untuk Kontrol Proses Industri**. Materi ini mencakup:

1. **Konsep dasar** RL dan perbedaannya dengan supervised/unsupervised.
2. **MDP** sebagai kerangka formal RL.
3. **Algoritma klasik:** DP, Monte Carlo, TD Learning.
4. **Algoritma fundamental:** Q-Learning, SARSA.
5. **Algoritma modern:** DQN, Policy Gradient, PPO.
6. **Eksplorasi vs eksploitasi** dan strategi-strateginya.
7. **Aplikasi nyata** di industri manufaktur, termasuk kontrol reaktor batch.
8. **Tantangan dan etika** dalam RL.

**Pesan untuk Mahasiswa:**
> "Reinforcement Learning adalah paradigma yang paling dekat dengan cara manusia belajar — melalui pengalaman dan umpan balik. Dalam industri manufaktur, RL membuka pintu untuk sistem kontrol cerdas yang adaptif dan otonom, seperti kontrol reaktor batch yang aman dan efisien. Dataset yang digunakan dalam praktikum ini (Batch Reactor Anomaly Data, 5.000 baris, ~500 KB) sangat ringan sehingga dapat diolah pada laptop dengan RAM 8GB atau 16GB."

**Pesan untuk Dosen:**
> "Modul ini bisa disesuaikan dengan tingkat pemahaman mahasiswa. Untuk pemula, fokus pada Q-Learning dan lingkungan reaktor batch. Untuk mahasiswa yang lebih mahir, tambahkan DQN, PPO, dan eksperimen dengan lingkungan alternatif seperti MfgRL atau Adaptive PLC Automation Dataset."

---

**Dokumen ini disusun untuk keperluan edukasi di Politeknik Manufaktur Bandung (Polman Bandung).**
**Versi:** 1.0 | **Minggu ke-2, Pertemuan ke-3 dari 4**
**Terakhir Diperbarui:** 2026
