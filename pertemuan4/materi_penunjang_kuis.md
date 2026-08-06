# Materi Penunjang Pembelajaran Machine Learning

## Dasar-Dasar ML, Data Collection, Data Cleaning, Data Processing, dan EDA

## Bagian 1: Dasar-Dasar Machine Learning

### 1.1 Pengertian Machine Learning

Machine Learning (ML) adalah cabang dari kecerdasan buatan (Artificial Intelligence) yang memungkinkan sistem untuk belajar dan meningkatkan performanya dari pengalaman (data) tanpa diprogram secara eksplisit. Alih-alih mengikuti instruksi yang kaku, model ML mengidentifikasi pola dalam data dan membuat prediksi atau keputusan berdasarkan pola tersebut.

Secara umum, ML dibagi menjadi tiga paradigma utama:

1. **Supervised Learning (Pembelajaran Terawasi)** : Model belajar dari data berlabel (input-output pairs). Tujuannya adalah mempelajari pemetaan dari input ke output sehingga dapat memprediksi output untuk input baru. Contoh: klasifikasi spam, regresi harga rumah.

2. **Unsupervised Learning (Pembelajaran Tak Terawasi)** : Model belajar dari data tanpa label, mencari struktur atau pola tersembunyi. Contoh: clustering pelanggan, reduksi dimensi.

3. **Reinforcement Learning (Pembelajaran Penguatan)** : Model belajar melalui interaksi dengan lingkungan, menerima reward atau punishment atas tindakannya.

### 1.2 Bias dan Variance

Konsep **bias-variance trade-off** adalah salah satu fondasi paling fundamental dalam machine learning. Konsep ini menjelaskan mengapa model bisa mengalami underfitting atau overfitting, dan bagaimana menemukan keseimbangan yang tepat.

**Bias** adalah kecenderungan model untuk membuat prediksi yang tidak akurat secara konsisten karena asumsi yang terlalu sederhana. Model dengan high bias tidak cukup fleksibel untuk menangkap pola sebenarnya dalam data, sehingga menghasilkan error yang tinggi baik pada data training maupun data testing. Ini disebut **underfitting**.

**Variance** adalah kecenderungan model untuk memberikan hasil yang sangat berbeda ketika dilatih dengan potongan data yang berbeda. Model dengan high variance terlalu sensitif terhadap fluktuasi kecil dalam data training, sehingga "menghafal" noise daripada belajar pola yang sebenarnya. Ini menghasilkan performa sangat baik pada data training tetapi buruk pada data testing—kondisi yang disebut **overfitting**.

| Karakteristik | High Bias (Underfitting) | High Variance (Overfitting) |
| :--- | :--- | :--- |
| Training Error | Tinggi | Sangat Rendah (mendekati 0) |
| Testing Error | Tinggi (sama dengan training) | Sangat Tinggi (jauh di atas training) |
| Penyebab | Model terlalu sederhana | Model terlalu kompleks |
| Solusi | Tambah kompleksitas (fitur, parameter) | Kurangi kompleksitas (regularisasi, pruning, lebih banyak data) |

Model yang ideal adalah yang memiliki **low bias dan low variance**—cukup kompleks untuk menangkap pola sebenarnya, namun cukup sederhana untuk generalisasi dengan baik pada data baru.

### 1.3 Overfitting dan Underfitting

**Overfitting** terjadi ketika model terlalu kompleks sehingga "menghafal" data training termasuk noise-nya. Akibatnya, model gagal menggeneralisasi ke data baru. Ciri-ciri overfitting:

- Training error sangat rendah (bahkan 0%)
- Testing error tinggi
- Model memiliki banyak parameter relatif terhadap jumlah data

**Underfitting** terjadi ketika model terlalu sederhana untuk menangkap struktur dasar data. Ciri-ciri underfitting:

- Training error tinggi
- Testing error tinggi (hampir sama dengan training error)
- Model gagal menangkap pola yang jelas dalam data

**Penyebab Overfitting:**

1. Terlalu banyak fitur dibandingkan jumlah sampel
2. Model terlalu kompleks (kedalaman pohon terlalu besar, derajat polinomial tinggi)
3. Data training mengandung noise yang dipelajari sebagai pola
4. Training dilakukan terlalu lama (terutama pada neural network)

**Penyebab Underfitting:**

1. Model terlalu sederhana (misal: regresi linear untuk data non-linear)
2. Fitur yang digunakan tidak informatif
3. Regularisasi terlalu kuat

### 1.4 Metrik Evaluasi dalam Machine Learning

#### Confusion Matrix

Confusion matrix adalah tabel yang merangkum performa model klasifikasi dengan memecah prediksi menjadi empat kategori:

| | Prediksi Positif | Prediksi Negatif |
| :--- | :--- | :--- |
| **Aktual Positif** | True Positive (TP) | False Negative (FN) - Type II Error |
| **Aktual Negatif** | False Positive (FP) - Type I Error | True Negative (TN) |

#### Metrik Turunan

**Akurasi**: Proporsi prediksi yang benar dari seluruh prediksi.
$$Akurasi = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision**: Proporsi prediksi positif yang benar. Mengukur seberapa akurat model saat memprediksi positif.
$$Precision = \frac{TP}{TP + FP}$$

**Recall (Sensitivitas)**: Proporsi kasus positif aktual yang berhasil dideteksi. Mengukur seberapa baik model menangkap kasus positif.
$$Recall = \frac{TP}{TP + FN}$$

**F1-Score**: Rata-rata harmonik dari precision dan recall. Memberikan penalti jika salah satu nilai rendah.
$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

**Mengapa Akurasi Bisa Menyesatkan?**

Pada data dengan ketidakseimbangan kelas (imbalance), akurasi menjadi metrik yang menyesatkan. Contoh: jika 99% data adalah kelas negatif dan 1% positif, model yang selalu memprediksi "negatif" akan mencapai akurasi 99%, tetapi sama sekali tidak berguna untuk mendeteksi kasus positif. Dalam kasus seperti ini, Recall, Precision, dan F1-Score jauh lebih informatif.

**Makro-F1 vs Mikro-F1**:

- **Makro-F1**: Menghitung F1 untuk setiap kelas secara terpisah, lalu merata-ratakannya. Setiap kelas mendapat bobot yang sama, terlepas dari jumlah sampelnya. Sensitif terhadap performa kelas minoritas.
- **Mikro-F1**: Menghitung F1 secara global dengan mengagregasi TP, FP, FN dari semua kelas. Didominasi oleh kelas mayoritas.

#### ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

ROC-AUC mengukur kemampuan model untuk membedakan kelas positif dan negatif di **semua nilai threshold** yang mungkin. Nilai AUC berkisar antara 0.5 (random guessing) hingga 1.0 (klasifikasi sempurna).

Keunggulan ROC-AUC:

- Tidak terpengaruh oleh ketidakseimbangan kelas
- Mengukur performa model secara holistik di semua threshold
- Threshold-independent (tidak bergantung pada pilihan cutoff 0.5)

### 1.5 Validasi Model dan Data Leakage

**Cross-Validation** adalah teknik untuk mengevaluasi performa model dengan membagi data menjadi beberapa fold (lipatan) dan melatih/menguji secara bergiliran.

**Data Leakage** adalah situasi di mana informasi dari luar data training (termasuk data test) secara tidak sengaja digunakan dalam proses training. Ini menyebabkan evaluasi model menjadi terlalu optimis dan tidak realistis.

Contoh data leakage:

1. Melakukan scaling (normalisasi/standarisasi) sebelum split train-test
2. Menggunakan informasi dari data test untuk mengisi missing value di data training
3. Menggunakan fitur yang secara tidak langsung mengandung informasi target (misal: menggunakan "total pembelian" untuk memprediksi "pembelian bulan depan" jika total sudah mencakup masa depan)

Untuk time-series, cross-validation standar dengan shuffle acak adalah **data leakage** karena data masa depan (yang seharusnya belum diketahui) bisa masuk ke data training dan digunakan untuk memprediksi data masa lalu. Solusi yang benar adalah menggunakan TimeSeriesSplit atau forward chaining.

## Bagian 2: Data Collection (Pengumpulan Data)

### 2.1 Pentingnya Kualitas Data

Prinsip fundamental dalam Machine Learning adalah **"Garbage In, Garbage Out"** . Kualitas model sangat bergantung pada kualitas data yang digunakan untuk melatihnya. Data yang bias, tidak representatif, atau mengandung error akan menghasilkan model yang bias dan tidak dapat diandalkan.

### 2.2 Jenis-Jenis Bias dalam Pengumpulan Data

#### Sampling Bias (Selection Bias)

Sampling bias terjadi ketika sampel yang dikumpulkan tidak mewakili populasi yang menjadi target analisis. Contoh: survei online hanya menjangkau pengguna internet, sehingga tidak mewakili populasi yang tidak memiliki akses internet. Ini menyebabkan kesimpulan yang distortif dan keliru.

#### Survivorship Bias

Survivorship bias adalah subset dari selection bias di mana analisis hanya fokus pada entitas yang "selamat" atau bertahan melalui suatu proses, mengabaikan mereka yang gagal atau tidak terlihat.

Contoh klasik: Menganalisis perusahaan yang sukses dan go public untuk memprediksi kesuksesan startup. Ini mengabaikan ribuan startup yang gagal, sehingga model tidak belajar dari pola kegagalan. Contoh lain: dalam analisis performa reksa dana, hanya dana yang masih aktif yang dianalisis, mengabaikan dana yang sudah tutup karena performa buruk.

#### Domain Shift (Covariate Shift)

Terjadi ketika distribusi data training berbeda secara signifikan dengan distribusi data di dunia nyata (target domain). Contoh: model deteksi spam dilatih dengan email dari satu perusahaan, lalu diterapkan ke email publik dengan pola spam yang berbeda.

### 2.3 Data Terstruktur vs Tidak Terstruktur

**Data Terstruktur**: Data yang terorganisir dalam format baris dan kolom (tabel). Contoh: spreadsheet, database SQL. Setiap kolom mewakili fitur (variabel) dan setiap baris mewakili satu observasi. Data ini relatif siap untuk analisis numerik, meskipun tetap memerlukan preprocessing.

**Data Tidak Terstruktur**: Data yang tidak memiliki format baku. Contoh: teks, gambar, audio, video. Data ini memerlukan tahap **Feature Extraction** atau **Embedding** untuk diubah menjadi representasi numerik yang dapat diproses oleh algoritma ML.

### 2.4 Entity Resolution

Entity resolution adalah proses mengidentifikasi dan menggabungkan record yang merujuk pada entitas yang sama dari sumber data yang berbeda. Masalah umum: produk yang sama direpresentasikan dengan ID berbeda di tiap toko (numerik vs alfanumerik vs teks bebas). Tanpa standardisasi, integrasi data menjadi tidak mungkin dan analisis menjadi kacau.

### 2.5 Missing Data Mechanisms

Memahami mekanisme hilangnya data (missing data mechanism) sangat penting karena menentukan strategi penanganan yang tepat.

**MCAR (Missing Completely At Random)** : Probabilitas data hilang adalah sama untuk semua observasi, terlepas dari nilai observed maupun unobserved. Contoh: data hilang karena kerusakan sensor secara acak. Penghapusan baris (listwise deletion) menghasilkan estimasi yang tidak bias jika data MCAR.

**MAR (Missing At Random)** : Probabilitas data hilang bergantung pada variabel lain yang teramati (observed), tetapi tidak pada nilai variabel itu sendiri. Contoh: probabilitas data gaji hilang lebih tinggi untuk level pendidikan tinggi (pendidikan teramati).

**MNAR (Missing Not At Random)** : Probabilitas data hilang bergantung pada nilai variabel itu sendiri yang tidak teramati. Contoh: sensor suhu mati ketika suhu terlalu tinggi—data hilang karena suhu itu sendiri tinggi.

| Mekanisme | Definisi | Contoh |
| :--- | :--- | :--- |
| MCAR | Hilang acak, tidak tergantung apapun | Sensor mati secara acak |
| MAR | Hilang tergantung variabel lain yang teramati | Gaji hilang lebih sering untuk pendidikan tinggi |
| MNAR | Hilang tergantung nilai variabel itu sendiri | Sensor mati saat suhu tinggi |

## Bagian 3: Data Cleaning (Pembersihan Data)

Data cleaning adalah proses mengidentifikasi dan mengoreksi (atau menghapus) data yang rusak, tidak akurat, atau tidak lengkap. Ini adalah tahap krusial sebelum EDA dan pemodelan.

### 3.1 Handling Missing Values

#### Metode Imputasi

1. **Listwise Deletion (Complete Case Analysis)** : Menghapus seluruh baris yang memiliki missing value. Efektif jika data MCAR dan jumlah missing kecil. Risiko: kehilangan banyak data dan bias jika MAR/MNAR.

2. **Mean/Median/Mode Imputation**: Mengganti missing value dengan mean (rata-rata), median (nilai tengah), atau modus (nilai paling sering).
   - **Mean**: Sensitif terhadap outlier. Cocok untuk data simetris tanpa outlier.
   - **Median**: Robust terhadap outlier dan skewness. Lebih stabil untuk data miring.
   - **Mode**: Untuk data kategorikal.

3. **Group-Based Imputation (Conditional Imputation)** : Mengisi missing value berdasarkan rata-rata/median dari kelompok yang sama (misal: rata-rata gaji per tingkat pendidikan). Ini adalah pendekatan yang tepat untuk data MAR.

4. **KNN Imputation**: Mengisi missing value berdasarkan nilai dari K tetangga terdekat. Efektif untuk data MCAR dan MAR.

5. **MICE (Multiple Imputation by Chained Equations)** : Metode iteratif yang mengisi missing value secara bergantian untuk setiap kolom. Dianggap paling reliable untuk performa prediktif di berbagai mekanisme missingness.

#### Prinsip Penting

- Imputasi harus dilakukan **setelah split train-test** untuk menghindari data leakage
- Pilihan metode imputasi harus didasarkan pada mekanisme missing data (MCAR/MAR/MNAR)
- Dokumentasikan alasan pemilihan metode imputasi

### 3.2 Deteksi dan Penanganan Outlier

Outlier adalah nilai yang sangat berbeda dari observasi lainnya. Outlier bisa berasal dari:

1. **Kesalahan input** (data entry error)
2. **Variasi alami** (misal: rumah mewah di dataset properti)
3. **Kejadian langka** yang sah secara domain

#### Metode Deteksi Outlier

**IQR (Interquartile Range)** : Metode berbasis kuartil.

- Q1 = kuartil ke-25
- Q3 = kuartil ke-75
- IQR = Q3 - Q1
- Batas bawah = Q1 - 1.5 × IQR
- Batas atas = Q3 + 1.5 × IQR
- Nilai di luar batas ini dianggap outlier

**Z-Score**: Mengukur seberapa jauh suatu nilai dari mean dalam satuan standar deviasi. Nilai dengan |z| > 3 biasanya dianggap outlier.

#### Strategi Penanganan Outlier

1. **Investigasi terlebih dahulu**: Apakah outlier adalah kesalahan atau variasi alami?
2. **Jika kesalahan input**: Koreksi atau hapus.
3. **Jika variasi alami**:
   - Pertahankan jika domain membenarkan (misal: rumah mewah)
   - Gunakan algoritma robust terhadap outlier (Tree-Based, SVM dengan kernel)
   - Lakukan transformasi (log, akar kuadrat) untuk mengurangi efek outlier
4. **Clipping/Winsorizing**: Batasi nilai outlier ke nilai maksimum/minimum yang masih wajar

**Peringatan**: Menghapus outlier tanpa investigasi adalah kesalahan umum yang dapat membuang informasi berharga.

### 3.3 Penanganan Data Duplikat

Duplikat adalah baris yang identik atau hampir identik. Dampak duplikat:

- Memberikan bobot berlebih pada pola tertentu
- Mendistorsi distribusi data
- Meningkatkan risiko overfitting

**Tindakan**: Deteksi dengan `df.duplicated()` dan hapus kelebihan duplikat (sisakan satu salinan).

### 3.4 Konversi Tipe Data

Tipe data yang salah adalah masalah umum. Contoh:

- Kolom tanggal terbaca sebagai `object`/string
- Kolom numerik terbaca sebagai `object` karena ada karakter aneh
- Kolom kategorikal terbaca sebagai integer

**Solusi**:

- Ubah ke `datetime` untuk kolom tanggal
- Ekstrak fitur turunan dari tanggal (usia, bulan, hari, kuartal)
- Ubah kategorikal ke `category` atau lakukan encoding

## Bagian 4: Data Processing (Preprocessing)

### 4.1 Feature Scaling

Feature scaling adalah proses menstandardisasi rentang fitur numerik. Algoritma berbasis jarak (KNN, SVM, Neural Network, PCA) **sangat sensitif** terhadap skala fitur. Algoritma berbasis pohon (Decision Tree, Random Forest, XGBoost) **tidak memerlukan scaling**.

#### StandardScaler (Z-Score Normalization)

Mengubah data sehingga memiliki mean = 0 dan standar deviasi = 1.
$$z = \frac{x - \mu}{\sigma}$$

- Tidak terbatas pada rentang tertentu
- Sensitif terhadap outlier (karena menggunakan mean dan std)
- Cocok untuk data yang mendekati distribusi normal

#### MinMaxScaler (Normalisasi)

Menskala data ke rentang tertentu, biasanya [0, 1].
$$x' = \frac{x - min}{max - min}$$

- Terbatas pada rentang [0, 1]
- Sangat sensitif terhadap outlier (min/max dipengaruhi outlier)
- Cocok untuk data dengan batas yang diketahui

#### Important Note

Baik StandardScaler maupun MinMaxScaler **tidak mengubah bentuk distribusi** (skewness tetap sama). Mereka hanya menggeser dan menskalakan data. Untuk mengubah skewness, diperlukan transformasi seperti log atau akar kuadrat.

**Golden Rule**: Fit scaler **hanya pada data training**, lalu gunakan parameter yang sama untuk transform data test.

### 4.2 Feature Encoding (Variabel Kategorikal)

#### Label Encoding

Mengganti setiap kategori dengan angka integer (0, 1, 2, ...).

- **Kelebihan**: Sederhana, hemat memori
- **Kekurangan**: Menimbulkan urutan numerik yang tidak bermakna (misal: Merah=0, Biru=1, Hijau=2)
- **Penggunaan**: Hanya untuk data **ordinal** (ada urutan alami)

#### One-Hot Encoding

Membuat kolom biner baru untuk setiap kategori.

- **Kelebihan**: Tidak mengasumsikan urutan
- **Kekurangan**: Ledakan dimensi (curse of dimensionality) untuk kategori dengan cardinality tinggi
- **Penggunaan**: Kategorikal nominal dengan jumlah kategori sedikit

**Dummy Variable Trap**: Jika One-Hot Encoding menghasilkan K kolom untuk K kategori, tanpa menghapus satu kolom, terjadi perfect multicollinearity (matriks singular). Solusi: `drop_first=True`.

#### Target Encoding (Mean Encoding)

Mengganti setiap kategori dengan rata-rata target untuk kategori tersebut.
$$Kategori\_encoded = mean(target | kategori)$$

- **Kelebihan**: Efektif untuk kategori dengan cardinality tinggi
- **Kekurangan**: Rentan terhadap target leakage (karena menggunakan informasi target)
- **Penggunaan**: High-cardinality categorical features dalam supervised learning

### 4.3 Feature Engineering

Feature engineering adalah proses menciptakan fitur baru dari data yang ada untuk meningkatkan performa model.

**Contoh**:

1. **Ekstraksi dari tanggal**: Dari `Tanggal_Lahir` → `Usia`, `Bulan_Lahir`, `Hari_dalam_Minggu`
2. **Interaksi fitur**: Dari `Panjang` × `Lebar` → `Luas`
3. **Binning**: Mengelompokkan usia kontinu menjadi kategori (0-18, 19-35, 36-55, >55)
4. **Agregasi**: Rata-rata, total, count per kelompok

**Prinsip**: Fitur baru harus memiliki makna domain dan relevan dengan target.

### 4.4 Feature Selection

Feature selection adalah proses memilih subset fitur yang paling informatif untuk mengurangi overfitting, mempercepat training, dan meningkatkan interpretabilitas.

**Metode**:

1. **Filter Method**: Memilih fitur berdasarkan korelasi dengan target (SelectKBest, uji F-statistik)
2. **Wrapper Method**: Menggunakan performa model untuk memilih fitur (RFE - Recursive Feature Elimination)
3. **Embedded Method**: Fitur selection terjadi selama training (L1 regularization / LASSO)

**PCA (Principal Component Analysis)** : Metode reduksi dimensi unsupervised yang menghasilkan komponen utama (kombinasi linear dari fitur asli). Kelemahan: tidak mempertimbangkan target, sehingga komponen yang dihasilkan mungkin tidak optimal untuk prediksi.

## Bagian 5: Exploratory Data Analysis (EDA)

### 5.1 Definisi dan Tujuan EDA

Exploratory Data Analysis (EDA) adalah proses investigasi data untuk memahami struktur, pola, anomali, dan karakteristiknya sebelum pemodelan. EDA melibatkan inspeksi, pembersihan, transformasi, dan visualisasi data untuk mengekstrak insight yang bermakna.

**Tujuan EDA**:

1. Memahami distribusi setiap fitur
2. Mendeteksi anomali, outlier, dan missing value
3. Mengidentifikasi hubungan antar fitur
4. Menemukan pola dan tren
5. Menghasilkan hipotesis untuk diuji lebih lanjut
6. Memandu pemilihan algoritma dan preprocessing

**Prinsip**: EDA bukan sekadar menjalankan kode dan menampilkan grafik—tujuan akhirnya adalah **insight dan rekomendasi tindakan**.

### 5.2 Analisis Univariat

Analisis univariat mempelajari satu variabel pada satu waktu.

**Untuk Numerik**:

- **Statistik Deskriptif**: mean, median, min, max, quartil, standar deviasi, skewness, kurtosis
- **Visualisasi**: Histogram (distribusi), Boxplot (outlier dan sebaran), Density Plot
- **Interpretasi**: Apakah data simetris? Ada outlier? Berapa skewness-nya?

**Untuk Kategorikal**:

- **Frekuensi**: `value_counts()` untuk melihat proporsi
- **Visualisasi**: Bar Chart, Pie Chart
- **Interpretasi**: Apakah ada ketidakseimbangan (imbalance) yang ekstrem?

### 5.3 Analisis Bivariat dan Multivariat

**Numerik vs Numerik**:

- **Korelasi Pearson**: Mengukur hubungan linear (-1 s.d. +1)
- **Korelasi Spearman**: Mengukur hubungan monotonik (rank-based), lebih robust terhadap non-linear
- **Visualisasi**: Scatter Plot, Heatmap Korelasi

**Peringatan**: Korelasi Pearson tinggi (misal: 0.92) **tidak selalu** berarti hubungan linear. Selalu periksa scatter plot. Pearson bisa menyesatkan untuk hubungan non-linear seperti kurva U.

**Kategorikal vs Numerik**:

- **Visualisasi**: Boxplot atau Violinplot yang mengelompokkan berdasarkan kategori
- **Uji Statistik**: t-test (2 kelompok), ANOVA (>2 kelompok)

**Kategorikal vs Kategorikal**:

- **Visualisasi**: Stacked Bar Chart, Mosaic Plot
- **Uji Statistik**: Chi-Square Test of Independence

### 5.4 Visualisasi dalam EDA

| Visualisasi | Kegunaan | Informasi yang Didapat |
| :--- | :--- | :--- |
| Histogram | Distribusi univariat | Skewness, modality, range |
| Boxplot | Sebaran dan outlier | Median, IQR, outlier, skewness (dari panjang whisker) |
| Scatter Plot | Hubungan dua variabel | Linear/non-linear, cluster, outlier |
| Heatmap | Korelasi multivariat | Multikolinearitas, fitur penting |
| Pairplot | Eksplorasi multivariat cepat | Distribusi univariat (diagonal) + hubungan bivariat (off-diagonal) |

### 5.5 Interpretasi Visualisasi

**Boxplot dan Skewness**:

- Whisker panjang ke atas → skewness positif (ekor kanan)
- Whisker panjang ke bawah → skewness negatif (ekor kiri)
- Titik di luar whisker → outlier

**Histogram dengan Banyak Nol**:

- Jika banyak nilai 0 dan beberapa nilai ekstrem → distribusi "L" atau "J"
- Solusi: segmentasi (analisis terpisah untuk yang 0 dan >0) atau transformasi log setelah menambahkan konstanta

### 5.6 EDA dan Pemilihan Algoritma

Hasil EDA secara langsung memandu pemilihan algoritma:

| Temuan EDA | Rekomendasi Algoritma |
| :--- | :--- |
| Hubungan linear, distribusi normal | Regresi Linier / Logistic Regression |
| Hubungan non-linear, banyak outlier | Tree-Based (Random Forest, XGBoost) |
| Banyak fitur, data sparse | Regularisasi (L1/L2) atau PCA |
| Data tidak seimbang | Class_weight, SMOTE, atau algoritma robust |
| Data time-series | LSTM, Prophet, ARIMA |

### 5.7 EDA yang Baik vs EDA yang Buruk

**EDA yang Baik**:

- Menghasilkan insight dan hipotesis
- Setiap visualisasi disertai interpretasi
- Menjawab pertanyaan "Jadi apa?" (So what?)
- Memberikan rekomendasi untuk tahap preprocessing dan modeling
- Fokus pada temuan kunci, bukan semua output

**EDA yang Buruk**:

- Hanya menjalankan kode tanpa interpretasi
- Menampilkan semua grafik default tanpa analisis
- Tidak ada kesimpulan atau rekomendasi
- Laporan berisi output mentah (`head()`, `info()`, `describe()`) tanpa narasi

## Daftar Pustaka untuk Pembelajaran Lanjutan

1. **Buku**: *An Introduction to Statistical Learning* (ISLR) - James, Witten, Hastie, Tibshirani
2. **Buku**: *Python for Data Analysis* - Wes McKinney (pandas)
3. **Dokumentasi**: Scikit-learn User Guide (<https://scikit-learn.org>)
4. **Dokumentasi**: Seaborn (<https://seaborn.pydata.org>)
5. **Platform**: Kaggle (notebook EDA dari kompetisi)
6. **Konsep**: Bias-Variance Tradeoff - <https://dascin.org> (Artikel mendetail)
