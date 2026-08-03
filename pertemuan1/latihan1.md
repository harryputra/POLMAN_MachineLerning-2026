# Latihan Dasar Terbimbing: “Halo, Machine Learning!”

**Pertemuan 1 – Ramah Pemula**

---

## 🎯 Tujuan Latihan

Setelah menyelesaikan latihan ini, kamu akan:

- Menjalankan Jupyter Notebook.
- Mengimpor *library* utama untuk machine learning.
- Memuat dan memahami *dataset* sederhana.
- Melatih model klasifikasi pertama kamu.
- Mengevaluasi kinerja model tersebut.
- Memahami alur kerja dasar machine learning.

---

## 🧰 Persiapan

Sebelum memulai, pastikan:

1. Lingkungan Python (Anaconda atau venv) sudah aktif (lihat panduan instalasi).
2. Jupyter Notebook sudah berjalan. Buka terminal/command prompt di folder kerja kamu, lalu ketik `jupyter notebook`.
3. Di browser, buat notebook baru: **New > Python 3** (atau nama environment kamu).

---

## 📓 Mulai Menulis Kode

Salin setiap blok kode ke dalam satu *cell* di notebook, lalu jalankan dengan **Shift + Enter**. Baca penjelasannya ya.

---

### 🔹 Langkah 1: Import Library yang Dibutuhkan

```python
# Library dasar untuk data dan matematika
import numpy as np
import pandas as pd

# Library untuk visualisasi
import matplotlib.pyplot as plt
import seaborn as sns

# Library machine learning dari scikit-learn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

**Apa ini?**
Kita mengimpor alat-alat yang akan dipakai. Ibarat mau masak, kita siapkan panci, pisau, dan bahan.

---

### 🔹 Langkah 2: Memuat Dataset Iris

```python
# Memuat dataset Iris
iris = load_iris()
X = iris.data      # fitur: panjang & lebar kelopak, panjang & lebar mahkota
y = iris.target    # label: 0 = setosa, 1 = versicolor, 2 = virginica

# Melihat nama fitur dan label
print("Nama fitur:", iris.feature_names)
print("Nama kelas:", iris.target_names)
print("Jumlah sampel:", len(X))
```

**Apa yang terjadi?**
Dataset Iris berisi 150 sampel bunga iris, masing-masing punya 4 ukuran. Kita pisahkan menjadi `X` (data input) dan `y` (label yang ingin diprediksi).

**Kenapa dataset ini?**
Iris adalah dataset “Hello World” di machine learning. Ukurannya kecil, bersih, dan mudah dipahami.

---

### 🔹 Langkah 3: Eksplorasi Data Sederhana

```python
# Ubah ke DataFrame agar lebih mudah dibaca
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(y, iris.target_names)

# Tampilkan 5 baris pertama
df.head()
```

```python
# Statistik ringkas
df.describe()
```

```python
# Visualisasi sebaran fitur
sns.pairplot(df, hue='species', markers=['o','s','D'])
plt.show()
```

**Apa yang bisa kita amati?**

- Setiap spesies punya rentang ukuran yang berbeda.
- Contoh: *setosa* umumnya memiliki petal (mahkota) yang lebih kecil.
- Dari grafik, kita sudah bisa melihat bahwa data mungkin bisa dipisahkan dengan model sederhana.

---

### 🔹 Langkah 4: Membagi Data Latih dan Data Uji

```python
# Split 80% untuk training, 20% untuk testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data latih:", X_train.shape[0], "sampel")
print("Data uji  :", X_test.shape[0], "sampel")
```

**Mengapa dipisah?**
Model akan belajar dari data latih, lalu kita uji dengan data yang belum pernah dilihatnya. Ini mensimulasikan performa model di dunia nyata. `random_state` digunakan agar hasil split bisa direproduksi.

---

### 🔹 Langkah 5: Memilih dan Membuat Model

Kita akan menggunakan **K-Nearest Neighbors (KNN)**. Model ini bekerja dengan prinsip: “Jika ada teman-teman terdekatmu mayoritas dari kelas A, maka kamu kemungkinan besar juga kelas A”.

```python
# Membuat model KNN dengan k=3 (melihat 3 tetangga terdekat)
model = KNeighborsClassifier(n_neighbors=3)
```

---

### 🔹 Langkah 6: Melatih Model

```python
# Latih model dengan data latih
model.fit(X_train, y_train)
print("Model selesai dilatih!")
```

Cepat, bukan? Di balik layar, KNN menyimpan semua data latih. Nanti saat prediksi, ia akan menghitung jarak.

---

### 🔹 Langkah 7: Memprediksi Data Uji

```python
# Prediksi kelas untuk data uji
y_pred = model.predict(X_test)

# Lihat 10 prediksi pertama
print("Prediksi   :", y_pred[:10])
print("Nilai asli :", y_test[:10])
```

---

### 🔹 Langkah 8: Evaluasi Model

```python
# Akurasi
akurasi = accuracy_score(y_test, y_pred)
print(f"Akurasi: {akurasi*100:.2f}%")
```

```python
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, xticklabels=iris.target_names, yticklabels=iris.target_names, cmap='Blues')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.show()
```

```python
# Laporan lengkap
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

**Interpretasi:**

- **Akurasi** menunjukkan seberapa sering model benar. 100%? Dataset Iris memang mudah dipisahkan.
- **Confusion matrix** menunjukkan jumlah prediksi benar di diagonal. Kesalahan (off-diagonal) mungkin nol atau sedikit.
- **Precision, Recall, F1-score** adalah metrik tambahan yang akan dipelajari lebih lanjut.

---

### 🔹 Langkah 9: Mencoba Memprediksi Data Baru (Opsional)

```python
# Misal kita dapat bunga baru dengan ukuran:
bunga_baru = [[5.1, 3.5, 1.4, 0.2]]  # sepal length, sepal width, petal length, petal width

prediksi = model.predict(bunga_baru)
print("Prediksi spesies:", iris.target_names[prediksi][0])
```

Apakah sesuai dengan pengetahuanmu tentang setosa?

---

## 🧪 Tantangan Tambahan (Coba Sendiri!)

1. Ubah nilai `n_neighbors` menjadi 5, 7, atau 1. Apakah akurasinya berubah?
2. Coba gunakan algoritma lain yang sudah tersedia, misalnya **Decision Tree**:

   ```python
   from sklearn.tree import DecisionTreeClassifier
   model = DecisionTreeClassifier(random_state=42)
   ```

   Bandingkan hasilnya.
3. Ganti dataset dengan yang lain: coba `load_wine()` atau `load_diabetes()` (untuk regresi). Sesuaikan metrik evaluasi.
4. Lakukan EDA lebih dalam: cek apakah ada hubungan antar fitur dengan korelasi (`df.corr()`).

---

## ✅ Kesimpulan

Kamu baru saja menyelesaikan siklus dasar machine learning:

1. Mengerti masalah dan data.
2. Mempersiapkan data.
3. Melatih model.
4. Mengevaluasi model.

Ini adalah fondasi yang akan terus kamu pakai, baik untuk model sederhana maupun deep learning modern. Selamat! 🎉

---

**📌 Catatan untuk Dosen/Instruktur:**

- Sesi ini bisa dilakukan dalam 1 pertemuan praktikum (90-120 menit).
- Siswa dianjurkan untuk mengetik ulang kode (bukan copy-paste) agar lebih terbiasa.
- Berikan umpan balik terhadap hasil mereka, terutama jika ada error, untuk membangun kepercayaan diri.

*Selamat belajar, calon praktisi machine learning!*

# Panduan Penulisan Laporan Praktikum

## “Halo, Machine Learning!” – Klasifikasi Bunga Iris

---

Laporan ini bertujuan untuk mendokumentasikan pemahaman kamu setelah menyelesaikan latihan dasar terbimbing. Ikuti kerangka berikut. Kamu tidak perlu menyalin seluruh kode, cukup bagian penting dan hasilnya. Tulis dengan bahasa yang jelas dan rapi.

---

## 📄 Format Laporan

### 1. Halaman Judul

- Judul praktikum: **“Klasifikasi Bunga Iris dengan K-Nearest Neighbors”**
- Nama, NIM, kelas
- Tanggal praktikum

---

### 2. Tujuan Praktikum

Tuliskan tujuan dari praktikum ini dengan kalimatmu sendiri. Contoh:
> Memahami alur dasar machine learning: memuat data, eksplorasi, membagi data latih dan uji, melatih model KNN, serta mengevaluasi performa model pada dataset Iris.

---

### 3. Dasar Teori (Ringkas)

Jelaskan secara singkat (2–3 paragraf) tentang:

- Apa itu machine learning dan supervised learning.
- Algoritma K-Nearest Neighbors: cara kerjanya (mencari mayoritas dari *k* tetangga terdekat).
- Dataset Iris: fitur, kelas, jumlah sampel.
- Konsep data latih, data uji, dan akurasi.

---

### 4. Langkah Percobaan dan Hasil

Tuliskan langkah‑langkah utama yang kamu lakukan. Untuk setiap langkah, cantumkan **cuplikan kode penting** dan **output/hasil** yang muncul (bisa teks atau gambar grafik). Jangan lupa beri penjelasan singkat.

#### 4.1 Import Library

```python
import numpy as np
... (sebutkan library yang diimpor)
```

*Jelaskan kegunaan library tersebut.*

#### 4.2 Memuat Dataset Iris

- Tampilkan potongan kode `load_iris()`.
- Tampilkan output: nama fitur, nama kelas, jumlah sampel.

#### 4.3 Eksplorasi Data

- Tampilkan `df.head()` dan `df.describe()`.
- Sertakan visualisasi `pairplot` dan jelaskan apa yang terlihat (misalnya: “Setosa memiliki petal length dan width yang jauh lebih kecil dibanding spesies lain.”).

#### 4.4 Membagi Data

- Tampilkan kode `train_test_split`.
- Tuliskan jumlah data latih dan data uji.

#### 4.5 Membuat dan Melatih Model KNN

- Kode `KNeighborsClassifier(n_neighbors=3)`.
- Proses `fit`.

#### 4.6 Prediksi dan Evaluasi

- Kode prediksi.
- Tampilkan akurasi dalam persen.
- Tampilkan *confusion matrix* (heatmap) dan jelaskan artinya.
- Sertakan *classification report* (precision, recall, f1-score) dan beri interpretasi singkat.

#### 4.7 Prediksi Data Baru (Opsional)

- Tampilkan contoh prediksi bunga baru dan hasilnya.

---

### 5. Analisis dan Pembahasan

Ini bagian terpenting. Jawab pertanyaan-pertanyaan berikut dalam bentuk narasi:

1. Mengapa kita perlu membagi data menjadi *train* dan *test*? Apa akibatnya jika tidak dilakukan?
2. Apa yang terjadi jika nilai `k` pada KNN diubah menjadi 1 atau 10? (Coba sendiri dan laporkan perubahan akurasi)
3. Fitur apa yang tampaknya paling membedakan spesies Iris berdasarkan visualisasi?
4. Apakah model mengalami *overfitting*? Mengapa pada dataset ini akurasi bisa mencapai 100%?
5. Apa kelebihan dan kekurangan KNN berdasarkan percobaan ini?

---

### 6. Kesimpulan

Rangkum hasil praktikum dalam 1–2 paragraf. Contoh:
> Praktikum berhasil menerapkan KNN untuk mengklasifikasikan bunga Iris. Model mencapai akurasi ...% pada data uji. Alur kerja ML (muat data → split → latih → evaluasi) telah dipahami. Perubahan parameter `k` mempengaruhi akurasi. Dataset Iris relatif mudah dipisahkan sehingga akurasi sempurna dapat dicapai.

---

### 7. Tugas Tambahan (Wajib)

Jika dosen memberikan tugas tambahan pada latihan (misalnya mencoba algoritma Decision Tree atau mengganti dataset), laporkan di sini dengan format serupa: kode, hasil, dan analisis singkat.

---

## 📝 Tips Penulisan

- Gunakan screenshot atau *snippet* untuk menampilkan grafik. Pastikan terbaca jelas.
- Jangan copy-paste seluruh kode; cukup bagian yang relevan.
- Beri label gambar (misal: “Gambar 1. Pairplot dataset Iris”).
- Jelaskan setiap output dengan kata-katamu sendiri. Ini menunjukkan pemahaman.
- Periksa ejaan dan format penulisan sebelum mengumpulkan.

---

**Selamat menulis laporan!** Jika ada pertanyaan, diskusikan dengan asisten atau teman sekelas. Laporan yang baik adalah cerminan pemahaman yang baik.
