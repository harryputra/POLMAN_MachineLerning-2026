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
