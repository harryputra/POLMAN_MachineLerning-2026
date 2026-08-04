# Dataset Pertemuan 1 — Pengenalan Ekosistem Machine Learning & Eksplorasi Data (EDA)

Enam berkas CSV berikut disiapkan agar persis sesuai dengan spesifikasi kolom, ukuran, dan karakteristik data (missing value, outlier, pola tersembunyi) yang dideskripsikan dalam **Buku Ajar Pertemuan 1**. Seluruh data bersifat sintetis (dibangkitkan dengan seed acak tetap agar dapat direproduksi), namun sengaja direkayasa agar memiliki sinyal/pola yang bermakna sehingga insight yang diminta pada tiap latihan dan studi kasus benar-benar dapat ditemukan mahasiswa.

| Berkas | Dipakai pada | Ukuran | Karakteristik khusus yang sengaja disisipkan |
|---|---|---|---|
| `retail_sales.csv` | Latihan Mandiri Terbimbing | 5.000 baris × 8 kolom | `CustomerID` hilang ±20%, `Description` hilang 10 baris; `Quantity` & `UnitPrice` right-skewed dengan outlier grosir/harga tinggi |
| `factory_production.csv` | Studi Kasus Mandiri 1 | 1.080 baris × 8 kolom (4 mesin × 3 shift × 90 hari) | Korelasi positif nyata antara `Suhu_Mesin` dan `Jumlah_Cacat` (r ≈ 0,42); missing acak pada `Suhu_Mesin`, `Kelembaban`, `Downtime` |
| `machine_downtime.csv` | Studi Kasus Mandiri 2 | 520 baris × 9 kolom (10 mesin, 6 bulan) | `Usia_Mesin_Tahun` sebagian bernilai negatif (kesalahan entri, ±2,5%); `Biaya_Perbaikan_Ribu_Rupiah` hilang ±18%; `Kerusakan Mekanis` mendominasi penyebab (±39%); mesin lebih tua cenderung downtime lebih lama |
| `sensor_kualitas_elektronik.csv` | Studi Kasus Mandiri 3 | 1.800 baris × 7 kolom | Unit berstatus "Gagal" punya `Tegangan_Output_Volt` lebih rendah & `Suhu_Komponen_Celcius` lebih tinggi secara sistematis (sinyal untuk `hue='Status_Akhir'`); missing acak pada `Tegangan_Output_Volt` (6%) & `Arus_Output_Ampere` (7%) |
| `air_quality.csv` | Studi Kasus Kelompok 1 | 2.160 baris × 14 kolom (90 hari, per jam) | Pola diurnal jelas — polutan memuncak pada jam sibuk 08.00–10.00 & 18.00–20.00; nilai hilang ditandai sentinel **`-200`** (bukan kosong/NaN), sesuai instruksi buku ajar untuk `df.replace(-200, np.nan)` |
| `logistik_pengiriman.csv` | Studi Kasus Kelompok 2 | 520 baris × 10 kolom (10 pemasok, 1 tahun) | `Selisih_Hari` & `Tanggal_Tiba_Aktual` hilang untuk ±6% pengiriman yang masih *in-transit*; Kapal Laut punya rata-rata keterlambatan tertinggi (2,65 hari) > Kereta (1,17) > Truk (0,34), dan jarak makin jauh makin berisiko terlambat |

## Cara penggunaan

Mahasiswa cukup mengunduh berkas CSV yang relevan dan meletakkannya satu folder dengan *notebook* mereka (atau mengunggahnya ke sesi Google Colab), lalu memuatnya dengan `pd.read_csv('nama_berkas.csv')` sebagaimana dicontohkan di setiap bagian buku ajar. Semua nama kolom, satuan, dan kode kategori pada berkas ini konsisten dengan tabel deskripsi dataset di dalam buku ajar — tidak diperlukan penyesuaian nama kolom apa pun.

## Reproduksibilitas

Data dibangkitkan dengan `numpy.random.default_rng(seed=42)`. Skrip generator Python tersedia bila sewaktu-waktu perlu dibangkitkan ulang atau divariasikan (misalnya untuk mencegah mahasiswa saling menyalin hasil analisis persis sama antar-angkatan).
