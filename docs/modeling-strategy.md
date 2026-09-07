# Modeling Strategy & Model Recommendations

Disusun berdasarkan temuan dari dua EDA:
- [`CBQD - EDA v2 (Manual).ipynb`](../notebook/CBQD%20-%20EDA%20v2%20(Manual).ipynb) / [Laporan v2](../reports/CBQD%20-%20EDA%20v2%20Report.html)
- [`CBQD - EDA v3 (Defect Decomposition).ipynb`](../notebook/CBQD%20-%20EDA%20v3%20(Defect%20Decomposition).ipynb) / [Laporan v3](../reports/CBQD%20-%20EDA%20v3%20(Defect%20Decomposition)%20Report.html)

Lihat juga [`preprocessing-recommendations.md`](./preprocessing-recommendations.md) — dokumen ini mengasumsikan langkah-langkah di sana (cluster-aware split, manifest bersih) sudah diterapkan.

## 1. Karakteristik Masalah (dari EDA, bukan asumsi)

| Karakteristik | Angka | Implikasi modeling |
|---|---|---|
| Ukuran dataset | 1.150-1.211 gambar train, 4 kelas | Kecil — rawan overfitting untuk model besar/dari-nol |
| Resolusi & isi gambar | 256×256, 1 bean per gambar, background polos, objek ~18-23% frame | Task visual sederhana secara komposisi — tidak perlu arsitektur untuk scene kompleks |
| Class balance | Rasio 1,033 | Tidak perlu class weighting/oversampling |
| Sinyal warna & tekstur | η²=0,22-0,27 (besar) pada mean RGB, variance, area_frac, bbox_ratio | Fitur low-level (warna, bentuk kasar) sudah cukup informatif — baseline non-deep-learning sudah masuk akal (73% akurasi hanya dari 11 fitur) |
| Struktur label | `defect` kemungkinan besar campuran 3 jenis lain yang rusak (EDA v3, 3 metode independen sepakat); + 68% ambiguitas near-dup cross-class justru antar `premium`/`peaberry`/`longberry` sendiri | Problem 4-kelas datar sebenarnya punya struktur laten 2 sumbu: **jenis bean** × **kondisi rusak/tidak** — bukan 4 kategori independen |
| Label noise | ~4,5% kandidat mislabel (proxy), beberapa near-duplicate cross-class kemungkinan salah label | Training harus toleran noise, bukan asumsi label 100% bersih |

## 2. Keputusan Framing: Flat 4-Class vs Hierarchical

Ini keputusan paling berdampak sebelum memilih model, jadi dibahas terpisah di depan.

- **Flat 4-class (baseline, mulai dari sini):** paling sederhana, sesuai skema label yang ada sekarang, cepat untuk dapat angka pertama. **Risiko yang sudah terbukti dari EDA:** recall `defect` akan tertahan (baseline hand-crafted feature saja sudah menunjukkan recall 0,57 — jauh di bawah 3 kelas lain 0,72-0,84) karena model dipaksa mempelajari kelas yang secara internal heterogen.
- **Hierarchical (jenis dulu, baru status rusak):** langsung mengikuti struktur laten yang ditemukan EDA v3. Berpotensi menaikkan recall `defect` signifikan karena tiap sub-model mengerjakan tugas yang lebih homogen. **Trade-off:** butuh 2 model/2 head, pipeline inference lebih kompleks, dan makin bergantung pada konfirmasi domain bahwa hipotesis "defect = lintas jenis" ini benar (lihat `preprocessing-recommendations.md` §4b).

**Rekomendasi:** jalankan flat 4-class dulu sebagai baseline wajib (Fase 1), baru masuk ke hierarchical/multi-task (Fase 2) sebagai perbaikan terarah pada titik lemah yang sudah diketahui — bukan tebak-tebakan.

## 3. Strategi Evaluasi

- **CV wajib cluster-aware** — pakai kolom `fold` dari `manifest_train_with_folds.csv` (lihat `preprocessing-recommendations.md` §3). Model apa pun yang dievaluasi dengan random split biasa, angkanya tidak bisa dipercaya untuk dataset ini.
- **Metrik utama: macro-F1**, bukan akurasi — karena kelemahan tersebar tidak rata (defect jauh lebih sulit), akurasi keseluruhan bisa menutupi masalah spesifik.
- **Baca confusion matrix sebagai dua masalah terpisah** (rekomendasi eksplisit dari laporan v3): (a) defect vs 3 kelas lain = soal deteksi kerusakan, (b) kesalahan antar premium/peaberry/longberry = soal identifikasi jenis. Satu angka akurasi menyembunyikan mana dari dua masalah ini yang sebenarnya dominan.
- **Gate wajib:** model baru HARUS mengalahkan baseline 73% (RandomForest + 11 fitur hand-crafted, EDA v2 Section 11) pada protokol CV yang sama. Kalau tidak, ada yang salah di pipeline (bukan modelnya kurang canggih).

## 4. 10 Rekomendasi Model

Diurutkan sebagai roadmap eksperimen (bukan ranking kualitas) — nomor kecil = coba lebih dulu.

### 1. Gradient Boosting (XGBoost/LightGBM) di atas 11 fitur hand-crafted
**Kenapa:** EDA v2 Section 11 sudah membuktikan fitur bentuk+warna+tekstur mencapai 73% dengan RandomForest. Gradient boosting hampir selalu mengungguli RF pada data tabular dengan effort tuning yang sama — cara termurah untuk menaikkan lower-bound sebelum masuk ke deep learning. Latih dalam hitungan menit di CPU.

### 2. MobileNetV3 (Small/Large), transfer learning
**Kenapa:** gambar secara visual sederhana (1 objek, background polos) — arsitektur ringan berorientasi mobile kemungkinan besar **cukup**, bukan under-powered. Relevan langsung kalau target akhirnya perangkat sortir bean di lapangan (edge deployment).

### 3. EfficientNet-B0/B1, transfer learning (fine-tune bertahap)
**Kenapa:** rasio akurasi/efisiensi terbaik di kelasnya untuk dataset kecil-menengah. Freeze backbone dulu, baru unfreeze layer akhir — dataset ~1.200 gambar terlalu kecil untuk fine-tune penuh dari awal training.

### 4. ResNet-18/34, transfer learning
**Kenapa:** arsitektur paling dipahami & paling banyak tooling-nya — baseline CNN yang stabil untuk dibandingkan dengan kandidat lain. Varian kecil (18/34, bukan 50+) supaya tidak overfit pada data sekecil ini.

### 5. ConvNeXt-Tiny, transfer learning
**Kenapa:** alternatif yang lebih modern dari ResNet dengan inductive bias konvolusional yang tetap terjaga (beda dari ViT murni) — cocok dicoba sebagai upgrade dari #4 kalau resource memadai, risiko overfitting lebih terkendali dibanding Vision Transformer murni pada data sekecil ini.

### 6. Vision Transformer kecil (ViT-Tiny/Small), transfer learning
**Kenapa dicoba:** menangkap pola global (mis. hubungan warna-bentuk keseluruhan bean) yang mungkin terlewat CNN lokal. **Catatan risiko:** ViT tanpa inductive bias konvolusional lebih rakus data — HANYA coba dengan augmentasi kuat + regularisasi ketat, dan bandingkan langsung ke #3/#5 sebelum diadopsi; jangan jadi pilihan pertama untuk dataset sekecil ini.

### 7. Two-stage hierarchical: classifier jenis (premium/peaberry/longberry) + classifier rusak/tidak terpisah
**Kenapa:** implementasi langsung dari rekomendasi utama laporan v3. Memecah problem 4-kelas yang secara internal tumpang tindih jadi dua sub-problem yang masing-masing lebih homogen. Backbone tiap stage bisa memakai arsitektur ringan (mis. #2 atau #3) karena tugasnya lebih sempit.

### 8. Multi-task single-backbone: satu backbone + dua output head (jenis & status rusak)
**Kenapa:** alternatif #7 yang lebih hemat data — berbagi representasi tingkat rendah (edge, warna) antar dua task lewat satu backbone, biasanya lebih data-efficient daripada dua model penuh terpisah saat total data cuma ~1.200 gambar. Coba ini dulu sebelum #7 kalau resource/waktu terbatas.

### 9. CNN + label smoothing / focal loss, dengan sample-weighting pada kandidat mislabel
**Kenapa:** EDA menemukan label noise nyata tapi berbasis proxy (bukan ground truth pasti). Melatih dengan asumsi label 100% bersih itu berisiko; label smoothing + turunkan bobot 54 kandidat mistakenness (bukan exclude keras — lihat `preprocessing-recommendations.md` §4a) adalah pendekatan yang lebih robust terhadap ketidakpastian label ini.

### 10. Ensemble: CNN terbaik (mis. hasil #3) + Gradient Boosting fitur hand-crafted (#1)
**Kenapa:** fitur hand-crafted (bentuk/warna/tekstur) terbukti membawa sinyal kuat dan mudah diinterpretasi; CNN belajar representasi yang mungkin overlap sebagian tapi juga menangkap pola yang tidak terkodekan manual. Late-fusion keduanya (avg probabilitas atau stacking) sering menambah robustness pada dataset kecil, dan komponen fitur hand-crafted-nya tetap bisa dijelaskan ke stakeholder non-teknis.

## 5. Roadmap Eksperimen Ringkas

1. **Fase 1 — Baseline wajib:** Model #1 (gradient boosting) dan salah satu dari #2/#3, keduanya dengan CV cluster-aware. Tujuan: pastikan pipeline data & evaluasi benar, dapatkan angka acuan.
2. **Fase 2 — Tangani titik lemah yang sudah diketahui:** kalau recall `defect` tetap jadi masalah utama, coba #7 atau #8 (hierarchical/multi-task). Kalau ambiguitas antar jenis (premium/peaberry/longberry) yang dominan di confusion matrix, prioritaskan audit label manual (`preprocessing-recommendations.md` §4c) sebelum ganti arsitektur.
3. **Fase 3 — Robustifikasi & refinement:** #9 (noise-robust training) sebagai lapisan tambahan di atas model terbaik Fase 1/2; #10 (ensemble) dan #5/#6 (arsitektur lanjutan) sebagai upaya marginal-gain setelah struktur masalah utama (framing, label noise) sudah ditangani — bukan langkah pertama.
