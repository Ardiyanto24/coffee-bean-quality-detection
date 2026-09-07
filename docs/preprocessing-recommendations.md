# Preprocessing Recommendations

Disusun berdasarkan temuan dari dua EDA:
- [`CBQD - EDA v2 (Manual).ipynb`](../notebook/CBQD%20-%20EDA%20v2%20(Manual).ipynb) / [Laporan v2](../reports/CBQD%20-%20EDA%20v2%20Report.html)
- [`CBQD - EDA v3 (Defect Decomposition).ipynb`](../notebook/CBQD%20-%20EDA%20v3%20(Defect%20Decomposition).ipynb) / [Laporan v3](../reports/CBQD%20-%20EDA%20v3%20(Defect%20Decomposition)%20Report.html)

Setiap rekomendasi ditautkan ke temuan EDA yang memicunya, supaya keputusan preprocessing tidak lepas konteks dari data aslinya.

## 1. Data Quality Gates (otomatis, bukan manual)

EDA v2 Section 02 mengonfirmasi 0 file corrupt dan 100% RGB di train (1.211) maupun test (200). Ini kondisi awal yang bagus, tapi **jangan diasumsikan tetap benar** setiap kali dataset di-refresh dari R2.

- Tambahkan stage DVC baru `validate_dataset` (setelah `generate_manifest`) yang menjalankan assertion: `n_corrupt == 0`, `set(modes) == {"RGB"}`, `set(resolutions) == {(256,256)}`. Gagalkan pipeline (non-zero exit) kalau ada yang menyimpang.
- Jalankan gate ini di CI setiap kali `dataset.dvc` berubah, bukan cuma sekali saat EDA.

## 2. Deduplication & Leakage Handling

| Temuan (EDA v2 Section 09) | Jumlah | Tindakan |
|---|---|---|
| Exact-duplicate (MD5 identik), semua within-train | 11 grup | **Exclude** — sisakan 1 representatif per grup |
| Near-duplicate cross-class dalam train | 33 pasang | **Exclude** kedua sisi pasangan dari train |
| Near-duplicate same-class dalam train | 31 pasang | **Jangan exclude** — lihat Section 3 (split strategy) |
| Near-duplicate train↔test | 6 pasang | **Jangan hapus otomatis** — tandai & review manual |

- Exclusion untuk 2 baris pertama sudah punya implementasi referensi di notebook v2 Section 13 (`manifest_train_clean.csv`, 1.211 → 1.150). Formalkan ini jadi script `scripts/clean_manifest.py` + stage DVC `clean_manifest`, supaya tidak bergantung pada notebook dijalankan manual.
- **Sebelum exclude cross-class near-duplicate secara final**, tinjau manual pasangan dengan Hamming distance terkecil (≤2) — EDA v3 Section 07 menemukan bahwa mayoritas (68%) ambiguitas cross-class ini justru terjadi *antar* `premium`/`peaberry`/`longberry`, bukan soal `defect`. Sebagian bisa jadi kesalahan label sungguhan (perlu dikoreksi labelnya, bukan dibuang), bukan cuma "gambar mirip yang aman dihapus".
- 6 pasangan leakage train↔test: **jangan hapus baris test** (test bersifat fixed untuk evaluasi). Simpan `manifest_test_flagged.csv` dan baca skor test pada baris yang ditandai dengan skeptis — pertimbangkan melaporkan metrik test dengan dan tanpa 6 baris ini untuk transparansi.

## 3. Train/Validation Split Strategy — wajib cluster-aware

31 pasangan near-duplicate same-class (Section 09) berarti **random split biasa akan bocor**: bean yang difoto berkali-kali bisa masuk ke train dan validation sekaligus, membuat skor validasi optimis palsu.

- Split HARUS memakai `StratifiedGroupKFold` dengan `group = cluster_id` (hasil union-find atas pasangan near-duplicate) — referensi implementasi ada di notebook v2 Section 10, sudah tervalidasi (0 cluster terbelah lintas fold di run nyata).
- Formalkan sebagai stage DVC `assign_folds`, output `manifest_train_with_folds.csv` — kolom `fold` ini yang dipakai satu-satunya sumber kebenaran split di seluruh eksperimen modeling, jangan re-split ad hoc per eksperimen.
- **Jangan buang kolom `cluster_id` setelah fold ditentukan** — tetap berguna kalau nanti ingin mengganti skema CV (mis. dari 5-fold ke leave-one-cluster-out) tanpa menghitung ulang union-find dari nol.

## 4. Label Quality Remediation

### 4a. Kandidat mislabel dari mistakenness proxy (EDA v2 Section 12)
54 dari 1.211 gambar train (4,5%) ditandai `mistake_score > 0,5`, didominasi kelas `defect` (32/54).

- **Jangan langsung exclude.** Proxy ini berbasis model baseline yang sendiri cuma 73% akurat — false positive pasti ada. Perlakukan sebagai daftar prioritas untuk **audit manual oleh annotator/domain expert**.
- Sambil menunggu audit manual, opsi yang lebih aman untuk training: **sample-weight turun** (mis. weight 0,5) untuk 54 sampel ini, bukan exclude keras — supaya kalau proxy-nya salah, sampel tidak hilang sepenuhnya dari training.

### 4b. Hipotesis defect = campuran bean rusak (EDA v3, didukung 3 metode independen)
Investigasi v3 menemukan `defect` kemungkinan besar bukan kelas tersendiri, melainkan gabungan `premium`/`peaberry`/`longberry` yang rusak (proporsi sub-grup: peaberry-like ~35-50%, premium-like ~27-32%, longberry-like ~23-26%, tergantung metode).

- **Sebelum mengubah skema label**, konfirmasi ke pemilik/pelabel data — EDA memberi bukti kuat tapi bukan pengganti definisi label dari sumbernya (lihat laporan v3 Section 10, rekomendasi #5).
- Kalau terkonfirmasi: jalankan ulang pipeline clustering v3 (nearest-centroid + KMeans + classifier-proba) sebagai **stage DVC formal** (bukan cuma notebook analysis), hasilkan kolom `defect_subgroup` (`premium_like`/`peaberry_like`/`longberry_like`) di manifest. Kolom ini dipakai untuk:
  - stratifikasi split yang lebih halus (pastikan tiap sub-grup terwakili proporsional di tiap fold),
  - opsi hierarchical modeling (lihat `modeling-strategy.md`),
  - analisis error yang lebih granular saat evaluasi model.

### 4c. Ambiguitas antar 3 kelas murni (EDA v3 Section 07)
Temuan baru: 17 dari 25 pasangan cross-class near-duplicate (68%) adalah `longberry`-`premium`, `peaberry`-`premium`, atau `longberry`-`peaberry` — bukan soal `defect` sama sekali.

- Ini masalah kualitas label **independen** dari poin 4b. Rekomendasi: ekspor daftar pasangan ini secara terpisah (`ambiguous_type_pairs.csv`) untuk direview annotator, dengan pertanyaan spesifik "apakah kedua bean ini benar-benar jenis yang berbeda, atau salah satu salah label?" — jangan dicampur dengan review kandidat mislabel 4a supaya reviewer tidak bingung dua masalah berbeda dalam satu daftar.

## 5. Image-Level Preprocessing

- **Ukuran objek kecil terhadap frame** (bean cuma ~18-23% dari kanvas 256×256 — EDA v2 Section 06). Resize langsung ke resolusi model standar (mis. 224×224) tetap menyisakan bean sangat kecil dalam piksel. Pertimbangkan **crop-to-bbox + margin** (mis. 20% padding di sekeliling bounding box hasil segmentasi threshold dari EDA v2) sebagai preprocessing opsional untuk memperbesar proporsi sinyal terhadap noise, terutama kalau memakai arsitektur ringan yang sensitif terhadap resolusi efektif objek.
- **Normalisasi channel**: dataset sudah 100% RGB seragam — normalisasi standar ImageNet mean/std aman dipakai untuk model pretrained.
- **Chromatic aberration (fringing ungu/biru) di tepi bean** (EDA v2 Section 05, konsisten di semua kelas): tidak perlu dikoreksi (bukan sinyal pembeda kelas, muncul merata), tapi catat di dokumentasi model supaya interpretability tools (Grad-CAM dll.) tidak salah membaca fringing sebagai fitur bermakna.

## 6. Augmentation Guidance

| Augmentasi | Rekomendasi | Alasan (EDA) |
|---|---|---|
| Rotasi / flip | **Aman, dianjurkan** | Bean diletakkan bebas, tidak ada orientasi "benar" yang baku |
| Random crop / translasi ringan | **Dianjurkan** | Mitigasi shortcut framing — `premium` sedikit lebih terpusat (EDA v2 Section 06, meski efeknya kecil η²=0,046) |
| Color jitter **agresif** / grayscale | **Hindari / sangat ringan saja** | Warna adalah sinyal kuat (η²=0,22-0,24, EDA v2 Section 07) dan terbukti genuine — bukan artefak (EDA v3 Section 06). Augmentasi warna kuat berisiko menghapus sinyal asli, bukan cuma noise |
| Gaussian blur kuat | **Hindari** | Variance/tekstur adalah sinyal kuat (η²=0,221, EDA v2 Section 08) |
| Cutout / random erasing ringan | **Boleh dicoba** | Berpotensi membantu model tidak overfit ke bagian spesifik permukaan bean, tapi belum ada bukti EDA yang mendukung/menolak — validasi via ablation |

## 7. Ringkasan Perubahan pada Pipeline DVC

Stage yang direkomendasikan ditambahkan ke `dvc.yaml`, urut setelah `generate_manifest`:

```
generate_manifest → validate_dataset → clean_manifest → assign_folds → (opsional) assign_defect_subgroups
```

Setiap stage menghasilkan CSV yang di-diff-kan lewat `dvc.lock` seperti manifest sekarang — supaya keputusan preprocessing (bukan cuma raw data) ikut ter-versi dan bisa direproduksi ulang, bukan hidup hanya di dalam notebook.
