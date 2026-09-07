# Strategi XAI — Alasan Pemilihan Metode

Disusun setelah full run 10 model ([`CBQD - Training.ipynb`](../notebook/CBQD%20-%20Training.ipynb) / [Laporan Training](../reports/CBQD%20-%20Training%20Report.html)) dan berdasarkan temuan dua EDA sebelumnya ([Laporan EDA v2](../reports/CBQD%20-%20EDA%20v2%20Report.html), [Laporan EDA v3](../reports/CBQD%20-%20EDA%20v3%20(Defect%20Decomposition)%20Report.html)). Dokumen ini menjelaskan **kenapa** tiap metode dipilih atau ditolak untuk tiap kombinasi sudut-pandang × model — bukan sekadar daftar metode. Lihat juga [`modeling-strategy.md`](./modeling-strategy.md) untuk konteks 10 model itu sendiri.

## 1. Konteks & Motivasi

Full run menghasilkan `09_noise_robust` sebagai juara (macro-F1 0,961), tapi gap ke runner-up `05_convnext_tiny` (0,948) cuma 1,3 poin persentase. Langkah wajar berikutnya adalah validasi 4-fold CV untuk memastikan peringkat ini bukan kebetulan dari satu pembagian data. Tapi ada pertanyaan yang lebih mendasar yang harus dijawab dulu: **4-fold CV cuma menguji stabilitas skor terhadap pembagian data — bukan apakah model "berpikir" dengan benar.**

Kalau sebuah model mendapat skor tinggi karena bergantung pada shortcut yang konsisten ada di SELURUH dataset (misalnya posisi bean dalam frame, atau artefak dari proses crop-to-bbox kita sendiri), maka 4-fold CV yang stabil justru memberi rasa percaya diri yang salah — shortcut itu akan "bekerja" secara konsisten di semua fold juga, karena datanya sama, cuma pembagiannya yang beda.

**Prinsip yang dipegang:** perbandingan antar model harus memastikan dulu cara berpikirnya benar, sebelum peringkat macro-F1 dipercaya sebagai perbandingan yang adil. Ini kenapa XAI diterapkan ke **seluruh 10 model**, bukan cuma pemenang — termasuk untuk menjelaskan temuan yang masih berupa observasi di laporan training (khususnya kegagalan ensemble di Section 6, yang saat ini cuma diketahui GEJALANYA — macro-F1 turun — tapi belum diketahui MEKANISMENYA).

Kecurigaan awal cuma satu (posisi/framing), tapi ditinjau ulang jadi **lima hipotesis kecurigaan** setelah disadari bahwa "posisi" cuma satu dari beberapa sinyal EDA yang berpotensi jadi shortcut:

| # | Hipotesis | Sumber kecurigaan |
|---|---|---|
| 1 | Posisi (`off_center`) dipakai sebagai shortcut ke `premium` | EDA v2 §06 |
| 2 | Konsep "rusak" bocor ke prediksi kelas non-defect | EDA v3 |
| 3 | Model tidak generalisasi ke gambar dunia nyata (`real_world`, tak pernah dilihat) | Praktik umum -- setiap model perlu diuji di luar distribusi evaluasi standarnya |
| 4 | Warna (`dark_color`) dipakai sebagai shortcut ke `defect`, bukan sinyal genuine | EDA v2 -- effect size terbesar dari semua fitur |
| 5 | Ukuran-di-frame (`large_area`) dipakai sebagai shortcut ke `peaberry` | EDA v2 §07/§08 |

**Prinsip cakupan (ditegaskan ulang setelah revisi):** kelima hipotesis ini diuji di **seluruh 10 model, tanpa kecuali** -- bukan cuma model yang paling mudah diadaptasi. Kalau suatu kombinasi hipotesis×model secara arsitektural tidak mungkin diuji (mis. target kelasnya tidak ada di ruang output sub-model tertentu), itu dilaporkan eksplisit sebagai **N/A** di tabel hasil -- bukan dilewati diam-diam. Menyempitkan cakupan demi menghemat waktu proses BUKAN alasan yang sah di project ini; batasan yang sah hanya batasan arsitektural nyata. Lihat Section 6 prinsip #7.

## 2. Lima Sudut Pandang, Lima Celah Berbeda

| Sudut pandang | Pertanyaan yang dijawab | Celah yang ditutup |
|---|---|---|
| **Spasial** | Bagian gambar mana yang dilihat model? | Bukti visual langsung — tapi cuma korelasional, gampang disalahtafsirkan sendirian |
| **Konsep semantik** | Seberapa besar konsep tertentu (posisi, bentuk, warna) berkontribusi ke keputusan? | Mengubah "kelihatannya model fokus ke tengah" dari dugaan visual jadi angka terukur |
| **Kausal/counterfactual** | Perubahan minimal apa yang membalik prediksi? | Bukti sebab-akibat langsung dari intervensi nyata, bukan sekadar korelasi heatmap |
| **Data lineage** | Contoh training mana yang paling mempengaruhi prediksi ini? | Membedakan masalah data (mislabel, near-duplicate) dari masalah model |
| **Agregat/statistik** | Apakah pola ini sistemik, atau kebetulan di 1–2 gambar? | Validasi bahwa temuan dari 4 sudut pandang di atas bukan anekdot |

Kelima sudut pandang ini sengaja dipertahankan semua (bukan dipilih salah satu) karena masing-masing menutup kelemahan sudut pandang lain: spasial saja rawan salah tafsir (Grad-CAM bisa menyala di area yang benar tapi karena alasan yang salah), TCAV mengkuantifikasi tapi butuh definisi konsep yang jelas, probe kausal membuktikan sebab-akibat tapi cuma untuk satu gambar pada satu waktu, data lineage menunjuk ke sumber masalah tapi tidak menjelaskan mekanisme keputusan, dan tanpa lapisan agregat, keempatnya rawan jadi kumpulan anekdot yang meyakinkan tapi tidak representatif — persis risiko yang sudah pernah terjadi di EDA v2 §12 (54 kandidat mistakenness dari proxy classifier yang sendiri tidak sempurna).

## 3. Kenapa Model Perlu Dikelompokkan per Keluarga Arsitektur

Lineup 10 model bukan 10 varian dari satu arsitektur — ada CNN murni, transformer, model tabular berbasis pohon, model dua-kepala, dan ensemble. Metode XAI paling populer (Grad-CAM) butuh feature map konvolusi yang cuma dimiliki sebagian dari mereka. Memaksakan satu metode untuk semua model akan menghasilkan kegagalan diam-diam (mis. Grad-CAM dijalankan di layer yang salah pada model tanpa struktur konvolusi yang sesuai) atau kode yang gagal total. Karena itu, ke-10 model dikelompokkan jadi lima keluarga, dan tiap keluarga punya "adapter" metode sendiri per sudut pandang — bukan lima notebook terpisah, tapi lima cabang logika di satu notebook yang sama.

| Keluarga | Anggota | Karakteristik yang menentukan adapter |
|---|---|---|
| A. CNN polos | 02, 03, 04, 05, 09 | Punya feature map konvolusi di layer akhir — kompatibel native dengan Grad-CAM |
| B. Transformer | 06 (DeiT-Tiny) | Representasi berbasis attention token, bukan feature map spasial konvolusi |
| C. Tabular/tree | 01 (Gradient Boosting) | Input berupa 11 fitur hand-crafted, bukan piksel — tidak punya "region gambar" sama sekali di dalam model |
| D. Multi-submodel | 07 (hierarchical, 2 model terpisah), 08 (multi-task, 1 backbone 2 head) | Punya lebih dari satu keputusan/head per gambar — butuh penjelasan per sub-bagian |
| E. Ensemble | 10 | Bukan model tunggal — gabungan probabilitas dari C dan salah satu anggota A |

## 4. Pemetaan Metode & Alasan per Keluarga

### A. CNN Polos (02, 03, 04, 05, 09)

- **Spasial → Grad-CAM.** Kenapa: native untuk arsitektur ini, murah dihitung (satu backward pass), dan sudah tervalidasi luas di literatur untuk backbone CNN pretrained ImageNet seperti ini. Dilengkapi **Integrated Gradients** sebagai cross-check — Grad-CAM dikenal punya failure mode saturasi (heatmap tetap "menyala" walau gradiennya sudah kecil), jadi butuh metode kedua yang tidak bergantung pada feature map layer tertentu. **Occlusion Sensitivity** ditambahkan sebagai cross-check ketiga yang sepenuhnya black-box (cuma forward pass berulang), untuk menangkap kasus di mana kedua metode berbasis gradien sama-sama menyesatkan.
- **Konsep → TCAV di layer intermediate.** Kenapa: keluarga ini punya representasi kontinu di layer tengah yang bisa dipakai untuk melatih Concept Activation Vector. Concept split dipakai ulang dari fitur hand-crafted EDA (`center_offset`, elongasi/aspect ratio dari PCA, `area_frac`, mean darkness) — bukan mengumpulkan gambar konsep baru — karena kita SUDAH punya kecurigaan konkret dari EDA v2 §06 (`premium` sedikit lebih terpusat), bukan tebakan buta yang butuh definisi konsep baru dari nol.
- **Kausal → probe reposisi/mask.** Kenapa: kita sudah punya logika segmentasi foreground/background dari `preprocess_dataset.py` (threshold grayscale untuk bounding box) — bisa dipakai ulang untuk menggeser posisi bean dalam frame atau menutup separuh gambar, lalu mengukur apakah probabilitas kelas berubah signifikan. Ini intervensi kausal langsung terhadap hipotesis framing shortcut, bukan cuma observasi pasif seperti heatmap.
- **Data lineage → nearest-neighbor di embedding penultimate layer.** Kenapa BUKAN influence functions: dataset kita kecil (929 gambar train) jadi influence functions formal sebenarnya *feasible* secara komputasi, tapi implementasinya (Hessian-vector product, damping, aproksimasi) rewel secara numerik untuk nilai tambah yang belum tentu sepadan di tahap eksplorasi ini. NN-embedding jauh lebih murah (cuma forward pass + pencarian tetangga terdekat) dan langsung menguji temuan EDA v2 §09/EDA v3 soal 68% ambiguitas near-duplicate cross-class: kalau tetangga terdekat sebuah `premium` yang salah klasifikasi ternyata mayoritas berlabel `peaberry`/`longberry`, itu bukti kuat masalah data, bukan model.
- **Agregat → korelasi centroid Grad-CAM vs `center_offset`/`area_frac` EDA, dihitung di ≥50 sampel per kelas.** Kenapa: satu heatmap yang "mencurigakan" tidak berarti apa-apa tanpa pembanding populasi — baru jadi bukti sistemik kalau polanya konsisten di banyak sampel dan berkorelasi dengan fitur yang sudah diketahui EDA, bukan kebetulan framing di 1-2 gambar.

### B. Transformer (06 DeiT-Tiny)

- **Spasial → Attention Rollout** (atau CAM berbasis attention), bukan Grad-CAM. Kenapa: DeiT tidak punya feature map spasial konvolusi di layer akhir — Grad-CAM klasik tidak punya tempat untuk "menempel". Attention Rollout mengagregasi bobot attention lintas layer transformer, mekanismenya secara fundamental berbeda dari Grad-CAM meski tujuannya sama (heatmap wilayah penting).
- **Konsep, Kausal, Data lineage → sama seperti keluarga A**, dengan embedding diganti jadi representasi token `[CLS]` alih-alih feature map konvolusi. Kenapa bisa disamakan: TCAV, probe kausal, dan NN-embedding semuanya cuma butuh (a) representasi kontinu di suatu layer, dan (b) kemampuan forward pass — keduanya tetap tersedia di arsitektur transformer, jadi tidak perlu adapter baru di luar penggantian sumber embedding.

### C. Tabular/Tree (01 Gradient Boosting)

- **Spasial → SHAP / feature-importance, BUKAN heatmap piksel.** Kenapa: model ini tidak pernah melihat piksel sama sekali — inputnya 11 angka hand-crafted (bentuk, warna, tekstur). Tidak ada "region gambar" untuk disorot di dalam model itu sendiri; SHAP di atas 11 fitur itu adalah analog yang paling setara (sama-sama menjawab "apa yang mendorong keputusan ini", cuma satuannya fitur bukan piksel).
- **Konsep → TCAV neural tidak dipakai, ditandai redundan — tapi TETAP dijawab lewat peringkat SHAP.** Kenapa TCAV redundan: TCAV adalah cara memproyeksikan representasi laten yang tidak transparan ke arah konsep yang bisa dipahami manusia. Model ini sudah transparan dari awal — 11 fiturnya SENDIRI adalah konsep yang bisa dipahami manusia (warna, luas area, densitas tepi). Memaksakan TCAV di sini menambah langkah tanpa menambah informasi baru. Tapi ini bukan alasan untuk melewati Hipotesis #1/#4/#5 di model ini: peringkat kepentingan SHAP untuk fitur yang bersesuaian (`center_offset`, `mean_r`, `area_frac`) adalah padanan langsungnya — peringkat 1 (paling penting) sama maknanya dengan TCAV score tinggi & signifikan di model neural. Redundansi metode (TCAV) tidak sama dengan redundansi hipotesis (H1/H4/H5 tetap wajib dijawab, cuma lewat instrumen yang berbeda).
- **Hipotesis #2 (kebocoran "rusak") → proyeksi linear di ruang 11-fitur, bukan gradien.** Kenapa: GB/LightGBM tidak bisa diturunkan (non-differentiable), jadi "arah rusak" dihitung dengan fungsi CAV yang sama (`compute_cav`) tapi diterapkan generik ke matriks fitur tabular alih-alih aktivasi neural — hasilnya arah vektor 11-dimensi yang memisahkan sampel defect/non-defect, lalu sensitivitas dihitung sebagai proyeksi (dot product) fitur test ke arah itu, dikorelasikan ke tingkat kesalahan klasifikasi ke `defect`.
- **Hipotesis #3 (generalisasi `real_world`) → hitung ulang 11 fitur dari gambar mentah `real_world/`, lalu `predict_proba` seperti biasa.** Kenapa ini valid tanpa adapter baru: fitur hand-crafted cuma butuh gambar (via `orig_path`, tersedia di semua split termasuk `real_world`), tidak butuh label — jadi confidence & distribusi prediksi bisa dibandingkan ke test persis seperti model neural.
- **Kausal → probe reposisi/mask, lalu hitung ulang 11 fitur, lalu cek prediksi berubah.** Kenapa masih relevan meski modelnya tabular: perturbasi terjadi di level GAMBAR (sebelum ekstraksi fitur), jadi tetap bisa menguji hipotesis yang sama seperti keluarga A (mis. apakah menggeser posisi bean mengubah `center_offset` yang dihitung ulang, dan apakah itu cukup mengubah prediksi).
- **Data lineage → vektor 11-fitur itu sendiri sebagai "embedding".** Kenapa ini justru lebih sederhana dari keluarga A: tidak perlu memilih layer mana yang jadi representasi — modelnya memang cuma beroperasi di ruang 11 dimensi itu.
- **Agregat → stabilitas peringkat SHAP antar sampel per kelas.** Kenapa: pertanyaannya bukan cuma "fitur apa yang penting", tapi apakah fitur yang sama konsisten jadi penentu utama di seluruh sampel satu kelas, atau berubah-ubah tanpa pola — indikasi apakah model punya strategi yang bisa diringkas atau tidak.

### D. Multi-submodel (07 Hierarchical, 08 Multi-task)

- **Semua sudut pandang → sama seperti keluarga A, diterapkan per sub-model/head.** Kenapa: baik 07 (dua EfficientNet-B0 terpisah) maupun 08 (satu backbone, dua head) pada akhirnya cuma kumpulan model keluarga A yang digabung — tidak perlu adapter metode baru, cukup jalankan pipeline keluarga A dua kali (per sub-bagian) dan bandingkan hasilnya satu sama lain.
- **Konsep (TCAV) → dijalankan per sub-model, dengan penanganan N/A eksplisit.** Kenapa perlu ditangani khusus (beda dari keluarga A): `type_model`/`type_wrapper` cuma 3-kelas (premium/peaberry/longberry) dan `damage_model`/`damage_wrapper` cuma 2-kelas (intact/defect) — tiap konsep (H1/H4/H5) cuma valid diuji di sub-model yang target kelasnya benar-benar ADA di ruang outputnya (mis. `dark_color`→`defect` cuma bisa diuji di sub-model damage, bukan type). Kombinasi yang target-kelasnya tidak ada dilaporkan **None/N/A**, bukan dipaksakan ke kelas terdekat atau dilewati diam-diam — ini keterbatasan arsitektural nyata, bukan penyempitan cakupan demi kepraktisan.
- **Hipotesis #2 (kebocoran "rusak") → dijalankan HANYA di sub-model yang punya kelas `defect`/rusak** (`damage_model` untuk 07, `damage_wrapper` untuk 08). Kenapa `type_model`/`type_wrapper` ditandai N/A untuk hipotesis ini: sub-model tipe memang tidak pernah melihat kelas rusak sama sekali (dipisah sejak desain hierarchical/multi-task), jadi "kebocoran ke prediksi non-defect" secara harfiah tidak bisa terjadi di ruang output sub-model itu.
- **Hipotesis #3 (generalisasi `real_world`) → proba gabungan lewat combiner yang sudah ada** (`hierarchical_predict_proba_pil`, `multitask_predict_proba_pil`), dijalankan per-gambar untuk `real_world/` sama seperti untuk `test`. Kenapa tidak perlu adapter baru: kedua fungsi combiner itu sudah menerima PIL image mentah dan mengembalikan probabilitas 4-kelas flat, cocok langsung dipakai ulang untuk split manapun.
- **Agregat → tingkat kesepakatan area-perhatian antar dua head/sub-model.** Kenapa ini jadi pertanyaan unik untuk keluarga D (tidak relevan untuk keluarga lain): desain hierarchical/multi-task berasumsi dua sub-task (jenis bean × status rusak) bisa dipisah secara bermakna. Kalau Grad-CAM head-tipe dan head-rusak konsisten menyorot wilayah gambar yang BERBEDA secara sistemik, itu bukti desainnya bekerja seperti niatnya. Kalau areanya nyaris identik, itu sinyal kedua head sebenarnya belajar hal yang sama — konsisten dengan temuan laporan training bahwa performa gabungan 07/08 tidak melampaui CNN flat biasa (Section 4), meski tiap sub-task sendiri sangat akurat (val-F1 0,994 dan 0,978).

### E. Ensemble (10)

- **Semua sudut pandang → gabungan penjelasan C (Gradient Boosting) dan A (ResNet-18, komponen CNN yang terpilih) berdampingan.** Kenapa tidak ada metode tunggal terpadu: model #10 bukan satu jaringan, jadi tidak ada satu set bobot untuk dianalisis — yang bisa dianalisis cuma dua komponennya secara terpisah lalu dibandingkan.
- **Hipotesis #2/#3 → dampak AKTUAL pada probabilitas gabungan (0,5×GB + 0,5×ResNet-18), bukan skor sensitivitas.** Kenapa berbeda dari keluarga lain: ensemble tidak punya representasi/gradien bersama untuk dihitung arah "rusak"-nya (beda dari GB yang punya proyeksi linear, atau CNN yang punya gradien) — satu-satunya besaran yang valid secara metodologis adalah dampak yang benar-benar terjadi di OUTPUT gabungan (tingkat misklasifikasi ke `defect` untuk H2, confidence & distribusi kelas di `real_world` untuk H3), dihitung dari probabilitas GB dan ResNet-18 yang sudah dikombinasikan sesuai definisi ensemble itu sendiri. Melaporkan angka sensitivitas palsu di sini akan menyiratkan mekanisme yang tidak benar-benar ada.
- **Agregat → tingkat setuju/tidak-setuju prediksi GB vs ResNet-18, dikorelasikan ke kasus ResNet-18 benar.** Kenapa ini prioritas tertinggi di seluruh rencana XAI: laporan training (Section 6) sudah menemukan ensemble *lebih jelek* dari ResNet-18 sendirian (0,891 vs 0,926), tapi baru sebatas observasi — analisis ini punya potensi langsung mengubahnya jadi penjelasan mekanistik (mis. "GB dan ResNet-18 tidak sepakat justru di kasus yang ResNet-18 benar, sehingga rata-rata 50/50 menariknya ke arah yang salah").

## 5. Metode yang Dipertimbangkan Tapi Tidak Dipakai

### Influence Functions (Hessian-based)

Dipertimbangkan untuk sudut pandang data lineage sebagai alternatif NN-embedding yang lebih rigorous (mengukur pengaruh kausal-perturbatif satu sampel training terhadap prediksi, bukan cuma kedekatan di ruang embedding). **Ditunda**, bukan ditolak permanen: implementasinya butuh aproksimasi Hessian-vector product yang secara numerik rewel (sensitif terhadap damping factor, konvergensi), dan nilai tambahnya di atas NN-embedding belum jelas sepadan di tahap eksplorasi ini. Direkomendasikan untuk dieskalasi HANYA kalau NN-embedding menemukan sesuatu yang ambigu dan butuh atribusi lebih presisi.

### Counterfactual Generatif (GAN/Diffusion-based)

Dipertimbangkan untuk sudut pandang kausal sebagai versi counterfactual yang lebih formal (menghasilkan gambar baru yang realistis dengan perubahan minimal yang membalik prediksi). **Tidak dipakai**: butuh melatih atau mengadaptasi model generatif terpisah — infrastruktur baru yang belum ada di project ini — untuk dataset yang cuma 929 gambar training. Probe manual (reposisi/mask terprogram) memberi bentuk bukti kausal yang sama (intervensi langsung pada input, bukan cuma korelasi), dengan biaya implementasi jauh lebih rendah karena memakai ulang logika segmentasi yang sudah ada di `preprocess_dataset.py`.

## 6. Prinsip Desain Notebook

1. **Cakupan seluruh 10 model, bukan cuma pemenang** — supaya peringkat macro-F1 di laporan training bisa dipercaya sebagai perbandingan yang adil, dan kegagalan ensemble bisa dijelaskan mekanismenya.
2. **Satu notebook, lima cabang adapter** — dikelompokkan per keluarga arsitektur (Section 3–4), bukan lima notebook terpisah atau (sebaliknya) satu implementasi seragam yang gagal diam-diam di model yang strukturnya tidak cocok.
3. **Probe kausal (reposisi/mask) sebagai metode paling universal** — satu-satunya yang jalan di ke-10 model nyaris tanpa modifikasi, jadi fondasi utama untuk perbandingan lintas keluarga arsitektur yang berbeda-beda.
4. **Agregat/statistik wajib untuk semua temuan** — dihitung di ≥50 sampel per kelas, dikorelasikan ke metadata EDA yang sudah ada, sebelum pola apa pun diklaim sistemik.
5. **TCAV memakai ulang fitur EDA sebagai concept split** — tidak mengumpulkan gambar konsep baru, karena kecurigaannya sudah konkret dari EDA v2 §06 dan §11.
6. **Perbandingan lintas keluarga dibaca sebagai konsistensi arah temuan, bukan kesetaraan angka** — Grad-CAM (piksel) dan SHAP (fitur tangan) adalah dua BENTUK bukti berbeda; kesimpulan yang valid adalah soal apakah keduanya menunjuk ke arah yang sama (mis. sama-sama menandai warna sebagai penentu utama `defect`), bukan membandingkan skalanya secara langsung.
7. **Tidak ada penyempitan cakupan demi kepraktisan waktu proses** — kelima hipotesis wajib diuji di seluruh 10 model. Satu-satunya alasan sah untuk tidak menguji suatu kombinasi hipotesis×model adalah keterbatasan ARSITEKTURAL nyata (mis. target kelas konsep tidak ada di ruang output sub-model tertentu — lihat Section 4D), dan itu pun harus dilaporkan eksplisit sebagai **N/A** di tabel hasil, bukan dilewati diam-diam. Notebook ini milik pemilik project dan dijalankan di atas kuota komputasinya sendiri — durasi run bukan pertimbangan yang sah untuk mengurangi cakupan analisis.

## 7. Referensi

- [`CBQD - Training.ipynb`](../notebook/CBQD%20-%20Training.ipynb) / [Laporan Training](../reports/CBQD%20-%20Training%20Report.html) — hasil 10 model yang jadi objek analisis XAI ini.
- [`CBQD - EDA v2 (Manual).ipynb`](../notebook/CBQD%20-%20EDA%20v2%20(Manual).ipynb) / [Laporan EDA v2](../reports/CBQD%20-%20EDA%20v2%20Report.html) — sumber fitur hand-crafted untuk TCAV concept split (§06, §07, §08) dan proxy label-noise (§12).
- [`CBQD - EDA v3 (Defect Decomposition).ipynb`](../notebook/CBQD%20-%20EDA%20v3%20(Defect%20Decomposition).ipynb) / [Laporan EDA v3](../reports/CBQD%20-%20EDA%20v3%20(Defect%20Decomposition)%20Report.html) — dasar hipotesis struktur hierarchical yang diuji ulang lewat XAI di Section 4 (keluarga D).
- [`modeling-strategy.md`](./modeling-strategy.md) — definisi 10 model dan roadmap fase yang jadi objek analisis.
- [`preprocessing-recommendations.md`](./preprocessing-recommendations.md) — logika segmentasi foreground/background yang dipakai ulang untuk probe kausal.
