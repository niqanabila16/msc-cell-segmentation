# Automation Testing — msc-cell-segmentation

## Riwayat perombakan (baca dulu kalau bingung kenapa strukturnya begini)

Versi pertama suite ini punya **satu file Selenium per FR (FR01-FR10)**,
dan tiap file mengunggah citra+label lalu menjalankan inferensi U-Net
DARI NOL. Itu menyebabkan dua masalah nyata yang ditemukan lewat
pemakaian sungguhan:

1. **Bug locator** (sudah diperbaiki): `click_button()` mencari tombol
   lewat `button[normalize-space(text())="..."]`, padahal Streamlit
   membungkus teks tombol di elemen anak (`<button><div><p>Label</p>
   </div></button>`), sehingga `text()` (hanya teks anak LANGSUNG) selalu
   kosong dan predikat itu TIDAK PERNAH cocok. Diganti jadi
   `normalize-space(.)` (string-value seluruh subtree elemen). Bug kedua
   yang ditemukan bersamaan: `Keys.HOME` tidak reliable mereset slider
   BaseWeb ke nilai minimum -- diganti jadi menekan `ARROW_LEFT` berkali-
   kali (dijamin mentok minimum lewat clamping) sebelum `ARROW_RIGHT`.
2. **Redundansi eksekusi** (structural, sudah dirombak): 1 file per FR
   berarti setiap FR yang berhubungan dengan hasil inferensi (FR02, FR05,
   FR06, FR07, FR08, FR09, FR10) mengulang upload+inferensi dari nol,
   padahal semuanya cuma memeriksa HASIL dari satu tindakan pengguna yang
   sama. Total 48 test x rata-rata 15-90 detik per tunggu = 17+ menit.

## Kenapa strukturnya sekarang begini (dengan sumber)

**Piramida test / minimalkan lapisan E2E.** Martin Fowler: *"you should
have many more low-level unit tests than high-level tests running
through a GUI"* -- [martinfowler.com/bliki/TestPyramid.html](https://martinfowler.com/bliki/TestPyramid.html).
Google Testing Blog menjelaskan alasannya: test lewat UI/browser lambat,
flaky, dan mahal untuk didiagnosis -- [testing.googleblog.com/2015/04/just-say-no-to-more-end-to-end-tests.html](https://testing.googleblog.com/2015/04/just-say-no-to-more-end-to-end-tests.html).
Konsekuensinya: **rumus matematika di `src/metrics.py` dan logika
validasi/decode di `src/data_utils.py` sekarang diuji sebagai unit test
Python biasa** (`tests/unit/`) -- tidak perlu Streamlit, tidak perlu
browser, jalan dalam < 1 detik untuk 33 test, DAN bisa memverifikasi
angka yang BENAR (bukan cuma "masuk akal 0-1") karena TP/FP/FN dikontrol
manual lewat array buatan tangan.

**Selenium seharusnya tidak dipakai untuk menyiapkan state.** Dokumentasi
resmi Selenium, *"Generating application state"*: *"Selenium should not
be used to prepare a test case... existing APIs should be leveraged to
create data for the AUT"* -- [selenium.dev/documentation/test_practices/encouraged/generating_application_state/](https://www.selenium.dev/documentation/test_practices/encouraged/generating_application_state/).
Karena Streamlit tidak punya API non-UI untuk menyuntik state upload,
adaptasinya di sini: upload+inferensi (satu-satunya bagian yang benar-
benar mahal) dijalankan **sekali per skenario**, lalu banyak hal
diperiksa dari SATU hasil itu -- bukan diulang per FR.

**Tapi "fresh browser per test" & "avoid sharing state" tetap
direkomendasikan resmi** -- [selenium.dev/.../fresh_browser_per_test/](https://www.selenium.dev/documentation/test_practices/encouraged/fresh_browser_per_test/)
dan [.../avoid_sharing_state/](https://www.selenium.dev/documentation/test_practices/encouraged/avoid_sharing_state/).
Jadi solusinya BUKAN berbagi satu browser antar banyak test function
(itu justru dilarang), tapi memakai **`subtests`** (fitur resmi pytest,
awalnya plugin `pytest-subtests` dari organisasi `pytest-dev`) supaya
SATU test function bisa punya beberapa pemeriksaan yang masing-masing
dilaporkan lolos/gagal SENDIRI-SENDIRI, tanpa mengulang setup mahal --
[docs.pytest.org/en/stable/how-to/subtests.html](https://docs.pytest.org/en/stable/how-to/subtests.html).
Traceability per-FR tetap ada lewat `msg=` di tiap subtest dan komentar
di kode, cuma eksekusinya digabung.

**Kenapa pemetaan test↔FR tetap dipertahankan (bukan cuma "ikut-ikutan
selera")**: ISTQB (badan sertifikasi resmi profesi software testing)
mendefinisikan *traceability* sebagai *"the ability to identify related
items in documentation and software, such as requirements with
associated tests"* -- [istqb-glossary.page/traceability](https://istqb-glossary.page/traceability/),
dan *requirements-based testing* sebagai pendekatan test case yang
diturunkan langsung dari kebutuhan -- [istqb-glossary.page/requirements-based-testing](https://istqb-glossary.page/requirements-based-testing/).

## Struktur folder

```
ape_msc/
├── app.py
├── config.py
├── src/
│   ├── data_utils.py, metrics.py, ...      (kode aplikasi)
│   └── test/                                
│       ├── conftest.py
│       ├── pytest.ini
│       ├── requirements-selenium.txt
│       ├── testconfig/settings.py           (semua label/teks terverifikasi dari app.py)
│       ├── pages/{base_page,app_page}.py    (Page Object Model)
│       ├── utils/test_data_generator.py     (data uji sintetis)
│       ├── fixtures/, screenshots/          (dibuat otomatis)
│       └── tests/
│           ├── unit/                        <-- TANPA browser, TANPA app.py aktif
│           │   ├── test_metrics.py              (FR06, 20 test)
│           │   └── test_data_validation.py      (FR03/FR04, 13 test)
│           └── e2e/                         <-- BUTUH `streamlit run app.py`
│               ├── test_upload_and_validation.py     (FR01, FR03, FR04)
│               ├── test_journey_with_ground_truth.py (FR02, FR05, FR06, FR07, FR08)
│               ├── test_journey_without_ground_truth.py
│               ├── test_overlay_alpha_slider.py       (FR09)
│               └── test_preprocessing_dropdown.py     (FR10)
```

Dinamakan folder `testconfig` karena `config.py` di root; menamai ulang menghindari tabrakan
`import config` di dalam proses pytest (yang justru DIBUTUHKAN
`tests/unit/` untuk `import src.data_utils` bisa jalan, karena
`src/data_utils.py` juga melakukan `import config`).

## Instalasi & menjalankan

```bash
source .venv/Scripts/activate
pip install -r src/test/requirements-selenium.txt
```

```bash
# unit test (cepat, TIDAK butuh app.py jalan, TIDAK butuh browser)
pytest src/test/tests/unit

# e2e (butuh `streamlit run app.py` di terminal lain)
pytest src/test/tests/e2e
pytest src/test/tests/e2e -m fr09              # cuma FR09
APE_MSC_HEADLESS=false pytest src/test/tests/e2e -m fr01   # lihat browser bekerja

# semuanya
pytest src/test

# laporan HTML satu-file (butuh pytest-html, sudah ada di requirements)
pytest src/test --html=src/test/report.html --self-contained-html
```

Test e2e yang gagal otomatis menyimpan **screenshot HALAMAN PENUH**
(bukan cuma viewport yang kebetulan terlihat) ke `src/test/screenshots/`.

## Temuan dari app.py (masih berlaku, sudah diverifikasi ulang)

- **`validate_image_file()`/`validate_label_file()` di-import tapi tidak
  pernah dipanggil di app.py.** `tests/unit/test_data_validation.py`
  membuktikan fungsinya sendiri BENAR; `tests/e2e/test_upload_and_validation.py`
  membuktikan app.py yang berjalan TIDAK memakainya -- penegakan FR03/FR04
  sepenuhnya bergantung pada `type=` bawaan `st.file_uploader()` (format)
  dan `server.maxUploadSize` di `.streamlit/config.toml` (ukuran, cek
  foldernya, saya lihat ada di explorer Anda). Kalau test ukuran file
  gagal, itu temuan valid, bukan test yang salah tulis.
- FR04 menyebut label hanya `.png`, implementasi nyata juga menerima
  `.json` (master COCO). Perlu diselaraskan di dokumen.
- `download_mask_png()` dipanggil dengan label `"Unduh Mask Hitam-Putih"`
  di app.py (override default `"Unduh Mask"` dari `ui_helpers.py`).
- FR02 mendukung 3 pola pencocokan: `<stem>.png`, `<stem>_mask.png`,
  master `annotations.json` -- ketiganya diuji sekaligus dalam SATU batch
  upload di `test_matching_patterns_and_negative_case_in_one_batch`.
- Slider (FR09) & dropdown (FR10) tidak otomatis memperbarui figure hasil
  -- `figure_with_legend()` cuma dipanggil di dalam blok `if run_btn:`,
  jadi test terkait sengaja klik run lagi sebelum membandingkan.

## Batasan yang masih berlaku

- **FR08**: prediksi model tidak deterministik dari sisi test; hanya
  kemunculan warna hijau (TP) yang dipastikan (label uji sengaja dibuat
  overlap dengan citra).
- Selector `data-testid` Streamlit bisa berubah antar versi. Kalau ada
  test yang gagal total (elemen tidak ketemu sama sekali, bukan soal
  logika), cek `streamlit --version` dan screenshot halaman penuh di
  `screenshots/` sebelum menyalahkan logika test-nya.

## Daftar sumber (diminta untuk dilampirkan)

1. Selenium Project. *"Page object models."* selenium.dev/documentation/test_practices/encouraged/page_object_models/
2. Selenium Project. *"Generating application state."* selenium.dev/documentation/test_practices/encouraged/generating_application_state/
3. Selenium Project. *"Fresh browser per test."* selenium.dev/documentation/test_practices/encouraged/fresh_browser_per_test/
4. Selenium Project. *"Avoid sharing state"* & *"Test independency."* selenium.dev/documentation/test_practices/encouraged/
5. Fowler, M. (2012). *"TestPyramid."* martinfowler.com/bliki/TestPyramid.html
6. Vocke, H. *"The Practical Test Pyramid."* martinfowler.com/articles/practical-test-pyramid.html
7. Google Testing Blog (2015). *"Just Say No to More End-to-End Tests."* testing.googleblog.com/2015/04/just-say-no-to-more-end-to-end-tests.html
8. pytest. *"How to use fixtures"* (scope & alasan menghindari setup berulang). docs.pytest.org/en/stable/how-to/fixtures.html
9. pytest. *"How to use subtests."* docs.pytest.org/en/stable/how-to/subtests.html
10. ISTQB Glossary. *"Traceability."* istqb-glossary.page/traceability/
11. ISTQB Glossary. *"Requirements-based Testing."* istqb-glossary.page/requirements-based-testing/
