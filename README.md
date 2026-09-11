# MLOps Starter Project - NovaVision Labs

## Deskripsi
Repository ini dibangun sebagai fondasi awal untuk manajemen siklus hidup *Machine Learning* (MLOps). Tujuannya adalah memfasilitasi integrasi yang mulus antara pengembangan model kecerdasan buatan, *backend* API, dan antarmuka *frontend* dalam satu lingkungan kerja yang terstruktur, *reproducible*, dan siap untuk kolaborasi tim.

## Struktur Project
* **`frontend/`**: Wajah aplikasi tempat kode antarmuka pengguna (UI/UX) berada.
* **`backend/`**: Mesin aplikasi yang berisi logika bisnis, pengelolaan basis data (seperti MySQL), dan *routing* API.
* **`model/`**: Otak dari sistem, tempat menyimpan kode pelatihan (*training*), *inference*, serta artefak model.
* **`data/`**: Bahan bakar sistem yang dipisahkan menjadi `raw/` (data mentah awal) dan `processed/` (data yang sudah dibersihkan).
* **`tests/`**: Pusat *quality control* yang memuat skenario pengujian kode dan *pipeline*.
* **`docs/`**: Kumpulan dokumentasi teknis, catatan keputusan arsitektur, dan manual sistem.

## Setup
1. Clone repository:
   ```bash
   git clone <URL-REPOSITORY>
   cd mlops-starter-project
