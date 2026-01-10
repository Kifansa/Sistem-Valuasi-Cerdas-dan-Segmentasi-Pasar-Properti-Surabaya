# PropInsight Enterprise v1.0

Sistem valuasi aset properti berbasis Machine Learning (Random Forest & K-Means) untuk wilayah Surabaya.

## Struktur Proyek
- `train_pipeline.py`: Script "Pabrik" untuk melatih dan menyimpan model AI.
- `app.py`: Server aplikasi web (Flask).
- `models/`: Folder tempat menyimpan otak AI (file .joblib).

## Cara Menjalankan (Deployment Guide)

1. **Install Dependensi**
    Pastikan Python sudah terinstall, lalu jalankan:
    ```bash
    pip install -r requirements.txt
2. **Latih Model (Hanya Sekali)**
    Sebelum menjalankan web, AI harus dilatih terlebih dahulu
    python train_pipeline.py
    tunggu sampai folder 'models/' muncul
3. **Jalankan Aplikasi**
    python app.py
    Buka browser dan akses: http://localhost:5000