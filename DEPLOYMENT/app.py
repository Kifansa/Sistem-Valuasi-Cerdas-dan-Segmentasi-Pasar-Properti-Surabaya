from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# SETUP PATH OTOMATIS
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, 'models')

# LOAD MODEL & METADATA
print("\n--------------------------------------------")
print("   STARTING PROPINSIGHT SERVER")
print("--------------------------------------------")

try:
    # Load Model Utama
    model_sup = joblib.load(os.path.join(model_dir, 'model_supervised.joblib'))
    model_unsup = joblib.load(os.path.join(model_dir, 'model_unsupervised.joblib'))
    
    # Load Metadata
    meta_options = joblib.load(os.path.join(model_dir, 'meta_options.joblib'))
    
    print("Models & Metadata Loaded.")

except FileNotFoundError as e:
    print(f"\n[CRITICAL ERROR] Gagal memuat file pendukung!")
    print(f"Pastikan Anda sudah menjalankan 'train_pipeline.py' yang baru.")
    print(f"Detail: {e}")
    exit()

@app.route('/', methods=['GET', 'POST'])
def index():
    context = {
        'kecamatan_list': meta_options['Kecamatan'],
        'sertifikat_list': meta_options['Sertifikat'],
        'perabotan_list': meta_options['Kondisi Perabotan'],
        'properti_list': meta_options['Kondisi Properti'],
        'hook_list': meta_options['Hook'],
        'internet_list': meta_options['Terjangkau Internet']
    }
    
    if request.method == 'POST':
        try:
            # Capture Input User 
            input_data = {
                # --- Numerik ---
                'Luas Tanah': float(request.form['luas_tanah']),
                'Luas Bangunan': float(request.form['luas_bangunan']),
                'Kamar Tidur': int(request.form['kamar_tidur']),
                'Kamar Mandi': int(request.form['kamar_mandi']),
                'Daya Listrik': float(request.form['listrik']),
                'Jumlah Lantai': float(request.form['jumlah_lantai']),
                
                # --- Kategorikal ---
                'Kecamatan': request.form['kecamatan'],
                'Sertifikat': request.form['sertifikat'],
                'Kondisi Perabotan': request.form['perabotan'],
                'Kondisi Properti': request.form['kondisi_properti'],
                'Hook': request.form['hook'],
                'Terjangkau Internet': request.form['internet']
            }
            
            # Buat DataFrame
            df_input = pd.DataFrame([input_data])

            # Supervised Prediction (Estimasi Harga)
            pred_harga_raw = model_sup.predict(df_input)[0]
            pred_harga = f"Rp {pred_harga_raw:,.0f}".replace(',', '.')

            # Unsupervised Prediction (Clustering)
            df_cluster_input = df_input[['Luas Tanah', 'Luas Bangunan', 'Kamar Tidur', 'Kamar Mandi']]
            cluster_id = model_unsup.predict(df_cluster_input)[0]

            # Interpretasi Bisnis
            cluster_map = {
                0: {'title': 'Compact / Entry-Level Asset', 
                    'desc': 'Properti efisien dengan luas terbatas. Cocok untuk keluarga baru atau investasi awal.',
                    'color': 'text-blue-600', 'bg': 'bg-blue-50'},
                1: {'title': 'Standard / Family Asset', 
                    'desc': 'Properti kelas menengah dengan keseimbangan fungsi ruang. Segmen paling likuid di pasar.',
                    'color': 'text-emerald-600', 'bg': 'bg-emerald-50'},
                2: {'title': 'Prime / Luxury Asset', 
                    'desc': 'Aset premium dengan spesifikasi luas dan fasilitas lengkap. Target pasar kelas atas.',
                    'color': 'text-purple-600', 'bg': 'bg-purple-50'}
            }
            cluster_info = cluster_map.get(cluster_id, cluster_map[0])

            # Update Context untuk dikirim ke UI
            context.update({
                'result_exists': True,
                'price': pred_harga,
                'cluster': cluster_info,
                'input': input_data
            })

        except Exception as e:
            context['error'] = f"Terjadi kesalahan input: {str(e)}"
            print(f"[ERROR] {e}")

    return render_template('index.html', **context)

if __name__ == '__main__':
    app.run(debug=True, port=5000)