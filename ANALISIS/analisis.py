import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, silhouette_score

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, '..', 'DATASET', 'Combined_Datalist_v1.1.csv')

    print("----------------------------------------------------")
    print("   LAB RISET ANALISIS - FULL FEATURES MODE")
    print("----------------------------------------------------")
    
    # LOAD DATA
    try:
        df = pd.read_csv(dataset_path, sep=';')
    except FileNotFoundError:
        print(f"\n[FATAL ERROR] File tidak ditemukan di: {dataset_path}")
        return

    # CLEANING BASIC
    df.columns = df.columns.str.strip()
    df['Price'] = df['Price'].astype(str).str.replace('.', '', regex=False)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df = df[df['Price'] > 0].copy()

    # IMPUTASI (MENGISI DATA KOSONG)
    # Numerik
    imputer_median = SimpleImputer(strategy='median')
    df['Daya Listrik'] = imputer_median.fit_transform(df[['Daya Listrik']])
    df['Jumlah Lantai'] = df['Jumlah Lantai'].fillna(1)
    
    # Kategorikal (Isi kosong dengan 'Unknown' atau modus agar tidak error saat Encoding)
    categorical_cols_raw = ['Kondisi Properti', 'Kondisi Perabotan', 'Hadap', 'Sertifikat', 'Hook', 'Terjangkau Internet']
    for col in categorical_cols_raw:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    print(f"-> Data siap: {df.shape[0]} baris.")
     
    # DEFINISI FITUR
    # Fitur Angka
    NUMERIC_FEATURES = [
        'Luas Tanah', 
        'Luas Bangunan', 
        'Kamar Tidur', 
        'Kamar Mandi', 
        'Daya Listrik', 
        'Jumlah Lantai'
    ]

    # Fitur Kategori (Teks)
    CATEGORICAL_FEATURES = [
        'Kecamatan',
        'Sertifikat',
        'Kondisi Perabotan',
        'Kondisi Properti',
        'Hook',
        'Terjangkau Internet'
    ]

    # SUPERVISED LEARNING (PREDIKSI HARGA)
    print("\n[SUPERVISED] Running Random Forest...")
    
    # Gabungkan semua fitur
    features_sup = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    target_sup = 'Price'
    
    X = df[features_sup].copy()
    y = df[target_sup]
    
    # ENCODING OTOMATIS (LOOPING)
    # Simpan encoder dalam dictionary 
    encoders = {} 
    
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        # Pakai astype(str) untuk menjaga jika ada data campuran
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le 

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modeling
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model_rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"-> RMSE (Error Rata-rata) : Rp {rmse:,.0f}")
    
    # Feature Importance (Cek Faktor Paling Berpengaruh)
    importances = pd.Series(model_rf.feature_importances_, index=features_sup).sort_values(ascending=False)
    print(f"\n-> Top 5 Faktor Penentu Harga:\n{importances.head(5).to_string()}")

    # UNSUPERVISED LEARNING (CLUSTERING)
    print("\n[UNSUPERVISED] Running K-Means...")
    
    # Untuk clustering, fokus pada FISIK BANGUNAN (Numerik)
    # Agar klasternya berdasarkan "Bentuk Rumah" (Besar/Kecil/Tinggi)
    features_unsup = NUMERIC_FEATURES 
    X_cluster = df[features_unsup].copy()
    
    # Scaling
    scaler = StandardScaler()
    X_cluster_scaled = scaler.fit_transform(X_cluster)
    
    # Modeling
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_cluster_scaled)
    
    # Evaluation
    sil_score = silhouette_score(X_cluster_scaled, df['Cluster'], sample_size=1000)
    print(f"-> Silhouette Score: {sil_score:.3f}")
    
    # Profiling (Melihat rata-rata tiap klaster)
    print("\n-> Profil Klaster (Rata-rata):")
    print(df.groupby('Cluster')[features_unsup + ['Price']].mean().sort_values(by='Price'))

    # EXPORT HASIL
    print("\n[EXPORT] Saving CSV...")

    # Export Supervised
    df_sup_res = X_test.copy()
    
    # KEMBALIKAN ANGKA JADI TEKS"
    for col in CATEGORICAL_FEATURES:
        le = encoders[col] 
        df_sup_res[col] = le.inverse_transform(df_sup_res[col])
        
    df_sup_res['Harga_Asli'] = y_test
    df_sup_res['Harga_Prediksi'] = y_pred
    df_sup_res['Selisih'] = df_sup_res['Harga_Asli'] - df_sup_res['Harga_Prediksi']
    
    path_sup = os.path.join(script_dir, 'Output_Supervised_Prediksi.csv')
    df_sup_res.to_csv(path_sup, index=False)
    print(f"-> Saved: {path_sup}")

    # Export Unsupervised
    cols = list(df.columns)
    if 'Cluster' in cols:
        cols.insert(1, cols.pop(cols.index('Cluster')))
    
    df_unsup_res = df[cols]
    path_unsup = os.path.join(script_dir, 'Output_Unsupervised_Clustering.csv')
    df_unsup_res.to_csv(path_unsup, index=False)
    print(f"-> Saved: {path_unsup}")

    print("\n--- SELESAI ---")

if __name__ == "__main__":
    main()