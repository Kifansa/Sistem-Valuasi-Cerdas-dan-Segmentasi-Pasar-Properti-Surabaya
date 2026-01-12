import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

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
    print("ANALISIS ")
    print("----------------------------------------------------")
    
    try:
        df = pd.read_csv(dataset_path, sep=';')
    except FileNotFoundError:
        print(f"\n[FATAL ERROR] File tidak ditemukan di: {dataset_path}")
        return

    df.columns = df.columns.str.strip()
    df['Price'] = df['Price'].astype(str).str.replace('.', '', regex=False)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df = df[df['Price'] > 0].copy()

    imputer_median = SimpleImputer(strategy='median')
    df['Daya Listrik'] = imputer_median.fit_transform(df[['Daya Listrik']])
    df['Jumlah Lantai'] = df['Jumlah Lantai'].fillna(1)
    
    categorical_cols_raw = ['Kondisi Properti', 'Kondisi Perabotan', 'Hadap', 'Sertifikat', 'Hook', 'Terjangkau Internet']
    for col in categorical_cols_raw:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    print(f"-> Data siap: {df.shape[0]} baris.")
     
    NUMERIC_FEATURES = [
        'Luas Tanah', 
        'Luas Bangunan', 
        'Kamar Tidur', 
        'Kamar Mandi', 
        'Daya Listrik', 
        'Jumlah Lantai'
    ]

    CATEGORICAL_FEATURES = [
        'Kecamatan',
        'Sertifikat',
        'Kondisi Perabotan',
        'Kondisi Properti',
        'Hook',
        'Terjangkau Internet'
    ]

    print("\n[ANALYSIS] Generating Correlation Matrix...")

    df_corr = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + ['Price']].copy()
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df_corr[col] = le.fit_transform(df_corr[col].astype(str))
        encoders[col] = le
    
    corr_matrix = df_corr.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix[['Price']].sort_values(by='Price', ascending=False), 
                annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Korelasi Fitur terhadap Harga (Price)')
    
    path_viz_corr = os.path.join(script_dir, 'viz_1_correlation.png')
    plt.savefig(path_viz_corr, bbox_inches='tight')
    plt.close()
    print(f"-> Gambar disimpan: {path_viz_corr}")

    print("\n[SUPERVISED] Running Random Forest...")
    
    features_sup = NUMERIC_FEATURES + CATEGORICAL_FEATURES    
    X = df_corr[features_sup].copy()
    y = df_corr['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    
    y_pred = model_rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"-> RMSE (Error Rata-rata) : Rp {rmse:,.0f}")

    mae = mean_absolute_error(y_test, y_pred)
    print(f"-> MAE  (Rata-rata Selisih Real) : Rp {mae:,.0f}")
    
    importances = pd.Series(model_rf.feature_importances_, index=features_sup).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    importances.head(10).plot(kind='barh', color='teal')
    plt.title('Top 10 Faktor Penentu Harga Properti')
    plt.xlabel('Tingkat Kepentingan (Importance Score)')
    plt.gca().invert_yaxis()

    path_viz_feat = os.path.join(script_dir, 'viz_2_feature_importance.png')
    plt.savefig(path_viz_feat)
    plt.close()
    print(f"-> Gambar disimpan: {path_viz_feat}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Garis diagonal
    plt.xlabel('Harga Aktual')
    plt.ylabel('Harga Prediksi')
    plt.title('Evaluasi Prediksi: Aktual vs Prediksi')
    plt.grid(True)
    
    path_viz_pred = os.path.join(script_dir, 'viz_3_prediction.png')
    plt.savefig(path_viz_pred)
    plt.close()
    print(f"-> Gambar disimpan: {path_viz_pred}")

    print("\n[UNSUPERVISED] Running Elbow Method...")
    
    features_unsup = NUMERIC_FEATURES 
    X_cluster = df[features_unsup].copy()
    
    scaler = StandardScaler()
    X_cluster_scaled = scaler.fit_transform(X_cluster)

    inertia = []
    K_range = range(1, 10)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_cluster_scaled)
        inertia.append(km.inertia_)
        
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertia, marker='o', linestyle='--')
    plt.xlabel('Jumlah Klaster (k)')
    plt.ylabel('Inertia (Sum of Squared Errors)')
    plt.title('Elbow Method untuk Menentukan k Optimal')
    plt.grid(True)
    
    path_viz_elbow = os.path.join(script_dir, 'viz_4_elbow.png')
    plt.savefig(path_viz_elbow)
    plt.close()
    print(f"-> Gambar disimpan: {path_viz_elbow}")
    
    print("[UNSUPERVISED] Running Clustering...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_cluster_scaled)

    sil_score = silhouette_score(X_cluster_scaled, df['Cluster'], sample_size=1000)
    print(f"-> Silhouette Score: {sil_score:.3f}")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Luas Tanah', y='Price', hue='Cluster', palette='viridis', alpha=0.6)
    plt.title('Sebaran Klaster: Luas Tanah vs Harga')
    plt.xlabel('Luas Tanah (m²)')
    plt.ylabel('Harga (Rupiah)')
    plt.xlim(0, 1000)
    plt.ylim(0, 10000000000)

    path_viz_cluster = os.path.join(script_dir, 'viz_5_cluster.png')
    plt.savefig(path_viz_cluster)
    plt.close()
    print(f"-> Gambar disimpan: {path_viz_cluster}")

    print("\n[EXPORT] Saving CSV...")

    df_sup_res = X_test.copy()
    
    for col in CATEGORICAL_FEATURES:
        le = encoders[col] 
        df_sup_res[col] = le.inverse_transform(df_sup_res[col])
        
    df_sup_res['Harga_Asli'] = y_test
    df_sup_res['Harga_Prediksi'] = y_pred
    df_sup_res['Selisih'] = df_sup_res['Harga_Asli'] - df_sup_res['Harga_Prediksi']
    
    path_sup = os.path.join(script_dir, 'Output_Supervised_Prediksi.csv')
    df_sup_res.to_csv(path_sup, index=False)
    print(f"-> Saved: {path_sup}")

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
