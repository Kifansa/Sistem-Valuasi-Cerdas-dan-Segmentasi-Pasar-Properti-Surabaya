import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

def main():
    # SETUP PATH
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    dataset_path = os.path.join(base_dir, '..', 'DATASET', 'Combined_Datalist_v1.1.csv')

    print("--------------------------------------------")
    print("TRAINING PIPELINE")
    print("--------------------------------------------")

    # LOAD & CLEAN DATA
    try:
        df = pd.read_csv(dataset_path, sep=';')
    except FileNotFoundError:
        print(f"\n[FATAL ERROR] Dataset tidak ditemukan di: {dataset_path}")
        return

    # Cleaning Standard
    df.columns = df.columns.str.strip()
    df['Price'] = df['Price'].astype(str).str.replace('.', '', regex=False)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df = df[df['Price'] > 0].copy()

    # Features Definition
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

    # Handle Missing Values pada Kolom Kategori
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    # PIPELINE SETUP
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ])

    # TRAINING MODELS
    print("Training Supervised Model (Regression)...")
    model_sup = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df['Price']
    model_sup.fit(X, y)

    print("Training Unsupervised Model (Clustering)...")
    preprocessor_unsup = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    model_unsup = Pipeline(steps=[
        ('preprocessor', preprocessor_unsup),
        ('cluster', KMeans(n_clusters=3, random_state=42, n_init=10))
    ])

    X_cluster = df[['Luas Tanah', 'Luas Bangunan', 'Kamar Tidur', 'Kamar Mandi']]
    model_unsup.fit(X_cluster)

    # SAVING MODELS & METADATA
    print("Saving Models to Disk...")
    joblib.dump(model_sup, os.path.join(model_dir, 'model_supervised.joblib'))
    joblib.dump(model_unsup, os.path.join(model_dir, 'model_unsupervised.joblib'))

    meta_options = {}
    for col in CATEGORICAL_FEATURES:
        unique_values = sorted(df[col].astype(str).unique().tolist())
        meta_options[col] = unique_values

    joblib.dump(meta_options, os.path.join(model_dir, 'meta_options.joblib'))

    print(f"\n[SUKSES] Model & Metadata tersimpan di: {model_dir}")

if __name__ == '__main__':
    main()