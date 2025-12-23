import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# 1. Konfigurasi Tracking URI
if os.getenv('GITHUB_ACTIONS') == 'true':
    # Lingkungan CI/CD: Kirim data ke DagsHub
    dagshub.init(repo_owner='widianditya', 
                 repo_name='Eksperimen_SML_I-Gede-Made-Widi-Anditya', 
                 mlflow=True)
else:
    # Simpan di localhost 
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

# 2. Aktifkan Autologging (Syarat Kriteria 2)
mlflow.sklearn.autolog()

# 3. Load Dataset menggunakan Path Relatif
data_path = 'namadataset_preprocessing/cleaned_heart.csv'
df = pd.read_csv(data_path)

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Inisialisasi Model dengan Parameter Terbaik (Hasil Tuning)
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=7, 
    criterion='gini', 
    random_state=42
)

# 5. Eksekusi Training dan Logging
with mlflow.start_run(run_name="HeartDisease_Final_Model"):
    model.fit(X_train, y_train)
    
    # Prediksi untuk evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Logging Manual agar muncul di kolom Dashboard DagsHub
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    
    print(f"Model Training Complete. Accuracy: {acc:.4f}, F1: {f1:.4f}")