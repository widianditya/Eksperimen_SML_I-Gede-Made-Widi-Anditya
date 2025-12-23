import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier # Contoh model Anda
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. ATUR ALAMAT TRACKING KE LOCALHOST 
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# 2. AKTIFKAN AUTOLOGGING
mlflow.sklearn.autolog()

# Memuat data 
df = pd.read_csv('namadataset_preprocessing/cleaned_heart.csv')
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. JALANKAN EKSPERIMEN DENGAN mlflow.start_run()
with mlflow.start_run(run_name="HeartDisease_Experiment"):
    # Definisikan Model
    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    
    # Latih Model
    # Semua parameter, metrik, dan model akan dicatat OTOMATIS oleh autolog
    model.fit(X_train, y_train)
    
    print("Eksperimen selesai dan tercatat di MLflow.")