from flask import Flask, request, jsonify
import joblib
import pandas as pd
import time

app = Flask(__name__)

# Load model dari folder Membangun_model
# Pastikan path ini benar sesuai struktur folder Anda
MODEL_PATH = '../Membangun_model/heart_disease_model/model.pkl'
model = joblib.load(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    try:
        data = request.get_json()
        df = pd.DataFrame(data)
        
        # Prediksi
        prediction = model.predict(df)
        
        # Simulasi latensi untuk monitoring
        duration = time.time() - start_time
        
        return jsonify({
            'prediction': prediction.tolist(),
            'status': 'success',
            'latency': duration
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    # Jalankan pada port 5002 sesuai contoh log Dicoding
    app.run(host='0.0.0.0', port=5002)