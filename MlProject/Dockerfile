# Menggunakan image python dasar
FROM python:3.12-slim

# Set folder kerja di dalam container
WORKDIR /app

# Salin file kebutuhan
COPY requirements.txt .

# Install library yang diperlukan
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh isi folder MLProject ke container
COPY . .

# Jalankan model serving saat container dimulai
CMD ["mlflow", "models", "serve", "-m", "best_heart_model", "--host", "0.0.0.0", "--port", "5002", "--no-conda"]