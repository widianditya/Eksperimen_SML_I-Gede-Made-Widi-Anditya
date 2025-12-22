from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary
import random
import time
import psutil

# 10 METRIKS UNTUK POIN ADVANCE
REQUEST_COUNT = Counter('api_requests_total', 'Total API Requests')
ERROR_COUNT = Counter('api_errors_total', 'Total API Errors')
PRED_HEALTHY = Counter('model_pred_healthy_total', 'Total Healthy Predictions')
PRED_SICK = Counter('model_pred_sick_total', 'Total Heart Disease Predictions')

LATENCY = Histogram('api_response_time_seconds', 'Response time in seconds')
CPU_USAGE = Gauge('system_cpu_usage_percent', 'Current CPU usage')
RAM_USAGE = Gauge('system_ram_usage_percent', 'Current RAM usage')
UPTIME = Gauge('system_uptime_seconds', 'System uptime in seconds')


INPUT_SIZE = Summary('api_request_payload_size_bytes', 'Size of request payload')
MODEL_VERSION = Gauge('model_version_info', 'Current model version', ['version'])

def collect_metrics():
    start_uptime = time.time()
    MODEL_VERSION.labels(version='1.0.0').set(1)
    
    while True:
        # Update metrik sistem
        CPU_USAGE.set(psutil.cpu_percent())
        RAM_USAGE.set(psutil.virtual_memory().percent)
        UPTIME.set(time.time() - start_uptime)
        
        # Simulasi trafik (Hapus bagian ini jika sudah dihubungkan ke API asli)
        REQUEST_COUNT.inc()
        if random.random() > 0.8:
            PRED_SICK.inc()
        else:
            PRED_HEALTHY.inc()
            
        time.sleep(5)
        
        LATENCY.observe(random.uniform(0.1, 0.5)) # Simulasi latensi 0.1 - 0.5 detik
        INPUT_SIZE.observe(random.randint(100, 500)) # Simulasi ukuran data
        if random.random() > 0.95:
            ERROR_COUNT.inc() # Simulasi error sesekali

if __name__ == '__main__':
    # Jalankan exporter di port 8000
    start_http_server(8000)
    print("Prometheus Exporter berjalan di port 8000")
    collect_metrics()