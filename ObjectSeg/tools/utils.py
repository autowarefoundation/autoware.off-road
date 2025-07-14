import os
import subprocess
import time

def launch_mlflow_ui(port=5000,log_dir="./mlruns"):
    os.makedirs(log_dir, exist_ok=True)
    cmd = ["mlflow", "ui", "--backend-store-uri", log_dir, "--port", str(port)]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)