"""Launcher script for MLflow UI with file store compatibility."""
import os
import sys
import subprocess

# Ensure MLflow allows local file store backend
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

def main():
    print("=" * 60)
    print("  Starting MLflow UI Dashboard on http://127.0.0.1:5000")
    print("=" * 60)
    cmd = [
        sys.executable, "-m", "mlflow", "ui",
        "--backend-store-uri", "./mlruns",
        "--port", "5000",
        "--host", "127.0.0.1",
        "--workers", "1"
    ]
    subprocess.run(cmd, env=os.environ)

if __name__ == "__main__":
    main()
