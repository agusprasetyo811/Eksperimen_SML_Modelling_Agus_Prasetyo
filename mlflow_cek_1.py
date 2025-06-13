import mlflow

# Fallback ke local tracking jika DagsHub down
try:
    mlflow.set_tracking_uri("https://dagshub.com/agusprasetyo811/kredit_pinjaman.mlflow")
    # Your MLflow code here
except:
    print("DagsHub unavailable, using local tracking")
    mlflow.set_tracking_uri("file:./mlruns")
    # Your MLflow code here