import random
import sys
import yaml
from pathlib import Path
import mlflow

with mlflow.start_run():
    train_params = yaml.safe_load(open(Path(__file__).resolve().parent.parent.parent / 'params.yaml'))['train']
    epochs = train_params['epochs']
    mlflow.log_param("epochs", epochs)
    for epoch in range(epochs):
        mlflow.log_metric("train/accuracy", epoch + random.random())
        mlflow.log_metric("train/loss", epochs - epoch - random.random())
        mlflow.log_metric("val/accuracy",epoch + random.random() )
        mlflow.log_metric("val/loss", epochs - epoch - random.random())