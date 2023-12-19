import pandas as pd
import mlflow
import os
import wget


def download_data(url, destination):
    # Удаление файла, если он уже существует
    if os.path.exists(destination):
        os.remove(destination)
    filename = wget.download(url)
    return pd.read_csv(filename, delimiter=',')


# Установка URI для реестра MLflow и трекинга
os.environ["MLFLOW_REGISTRY_URI"] = "/home/ksmk99/flow/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")

# Установка названия эксперимента в MLflow
mlflow.set_experiment("get_data")

# Начало записи MLflow
with mlflow.start_run():
    # Загрузка данных и запись артефакта
    data_url = "https://drive.google.com/file/d/1OBWadMwXy9G4yTQaGChZHg8mSl2up3zn/view?usp=sharing"
    df_full = download_data(data_url, "train.csv")

    # Запись кода в виде артефакта
    mlflow.log_artifact(local_path="/home/ksmk99/flow/scripts/get_data.py", artifact_path="get_data_code")

# Сохранение DataFrame в CSV файл
df_full.to_csv('/home/ksmk99/flow/datasets/data.csv', index='Row_id')
