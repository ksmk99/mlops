import pandas as pd
import pickle as pkl
import mlflow
import os
from sklearn.metrics import mean_absolute_percentage_error

def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pkl.load(f)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    with mlflow.start_run():
        mlflow.log_metric('MAPE', mape)
        mlflow.log_artifact(local_path="/home/ksmk99/flow/scripts/test_model.py",
                            artifact_path="test_model_code")

    return mape

# Установка URI для реестра MLflow и трекинга
os.environ["MLFLOW_REGISTRY_URI"] = "/home/ksmk99/flow/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("test_model")

# Загрузка тестовых данных
test_df = pd.read_csv('/home/ksmk99/flow/datasets/data_test.csv')
X_test, y_test = test_df.drop(['num_sold'], axis=1), test_df[['num_sold']]

# Загрузка модели
model_path = '/home/ksmk99/flow/models/rf_regressor.pickle'
model = load_model(model_path)
mape = evaluate_model(model, X_test, y_test)

print("MAPE Score:", mape)
