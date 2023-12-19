from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pickle as pkl
import mlflow
import os


def load_train_data(file_path):
    return pd.read_csv(file_path)


def train_random_forest(X_train, y_train, n_estimators=50, random_state=0):
    rf_regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    with mlflow.start_run():
        mlflow.sklearn.log_model(rf_regressor,
                                 artifact_path="rf_regressor",
                                 registered_model_name="rf_regressor")

        mlflow.log_artifact(local_path="/home/ksmk99/flow/scripts/train_model.py",
                            artifact_path="train_model_code")

    rf_regressor.fit(X_train, y_train)

    return rf_regressor


def save_model(model, model_path):
    with open(model_path, 'wb') as f:
        pkl.dump(model, f)


# Установка URI для реестра MLflow и трекинга
os.environ["MLFLOW_REGISTRY_URI"] = "/home/ksmk99/flow/mlflow/"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("train_model")

# Загрузка данных для обучения
train_df = load_train_data('/home/ksmk99/flow/datasets/data_train.csv')
X_train, y_train = train_df.drop(['num_sold'], axis=1), train_df[['num_sold']]

trained_rf_model = train_random_forest(X_train, y_train)

model_path = '/home/ksmk99/flow/models/rf_regressor.pickle'
save_model(trained_rf_model, model_path)