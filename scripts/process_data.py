import pandas as pd
import numpy as np

def load_data(file_path):
    """Загрузка данных из CSV файла."""
    return pd.read_csv(file_path, index_col=0)

def preprocess_data(df):
    """Предобработка данных."""
    df = df[df['num_sold'] <= 920]
    df['date'] = pd.to_datetime(df['date'])
    df[['year', 'month', 'day']] = df['date'].dt.year, df['date'].dt.month, df['date'].dt.day

    df = df.drop(['date', 'row_id'], axis=1).reset_index(drop=True)
    df['num_sold'] = np.log(df['num_sold'])
    df[['year', 'month', 'day']] = df[['year', 'month', 'day']].astype('category')

    # Создание фиктивных переменных для категориальных признаков
    df = pd.get_dummies(df)

    return df

def save_processed_data(df, output_path):
    df.to_csv(output_path, index='Row_id')


input_file = '/home/ksmk99/flow/datasets/data.csv'
df = load_data(input_file)
df_processed = preprocess_data(df)

output_file = '/home/ksmk99/flow/datasets/data_processed.csv'
save_processed_data(df_processed, output_file)