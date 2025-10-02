import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('./datasets/02-14-2018.csv')
df = df.drop(['Timestamp'], axis=1) # Удаление метки времени


# Вывод DataFrame в виде таблицы. На входе - DataFrame; r_start, r_end - диапазон строк для вывода
def print_df(dataf: pd.DataFrame, r_start: int = 0, r_end: int = 30) -> None:
    print(tabulate(dataf[r_start:r_end], headers=dataf.columns.tolist(), tablefmt='fancy_grid'))


# Обработка DataFrame
def processing(dataf: pd.DataFrame) -> pd.DataFrame:
    cols = list(dataf.columns)
    # Замена всех inf-значений на NaN
    dataf.mask(cond=(dataf == np.inf) | (dataf == -np.inf), other=np.nan, inplace=True)

    # Замена всех NaN значений на ближайшее (зачем это здесь?)
    # dataf.bfill(inplace=True)

    # Сортированный список всех уникальных Label в dataf
    unique_labels = sorted(dataf['Label'].unique())

    # Словарь с пронумерованными Label
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    # Упрощение всех Label в dataf до номера в словаре
    dataf['Label'] = dataf['Label'].map(label_to_index)

    # Сохранение колонки Label с удалением индексов
    labels = dataf['Label'].reset_index(drop=True)
    scaler = MinMaxScaler()

    # Сжатие всех значений в dataf до диапазона 0 - 1 (не трогая Label)
    df_scaled = scaler.fit_transform(dataf.drop('Label', axis=1))

    # fit_transform даёт на выходе матрицу NumPy, преобразуем её в DataFrame
    df_scaled = pd.DataFrame(df_scaled)

    # Возвращение колонки Label
    df_scaled['Label'] = labels

    # Устранение потери наименований колонок после конвертации матрицы NumPy
    df_scaled.columns = cols
    return df_scaled

def tests():
    global df
    processed = processing(df)
    print_df(processed)

if __name__ == "__main__":
    tests()







