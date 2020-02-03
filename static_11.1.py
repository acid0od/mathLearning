import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing as pr
from scipy.interpolate import UnivariateSpline


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        if (feature_name != 'aria'):
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


figsize = (14, 4.75)
df = pd.read_csv('data_11.csv', sep='|', header=None, dtype={'aria': np.str, 'u': np.float, 't1': np.float, 'r2': np.float, 'r2': np.float},
                 names=['aria', 'u', 't1', 'r1', 'r2'])
df.replace('N/A', np.NaN)

# Нормализуем данные
df_new = normalize(df)
df_new.index = df.aria

# Регулируем количество колонок для печати результатов обсчета
pd.set_option("display.max_columns", 102)

print(df_new)

fig, ax = plt.subplots(figsize=figsize)

# Исключаем информацию об областях, сохраняем для дальнейшего использования
varieties = list(df_new.pop('aria'))

# Извлекаем измерения как массив NumPy
samples = df_new.values

# Реализация иерархической кластеризации при помощи функции linkage
mergings = linkage(samples, method='complete')

# Строим дендрограмму, указав параметры удобные для отображения
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=10,
           )

plt.show()

fig.savefig('/tmp/data_11.1.svg', dpi=fig.dpi)