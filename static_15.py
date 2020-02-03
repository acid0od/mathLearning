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
        if feature_name != 'aria':
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


figsize = (14, 6)
df = pd.read_csv('data_15.csv', sep='|', header=None, dtype={'aria': np.str,
                                                             'u1': np.float,
                                                             'u2': np.float,
                                                             'u3': np.float,
                                                             'u4': np.float,
                                                             'u5': np.float,
                                                             'u6': np.float,
                                                             'u7': np.float,
                                                             'u8': np.float,
                                                             'u9': np.float,
                                                             'u10': np.float,
                                                             'u11': np.float,
                                                             'u12': np.float,
                                                             'u13': np.float,
                                                             'u14': np.float,
                                                             'u15': np.float,
                                                             'u16': np.float,
                                                             'u17': np.float,
                                                             'u18': np.float,
                                                             'u19': np.float,
                                                             'u20': np.float,
                                                             },
                 names=['aria',
                        'u1',
                        'u2',
                        'u3',
                        'u4',
                        'u5',
                        'u6',
                        'u7',
                        'u8',
                        'u9',
                        'u10',
                        'u11',
                        'u12',
                        'u13',
                        'u14',
                        'u15',
                        'u16',
                        'u17',
                        'u18',
                        'u19',
                        'u20'
                        ])

df.replace('N/A', np.NaN)
# Нормализуем данные
df['u5'] = df['u2'] / 42200000
df['u7'] = df['u6'] / 29720
df['u9'] = df['u8'] / 461
df['u11'] = df['u10'] / 24640930
df['u13'] = df['u12'] / 173
df['u15'] = df['u14'] / 29
df['u17'] = df['u16'] / df['u1']
df['u18'] = df['u10'] / df['u16']
df['u20'] = df['u18'] / df['u19']

new = df[['aria', 'u3', 'u4', 'u5', 'u7', 'u9', 'u11', 'u13', 'u17', 'u18', 'u20']]
df_new = normalize(new)
df_new.index = df.aria

print(new.to_string())
# Регулируем количество колонок для печати результатов обсчета
pd.set_option("display.max_columns", 102)

# print(df_new)

df.to_csv('output_15.csv', sep='|', index=None, header=True)
df_new.to_csv('output_15_normalize.csv', sep='|', index=None, header=True)

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
           orientation='right',
           leaf_rotation=0,
           leaf_font_size=10,
           )

plt.show()

fig.savefig('/tmp/data_15.svg', dpi=fig.dpi)
