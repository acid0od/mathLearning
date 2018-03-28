import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing as pr


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


figsize = (14, 4.75)
df = pd.read_csv('data_03.csv', sep='|', header=None,
                 dtype={'year': np.int64, 'urban': np.float, 'sh': np.float, 'R': np.float, 'K': np.float,
                        'disaster': np.float},
                 names=['year', 'urban', 'sh', 'R', 'K', 'disaster'])
df.replace('N/A', np.NaN)

df_new = normalize(df)
df_new = df_new.sort_values(by=['urban'])
df_new.index = df_new.urban

fig, ax = plt.subplots(figsize=figsize)

ax.plot(df_new.R)
ax.plot(df_new.R, ls="", marker="o", label="points")

ax.plot(df_new.disaster)
ax.plot(df_new.disaster, ls="", marker="^", label="points")

plt.show()
fig.savefig('data_03.01.svg', dpi=fig.dpi)
