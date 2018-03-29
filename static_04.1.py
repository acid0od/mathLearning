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
df = pd.read_csv('data_04.csv', sep='|', header=None,
                 dtype={'year': np.int64, 'urban': np.float, 'sh': np.float, 'R': np.float, 'K': np.float,
                        'disaster': np.float, 'Ris': np.float},
                 names=['year', 'urban', 'sh', 'R', 'K', 'disaster', 'Ris'])
df.replace('N/A', np.NaN)

df_new = normalize(df)
df_new.index = df.year

fig, ax = plt.subplots(figsize=figsize)


major_ticks = np.arange(2000, 2020, 1)
minor_ticks = np.arange(2000, 2020, 0.5)

ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.grid(which='both')
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)

ax.set_xlabel('Years')
ax.set_ylabel('Rating')

ax.plot(df_new.R)
ax.plot(df_new.R, ls="", marker="o")

ax.plot(df_new.Ris)
ax.plot(df_new.Ris, ls="", marker="^")
plt.xticks(rotation=90)

ax.legend(loc='best')

plt.show()
fig.savefig('data_04.01.svg', dpi=fig.dpi)
