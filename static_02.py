import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing as pr

figsize = (14, 4.75)
df = pd.read_csv('data_02.csv', sep='|', header=None, dtype={'year': np.int64, 'hc': np.float, 'disaster': np.float},
                 names=['year', 'hc', 'disaster'])
df.replace('N/A', np.NaN)
df.index = df.year

fig, ax = plt.subplots(figsize=figsize)

major_ticks = np.arange(1990, 2020, 1)
minor_ticks = np.arange(1990, 2020, 0.5)

ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.grid(which='both')
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)

ax.plot(df.hc)
ax.plot(df.hc, ls="", marker="o", label="points")

ax.plot(df.disaster)
ax.plot(df.disaster, ls="", marker="^", label="points")
plt.xticks(rotation=90)
plt.show()
fig.savefig('data_02.svg', dpi=fig.dpi)