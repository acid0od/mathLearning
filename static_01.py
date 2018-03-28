import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing as pr

figsize = (14, 4.75)
# t = pd.read_csv('data.csv', sep='|', dtype={'year':np.int64, 'urban':np.float, 'population':np.float}, names={'year', 'd', 'm'})
t = pd.read_csv('data.csv', sep='|', header=None, dtype={'year': np.int64, 'urban': np.float, 'population': np.float})
t.index = t[0]

scaler = pr.MinMaxScaler()
scaled_values = scaler.fit_transform(t)
t.loc[:,:] = scaled_values

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)

fig, ax = plt.subplots(figsize=figsize)

# Major ticks every 20, minor ticks every 5

major_ticks = np.arange(1980, 2020, 1)
minor_ticks = np.arange(1980, 2020, 0.5)

ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
# ax.set_yticks(major_ticks)
# ax.set_yticks(minor_ticks, minor=True)

# And a corresponding grid
ax.grid(which='both')

# Or if you want different settings for the grids:
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)

ax.plot(t[1])
ax.plot(t[2])
plt.xticks(rotation=90)
plt.show()
