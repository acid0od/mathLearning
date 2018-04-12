import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing as pr
from scipy.interpolate import spline
from scipy.interpolate import UnivariateSpline
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


figsize = (14, 4.75)
df = pd.read_csv('data_02.csv', sep='|', header=None, dtype={'year': np.int64, 'hc': np.float, 'disaster': np.float},
                 names=['year', 'hc', 'disaster'])
df.replace('N/A', np.NaN)
df_new = normalize(df)
df_new.index = df.year

df_new = df_new.sort_values(by=['year'])

# x = np.linspace(-3, 3, 50)
# y = np.exp(-x**2) + 0.1 * np.random.randn(50)
# plt.plot(x, y, 'ro', ms=5)
#
# spl = UnivariateSpline(x, y)
# xs = np.linspace(-3, 3, 1000)
# plt.plot(xs, spl(xs), 'g', lw=3)
#
# spl.set_smoothing_factor(0.5)
# plt.plot(xs, spl(xs), 'b', lw=3)

x = np.linspace(df.year.min(), df.year.max(), num=21)
y = df_new.disaster

spl = UnivariateSpline(x, y)

fig, ax = plt.subplots(figsize=figsize)
xs = np.linspace(x.min(), x.max(), num=1000)

plt.plot(x, y, 'ro', ms=5)
plt.plot(xs, spl(xs), 'g', lw=3)

spl.set_smoothing_factor(0.0005)
plt.plot(xs, spl(xs), 'b', lw=3)

# x_smooth = np.linspace(df.year.min(), df.year.max(), num=300)
# spl = UnivariateSpline(x_smooth, df.disaster)
#
# xs = np.linspace(df.year.min(), df.year.max(), num=300)
#
# fig, ax = plt.subplots(figsize=figsize)
#
# plt.plot(xs, spl(xs), 'g', lw=3)
#
#
# x_smooth = np.linspace(df.year.min(), df.year.max(), num=300)
# y_smooth = spline(df.year, df.disaster, x_smooth)
#
#
#
# print(y_smooth)

major_ticks = np.arange(1997, 2018, 1)
minor_ticks = np.arange(1997, 2018, 0.5)
#
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.grid(which='both')
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)
#
# ax.plot(df_new.hc)
# ax.plot(df_new.hc, ls="", marker="o", label="points")
#
# ax.plot(df_new.disaster)
# ax.plot(df_new.disaster, ls="", marker="^", label="points")
plt.xticks(rotation=90)
# plt.plot(x_smooth, y_smooth)
# plt.plot(df.year, df.disaster)

plt.show()
fig.savefig('data_02.01.svg', dpi=fig.dpi)
