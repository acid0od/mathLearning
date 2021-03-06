import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing as pr
import statsmodels.formula.api as smf
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression


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
# ax.set_ylabel('Rating')
#
# ax.plot(df_new.R, label='Rt E-5')
# ax.plot(df_new.Ris, label='Ri E-6')
# ax.legend(loc='best')
#
# ax.plot(df_new.R, ls="", marker="o")
# ax.plot(df_new.Ris, ls="", marker="^")
# plt.xticks(rotation=90)

# regression = smf.ols(y=df_new.R, x=df_new.year)
# regression.summary
x = df_new.year
y = df_new.R

# model = LinearRegression(normalize=True)
# model.fit(df_new.year, df_new.R)
# # calculate trend
# trend = model.predict(df_new.R)

model = sm.formula.ols(formula='R ~ year', data=df_new)
res = model.fit()
print(res.summary())
df.assign(fit=res.fittedvalues).plot(x='year', y='R', ax=ax)

model_ = sm.formula.ols(formula='Ris ~ year', data=df_new)
res_ = model_.fit()
print(res_.summary())
df.assign(fit=res_.fittedvalues).plot(x='year', y='Ris', ax=ax)


#coefficients, residuals, _, _, _ = np.polyfit(range(len(df.index)), df, 1, full=True)
# plt.plot([coefficients[0]*x + coefficients[1] for x in range(len(df))])
# params = np.polyfit(x, y, 2020)
#
# xp = np.linspace(x.min(), 1990, 2020)
# yp = np.polyval(params, xp)
# sig = np.std(y - np.polyval(params, x))
# plt.fill_between(xp, yp - sig, yp + sig,
#                  color='k', alpha=0.2)
#

# sns.lmplot(x='year', y='R', data=df, x_estimator=np.mean, logistic=True, y_jitter=.03)
# sns.jointplot(x='year', y='R', data=df, kind="reg");
plt.show()
fig.savefig('data_04.01.svg', dpi=fig.dpi)
