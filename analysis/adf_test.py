from statsmodels.tsa.stattools import adfuller
import pandas as pd

"""
    -------------------------------    ADF TEST    -------------------------------

    For a Time series to be stationary, its ADF test should have:
    1. p-value to be low (according to the null hypothesis)
    2. The critical values at 1%, 5% and 10% should be as close possible to the Test Statistic

    Theory:

    Test Statistic < Critical Values
    and
    p-value < 0.05

"""


def adf_test(series, country, regression):
    dftest = adfuller(series, autolag='AIC', regression=regression)
    df_output = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        df_output['Critical Value (%s)' % key] = value

    print(f'Results of Dickey Fuller Test for {country}: ')
    print(df_output)
    print("-----------------------------------------------------------------------------------")
