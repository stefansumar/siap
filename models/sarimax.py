import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from analysis.measures import measures
from utils.split_data import split_data


def sarimax(inflation_df, exog_df, country, pdq, PDQS):
    train_data, test_data, train_exog, test_exog = split_data(inflation_df, exog_df)

    sarimax_model = sm.tsa.statespace.SARIMAX(
        train_data,
        order=pdq,
        seasonal_order=PDQS,
        exog=train_exog
    )
    results_sarimax = sarimax_model.fit(disp=0)
    predictions_sarimax = results_sarimax.predict(start='2015', end='2021', exog=test_exog)
    predictions_sarimax.index = test_data.index
    predictions_sarimax = predictions_sarimax

    plt.plot(train_data)
    plt.plot(test_data)
    plt.plot(predictions_sarimax)
    plt.legend(['Training', 'Test', 'SARIMAX'])
    plt.xlabel('Year')
    plt.ylabel(f'Inflation')
    plt.suptitle(f'SARIMAX - {country} - Inflation')
    plt.show()

    measures("SARIMAX", country, test_data, predictions_sarimax)
