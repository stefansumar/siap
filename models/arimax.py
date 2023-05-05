import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from utils.split_data import split_data


def arimax(inflation_df, exog_df, country, pdq):
    train_data, test_data, train_exog, test_exog = split_data(inflation_df, exog_df)

    arimax_model = sm.tsa.statespace.SARIMAX(
        train_data,
        order=pdq,
        seasonal_order=(0, 0, 0, 0),
        exog=train_exog
    )
    results_ARIMAX = arimax_model.fit(disp=0)
    predictions_ARIMAX = results_ARIMAX.predict(start='2015', end='2021', exog=test_exog)
    predictions_ARIMAX.index = test_data.index
    predictions_ARIMAX = predictions_ARIMAX

    plt.plot(train_data)
    plt.plot(test_data)
    plt.plot(predictions_ARIMAX)
    plt.legend(['Training', 'Test', 'ARIMAX'])
    plt.xlabel('Year')
    plt.ylabel('Inflation')
    plt.suptitle(f'ARIMAX - {country} - Inflation')
    plt.show()

    print(f"ARIMAX - {country} - Mean Squared Error = {round(mean_squared_error(test_data, predictions_ARIMAX), 3)}")
    print("-----------------------------------------------------------------------------------")
