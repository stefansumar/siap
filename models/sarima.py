import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

from utils.split_data import split_data


def sarima(inflation_df, country, pdq, PDQS):

    train_data, test_data = split_data(inflation_df)

    sarima_model = sm.tsa.SARIMAX(
        train_data,
        order=pdq,
        seasonal_order=PDQS,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=0)

    sarima_pred = sarima_model.get_forecast(steps=len(test_data.index))
    sarima_pred_df = sarima_pred.conf_int(alpha=0.05)
    sarima_pred_df["Predictions"] = sarima_model.predict(start=sarima_pred_df.index[0],
                                                         end=sarima_pred_df.index[-1])
    sarima_pred_df.index = test_data.index
    predictions_SARIMA = sarima_pred_df[["Predictions"]]

    plt.plot(train_data)
    plt.plot(test_data)
    plt.plot(predictions_SARIMA)
    plt.legend(['Training data', 'Test data', 'SARIMA'])
    plt.xlabel('Year')
    plt.ylabel('Inflation')
    plt.suptitle(f'SARIMA - {country} - Inflation')
    plt.show()

    print(f"SARIMA - {country} - Mean Squared Error = {round(mean_squared_error(test_data, predictions_SARIMA), 3)}")
    print("-----------------------------------------------------------------------------------")
