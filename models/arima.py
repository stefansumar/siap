import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

from analysis.measures import measures
from utils.split_data import split_data


def arima(inflation_df, country, pdq):
    train_data, test_data = split_data(inflation_df)

    arima_model = ARIMA(train_data, order=pdq)
    results_ARIMA = arima_model.fit()
    arima_predictions_df = results_ARIMA.get_forecast(steps=len(test_data)).conf_int(alpha=0.05)
    arima_predictions_df['Predictions'] = results_ARIMA.predict(start=arima_predictions_df.index[0],
                                                                end=arima_predictions_df.index[-1])
    arima_predictions_df.index = test_data.index
    predictions_ARIMA = arima_predictions_df[['Predictions']]

    plt.plot(train_data)
    plt.plot(test_data)
    plt.plot(predictions_ARIMA)
    plt.legend(['Training data', 'Test data', 'ARIMA'])
    plt.xlabel('Year')
    plt.ylabel('Inflation')
    plt.suptitle(f'{country} - Inflation')
    plt.show()

    measures("ARIMA", country, test_data, predictions_ARIMA)
