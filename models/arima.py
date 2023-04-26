import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

from plots.plot_model_prediction import plot_model_prediction
from plots.plot_model_prediction_barchart import plot_model_prediction_barchart


def arima(train_data, test_data, country):
    print("< ==================== ARIMA ==================== >\n")

    arima_model = sm.tsa.ARIMA(
        train_data,
        order=(4, 1, 3),
        freq=None
    ).fit()

    arima_pred = arima_model.get_forecast(len(test_data.index))
    arima_pred_df = arima_pred.conf_int(alpha=0.05)
    arima_pred_df["Predictions"] = arima_model.predict(start=arima_pred_df.index[0], end=arima_pred_df.index[-1])
    arima_pred_df.index = test_data.index
    y_pred_arima = arima_pred_df[["Predictions"]]

    plot_model_prediction(train_data, test_data, y_pred_arima, "ARIMA", country)
    plot_model_prediction_barchart(test_data, y_pred_arima, "ARIMA", country)

    print(f"Mean Squared Error = {round(mean_squared_error(test_data, y_pred_arima), 3)}\n")
