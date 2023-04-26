import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

from plots.plot_model_prediction import plot_model_prediction
from plots.plot_model_prediction_barchart import plot_model_prediction_barchart


def sarima(train_data, test_data, country):
    print("< ==================== SARIMA ==================== >")
    sarima_model = sm.tsa.SARIMAX(
        train_data,
        order=(2, 1, 2),
        seasonal_order=(2, 1, 2, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=0)

    sarima_pred = sarima_model.get_forecast(steps=len(test_data.index))
    sarima_pred_df = sarima_pred.conf_int(alpha=0.05)
    sarima_pred_df["Predictions"] = sarima_model.predict(start=sarima_pred_df.index[0], end=sarima_pred_df.index[-1])
    sarima_pred_df.index = test_data.index
    y_pred_sarima = sarima_pred_df[["Predictions"]]

    plot_model_prediction(train_data, test_data, y_pred_sarima, "SARIMA", country)
    plot_model_prediction_barchart(test_data, y_pred_sarima, "SARIMA", country)

    print(f"Mean Squared Error = {round(mean_squared_error(test_data, y_pred_sarima), 3)}\n")
