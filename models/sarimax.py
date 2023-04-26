import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

from plots.plot_model_prediction import plot_model_prediction
from plots.plot_model_prediction_barchart import plot_model_prediction_barchart


def sarimax(train_data, train_exog, test_data, test_exog, country):
    print("< ==================== SARIMAX ==================== >\n")

    sarimax_model = sm.tsa.SARIMAX(
        train_data,
        order=(2, 1, 2),
        seasonal_order=(2, 1, 2, 52),
        exog=train_exog,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=0)

    sarimax_pred = sarimax_model.get_forecast(steps=len(test_data), exog=test_exog)
    sarimax_pred_df = sarimax_pred.conf_int(alpha=0.05)
    sarimax_pred_df["Predictions"] = sarimax_model.predict(
        start=sarimax_pred_df.index[0],
        end=sarimax_pred_df.index[-1],
        exog=test_exog)
    sarimax_pred_df.index = test_data.index
    y_pred_sarimax = sarimax_pred_df[["Predictions"]]

    plot_model_prediction(train_data, test_data, y_pred_sarimax, "SARIMAX", country)
    plot_model_prediction_barchart(test_data, y_pred_sarimax, "SARIMAX", country)

    print(f"Mean Squared Error = {round(mean_squared_error(test_data, y_pred_sarimax), 3)}\n")
