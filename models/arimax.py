import statsmodels.api as sm
from sklearn.metrics import mean_squared_error


from plots.plot_model_prediction import plot_model_prediction
from plots.plot_model_prediction_barchart import plot_model_prediction_barchart


def arimax(train_data, train_exog, test_data, test_exog, country):
    print("< ==================== ARIMAX ==================== >\n")

    arimax_model = sm.tsa.SARIMAX(
        train_data,
        order=(2, 1, 2),
        seasonal_order=(0, 0, 0, 0),
        exog=train_exog,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=0)

    arimax_pred = arimax_model.get_forecast(steps=len(test_data), exog=test_exog)
    arimax_pred_df = arimax_pred.conf_int(alpha=0.05)
    arimax_pred_df["Predictions"] = arimax_model.predict(
        start=arimax_pred_df.index[0],
        end=arimax_pred_df.index[-1],
        exog=test_exog)
    arimax_pred_df.index = test_data.index
    y_pred_arimax = arimax_pred_df[["Predictions"]]

    plot_model_prediction(train_data, test_data, y_pred_arimax, "ARIMAX", country)
    plot_model_prediction_barchart(test_data, y_pred_arimax, "ARIMAX", country)

    print(f"Mean Squared Error = {round(mean_squared_error(test_data, y_pred_arimax), 3)}\n")
