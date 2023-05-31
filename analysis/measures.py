from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def measures(model, country,  real, predictions):
    print(f"{model} - {country} - RMSE = {round(mean_squared_error(real, predictions), 3)}")
    print(f"{model} - {country} - MAE = {round(mean_absolute_error(real, predictions), 3)}")
    print(f"{model} - {country} - MAPE = {round(mean_absolute_percentage_error(real, predictions), 3)}")
    print("-----------------------------------------------------------------------------------")