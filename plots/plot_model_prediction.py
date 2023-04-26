import matplotlib.pyplot as plt


def plot_model_prediction(train_data, test_data, prediction, model, country):
    plt.plot(train_data, color='blue')
    plt.plot(test_data, color='red')
    plt.plot(prediction, color='green', label=f'{model} Predictions')
    plt.ylabel('Inflation')
    plt.xlabel('Year')
    plt.xticks(rotation=45)
    plt.title(f'{model} Prediction - {country}')
    plt.legend()
    plt.show()
