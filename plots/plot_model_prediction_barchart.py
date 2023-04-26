import numpy as np
import matplotlib.pyplot as plt


def plot_model_prediction_barchart(t_v, p_v, model, country):
    test_values = t_v.tolist()
    predicted_values = p_v['Predictions'].round(3).tolist()

    # set width of bar
    bar_width = 0.25
    fig = plt.subplots()

    # Set position of bar on X axis
    real_bar = np.arange(len(test_values))
    predicted_bar = [x + bar_width for x in real_bar]

    # Make the plot
    plt.bar(real_bar, test_values, color='dodgerblue', width=bar_width,
            label='Inflation')
    plt.bar(predicted_bar, predicted_values, color='darkorange', width=bar_width,
            label='Predicted inflation')

    # Adding Xticks
    plt.xlabel('Year')
    plt.ylabel('Inflation')
    plt.xticks([r + bar_width for r in range(len(test_values))],
               ['2015', '2016', '2017', '2018', '2019', '2020', '2021'])
    plt.legend()
    plt.suptitle(f'{model} Prediction - {country}')
    plt.show()
