import matplotlib.pyplot as plt


def plot_features(exog_variables, country):
    col_names = list(exog_variables.columns.values)

    exog_variables.plot(subplots=True,
                        title=col_names,
                        legend=False,
                        layout=(6, 3),
                        sharex=True,
                        figsize=(11, 8))
    plt.suptitle(f'Economic Indicators Over Time - {country}', fontsize=20)
    plt.show()
