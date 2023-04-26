import matplotlib.pyplot as plt


def plot_country_data(inflation_dataset, inflation_dp, country):
    fig, axs = plt.subplots(2, 1)

    inflation_dataset.plot(x='Year', y=inflation_dp, ax=axs[0])
    axs[0].set_ylabel(inflation_dp)
    inflation_dataset.plot(x='Year', y=inflation_dp, kind='bar', ax=axs[1])
    axs[1].set_ylabel(inflation_dp)
    plt.suptitle(f'Inflation at end of the period - {country}')
    plt.show()
