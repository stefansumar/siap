import matplotlib.pyplot as plt


def plot_country_data(inflation_dp, country):
    plt.plot(inflation_dp)
    plt.suptitle(f'{country} - Inflation at end of the period')
    plt.show()
