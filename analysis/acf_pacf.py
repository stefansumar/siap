from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt


def acf_pacf(inflation_df):
    fig, (ax1, ax2) = plt.subplots(2)
    plot_acf(inflation_df, ax=ax1)
    plot_pacf(inflation_df.diff().diff().dropna(), lags=len(inflation_df) / 4, ax=ax2, method='ywm')
    plt.show()
