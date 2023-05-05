import pandas as pd
import seaborn as sns
import warnings

from analysis.acf_pacf import acf_pacf
from analysis.adf_test import adf_test
from models.arima import arima
from models.arimax import arimax
from models.sarima import sarima
from models.sarimax import sarimax
from plots.plot_country_data import plot_country_data
from utils.transform_dataset import transform_dataset


sns.set()
warnings.filterwarnings('ignore')

dataset = pd.read_csv("economic-data.csv")

serbia = 'Serbia'
croatia = 'Croatia'
bosnia = 'Bosnia and Herzegovina'

serbia_inflation_df, serbia_exog_df = transform_dataset(dataset, serbia)
croatia_inflation_df, croatia_exog_df = transform_dataset(dataset, croatia)
bh_inflation_df, bh_exog_df = transform_dataset(dataset, "Bosnia and Herzegovina")

# Serbia
plot_country_data(serbia_inflation_df, serbia)
adf_test(serbia_inflation_df.diff().dropna(), serbia)
acf_pacf(serbia_inflation_df.diff().dropna())
arima(serbia_inflation_df, serbia, 6, 1, 7)
arimax(serbia_inflation_df, serbia_exog_df, serbia, 6, 1, 7)
sarima(serbia_inflation_df, serbia, 0, 0, 0, 0)
sarimax(serbia_inflation_df, serbia_exog_df, serbia, 0, 0, 0, 0)

# Croatia
plot_country_data(croatia_inflation_df, croatia)
adf_test(croatia_inflation_df.diff().dropna(), croatia)
acf_pacf(serbia_inflation_df.diff().dropna())
arima(serbia_inflation_df, croatia, 2, 1, 2)
arimax(croatia_inflation_df, croatia_exog_df, croatia, 2, 1, 2)
sarima(croatia_inflation_df, croatia, 0, 0, 0, 0)
sarimax(croatia_inflation_df, croatia_exog_df, croatia, 0, 0, 0, 0)

# Bosnia and Herzegovina
plot_country_data(bh_inflation_df, bosnia)
adf_test(bh_inflation_df, bosnia)
acf_pacf(bh_inflation_df)
arima(bh_inflation_df, bosnia, 5, 0, 5)
arimax(bh_inflation_df, bh_exog_df, bosnia, 5, 0, 5)
sarima(bh_inflation_df, bosnia, 0, 0, 0, 0)
sarimax(bh_inflation_df, bh_exog_df, bosnia, 0, 0, 0, 0)
