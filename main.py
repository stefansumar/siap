import pandas as pd
import seaborn as sns
import warnings

from models.arima import arima
from models.arimax import arimax
from models.sarima import sarima
from models.sarimax import sarimax
from plots.plot_country_data import plot_country_data
from plots.plot_features import plot_features

sns.set()

warnings.filterwarnings('ignore')

dataset = pd.read_csv("economic-data.csv")

country = "Croatia"

#   Iz dataseta uzimamo drzavu koju zelimo, ovo cemo napraviti tako da bude genericno (Za pocetak uraditi predikcije za drzave Balkana)
dataset = dataset[dataset.Country == country]

#   Uzimamo samo vrednosti iz dataseta koje su izrazene u procentima
percent = "Percent"
percent_change = "Percent change"
pgdp = "Percent of GDP"
ppgdp = "Percent of potential GDP"
ptlp = "Percent of total labor force"

dataset = dataset[(dataset['Units'] == percent) | (dataset['Units'] == percent_change) | (dataset['Units'] == pgdp) | (
        dataset['Units'] == ppgdp) | (dataset['Units'] == ptlp)]

#   Uzimamo sve godine od 1997 do 2021 - Za njih imamo predikcije za Srbiju
dataset_temp = dataset.loc[:, "1997": "2021"]

#   Spajamo tabelu sa vrednostima i dodajemo data pointove
dataset_temp = dataset_temp.join(dataset[["Subject Descriptor"]])

dataset = dataset_temp.transpose()

dataset.columns = dataset.iloc[-1]

#   Poslednja vrsta je visak jer ona sadrzi nazive kolona, stoga je uklanjamo
dataset = dataset.iloc[:-1]

#   Removing unnecessary column because it contains only null values. Inplace True to change original dataset
dataset.drop(columns=['Output gap in percent of potential GDP'], inplace=True)

inflation_dp = 'Inflation, end of period consumer prices'

col_names = list(dataset.columns.values)
col_names.remove(inflation_dp)

#   Menjamo preostale null vrednosti sa prosecnim vrednostima kolone
for col_name in dataset:
    col_mean = dataset[col_name].astype('float').mean()
    col_mean_rounded = round(col_mean, 3)
    dataset[col_name] = dataset[col_name].fillna(col_mean_rounded)

#   Vrsimo konverziju vrednosti iz dataseta u float kako bismo mogli da plotujemo
dataset = dataset.astype(float)

#   Pripremljeni podaci za podelu na trening i test
inflation_dataset = dataset[inflation_dp]
exog_variables = dataset.drop(columns=inflation_dp)

#   Vizualizacija podataka o inflaciji za jednu drzavu (ciljna varijabla)
plot_country_data(inflation_dataset, inflation_dp, country)

#   Nakon ovog koraka izdvajamo
plot_features(exog_variables, col_names, country)

#   Deljenje na trening i test skup
train_data = inflation_dataset[:'2015']
test_data = inflation_dataset['2015':]
train_exog = exog_variables[:'2015']
test_exog = exog_variables['2015':]

  # ARIMA - Treniranje modela i rezultati

arima(train_data, test_data, country)

#   ARIMAX - Treniranje modela i rezultati

arimax(train_data, train_exog, test_data, test_exog, country)

#   SARIMA - Treniranje modela i rezultati

sarima(train_data, test_data, country)

#   SARIMAX - Treniranje modela i rezultati

sarimax(train_data, train_exog, test_data, test_exog, country)
