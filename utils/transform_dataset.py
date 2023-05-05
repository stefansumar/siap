import pandas as pd


def transform_dataset(dataset, country):
    #   Iz dataseta uzimamo drzavu koju zelimo
    dataset = dataset[dataset.Country == country]

    #   Uzimamo samo vrednosti iz dataseta koje su izrazene u procentima
    percent = "Percent"
    percent_change = "Percent change"
    pgdp = "Percent of GDP"
    ppgdp = "Percent of potential GDP"
    ptlp = "Percent of total labor force"

    dataset = dataset[
        (dataset['Units'] == percent) | (dataset['Units'] == percent_change) | (dataset['Units'] == pgdp) | (
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
    inflation_dataset = pd.Series(dataset[inflation_dp])
    exog_variables = dataset.drop(columns=inflation_dp)

    inflation_dataset.index = pd.to_datetime(inflation_dataset.index, format='%Y')
    inflation_dataset = inflation_dataset.resample('M').first().interpolate('linear')

    exog_variables.index = pd.to_datetime(exog_variables.index, format="%Y")
    exog_variables = exog_variables.resample('M').first().interpolate('linear')

    return tuple((inflation_dataset, exog_variables))
