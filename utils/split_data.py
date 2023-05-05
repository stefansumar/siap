def split_data(inflation_df, exog_df=[]):
    train_data = inflation_df[:'2015']
    test_data = inflation_df['2015':]
    if len(exog_df):
        train_exog = exog_df[:'2015']
        test_exog = exog_df['2016':]

        return tuple((train_data, test_data, train_exog, test_exog))

    return tuple((train_data, test_data))
