def drop_columns(df, columns):
    df.drop(columns, 1, inplace=True)
