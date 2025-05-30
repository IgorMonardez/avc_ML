def unique_values(df, columns):

    for column in columns:
        print(f"Valores possíveis em {column}:", df[column].unique())
