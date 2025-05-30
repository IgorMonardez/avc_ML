

def unique_values(df, *args):
    for column in args:
        print(f"Valores possíveis em {column}:", df[column].unique())
