def unique_values(df, columns):

    for column in columns:
        print(f"Valores possíveis em {column}:", df[column].unique())

def show_percentage(df, columns):

    for column in columns:
        lista_item = df[column].unique()
        dicionario_porcentagem = {}

        for item in lista_item:
            dicionario_porcentagem[item] = df[df[column] == item].shape[0]


        total = len(df)

        print(f"\nPORCENTAGEM EM {column.upper()}")
        for item in lista_item:
            porcentagem = (dicionario_porcentagem[item] / total) * 100
            print(f"Porcentagem de {str(item).upper()}: {porcentagem:.2f}%")