from sklearn.preprocessing import StandardScaler
import pandas as pd


def one_hot_encoding(features):
    x = pd.get_dummies(features, columns=['Sex', 'Diet', 'Country', 'Continent', 'Hemisphere'])

    return x


def split_blood_pressure(df):
    df[['Systolic Pressure', 'Diastolic Pressure']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
    df.drop(columns=['Blood Pressure'], inplace=True)
    return df


def features_and_target(df):
    # Criar as colunas Systolic e Diastolic
    df = split_blood_pressure(df)

    # Dropar colunas desnecessárias
    features = df.drop(columns=['Heart Attack Risk', 'Patient ID'])
    features = one_hot_encoding(features)

    # Seleciona apenas colunas numéricas (após o one-hot)
    numeric_cols = features.select_dtypes(include=['int64', 'float64']).columns

    # Aplica StandardScaler nas colunas numéricas
    scaler = StandardScaler()
    features[numeric_cols] = scaler.fit_transform(features[numeric_cols])

    features.to_csv('database/heart_attack_db_features.csv', index=False)

    # Seleciona apenas colunas numéricas (após o one-hot)
    numeric_cols = features.select_dtypes(include=['int64', 'float64']).columns

    # Diagnóstico das escalas numéricas antes da normalização
    # numeric_cols = features.select_dtypes(include=['int64', 'float64']).columns
    # print("Diagnóstico das escalas numéricas antes da normalização:\n")
    # print(features[numeric_cols].describe())

    # Aplica StandardScaler nas colunas numéricas
    scaler = StandardScaler()
    features[numeric_cols] = scaler.fit_transform(features[numeric_cols])

    # Target
    target = df['Heart Attack Risk']
    target.to_csv('database/heart_attack_db_target.csv', index=False)

    return features, target