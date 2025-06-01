from sklearn.preprocessing import StandardScaler
import pandas as pd


def one_hot_encoding(features):
    x = pd.get_dummies(features, columns=['Sex', 'Diet', 'Country', 'Continent', 'Hemisphere'])
    return x

def features_and_target(df):
    # Remove colunas desnecessárias
    features = df.drop(columns=['Heart Attack Risk', 'Patient ID', 'Blood Pressure'])

    # One-Hot Encoding
    features = one_hot_encoding(features)

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