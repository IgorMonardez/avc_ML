from sklearn.preprocessing import StandardScaler
import pandas as pd


def one_hot_encoding(features):
    x = pd.get_dummies(features)

    return x


def split_blood_pressure(df):
    df[['Systolic Pressure', 'Diastolic Pressure']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)

    df['BP Ratio'] = df['Systolic Pressure'] / df['Diastolic Pressure']
    df.drop(columns=['Blood Pressure'], inplace=True)

    return df


def features_and_target(df):
    # Criar as colunas Systolic e Diastolic
    df = split_blood_pressure(df)

    # Dropar colunas desnecessárias
    features = df.drop(columns=['Patient ID', 'Heart Attack Risk'])
 
    features = one_hot_encoding(features)

    # Seleciona apenas colunas numéricas (após o one-hot)
    numeric_cols = features.select_dtypes(include=['int64', 'float64']).columns
    non_binary_numeric_cols = [col for col in numeric_cols if features[col].nunique() > 2]

    # Aplica StandardScaler nas colunas numéricas
    scaler = StandardScaler()
    features[non_binary_numeric_cols] = scaler.fit_transform(features[non_binary_numeric_cols])

    features.to_csv('database/heart_attack_db_features.csv', index=False)

    # Target
    target = df['Heart Attack Risk']
    target.to_csv('database/heart_attack_db_target.csv', index=False)

    return features, target
