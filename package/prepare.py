import pandas as pd


def one_hot_encoding(features):
    x = pd.get_dummies(features, columns=['Sex', 'Diet', 'Country', 'Continent', 'Hemisphere'])

    return x


def features_and_target(df):
    # Features
    features = df.drop(columns=['Heart Attack Risk', 'Patient ID', 'Blood Pressure'])
    features = one_hot_encoding(features)
    features.to_csv('database/heart_attack_db_features.csv', index=False)

    # Target
    target = df['Heart Attack Risk']
    target.to_csv('database/heart_attack_db_target.csv', index=False)

    return features, target