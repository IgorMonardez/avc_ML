from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import xgboost as xgb
import sys
from package import database_processing
from package import prepare
from imblearn.over_sampling import SMOTE

def random_forest_rank_features():
    df = pd.read_csv('database/heart_attack_prediction_dataset.csv')

    X = df.drop(['Patient ID', 'Heart Attack Risk', 'Previous Heart Problems', 'Alcohol Consumption',
                 'Family History', 'Medication Use', 'Obesity', 'Diabetes', 'Diet', 'Sex', 'Continent', 'Country',
                 'Hemisphere'], axis=1)

    y = df['Heart Attack Risk']

    # X = prepare.one_hot_encoding(X)

    X = prepare.split_blood_pressure(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(random_state=42)

    scores = cross_val_score(model, X_train, y_train, cv=5)

    model.fit(X_train, y_train)

    # Obter a importância dos atributos do RandomForest
    rf_importances = model.feature_importances_

    # Criar um DataFrame para visualizar a importância dos atributos
    feature_importances_rf = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_importances
    })

    feature_importances_rf = feature_importances_rf.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 8))
    plt.title("Importância dos Atributos - RandomForest")
    plt.barh(feature_importances_rf['feature'], feature_importances_rf['importance'], color='b', align='center')
    plt.xlabel("Importância")
    plt.gca().invert_yaxis()

    # Ajustar o espaço à esquerda do gráfico
    plt.subplots_adjust(left=0.3)

    plt.show()


def xgboost_rank_features():
    df = pd.read_csv('database/heart_attack_prediction_dataset.csv')

    X = df.drop(['Patient ID', 'Heart Attack Risk', 'Previous Heart Problems', 'Alcohol Consumption',
                 'Family History', 'Medication Use', 'Obesity', 'Diabetes', 'Diet', 'Sex', 'Continent', 'Country',
                'Hemisphere'], axis=1)

    y = df['Heart Attack Risk']

    # X = prepare.one_hot_encoding(X)

    X = prepare.split_blood_pressure(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    xgb_model = xgb.XGBClassifier(random_state=42)

    # Treinar o modelo XGBoost com os dados de treinamento
    xgb_model.fit(X_train, y_train.values.ravel())

    # Obter a importância dos atributos do XGBoost
    xgb_importances = xgb_model.feature_importances_

    feature_importances_xgb = pd.DataFrame({
        'feature': X_train.columns,
        'importance': xgb_importances
    })
    feature_importances_xgb = feature_importances_xgb.sort_values('importance', ascending=False)

    # Plotar a importância dos atributos para XGBoost
    plt.figure(figsize=(10, 8))
    plt.title("Importância dos Atributos - XGBoost")
    plt.barh(feature_importances_xgb['feature'], feature_importances_xgb['importance'], color='r', align='center')
    plt.xlabel("Importância")
    plt.gca().invert_yaxis()

    # Ajustar o espaço à esquerda do gráfico
    plt.subplots_adjust(left=0.3)

    plt.show()


def xg_boost():
    df = pd.read_csv('database/heart_attack_prediction_dataset.csv')

    #X = df.drop(['Patient ID', 'Heart Attack Risk', 'Previous Heart Problems', 'Alcohol Consumption',
    #             'Family History', 'Medication Use', 'Obesity', 'Diabetes', 'Diet', 'Sex', 'Continent', 'Country',
    #             'Hemisphere'], axis=1)

    X = df.drop(['Heart Attack Risk','Patient ID','Country', 'Continent', 'Hemisphere'], axis=1)
    X = prepare.split_blood_pressure(X)
    X = pd.get_dummies(X)

    y = df['Heart Attack Risk']

    # X = prepare.one_hot_encoding(X)

    # X = prepare.split_blood_pressure(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    neg, pos = np.bincount(y_train)
    ratio = pos/neg
    xgb_model = xgb.XGBClassifier(random_state=42)
    # Treinar o modelo XGBoost com os dados de treinamento
    xgb_model.fit(X_train, y_train.values.ravel())
  #   y_previsto = xgb_model.predict(X_test)
    y_probs = xgb_model.predict_proba(X_test)
    threshold = 0.3
    y_previsto = (y_probs[:, 1]>= threshold).astype(int)

    # Métricas
    accuracy = accuracy_score(y_test, y_previsto)
    print(f"Acurácia: {accuracy:.2f}")

    print("Relatório de Classificação:")
    print(classification_report(y_test, y_previsto))

    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_previsto))
