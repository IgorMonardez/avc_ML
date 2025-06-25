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


def random_forest_rank_features():
    df = pd.read_csv('database/heart_attack_prediction_dataset.csv')

    X,y = prepare.features_and_target(df)

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

    X,y = prepare.features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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

    X, y = prepare.features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    xgb_model = xgb.XGBClassifier(random_state=42)

    # Treinar o modelo XGBoost com os dados de treinamento
    xgb_model.fit(X_train, y_train.values.ravel())

    y_previsto = xgb_model.predict(X_test)

    # Métricas
    accuracy = accuracy_score(y_test, y_previsto)
    print(f"Acurácia: {accuracy:.2f}")

    print("Relatório de Classificação:")
    print(classification_report(y_test, y_previsto))

    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_previsto))


def random_forest():
    df = pd.read_csv('database/heart_attack_prediction_dataset.csv')

    X,y = prepare.features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    model = RandomForestClassifier(random_state=42)

    scores = cross_val_score(model, X_train, y_train, cv=5)

    model.fit(X_train, y_train)

    y_previsto = model.predict(X_test)

    # Métricas
    accuracy = accuracy_score(y_test, y_previsto)
    print(f"Acurácia: {accuracy:.2f}")

    print("Relatório de Classificação:")
    print(classification_report(y_test, y_previsto))

    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_previsto))