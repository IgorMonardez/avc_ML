import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.tree import DecisionTreeClassifier


def train(features_train, target_train, max_iteration=1000, c_value=10.0):
    classifier = LogisticRegression(max_iter=max_iteration, class_weight='balanced', C=c_value) # Cria um modelo de Regressão Logística com o parâmetro max_iter definido para 1000 e class_weight definido como 'balanced' para lidar com classes desbalanceadas
    classifier.fit(features_train, target_train) # Treina o modelo de Regressão Logística usando os dados de treinamento (features_train e target_train)

    return classifier

def predict(features_test, features_train, target_train, max_iteration=1000, c_value=10.0):
    classifier = train(features_train, target_train, max_iteration, c_value)
    target_predict = classifier.predict(features_test)

    return target_predict

def info(target_test, target_predict):
    print("Accuracy:", accuracy_score(target_test, target_predict))
    print(classification_report(target_test, target_predict))

def cross_validation(features, target):
    clf_cv = DecisionTreeClassifier(
        criterion='gini',
        max_depth=2,
        min_samples_leaf=1,
        min_samples_split=5,
        random_state=42
    )

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

    scores = cross_validate(clf_cv, features, target, cv=cv, scoring=scoring)

    print("===== Validação Cruzada =====")
    for metric in scoring:
        mean_score = np.mean(scores[f'test_{metric}'])
        std_score = np.std(scores[f'test_{metric}'])
        print(f"{metric}: {mean_score:.3f} (±{std_score:.3f})")


def grid_search(features, target):
    param_grid = {
        'max_depth': [2, 3, 4, 5],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [1, 5, 10, 15, 20],
    }


    clf = DecisionTreeClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator= clf,
        param_grid=param_grid,
        cv=30,
        scoring='accuracy'
    )

    grid_search.fit(features, target)

    print("Melhores hiperparâmetros: ", grid_search.best_params_)
    print("Melhor acurácia: ", grid_search.best_score_)
