from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def train(features_train, target_train, max_iteration=1000):
    classifier = LogisticRegression(max_iter=max_iteration)
    classifier.fit(features_train, target_train) # Treina o modelo de Regressão Logística usando os dados de treinamento (features_train e target_train)

    return classifier

def predict(features_test, features_train, target_train, max_iteration=1000):
    classifier = train(features_train, target_train, max_iteration)
    target_predict = classifier.predict(features_test)

    return target_predict

def info(target_test, target_predict):
    print("Accuracy:", accuracy_score(target_test, target_predict))
    print(classification_report(target_test, target_predict))

def cross_validation(features, target):
    clf_cv = DecisionTreeClassifier(
        criterion='gini',
        max_depth=3,
        min_samples_leaf=1,
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
    