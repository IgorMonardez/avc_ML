from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def train(features_train, target_train, max_iteration=1000):
    classifier = LogisticRegression(max_iter=max_iteration, class_weight='balanced') # Cria um modelo de Regressão Logística com o parâmetro max_iter definido para 1000 e class_weight definido como 'balanced' para lidar com classes desbalanceadas
    classifier.fit(features_train, target_train) # Treina o modelo de Regressão Logística usando os dados de treinamento (features_train e target_train)

    return classifier

def predict(features_test, features_train, target_train, max_iteration=1000):
    classifier = train(features_train, target_train, max_iteration)
    target_predict = classifier.predict(features_test)

    return target_predict

def info(target_test, target_predict):
    print("Accuracy:", accuracy_score(target_test, target_predict))
    print(classification_report(target_test, target_predict))
