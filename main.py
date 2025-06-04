from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

import package.database_processing as database_processing
import package.prepare as database_prepare
import pandas as pd
import numpy as np

from package import classification_model

columns = ["Age", "Sex", "Cholesterol", "Blood Pressure", "Heart Rate", "Diabetes",
           "Family History", "Smoking", "Obesity", "Alcohol Consumption", "Exercise Hours Per Week",
           "Diet", "Previous Heart Problems", "Medication Use", "Stress Level",
           "Sedentary Hours Per Day", "Income", "BMI", "Triglycerides",
           "Physical Activity Days Per Week", "Sleep Hours Per Day", "Country", "Continent",
           "Hemisphere", "Heart Attack Risk"]

# columns_used = columns.remove("Blood Pressure")
df = pd.read_csv('database/heart_attack_prediction_dataset.csv')
#
# database_processing.unique_values(df, columns)
#
features, target = database_prepare.features_and_target(df)
# print(features)
# print(target)
# param_grid = {
#     'max_depth': [2, 3, 4, 5],
#     'min_samples_split': [5, 10, 15],
#     'min_samples_leaf': [1, 5, 10, 15, 20],
# }


# clf = DecisionTreeClassifier(random_state=42)
#
# grid_search = GridSearchCV(
#     estimator= clf,
#     param_grid=param_grid,
#     cv=30,
#     scoring='accuracy'
# )
#
# grid_search.fit(features, target)
#
# print("Melhores hiperparâmetros: ", grid_search.best_params_)
# print("Melhor acurácia: ", grid_search.best_score_)
# classification_model.grid_search(features, target)
classification_model.cross_validation(features, target)
# features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3, random_state=42)
# target_predidction_by_classfication = classification_model.predict(features_test, features_train, target_train, max_iteration=1000)
# classification_model.info(target_test, target_predidction_by_classfication)
# classification_model.cross_validation(features, target)