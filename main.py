from sklearn.model_selection import train_test_split

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

columns_used = columns.remove("Blood Pressure")
df = pd.read_csv('database/heart_attack_prediction_dataset.csv')

database_processing.unique_values(df, columns)

features, target = database_prepare.features_and_target(df)
print(features)
print(target)

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)
target_predidction_by_classfication = classification_model.predict(features_test, features_train, target_train, max_iteration=1000)
classification_model.info(target_test, target_predidction_by_classfication)
classification_model.cross_validation(features, target)