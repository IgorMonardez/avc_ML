import package.database_processing as database_processing

import pandas as pd
import numpy as np

df = pd.read_csv('database/heart_attack_prediction_dataset.csv')

database_processing.unique_values(df, "Age", "Sex", "Cholesterol", "Blood Pressure", "Heart Rate", "Diabetes",
                              "Family History", "Smoking", "Obesity", "Alcohol Consumption", "Exercise Hours Per Week",
                              "Diet", "Previous Heart Problems", "Medication Use", "Stress Level",
                              "Sedentary Hours Per Day", "Income", "BMI", "Triglycerides",
                              "Physical Activity Days Per Week", "Sleep Hours Per Day", "Country", "Continent",
                              "Hemisphere", "Heart Attack Risk")
