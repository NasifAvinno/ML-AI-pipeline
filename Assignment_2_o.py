import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Datasheet loading

file_path = r'd:\archive\heart.csv'
df = pd.read_csv(file_path)

# Datasheet printing

print("First Five Rows of the Dataset : ")
print(df.head())

print("\n Last Five Rows of the Dataset : ")
print(df.tail())

print("\n Shape of the Dataset : ")
print(df.shape)

print("\n Columns : ")
print(df.columns.tolist())

print("\n Database Information : ")
df.info()

print("\n Missing values of Column : ")
print(df.isnull().sum())


# Catagorywise analysis

print("Heart Disease Count : ")
print(df['target'].value_counts())

print("\n Patient without Heart Disease : ")
print(df[df['target'] == 0].shape[0])

print("\n Patient with Heart Disease : ")
print(df[df['target'] == 1].shape[0])

print("\n Gender Count : ")
print(df['sex'].value_counts())

print("\n Female Count : ")
print(df[df['sex'] == 0].shape[0])

print("\n Male Count : ")
print(df[df['sex'] == 1].shape[0])


# Data slection using iloc

print("First 10 Rows : ")
print(df.iloc[0:10, :])

print("\n Age Sex & Cholesterol : ")
print(df.iloc[:, [0, 1, 4]])

print("\n Rows 20 to 30 and columns 0 to 4:")
print(df.iloc[20:31, 0:5])


# Data Filtering

older_than_50 = df[df['age'] > 50]
print("Older Than 50 : ")
print(older_than_50)

heart_disease_high_cholesterol = df[
    (df['target'] == 1) & (df['chol'] > 240)
]

print("Patients with Heart Disease and High Cholesterol : ")
print(heart_disease_high_cholesterol)

count = len(heart_disease_high_cholesterol)
print("Number of Patients with Heart Disease and High Cholesterol : ")
print(count)