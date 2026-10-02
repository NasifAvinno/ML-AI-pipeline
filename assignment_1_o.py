import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# TASK 1: DATA LOADING AND INITIAL INSPECTION


url = "https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv"

df = pd.read_csv(url)


print("First 5 Rows:")
print(df.head())

# Data information
print("\nDataset Information:")
df.info()


print("\nDescriptive Statistics:")
print(df.describe())


print("\nMissing Values:")
print(df.isnull().sum())



# TASK 2: HANDLING MISSING VALUES


cabin_missing = df['Cabin'].isnull().sum()
cabin_percentage = (cabin_missing / len(df)) * 100

print("\nCabin Missing Values:", cabin_missing)
print("Cabin Missing Percentage:", round(cabin_percentage, 2), "%")


df = df.drop('Cabin', axis=1)


embarked_mode = df['Embarked'].mode()[0]

print("\nMost Frequent Embarked Port:", embarked_mode)


df['Embarked'] = df['Embarked'].fillna(embarked_mode)


age_median = df['Age'].median()

print("Median Age:", age_median)


df['Age'] = df['Age'].fillna(age_median)


print("\nMissing Values After Cleaning:")
print(df.isnull().sum())



# TASK 3: UNIVARIATE ANALYSIS


survival_rate = df['Survived'].mean() * 100

print("\nOverall Survival Rate:",
      round(survival_rate, 2), "%")



plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df)

plt.title('Survival Distribution')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Number of Passengers')

plt.show()



plt.figure(figsize=(6, 4))
sns.countplot(x='Pclass', data=df)

plt.title('Passenger Class Distribution')
plt.xlabel('Passenger Class')
plt.ylabel('Number of Passengers')

plt.show()

print("\nPassenger Class Counts:")
print(df['Pclass'].value_counts())



plt.figure(figsize=(8, 5))
plt.hist(df['Age'], bins=20)

plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')

plt.show()



# TASK 4: BIVARIATE AND MULTIVARIATE ANALYSIS


print("\nSurvival by Sex:")
sex_survival = pd.crosstab(df['Sex'], df['Survived'])
print(sex_survival)

print("\nSurvival Rate by Sex:")
sex_survival_rate = df.groupby('Sex')['Survived'].mean() * 100
print(sex_survival_rate)


plt.figure(figsize=(7, 5))
sns.countplot(x='Sex', hue='Survived', data=df)

plt.title('Survival by Sex')
plt.xlabel('Sex')
plt.ylabel('Number of Passengers')

plt.show()



print("\nSurvival Rate by Passenger Class:")

class_survival_rate = df.groupby('Pclass')['Survived'].mean() * 100

print(class_survival_rate)


plt.figure(figsize=(7, 5))
sns.barplot(x='Pclass', y='Survived', data=df)

plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')

plt.show()



plt.figure(figsize=(8, 5))

sns.boxplot(
    x='Survived',
    y='Age',
    data=df
)

plt.title('Age Distribution of Survivors vs Non-Survivors')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Age')

plt.show()


plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x='Age',
    hue='Survived',
    bins=20,
    kde=True
)

plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')

plt.show()



print("\nSurvival Rate by Port of Embarkation:")

embarked_survival = df.groupby('Embarked')['Survived'].mean() * 100

print(embarked_survival)


plt.figure(figsize=(7, 5))

sns.barplot(
    x='Embarked',
    y='Survived',
    data=df
)

plt.title('Survival Rate by Port of Embarkation')
plt.xlabel('Port of Embarkation')
plt.ylabel('Survival Rate')

plt.show()



plt.figure(figsize=(8, 5))

sns.barplot(
    x='Pclass',
    y='Survived',
    hue='Sex',
    data=df
)

plt.title('Survival Rate by Class and Sex')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')

plt.show()



# TASK 5: FINAL RESULTS


print("FINAL EDA RESULTS")

print("\nOverall Survival Rate:",
      round(survival_rate, 2), "%")

print("\nSurvival Rate by Sex:")
print(sex_survival_rate.round(2))

print("\nSurvival Rate by Class:")
print(class_survival_rate.round(2))

print("\nSurvival Rate by Embarked:")
print(embarked_survival.round(2))

print("\nEDA Completed Successfully!")