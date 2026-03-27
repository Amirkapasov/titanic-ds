import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('data/train.csv')

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)

rare_titles = ["Lady", "Countess", "Capt", "Col", "Don", "Dr",
               "Major", "Rev", "Sir", "Jonkheer", "Dona"]
df["Title"] = df["Title"].replace(rare_titles, "Rare")
df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
df['Title'] = df['Title'].map({
    'Mr':0,
    'Miss':1,
    'Mrs':2,
    'Master':3,
    'Rare':4,
})
df['Family'] = df['SibSp']+df['Parch']+1
df['Along'] = (df['Family'] == 1).astype(int)

df['AgeBand'] = pd.cut(df['Age'], [0, 12, 20, 40, 60, 100])
df['AgeBand'] = df['AgeBand'].astype(str)
df['AgeBand'] = df['AgeBand'].map({
    '(0, 12]': 0,
    '(12, 20]': 1,
    '(20, 40]': 2,
    '(40, 60]': 3,
    '(60, 100]': 4
})
df['Class_Sex'] = df['Pclass'].astype(str) + "_" + df['Sex'].astype(str)
df['Class_Sex'] = df['Class_Sex'].map({
    '1_1': 0,
    '1_0': 1,
    '2_1': 2,
    '2_0': 3,
    '3_1': 4,
    '3_0': 5
})


df['FareBand'] = pd.qcut(df['Fare'], 4,labels=False)



df = df.drop(columns=['Along', 'Parch', 'Embarked', 'AgeBand', 'FareBand', 'Name', 'Ticket', 'Cabin', 'PassengerId'])

X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=6,
    min_samples_split=8,
    min_samples_leaf=3,
    random_state=42
)
rf_model.fit(X_train, y_train)
print(f"RandomForest: {accuracy_score(y_test, rf_model.predict(X_test)):.1%}")

test_df = pd.read_csv('data/test.csv')
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())
test_df['Embarked'] = test_df['Embarked'].fillna(test_df['Embarked'].mode()[0])
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})
test_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
test_df["Title"] = test_df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)

rare_titles = ["Lady", "Countess", "Capt", "Col", "Don", "Dr",
               "Major", "Rev", "Sir", "Jonkheer", "Dona"]
test_df["Title"] = test_df["Title"].replace(rare_titles, "Rare")
test_df["Title"] = test_df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
test_df['Title'] = test_df['Title'].map({
    'Mr':0,
    'Miss':1,
    'Mrs':2,
    'Master':3,
    'Rare':4,
})
test_df['Family'] = test_df['SibSp']+test_df['Parch']+1
test_df['Along'] = (test_df['Family'] == 1).astype(int)
test_df['AgeBand'] = pd.cut(test_df['Age'], [0, 12, 20, 40, 60, 100])
test_df['AgeBand'] = test_df['AgeBand'].astype(str)
test_df['AgeBand'] = test_df['AgeBand'].map({
    '(0, 12]': 0,
    '(12, 20]': 1,
    '(20, 40]': 2,
    '(40, 60]': 3,
    '(60, 100]': 4
})
test_df['Class_Sex'] = test_df['Pclass'].astype(str) + "_" + test_df['Sex'].astype(str)
test_df['Class_Sex'] = test_df['Class_Sex'].map({
    '1_1': 0,
    '1_0': 1,
    '2_1': 2,
    '2_0': 3,
    '3_1': 4,
    '3_0': 5
})


test_df['FareBand'] = pd.qcut(test_df['Fare'], 4,labels=False)





test_df = test_df.drop(columns=['Along', 'Parch', 'Embarked', 'AgeBand', 'FareBand', 'Name', 'Ticket', 'Cabin'])
passenger_ids = test_df['PassengerId']
X_kaggle = test_df.drop(columns=['PassengerId'])

predictions = rf_model.predict(X_kaggle)
submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': predictions
})
submission.to_csv('submission.csv', index=False)
print(submission.head())

feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
print(feature_importance.sort_values(ascending=False))