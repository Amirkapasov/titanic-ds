
import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


def find_data_file(filename: str) -> str:
    candidates = [
        filename,
        os.path.join(".", filename),
        os.path.join("/mnt/data", filename),
        os.path.join("/kaggle/input/titanic", filename),
        os.path.join("/kaggle/input/titanic-machine-learning-from-disaster", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find {filename}. Put it рядом со скриптом, "
        f"or in /mnt/data, or use Kaggle Titanic input directory."
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["cabin_multiple"] = df["Cabin"].apply(lambda x: 0 if pd.isna(x) else len(str(x).split()))
    df["cabin_adv"] = df["Cabin"].apply(lambda x: "n" if pd.isna(x) else str(x)[0])

    df["numeric_ticket"] = df["Ticket"].apply(lambda x: 1 if str(x).isnumeric() else 0)
    df["ticket_letters"] = df["Ticket"].apply(
        lambda x: "".join(str(x).split(" ")[:-1]).replace(".", "").replace("/", "").lower()
        if len(str(x).split(" ")[:-1]) > 0 else "none"
    )
    df["ticket_letters"] = df["ticket_letters"].replace("", "none")

    df["name_title"] = df["Name"].apply(lambda x: str(x).split(",")[1].split(".")[0].strip())
    df["family_size"] = df["SibSp"] + df["Parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    df["norm_fare"] = np.log1p(df["Fare"])
    df["Pclass"] = df["Pclass"].astype(str)

    return df


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["train_test"] = 1
    test_df["train_test"] = 0
    if "Survived" not in test_df.columns:
        test_df["Survived"] = np.nan

    all_data = pd.concat([train_df, test_df], ignore_index=True)
    all_data = add_features(all_data)

    feature_cols = [
        "Pclass", "Sex", "Age", "SibSp", "Parch", "norm_fare", "Embarked",
        "cabin_adv", "cabin_multiple", "numeric_ticket", "ticket_letters",
        "name_title", "family_size", "is_alone", "train_test"
    ]

    all_dummies = pd.get_dummies(all_data[feature_cols], drop_first=False)

    scaled = all_dummies.copy()
    scale_cols = ["Age", "SibSp", "Parch", "norm_fare", "family_size", "cabin_multiple"]
    scaler = StandardScaler()
    scaled[scale_cols] = scaler.fit_transform(scaled[scale_cols])

    X_train = all_dummies[all_dummies["train_test_1"] == True].drop(columns=["train_test_0", "train_test_1"])
    X_test = all_dummies[all_dummies["train_test_0"] == True].drop(columns=["train_test_0", "train_test_1"])

    X_train_scaled = scaled[scaled["train_test_1"] == True].drop(columns=["train_test_0", "train_test_1"])
    X_test_scaled = scaled[scaled["train_test_0"] == True].drop(columns=["train_test_0", "train_test_1"])

    y_train = all_data.loc[all_data["train_test"] == 1, "Survived"].astype(int).reset_index(drop=True)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    X_train_scaled = X_train_scaled.reset_index(drop=True)
    X_test_scaled = X_test_scaled.reset_index(drop=True)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train


def evaluate_models(X_train, X_train_scaled, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "LogisticRegression": (LogisticRegression(max_iter=2000, solver="liblinear"), X_train_scaled),
        "KNN": (KNeighborsClassifier(n_neighbors=7, weights="distance"), X_train_scaled),
        "RandomForest": (RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42
        ), X_train),
        "SVC": (SVC(C=1, gamma="scale", probability=True, random_state=42), X_train_scaled),
    }

    results = {}
    for name, (model, X_used) in models.items():
        score = cross_val_score(model, X_used, y_train, cv=cv, scoring="accuracy")
        results[name] = (score.mean(), score.std())
        print(f"{name}: mean={score.mean():.4f}, std={score.std():.4f}")

    return models


def train_and_predict(train_df, test_df):
    X_train, X_test, X_train_scaled, X_test_scaled, y_train = build_features(train_df, test_df)

    print("Cross-validation:")
    models = evaluate_models(X_train, X_train_scaled, y_train)

    lr = LogisticRegression(max_iter=2000, solver="liblinear")
    knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42
    )
    svc = SVC(C=1, gamma="scale", probability=True, random_state=42)

    lr.fit(X_train_scaled, y_train)
    knn.fit(X_train_scaled, y_train)
    rf.fit(X_train, y_train)
    svc.fit(X_train_scaled, y_train)

    # Voting on scaled features with models that benefit from scaling
    voting = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=2000, solver="liblinear")),
            ("knn", KNeighborsClassifier(n_neighbors=7, weights="distance")),
            ("svc", SVC(C=1, gamma="scale", probability=True, random_state=42)),
        ],
        voting="soft"
    )
    voting.fit(X_train_scaled, y_train)
    pred_voting = voting.predict(X_test_scaled).astype(int)

    pred_rf = rf.predict(X_test).astype(int)

    submission_voting = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": pred_voting
    })
    submission_rf = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": pred_rf
    })

    submission_voting.to_csv("submission_voting.csv", index=False)
    submission_rf.to_csv("submission_rf.csv", index=False)

    print("\nSaved files:")
    print("- submission_voting.csv")
    print("- submission_rf.csv")

    return submission_voting, submission_rf


if __name__ == "__main__":
    train_path = find_data_file("train.csv")
    test_path = find_data_file("test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    submission_voting, submission_rf = train_and_predict(train_df, test_df)

    print("\nPreview:")
    print(submission_voting.head())
