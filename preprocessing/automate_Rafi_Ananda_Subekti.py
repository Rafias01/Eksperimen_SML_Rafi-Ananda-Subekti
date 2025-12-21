import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_data(
    input_path: str,
    test_size: float = 0.2,
    random_state: int = 42
):

    df = pd.read_csv(input_path)

    df = df[df["Age"] > 0]
    df = df[df["Balance"] >= 0]
    df = df.drop_duplicates()

    df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)

    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])

    df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    numerical_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "EstimatedSalary"
    ]

    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data(
        "../bank_dataset_raw/bank_dataset_raw.csv"
    )

    print("Preprocessing selesai")
    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)