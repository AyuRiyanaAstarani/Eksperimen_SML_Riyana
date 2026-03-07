# Import library
import pandas as pd
from sklearn.model_selection import train_test_split

# Fungsi preprocessing
def preprocess_data(input_path, output_train, output_test):

    # Load dataset
    df = pd.read_csv(input_path)

    # Drop kolom tidak relevan
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

    # Encoding categorical
    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)

    # Pisahkan fitur dan target
    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Gabungkan kembali fitur dan target
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    # Simpan hasil preprocessing
    train_data.to_csv(output_train, index=False)
    test_data.to_csv(output_test, index=False)

    print("Preprocessing selesai. Dataset berhasil disimpan.")


# Menjalankan fungsi
if __name__ == "__main__":

    input_path = "../dataset_raw/Churn_Modelling.csv"

    output_train = "dataset_preprocessing/train_preprocessed.csv"
    output_test = "dataset_preprocessing/test_preprocessed.csv"

    preprocess_data(input_path, output_train, output_test)