import pandas as pd

from src.preprocess import clean_text
from src.train import train_model
from src.evaluate import evaluate_model


def main():
    print("Starting Project...")

    train_df = pd.read_csv(
    "data/train_data.txt",
    sep=" ::: ",
    engine="python",
    names=["id", "title", "genre", "plot"]
    )
    test_df = pd.read_csv(
    "data/test_data.txt",
    sep=" ::: ",
    engine="python",
    names=["id", "title", "plot"]
    )
    solution_df = pd.read_csv(
    "data/test_data_solution.txt",
    sep=" ::: ",
    engine="python",
    names=["id", "title", "genre", "plot"]
    )

    print("Data Loaded")
    print("\nTRAIN DATA SAMPLE:")
    print(train_df.head())

    print("\nGENRE UNIQUE VALUES:")
    print(train_df['genre'].unique()[:20])

    print("\nNUMBER OF UNIQUE GENRES:", train_df['genre'].nunique())

    train_df['plot'] = train_df['plot'].str.strip()
    train_df['genre'] = train_df['genre'].str.strip()
    test_df['plot'] = test_df['plot'].str.strip()
    solution_df['genre'] = solution_df['genre'].str.strip()

    print("Text Cleaned")
    
    
    train_df['clean_plot'] = train_df['plot'].apply(clean_text)
    test_df['clean_plot'] = test_df['plot'].apply(clean_text)

    print("Text Preprocessed")

    X_train = train_df['clean_plot']
    y_train = train_df['genre']

    model, tfidf = train_model(X_train, y_train)
    print("Model Trained")

    X_test = test_df['clean_plot']
    y_test = solution_df['genre']

    print("Test size:", len(X_test))
    print("Solution size:", len(y_test))
    assert len(X_test) == len(y_test)

    y_pred = evaluate_model(model, tfidf, X_test, y_test)

    print("Done!")


if __name__ == "__main__":
    main()