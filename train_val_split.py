from sklearn.model_selection import train_test_split
import load_data
import pandas as pd

def split_data_val(train_df):
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.20,
        random_state=42,
        stratify=train_df["sentiment"]
    )
    return train_df, val_df

def save_data_split():
    train_df,test_df = load_data.collect_data()
    train_df, val_df = split_data_val(train_df)

    test_df.to_csv("data/split/test.csv")
    train_df.to_csv("data/split/train.csv")
    val_df.to_csv("data/split/val.csv")
save_data_split()