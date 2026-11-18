import pandas as pd

def load_data(path):
    with open(path, 'r',encoding="utf-8") as file:
        return pd.DataFrame(
            {
                "text":[line.rstrip("\r\n") for line in file],
            }
        )
def collect_data():
    train_neg =load_data("data/raw/train_Neg.en")
    train_pos =load_data("data/raw/train_Pos.en")
    test_neg =load_data("data/raw/ttestNeg.en")
    test_pos =load_data("data/raw/ttestPos.en")
    # Assign sentiment labels
    train_neg["sentiment"] = 0
    train_pos["sentiment"] = 1

    test_neg["sentiment"] = 0
    test_pos["sentiment"] = 1

    # Combine datasets
    train_df = pd.concat([train_neg, train_pos], ignore_index=True)
    test_df = pd.concat([test_neg, test_pos], ignore_index=True)
    return train_df, test_df

