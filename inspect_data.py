import pandas as pd
import load_data


def initial_look():
    train_df, test_df = load_data.load_data()


    # Display results
    print("Training dataset:", train_df.shape)
    print("Test dataset:", test_df.shape)

    print("\nTraining sentiment distribution:")
    print(train_df["sentiment"].value_counts())

    print("\nTest sentiment distribution:")
    print(test_df["sentiment"].value_counts())

    print("\nFirst five training tweets:")
    print(train_df.head())


def validation_checks():
    import pandas as pd
    from load_data import collect_data

    train_df, test_df = collect_data()

    # Check missing values
    print("Training missing values:")
    print(train_df.isnull().sum())

    print("\nTest missing values:")
    print(test_df.isnull().sum())

    # Check empty tweets
    print("\nEmpty training tweets:",
          train_df["text"].str.strip().eq("").sum())

    print("Empty test tweets:",
          test_df["text"].str.strip().eq("").sum())

    # Check duplicates
    print("\nDuplicate training tweets:",
          train_df.duplicated(subset=["text"]).sum())

    print("Duplicate test tweets:",
          test_df.duplicated(subset=["text"]).sum())

    # Check train/test overlap
    overlap = set(train_df["text"]) & set(test_df["text"])

    print("\nTrain/test overlap:", len(overlap))

    # Check conflicting labels
    conflicts = train_df.groupby("text")["sentiment"].nunique()

    print("Conflicting labels:", (conflicts > 1).sum())

