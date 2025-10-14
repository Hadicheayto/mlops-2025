from pathlib import Path
import pandas as pd
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    logging.info(f"Loaded train: {train.shape}, test: {test.shape}")
    return train, test

def unify_data(train, test):
    combined = pd.concat([train, test], sort=False).reset_index(drop=True)
    return combined

def extract_title(df):
    df['Title'] = df['Name'].str.split(", ").str[1].str.split(".").str[0]
    rare_titles = ['Lady', 'the Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
    return df

def family_size_feature(df):
    df['Family_size'] = df['SibSp'] + df['Parch'] + 1
    def label_size(n):
        if n==1: return 'Alone'
        elif n<5: return 'Small'
        else: return 'Large'
    df['Family_size'] = df['Family_size'].apply(label_size)
    return df

def drop_unused(df):
    to_drop = [c for c in ['Name','Ticket','SibSp','Parch'] if c in df.columns]
    df.drop(columns=to_drop, inplace=True)
    return df

def final_cleanup(df):
    if 'Age' in df.columns:
        df['Age'] = df['Age'].astype(int)
    return df

def write_output(df, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(out_dir)/'features.csv', index=False)
    logging.info(f"Wrote features.csv to {out_dir}")

def main(train_path, test_path, out_dir):
    train, test = load_data(train_path, test_path)
    combined = unify_data(train, test)
    combined = extract_title(combined)
    combined = family_size_feature(combined)
    combined = drop_unused(combined)
    combined = final_cleanup(combined)
    write_output(combined, out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Titanic Feature Engineering")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out-dir", default="./features")
    args = parser.parse_args()
    main(args.train, args.test, args.out_dir)
