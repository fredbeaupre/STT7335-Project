import numpy as np
import pandas as pd


def csv_to_pandas(datapath="./bank-additional-full.csv"):
    df = pd.read_csv(datapath, header=0, sep=';')
    return df


def main():
    df = csv_to_pandas()
    print(df)
    print(df.shape)
    for col in df.columns:
        print(col)


if __name__ == "__main__":
    main()
