import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch


def create_dataloader(path="./test_data.csv"):
    df = pd.read_csv(path)
    return df


def compute_metrics():
    pass


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = create_dataloader()
    y = loader["y"].to_numpy()
    unique, counts = np.unique(y, return_counts=True)
    print(unique)
    print(counts)


if __name__ == "__main__":
    main()
