import torch
from torch import tensor
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from random import sample


def yes_no_to_number(value):
    return 1. if value == 'yes' else 0.


def transform_bank_data(df, drop_na=False, scaler=None):
    if drop_na is True:
        df.dropna(inplace=True)
    # Convert binary variables to 0-1
    df["y"] = df["y"].map(yes_no_to_number)
    df["loan"] = df["loan"].map(yes_no_to_number)
    df["housing"] = df["housing"].map(yes_no_to_number)

    targets = df['y']
    df = df.drop(columns=["y"])
    data = pd.get_dummies(df, drop_first=False)

    # Scale data between 0-1
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(data)
    data = pd.DataFrame(scaler.transform(data), columns=data.columns)
    print(type(data))

    return data, targets, scaler


class BankDataset(Dataset):
    def __init__(self, path, mask_input=False, target_input=False, drop_na=False, scaler=None):
        df_bank = pd.read_csv(path, index_col="Unnamed: 0")
        self.bank_data, self.targets, self.scaler = transform_bank_data(df_bank, drop_na=drop_na, scaler=scaler)

        self.mask_input = mask_input
        self.target_input = target_input

    def __getitem__(self, index):
        item = self.bank_data.iloc[index]
        # Target is input data when self.target_input is True, otherwise real target
        if self.target_input is False:
            target = tensor(self.targets.iloc[index], dtype=torch.float).reshape(1)
        else:
            target = tensor(item.to_numpy(), dtype=torch.float)
        # Apply mask if self.mask_data is True
        if self.mask_input is True:
            data = tensor(get_masked_data(item).to_numpy(), dtype=torch.float)
        else:
            data = tensor(item.to_numpy(), dtype=torch.float)
        return data, target

    def __len__(self):
        return len(self.bank_data)


def get_masked_data(data):
    columns = ["age", "job", "marital", "education", "housing", "loan", "emp.var.rate", "cons.price.idx",
               "cons.conf.idx", "euribor3m", "nr.employed"]
    col_mask = sample(columns, 1)[0]
    data = data.copy(deep=True)
    for items in data.iteritems():
        if col_mask in items[0]:
            data[items[0]] = 0.
    return data


def train_valid_test_split(df, prop_train, prop_valid, random_state=None):
    prop_test = 1 - prop_train - prop_valid
    assert 0 <= prop_test <= 1
    df_, df_test = train_test_split(df, test_size=prop_test, shuffle=True, random_state=random_state)
    df_train, df_valid = train_test_split(df_, test_size=prop_valid/(prop_valid + prop_train), shuffle=True,
                                          random_state=random_state)
    return df_train, df_valid, df_test


if __name__ == "__main__":
    test_data = BankDataset("bank_additional_clean.csv", target_input=True, mask_input=False)
    test = test_data[1]
    print(test[0])
    # print(test[1])

    create_train_test = False
    if create_train_test is True:
        df_split = pd.read_csv("bank_additional_clean.csv", index_col="Unnamed: 0")
        train, valid, test = train_valid_test_split(df_split, prop_train=0.7, prop_valid=0.15)
        train.to_csv("bank_additional_clean_train.csv")
        valid.to_csv("bank_additional_clean_valid.csv")
        test.to_csv("bank_additional_clean_test.csv")