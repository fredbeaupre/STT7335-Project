import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch


CONTINUOUS_VARS = ["age", "campaign", "previous", "emp.var.rate",
                   "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
CATEGORICAL_VARS = ["job", "marital", "education", "default",
                    "housing", "loan", "contact", "month", "day_of_week", "poutcome"]


def yes_no_to_number(value):
    if value == "yes":
        return 1
    elif value == "no":
        return 0
    else:
        return np.nan


def embed_bank_data(df):
    label_encoders = {}
    for cat_col in CATEGORICAL_VARS:
        label_encoders[cat_col] = LabelEncoder()
        df[cat_col] = label_encoders[cat_col].fit_transform(df[cat_col])
    return df


def remove_columns(df, columns=["pdays", "duration"]):
    df = df.drop(columns=columns)
    return df


def prepare_dataframe(df):
    df = remove_columns(df)
    label_encoders = {}
    for cat_col in CATEGORICAL_VARS:
        label_encoders[cat_col] = LabelEncoder()
        df[cat_col] = label_encoders[cat_col].fit_transform(df[cat_col])
    cat_dims = [int(df[col].nunique()) for col in CATEGORICAL_VARS]
    emb_dims = [(x, (x + 1) // 2) for x in cat_dims]
    return df, emb_dims


def convert_output(df, output_col):
    df[output_col] = df[output_col].map(yes_no_to_number)
    return df


class TabularBankDataset(torch.utils.data.Dataset):
    def __init__(self, data, cat_cols=CATEGORICAL_VARS, cont_cols=CONTINUOUS_VARS, output_col="y"):
        data = convert_output(data, output_col)
        dataframe, self.emb_dims = prepare_dataframe(data)

        self.n = dataframe.shape[0]
        if output_col:
            self.y = dataframe[output_col].astype(
                np.float32).values.reshape(-1, 1)
        else:
            self.y = np.zeros((self.n, 1))

        self.cat_cols = cat_cols if cat_cols else []
        self.cont_cols = cont_cols if cont_cols else []

        if self.cont_cols:
            self.cont_x = dataframe[self.cont_cols].astype(np.float32).values
        else:
            self.cont_x = np.zeros((self.n, 1))

        if self.cat_cols:
            self.cat_x = dataframe[cat_cols].astype(np.int64).values
        else:
            self.cat_x = np.zeros((self.n, 1))

    def __len__(self):
        return self.n

    def get_emb_dims(self):
        return self.emb_dims

    def __getitem__(self, idx):
        return [self.y[idx], self.cont_x[idx], self.cat_x[idx]]
