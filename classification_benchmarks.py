import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data import TabularBankDataset
import torch
import models
from data2vec import Data2Vec
from sklearn.preprocessing import LabelEncoder
import utils
from sklearn import svm


CATEGORICAL_VARS = ["job", "marital", "education", "default",
                    "housing", "loan", "contact", "month", "day_of_week", "poutcome"]


metrics = {
    "balanced-net": {
        "accuracy": None,
        "precision": None,
        "Recall": None,
        "tpr": None,
        "fpr": None,
        "response": None,
        "lift": None,
    },
    "deepnet": {
        "accuracy": None,
        "precision": None,
        "Recall": None,
        "tpr": None,
        "fpr": None,
        "response": None,
        "lift": None,
    },
    "balanced-SVM": {
        "accuracy": None,
        "precision": None,
        "Recall": None,
        "tpr": None,
        "fpr": None,
        "response": None,
        "lift": None,
    },
    "SVM": {
        "accuracy": None,
        "precision": None,
        "Recall": None,
        "tpr": None,
        "fpr": None,
        "response": None,
        "lift": None,
    },
    "logreg": {
        "accuracy": None,
        "precision": None,
        "Recall": None,
        "tpr": None,
        "fpr": None,
        "response": None,
        "lift": None,
    },
    "balanced-logreg": {
        "accuracy": None,
        "precision": None,
        "Recall": None,
        "tpr": None,
        "fpr": None,
        "response": None,
        "lift": None,
    },
    "balanced-randomforest": {
        "accuracy": None,
        "precision": None,
        "Recall": None,
        "tpr": None,
        "fpr": None,
        "response": None,
        "lift": None,
    },
    "randomforest": {
        "accuracy": None,
        "precision": None,
        "Recall": None,
        "tpr": None,
        "fpr": None,
        "response": None,
        "lift": None,
    },
}


def label_encode_data(df):
    label_encoders = {}
    for cat_col in CATEGORICAL_VARS:
        label_encoders[cat_col] = LabelEncoder()
        df[cat_col] = label_encoders[cat_col].fit_transform(df[cat_col])
    return df


def convert_output(df):
    df["y"] = df["y"].map(utils.yes_no_to_number)
    return df


def create_dataloader(path="./saved_models/data2vec_classification/test_data.csv"):
    df = pd.read_csv(path)
    dataset = TabularBankDataset(data=df)
    batch_size = 256
    emb_dims = dataset.get_emb_dims()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False)
    return dataloader, emb_dims


def create_ML_test_data():
    balanced_path = "./saved_models/data2vec_balanced_classification/test_data.csv"
    path = "./saved_models/data2vec_classification/test_data.csv"
    df = pd.read_csv(path)
    df_y = df["y"]
    df_y = convert_output(df_y)
    df_x = df.drop(["y"], axis=1)
    df_x = label_encode_data(df_x)

    balanced_df = pd.read_csv(path)
    balanced_df_y = balanced_df["y"]
    balanced_df_y = convert_output(balanced_df)
    balanced_df_x = balanced_df.drop(["y"], axis=1)
    balanced_df_x = label_encode_data(balanced_df_x)
    return (balanced_df_x, balanced_df_y), (df_x, df_y)


def create_ML_train_data():
    balanced_path = "./saved_models/data2vec_balanced_classification/train_data.csv"
    path = "./saved_models/data2vec_classification/train_data.csv"
    df = pd.read_csv(path)
    df_y = df["y"]
    df_y = convert_output(df_y)
    df_x = df.drop(["y"], axis=1)
    df_x = label_encode_data(df_x)

    balanced_df = pd.read_csv(path)
    balanced_df_y = balanced_df["y"]
    balanced_df_y = convert_output(balanced_df)
    balanced_df_x = balanced_df.drop(["y"], axis=1)
    balanced_df_x = label_encode_data(balanced_df_x)
    return (balanced_df_x, balanced_df_y), (df_x, df_y)


def get_deepNet(emb_dims, device, path="./saved_models/data2vec_classification/best_model.pth"):
    encoder = models.FeedForwardNet(
        emb_dims=emb_dims,
        num_continuous=8,
        lin_layer_sizes=[50, 100],
        output_size=16,
        emb_dropout=0.04,
        lin_layer_dropouts=[0.001, 0.01]
    ).to(device)
    model = Data2Vec(encoder=encoder, device=device).to(device)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def compute_SVM_metrics(xtrain, ytrain, xtest, ytest, balanced=True):
    svm_classifier = svm.SVC()
    svm_classifier.fit(xtrain, ytrain)
    preds = svm_classifier.predict(xtest)


def compute_metrics():
    pass


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Dataloaders and models for the DL data2vec method, testing on both a balanced and imbalanced dataset
    dataloader, emb_dims = create_dataloader()
    balanced_dataloader, emb_dims = create_dataloader(
        path="./saved_models/data2vec_balanced_classification/test_data.csv")
    X, Y = create_ML_data()
    encoder = models.FeedForwardNet(
        emb_dims=emb_dims,
        num_continuous=8,
        lin_layer_sizes=[50, 100],
        output_size=16,
        emb_dropout=0.04,
        lin_layer_dropouts=[0.001, 0.01]
    ).to(device)
    model = get_deepNet(emb_dims, device)
    balanced_model = get_deepNet(
        emb_dims, device, path="./saved_models/data2vec_balanced_classification/best_model.pth")

    # Data and models for the ML benchmarks, testing on both a balanced and imbalanced dataset
    (balanced_xtrain, balanced_ytrain), (xtrain, ytrain) = create_ML_train_data()
    (balanced_xtest, balanced_ytest), (xtest, ytest) = create_ML_train_data()


if __name__ == "__main__":
    main()
