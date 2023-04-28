import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data import TabularBankDataset
import torch
import models
from data2vec import Data2Vec
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


CATEGORICAL_VARS = ["job", "marital", "education", "default",
                    "housing", "loan", "contact", "month", "day_of_week", "poutcome"]


all_metrics = {
    "balanced-net": {
        "accuracy": None,
        "precision": None,
        "recall": None,
    },
    "deepnet": {
        "accuracy": None,
        "precision": None,
        "recall": None,
    },
    "balanced-SVM": {
        "accuracy": None,
        "precision": None,
        "recall": None,
    },
    "SVM": {
        "accuracy": None,
        "precision": None,
        "recall": None,
    },
    "bayes": {
        "accuracy": None,
        "precision": None,
        "recall": None,
    },
    "balanced-bayes": {
        "accuracy": None,
        "precision": None,
        "recall": None,
    },
    "balanced-randomforest": {
        "accuracy": None,
        "precision": None,
        "recall": None,
    },
    "randomforest": {
        "accuracy": None,
        "precision": None,
        "recall": None,
    },
}


def yes_no_to_number(value):
    if value == "yes":
        return 1
    elif value == "no":
        return 0
    else:
        return np.nan


def label_encode_data(df):
    label_encoders = {}
    for cat_col in CATEGORICAL_VARS:
        label_encoders[cat_col] = LabelEncoder()
        df[cat_col] = label_encoders[cat_col].fit_transform(df[cat_col])
    return df


def convert_output(df):
    df["y"] = df["y"].map(yes_no_to_number)
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
    df = convert_output(df)
    df_y = df["y"]
    df_x = df.drop(["pdays", "duration", "y"], axis=1)
    df_x = label_encode_data(df_x)

    balanced_df = pd.read_csv(balanced_path)
    print(balanced_df[balanced_df["y"] == "yes"].shape)
    print(balanced_df[balanced_df["y"] == "no"].shape)
    balanced_df = convert_output(balanced_df)
    balanced_df_y = balanced_df["y"]
    balanced_df_x = balanced_df.drop(["pdays", "duration", "y"], axis=1)
    balanced_df_x = label_encode_data(balanced_df_x)
    return (balanced_df_x, balanced_df_y), (df_x, df_y)


def create_ML_train_data():
    balanced_path = "./saved_models/data2vec_balanced_classification/train_data.csv"
    path = "./saved_models/data2vec_classification/train_data.csv"
    df = pd.read_csv(path)
    df = convert_output(df)
    df_y = df["y"]
    df_x = df.drop(["pdays", "duration", "y"], axis=1)
    df_x = label_encode_data(df_x)

    balanced_df = pd.read_csv(balanced_path)
    print(balanced_df[balanced_df["y"] == "yes"].shape)
    print(balanced_df[balanced_df["y"] == "no"].shape)

    balanced_df = convert_output(balanced_df)
    balanced_df_y = balanced_df["y"]
    balanced_df_x = balanced_df.drop(["pdays", "duration", "y"], axis=1)
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
    key = "balanced-SVM" if balanced else "SVM"
    ytest = ytest.to_numpy()
    all_metrics[key]["accuracy"] = metrics.compute_accuracy(preds, ytest)
    all_metrics[key]["precision"] = metrics.compute_precision(preds, ytest)
    all_metrics[key]["recall"] = metrics.compute_recall(preds, ytest)


def compute_forest_metrics(xtrain, ytrain, xtest, ytest, balanced=True):
    forest_classifier = RandomForestClassifier(max_depth=10)
    forest_classifier.fit(xtrain, ytrain)
    preds = forest_classifier.predict(xtest)
    key = "balanced-randomforest" if balanced else "randomforest"
    ytest = ytest.to_numpy()
    all_metrics[key]["accuracy"] = metrics.compute_accuracy(preds, ytest)
    all_metrics[key]["precision"] = metrics.compute_precision(preds, ytest)
    all_metrics[key]["recall"] = metrics.compute_recall(preds, ytest)


def compute_bayes_metrics(xtrain, ytrain, xtest, ytest, balanced=True):
    bayes = GaussianNB()
    bayes.fit(xtrain, ytrain)
    preds = bayes.predict(xtest)
    key = "balanced-bayes" if balanced else "bayes"
    ytest = ytest.to_numpy()
    all_metrics[key]["accuracy"] = metrics.compute_accuracy(preds, ytest)
    all_metrics[key]["precision"] = metrics.compute_precision(preds, ytest)
    all_metrics[key]["recall"] = metrics.compute_recall(preds, ytest)


def compute_metrics():
    pass


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data and models for the ML benchmarks, testing on both a balanced and imbalanced dataset
    (balanced_xtrain, balanced_ytrain), (xtrain, ytrain) = create_ML_train_data()
    (balanced_xtest, balanced_ytest), (xtest, ytest) = create_ML_test_data()

    compute_SVM_metrics(xtrain, ytrain, xtest, ytest, balanced=False)
    compute_SVM_metrics(balanced_xtrain, balanced_ytrain,
                        balanced_xtest, balanced_ytest, balanced=True)

    compute_forest_metrics(xtrain, ytrain, xtest, ytest, balanced=False)
    compute_forest_metrics(balanced_xtrain, balanced_ytrain,
                           balanced_xtest, balanced_ytest, balanced=True)

    compute_bayes_metrics(xtrain, ytrain, xtest, ytest, balanced=False)
    compute_bayes_metrics(balanced_xtrain, balanced_ytrain,
                          balanced_xtest, balanced_ytest, balanced=True)
    print(all_metrics)


if __name__ == "__main__":
    main()
