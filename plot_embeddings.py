import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def jitter(df, colname):
    std = 0.01
    df["jitter"] = np.random.normal(loc=0.0, scale=std, size=df.shape[0])
    df[colname] = df[colname] + df["jitter"]
    df.drop(columns=["jitter"], inplace=True)


df_train_emb = pd.read_csv("embedding_train.csv")
jitter(df_train_emb, "emb1")
jitter(df_train_emb, "emb2")


df_train_emb.plot.scatter(x="emb1", y="emb2", color="#3FB59E")
plt.xlabel("Dim. 1 embedding")
plt.ylabel("Dim. 2 embedding")
plt.tight_layout()

plt.show()
# plt.savefig("embedding_train.jpeg", format="jpeg", dpi=500)
