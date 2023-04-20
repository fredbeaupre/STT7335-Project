# Visualisation des données pour identifier les données extrèmes et/ou aberrantes

from matplotlib import pyplot as plt
import pandas as pd


# Mapping functions
def map_pdays(value):
    new_value = -1 if value == 999 else value
    return new_value


def map_unknown(value):
    new_value = pd.NA if value == "unknown" else value
    return new_value


def map_shorten_education(value):
    new_value = "professional" if value == "professional.course" else value
    return new_value


# Charger les données
df = pd.read_csv("bank_additional_full.csv")


# Identifier les doublons
df["duplicated"] = df.duplicated(keep=False)
print(f"Nb. doublons : {len(df[df['duplicated'] == True])}")
# # Visualiser les lignes qui sont des doublons
# print(df[df["duplicated"] == True].to_string())

# Supprimer les doublons
df.drop_duplicates(inplace=True)
df.drop(columns="duplicated", inplace=True)


# Supprimer les variables non-pertinentes
df.drop(columns=["default", "contact", "month", "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome"],
        inplace=True)

# dfna1 = df[df.isna().any(axis=1)]
# print(dfna1.head().to_string())
# Changer les "unknown" par des objets pd.NA
for column_name in df.columns:
    df[column_name] = df[column_name].map(map_unknown)
# dfna2 = df[df.isna().any(axis=1)]
# print(dfna2.head().to_string())
print("Nb rows with NA : ", len(df[df.isna().any(axis=1)].index))


# # Changer pdays 999 à -1 pour visualisation
# df["pdays"] = df["pdays"].map(map_pdays)


# # Afficher description données
# print("Type de Variables :\n", df.dtypes, "\n")
# print("Variables numériques :\n", df.describe().to_string(), "\n")
# print("Variables catégorielles :\n", df.describe(include="object").to_string(), "\n")


# Create graph
nrows = 2
ncols = 6
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 9), sharex=False, sharey=False)

i = 0
for name_col, dtype_col in df.dtypes.iteritems():
    row = int((i/ncols) % nrows)
    col = i % ncols
    axs[row, col].set_title(name_col)
    if dtype_col == "object":
        if name_col == "education":
            df["education"].dropna().map(map_shorten_education).value_counts().plot(kind="bar", ax=axs[row, col])
        else:
            df[name_col].value_counts().plot(kind="bar", ax=axs[row, col])
    else:
        df[name_col].plot(kind="box", ax=axs[row, col])
        axs[row, col].set_xticks([])
    i += 1

plt.tight_layout()
plt.show()
# plt.savefig("variables_visual.jpeg", format="jpeg", dpi=500)


# Changer l'éducation en variable numérique après avoir fait le graphique
map_education = {'illiterate': 1, 'basic.4y': 2, 'basic.6y': 3, 'basic.9y': 4, 'high.school': 5,
                 'professional.course': 6, 'university.degree': 7, pd.NA: pd.NA}
df["education"] = df["education"].map(map_education)
# dfna3 = df[df.isna().any(axis=1)]
# print(dfna3.head().to_string())


# Save formated data
df.to_csv("bank_additional_clean.csv")