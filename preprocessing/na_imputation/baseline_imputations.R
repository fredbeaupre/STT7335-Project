### Préparation de l'environnement
## Loader les packages
library(mice)
library(VIM)
library(dplyr)
## Importer les données
data_clean <- read.csv("bank_additional_clean.csv")

## Changer les valeurs manquantes en NA
data_clean <- data_clean %>% naniar::replace_with_na_all(condition = ~.x == "")

## Bon types de données
data_clean$job <- as.factor(data_clean$job)
data_clean$marital <- as.factor(data_clean$marital)
data_clean$housing <- as.factor(data_clean$housing)
data_clean$loan <- as.factor(data_clean$loan)
data_clean$education <- factor(data_clean$education, ordered = T) # 7 = uni degree

### Analyse de la non-réponse
## Patrons
mice::md.pattern(data_clean, rotate.names = T)
naniar::gg_miss_upset(data_clean)

## Nombres de lignes avec NA
sum(apply(data_clean, 1, function(x) sum(is.na(x)) >= 1))

## Prop NA par variables
sapply(data_clean, function(x) sum(is.na(x))/length(x))

# test de little
test_little <- naniar::mcar_test(data_clean)
test_little
# p-value = 0 donc rejette H_0 : données MCAR
# Prend pour acquis MAR pour imputation

## Quelques graphiques intéressants sur les prop de NA
test <- Hmisc::naclus(data_clean)
Hmisc::naplot(test)


### Traitement de la non réponse - Méthode traditionnelle

## Ajouter une catégorie "inconnu"
# Comme toutes nos variables contenants des valeurs manquantes sont catégorielles, cette option est possible.

## Imputation Hot-Deck (Par la distribution)
data_s <- data_clean %>% select(age, job, marital, education, housing, loan)
data_s <- VIM::hotdeck(data_s, vars2imp)
data_clean_hotdeck <- cbind("X" = data_clean$X, data_s, "y" = data_clean$y)
sum(is.na(data_clean_hotdeck))

## Imputation par régression logistique multinomiale
data_imput <- data_clean %>% select(-c(X, y))
colnames(data_imput)
na_vec <- c("", "polyreg", "polyreg", "polr", "polyreg", "polyreg", "", "", "", "", "")

#mids <- mice(data_imput, method = na_vec, m = 1, maxit = 5)
#data_imp <- complete(mids)
#sum(is.na(data_imp))
#naniar::gg_miss_upset(data_imp)


## Imputation multiple
mids_mul <- mice(data_imput, method = na_vec, m = 5, maxit = 5)
temp_mids <- mids_mul

# Utiliser la plus commune des 5 imputations pour valeur
temp_mids$imp$education <- cbind(mids_mul$imp$education, "6" = apply(mids_mul$imp$education, 1, function(x) x[which.max(table(x))]))
temp_mids$imp$job <- cbind(mids_mul$imp$job, "6" = apply(mids_mul$imp$job, 1, function(x) x[which.max(table(x))]))
temp_mids$imp$loan <- cbind(mids_mul$imp$loan, "6" = apply(mids_mul$imp$loan, 1, function(x) x[which.max(table(x))]))
temp_mids$imp$housing <- cbind(mids_mul$imp$housing, "6" = apply(mids_mul$imp$housing, 1, function(x) x[which.max(table(x))]))
temp_mids$imp$marital <- cbind(mids_mul$imp$marital, "6" = apply(mids_mul$imp$marital, 1, function(x) x[which.max(table(x))]))

mids_mul$m
temp_mids$m <- 6
data_imp_mul <- complete(temp_mids, action = 6)
sum(is.na(data_imp_mul))
#naniar::gg_miss_upset(data_imp_mul)

data_clean_mul <- cbind("X" = data_clean$X, data_imp_mul, "y" = data_clean$y)

## Validation des prop avant et après imput
vars2imp <- c("education", "job", "loan", "housing", "marital")

val_init <- apply(data_clean %>% select(all_of(vars2imp)), 2, function(x) { 
  x_ <- x[!is.na(x)]
  round(100*table(x_)/length(x_) , 3)
} 
)

val_end <- apply(data_imp_mul %>% select(all_of(vars2imp)), 2, function(x) { 
  x_ <- x[!is.na(x)]
  round(100*table(x_)/length(x_) , 3)
} 
)

# education
val_init$education
val_end$education

# job
val_init$job
val_end$job

# loan
val_init$loan
val_end$loan

# housing
val_init$housing
val_end$housing

# marital
val_init$marital
val_end$marital


### Save data
write.csv(data_clean_hotdeck, file = "bank_additionnal_hotdeckImputation.csv")
write.csv(data_clean_mul, file = "bank_additionnal_multipleImputation.csv")
