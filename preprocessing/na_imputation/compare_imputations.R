#### Comparer les différents jeux de données imputés par les différentes méthodes 
#### (nouvelle classe, hotdeck, multipe, modèle auto-supervisé)

### Préparation de l'environnement
## Librairies
librairies <- c("dplyr", "naniar", "mice", "ggplot2", "imbalance", "varImp", "ranger") 
library(dplyr)
library(imbalance) # https://rdrr.io/cran/imbalance/
library(ggplot2)
library(ranger)

## Importer les données
#data_og <- read.csv("bank_additional_clean.csv")
data_hd <- read.csv("bank_additionnal_hotdeckImputation.csv", row.names = 1)
data_mi <- read.csv("bank_additionnal_multipleImputation.csv", row.names = 1)
data_na <- read.csv("bank_additionnal_NA2factor.csv", row.names = 1)

## Type de variables appropriés
data_hd <- data_hd %>% 
  select(-X) %>% 
  mutate(
    job = factor(job),
    marital = factor(marital),
    housing = factor(housing),
    loan = factor(loan),
    education = factor(education, ordered = T),
    y = factor(y)
  )
data_mi <- data_mi %>% 
  select(-X) %>% 
  mutate(
    job = factor(job),
    marital = factor(marital),
    housing = factor(housing),
    loan = factor(loan),
    education = factor(education, ordered = T),
    y = factor(y)
  )
data_na <- data_na %>% 
  select(-X) %>% 
  mutate(
    job = factor(ifelse(is.na(job), "NA", as.character(job))),
    marital = factor(ifelse(is.na(marital), "NA", as.character(marital))),
    housing = factor(ifelse(is.na(housing), "NA", as.character(housing))),
    loan = factor(ifelse(is.na(loan), "NA", as.character(loan))),
    education = factor(ifelse(is.na(education), "NA", as.character(education)), ordered = T),
    y = factor(y)
  )

#sapply(data_og, class)
sapply(data_hd, class)
sapply(data_mi, class)
sapply(data_na, class)

## Reproductibilité
set.seed(2023)

### Fonctions utilitaires
## Fonction d'évaluation des métriques
metriques <- function(cm, b = 0.5){
  #-----------------------------------
  ## IN
  # * Matrice de confusion :
  # TN FP
  # FN TP
  #-----------------------------------
  ## OUT
  # * Vecteur des mesures de performance
  #-----------------------------------
  precision <- cm[2,2]/(cm[2,2]+cm[1,2])
  recall <- cm[2,2]/(cm[2,2]+cm[2,1])
  fb <- (1+b^2)*precision*recall/(b^2*precision+recall)
  accuracy <- (cm[1,1]+cm[2,2])/(cm[1,1]+cm[1,2]+cm[2,1]+cm[2,2])
  specificity <- cm[1,1]/(cm[1,1]+cm[1,2])
  npv <- cm[1,1]/(cm[1,1]+cm[2,1])
  
  met <- c(
    "prec" = precision, 
    "reca" = recall, 
    "f0_5" = fb, 
    "accu" = accuracy, 
    "sepc" = specificity, 
    "npv" = npv)
  met <- ifelse(is.nan(met), 0 , met)
  
  return(met)
}



## Split train-validation test
n <- nrow(data_na)
shuffle_data <- sample(1:n, n)

split_1 <- round(n*0.6)
split_2 <- split_1 + round(n*0.2)

train_ind <- shuffle_data[1:split_1]
val_ind <- shuffle_data[(split_1+1):split_2]
test_ind <- shuffle_data[(split_2+1):n]

#train_data <- data_hd[train_ind, ]
#val_data <- data_hd[val_ind, ]
#test_data <- data_hd[test_ind, ]

## Boucle d'entrainement
metriques_cv <- c(0,0,0,0,0,0)
metriques_best <- c(0,0,0,0,0,0)
n_best <- 0
w_best <- 0
compteur <- 1
col.names = c("i", "n", "w", "prec", "rec", "f0_5", "acc", "spec", "npv")
resultats <- read.table(text = "", col.names = col.names)
res_list <- list()
res_df <- as.data.frame(matrix(rep(0, length(col.names)), nrow = 1))
colnames(res_df) <- col.names
for (i in 1:3) {
  if (i == 1){
    train_data <- data_hd[train_ind, ]
    val_data <- data_hd[val_ind, ]
  }
  else if (i == 2){
    train_data <- data_mi[train_ind, ]
    val_data <- data_mi[val_ind, ]
  }
  else if (i == 2){
    train_data <- data_na[train_ind, ]
    val_data <- data_na[val_ind, ]
  }
  else if (i == 4){
    print("Pas le data encore")
  }
  
  print(paste0("-- Méthode ", i, " --"))
  
  for (n in c(50, 500, 700, 1000)){ # nombre d'arbres
    for (w in c(0.01, 0.1, 0.25, 0.35, 0.5)){ # 1-w poids d'un échec
      cat(sprintf("Itération : %s / 20 \n", compteur))
  
      ## Entrainement du modèle
      rf <- ranger(y ~., data=train_data,
                   num.trees=n,
                   splitrule = "gini",
                   importance ="impurity",
                   min.node.size = 1,
                   oob.error = FALSE, 
                   classification = TRUE,
                   sample.fraction = c(0.5, 0.5), # Balanced Random Forest
                   class.weights = c(w,1-w)) # Weighted Random Forest
      
      ## Prédictions
      train_pred <- predict(rf, train_data)$predictions
      val_pred <- predict(rf, val_data)$predictions
      
      
      ## Métriques sur l'ensemble de validation
      # precision, recall, accuracy, specificity, npv
      val_cm <- table(val_data$y, val_pred)
      metriques_cv <- metriques_cv + metriques(val_cm)
      
      ## Fin de la validation croisée pour une combinaison d'hyperparamètres
      
      # MAJ de la df des résultats
      resultats[nrow(resultats) + 1,] = c(i, n, w, metriques_cv)
      
      ## Critère de sélection du meilleur modèle
      if(metriques_best[3] < metriques_cv[3] #& metriques_cv[4] >= 0.8
         )
      {
        metriques_best <- metriques_cv
        n_best <- n
        w_best <- w
      }
        
      # Remise à zéro
        metriques_cv <- c(0,0,0,0,0,0)
      # Affichage
      compteur <- compteur + 1
    }}
  metriques_best  <- c(0,0,0,0,0,0)
  compteur <- 1
  n_best <- 0
  w_best <- 0
  res_df <- bind_rows(res_df, resultats)
  res_list <- c(res_list, resultats)
  resultats <- read.table(text = "", col.names = col.names)
}
# res

res_df %>% 
  filter(i != 0) %>%
  arrange(desc(f0_5), desc(acc), desc(prec))

res_1 <- res_df %>% 
  filter(i == 1) %>%
  arrange(desc(f0_5), desc(acc), desc(prec)) %>% 
  head(1)
res_2 <- res_df %>% 
  filter(i == 2) %>%
  arrange(desc(f0_5), desc(acc), desc(prec)) %>% 
  head(1)
res_3 <- res_df %>% 
  filter(i == 3) %>%
  arrange(desc(f0_5), desc(acc), desc(prec)) %>% 
  head(1)
# hyper prams retenus
cat(sprintf(
"
\n -- Méthode 1 -- \n
\n===== Choix d'hyperparamètres ===== \n
n: %s \n
w: %s \n
\n===== Performances espérées en validation ===== \n
Precision: %.3f \n
Recall: %.3f \n
F0.5: %.3f \n
Accuracy: %.3f \n
Specificity: %.3f \n
NPV: %.3f \n", 
res_1$n, res_1$w, res_1$prec, res_1$rec, res_1$f0_5, res_1$acc, res_1$spec, res_1$npv))

cat(sprintf(
  "
\n -- Méthode 2 -- \n
\n===== Choix d'hyperparamètres ===== \n
n: %s \n
w: %s \n
\n===== Performances espérées en validation ===== \n
Precision: %.3f \n
Recall: %.3f \n
F0.5: %.3f \n
Accuracy: %.3f \n
Specificity: %.3f \n
NPV: %.3f \n", 
  res_2$n, res_2$w, res_2$prec, res_2$rec, res_2$f0_5, res_2$acc, res_2$spec, res_2$npv))

cat(sprintf(
  "
\n -- Méthode 3 -- \n
\n===== Choix d'hyperparamètres ===== \n
n: %s \n
w: %s \n
\n===== Performances espérées en validation ===== \n
Precision: %.3f \n
Recall: %.3f \n
F0.5: %.3f \n
Accuracy: %.3f \n
Specificity: %.3f \n
NPV: %.3f \n", 
res_3$n, res_3$w, res_3$prec, res_3$rec, res_3$f0_5, res_3$acc, res_3$spec, res_3$npv))




### Modèle final 

## Méthode 1 - hotdeck
train_data <- data_hd[train_ind, ]
val_data <- data_hd[val_ind, ]
test_data <- data_hd[test_ind, ]

rf_final_1 <- ranger(y ~., data=train_data,
                   num.trees=res_1$n,
                   splitrule = "gini",
                   importance ="impurity",
                   sample.fraction = c(0.5,0.5),
                   class.weights = c(res_1$w,1-res_1$w))

# généralisation
# Prédictions
test_pred_1 <- predict(rf_final_1, test_data)$predictions

# Évaluation
test_cm_1 <- table(test_data$y, test_pred_1)

(test_metriques_1 <- metriques(test_cm_1))


## Méthode 2 - Multiple imputation
train_data <- data_mi[train_ind, ]
val_data <- data_mi[val_ind, ]
test_data <- data_mi[test_ind, ]

rf_final_2 <- ranger(y ~., data=train_data,
                     num.trees=res_2$n,
                     splitrule = "gini",
                     importance ="impurity",
                     sample.fraction = c(0.5,0.5),
                     class.weights = c(res_2$w,1-res_2$w))

# généralisation
# Prédictions
test_pred_2 <- predict(rf_final_2, test_data)$predictions

# Évaluation
test_cm_2 <- table(test_data$y, test_pred_2)

(test_metriques_2 <- metriques(test_cm_2))


## Méthode 1 - hotdeck
train_data <- data_na[train_ind, ]
val_data <- data_na[val_ind, ]
test_data <- data_na[test_ind, ]

rf_final_3 <- ranger(y ~., data=train_data,
                     num.trees=res_3$n,
                     splitrule = "gini",
                     importance ="impurity",
                     sample.fraction = c(0.5,0.5),
                     class.weights = c(res_3$w,1-res_3$w))

# généralisation
# Prédictions
test_pred_3 <- predict(rf_final_3, test_data)$predictions

# Évaluation
test_cm_3 <- table(test_data$y, test_pred_3)

(test_metriques_3 <- metriques(test_cm_3))


### Final eval
test_metriques_1
test_metriques_2
test_metriques_3


