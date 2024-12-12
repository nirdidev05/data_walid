# Charger les bibliothèques nécessaires
library(FactoMineR)
library(factoextra)

# Définir le chemin du fichier CSV
file_path <- "C:\\Users\\ZOUBIR\\Desktop\\Table_Finale1.csv"

# Charger les données
data <- read.csv(file_path)

# Définir les identifiants comme noms de lignes
rownames(data) <- data$Matricule
data <- data[, -which(names(data) == "Matricule")]

# Vérifier et convertir la colonne 'Affectation'
data$Affectation <- as.factor(data$Affectation)
affectation <- data$Affectation
data <- data[, -which(names(data) == "Affectation")]

# Vérification de la matrice de corrélation pour l'ACP
cor_matrix <- cor(data)
print(cor_matrix)

# Effectuer l'ACP
res.pca <- PCA(data, scale.unit = TRUE, graph = FALSE)

# Calculer les COS²
cos2_variables <- res.pca$var$cos2 # COS² des variables

# Contributions absolues (déjà disponibles)
contrib_variables <- res.pca$var$contrib

# Vérifier les valeurs totales pour chaque dimension
total_contrib <- colSums(contrib_variables)

# Exporter les résultats dans deux fichiers séparés pour éviter les confusions
output_cos2 <- "C:\\Users\\ZOUBIR\\Desktop\\COS2_Variables.csv"
output_contrib <- "C:\\Users\\ZOUBIR\\Desktop\\Contributions_Variables.csv"
write.csv(cos2_variables, output_cos2, row.names = TRUE)
write.csv(contrib_variables, output_contrib, row.names = TRUE)

# Messages de confirmation
print(paste("Le tableau des COS² a été exporté vers :", output_cos2))
print(paste("Le tableau des contributions absolues a été exporté vers :", output_contrib))

# Graphique des valeurs propres pour évaluer l'importance des composantes
dev.new() # Ouvrir une fenêtre pour le graphique des valeurs propres
fviz_eig(res.pca,
         addlabels = TRUE,
         title = "Scree Plot : Valeurs propres")

# Graphique des individus
dev.new() # Ouvrir une fenêtre pour le graphique des individus
fviz_pca_ind(res.pca,
             col.ind = affectation,
             palette = c("#FF5733", "#228B22", "#3357FF", "#8E44AD"),
             addEllipses = TRUE,
             legend.title = "Affectation",
             repel = TRUE,
             title = "Représentation des individus")

# Graphique du cercle des corrélations pour les variables
dev.new() # Ouvrir une fenêtre pour le cercle des corrélations
fviz_pca_var(res.pca,
             col.var = "black",
             repel = TRUE,
             title = "Cercle des corrélations")

# Analyse des contributions des variables aux composantes principales
dev.new() # Ouvrir une fenêtre pour les contributions des variables
fviz_contrib(res.pca, choice = "var", axes = 1, top = 10, title = "Contributions des variables - Composante 1")
fviz_contrib(res.pca, choice = "var", axes = 2, top = 10, title = "Contributions des variables - Composante 2")

# Interprétation des résultats : Résumé des composantes
summary(res.pca)
