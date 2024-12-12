import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prince import MCA, CA


# Step 2: Import the data
# Make sure the file is named correctly and path is correct
data = pd.read_csv('/home/zoubir/DataSet.csv')



# Résumé statistique des données
print("\nRésumé des données :")
print(data.describe(include='all'))

# Fréquences pour une variable (exemple : Sexe)
print("\nFréquence des catégories pour la variable Sexe :")
print(data['Sexe'].value_counts())

# Étape 3 : Visualisation des données

# Pie chart
plt.figure(figsize=(6, 6))
plt.pie(
    data['Sexe'].value_counts(),
    labels=data['Sexe'].value_counts().index,
    autopct='%1.1f%%',  # Show percentage
    startangle=90,  # Start angle for pie chart
    colors=['skyblue', 'lightpink', 'lightgreen', 'gold', 'plum', 'orange', 'cyan'],  # Custom colors for genders
)
plt.title("Gender Distribution")
plt.show()
plt.close()
# Répétez pour d'autres variables (exemple : Type de Visiteur)
plt.figure(figsize=(6, 6))
plt.pie(
    data['Age'].value_counts(),
    labels=data['Age'].value_counts().index,
    autopct='%1.1f%%',  # Show percentage
    startangle=90,  # Start angle for pie chart
    colors=['gold', 'plum', 'orange','skyblue', 'lightpink', 'lightgreen', 'gold', 'plum', 'orange', 'cyan'],  # Custom colors for genders
)
plt.title("Age Distribution")
plt.show()
plt.close()

sns.countplot(data=data, x='Type de Visiteur', palette='viridis')
plt.title("Users type")
plt.show()

plt.figure(figsize=(6, 6))
plt.pie(
    data['Access'].value_counts(),
    labels=data['Access'].value_counts().index,
    autopct='%1.1f%%',  # Show percentage
    startangle=90,  # Start angle for pie chart
    colors=['gold', 'plum','skyblue', 'lightpink', 'lightgreen', 'gold', 'plum', 'orange', 'cyan'],  # Custom colors for genders
)
plt.title("Access devices")
plt.show()
plt.close()


plt.figure(figsize=(6, 6))
plt.pie(
    data['Utiliser deja'].value_counts(),
    labels=data['Utiliser deja'].value_counts().index,
    autopct='%1.1f%%',  # Show percentage
    startangle=90,  # Start angle for pie chart
    colors=['skyblue', 'gold',  'lightpink', 'lightgreen', 'gold', 'plum', 'orange', 'cyan'],  # Custom colors for genders
)
plt.title("Used Before")
plt.show()
plt.close()



sns.countplot(data=data, x='Satisfaction des attentes', palette='viridis')
plt.title("Satisfaction review")
plt.show()

sns.countplot(data=data, x='Navigation', palette='viridis')
plt.title("Navigation review")
plt.show()


sns.countplot(data=data, x='Accessibilite aux informations', palette='viridis')
plt.title("Information accessibility review")
plt.show()

sns.countplot(data=data, x='Experiance de reservation', palette='viridis')
plt.title("Reservation review")
plt.show()

sns.countplot(data=data, x='Design', palette='viridis')
plt.title("Design review")
plt.show()


sns.countplot(data=data, x='Lisibilite du texte', palette='viridis')
plt.title("Lisibility review")
plt.show()


#==================================================================================================================================


# Étape 4 : Transformation en tableau disjonctif
# Transformer les données qualitatives en variables factorielles
data_factored = pd.get_dummies(data, drop_first=True)

# Sauvegarder les données transformées en CSV
data_factored.to_csv('/home/zoubir/data_factored.csv', index=False)

mca = MCA(n_components=2)
mca_fit = mca.fit(data_factored)

# Étudier les valeurs propres
print("\nValeurs propres :")
print(mca_fit.eigenvalues_)  # Eigenvalues (also represent the explained variance)
# Étape 6 : Visualisation des résultats
# Représentation biplot individus-variables
fig, ax = plt.subplots(figsize=(10, 8))

# Utilisation de row_coordinates pour obtenir les coordonnées des individus
row_coords = mca_fit.row_coordinates(data_factored)

# Tracer les coordonnées des individus
ax.scatter(row_coords.iloc[:, 0], row_coords.iloc[:, 1], color='blue', label="Individus")
plt.title("Biplot individus-variables (AFCM)")
plt.xlabel("Composante 1")
plt.ylabel("Composante 2")
plt.show()

# Étape 7 : Contributions aux dimensions
# Obtenir les contributions des variables
contributions = mca_fit.column_contributions_
print("\nContributions des variables :")
print(contributions)

# Étape 8 : Interprétation des axes et associations
# Obtenez les coordonnées des variables
coordinates = mca.column_coordinates(data_factored)

# Visualisation des coordonnées des variables
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(coordinates.iloc[:, 0], coordinates.iloc[:, 1], color='blue')  # Coordonnées des variables
for i, txt in enumerate(coordinates.index):
    ax.annotate(txt, (coordinates.iloc[i, 0], coordinates.iloc[i, 1]))

plt.title("Représentation des variables dans l'espace des axes")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.show()



# Étape 9 : Croisement de deux questions pertinentes
# Exemple avec Sexe et Type de Visiteur
contingency_table = pd.crosstab(data['Sexe'], data['Type de Visiteur'])
print("\nTableau de contingence Sexe x Type de Visiteur :")
print(contingency_table)
from prince import CA
import pandas as pd
import matplotlib.pyplot as plt

# Appliquer l'AC sur la table de contingence
ca = CA(n_components=2)
ca_fit = ca.fit(contingency_table)

# Récupérer les coordonnées des lignes (individus)
coordinates = ca.row_coordinates(contingency_table)

# Affichage des valeurs propres (inertie)
eigenvalues = ca.eigenvalues_
print("Valeurs propres (inertie) :", eigenvalues)

# Calculer l'inertie expliquée pour chaque dimension
explained_inertia = eigenvalues / eigenvalues.sum()
print("Inertie expliquée par chaque dimension :", explained_inertia)

# Vérification des coordonnées
print("Forme des coordonnées :", coordinates.shape)

# Vérifier si la dimension est suffisante pour un graphique 2D
if coordinates.shape[1] >= 2:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coordinates.iloc[:, 0], coordinates.iloc[:, 1], color='blue')  # Coordonnées des individus

    # Ajouter des annotations (si nécessaire)
    for i, txt in enumerate(coordinates.index):
        ax.annotate(txt, (coordinates.iloc[i, 0], coordinates.iloc[i, 1]))

    plt.title("Biplot (Analyse des Correspondances)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.show()
else:
    # Affichage en 1D si seulement une dimension est présente
    print("Affichage en 1D, car il n'y a qu'une seule dimension disponible.")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coordinates.iloc[:, 0], [0]*len(coordinates), color='blue')  # Affichage 1D

    # Ajouter des annotations (si nécessaire)
    for i, txt in enumerate(coordinates.index):
        ax.annotate(txt, (coordinates.iloc[i, 0], 0))

    plt.title("Affichage des coordonnées en 1D (Analyse des Correspondances)")
    plt.xlabel("Dimension 1")
    plt.yticks([])  # Retirer l'axe Y car il n'a pas de sens en 1D
    plt.grid(True)
    plt.show()