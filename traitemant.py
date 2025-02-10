import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


# Load dataset
def load_data():
    df = pd.read_excel("/Users/mac/Desktop/DATA.xlsx")
    df["Date.CMD"] = pd.to_datetime(df["Date.CMD"])
    df["Mois"] = df["Date.CMD"].dt.month
    df["Année"] = df["Date.CMD"].dt.year
    df["Category"] = df["Produit"].str.split().str[0]
    df["Wilaya"] = df["Adresse"].str.split(",").str[-1].str.strip()
    return df


df = load_data()

# Streamlit UI
st.title("📊 Analyse des Ventes")

# Sidebar menu
option = st.sidebar.selectbox(
    "Choisissez une analyse :",
    [
        "Liste des produits vendus après une date",
        "Produit ayant rapporté le plus d’argent",
        "Client ayant effectué le plus d’achats",
        "Ventes quantitatives par mois et année",
        "Meilleur client par mois et catégorie",
        "Catégorie de produit la plus rentable"
    ]
)

if option == "Liste des produits vendus après une date":
    date_str = st.date_input("Sélectionnez une date", datetime.today())
    date_filter = pd.to_datetime(date_str)
    filtered_df = df[df["Date.CMD"] > date_filter]
    st.dataframe(filtered_df[['Produit', 'Quantité']])

    # Plot
    fig, ax = plt.subplots()
    if not filtered_df.empty:
        fig, ax = plt.subplots()
        filtered_df.groupby("Produit")["Quantité"].sum().sort_values().plot(kind='barh', color='skyblue', ax=ax)
        ax.set_title(f"Produits vendus après {date_str}")
        st.pyplot(fig)
    else:
        st.warning("Aucun produit vendu après cette date.")


elif option == "Produit ayant rapporté le plus d’argent":
    top_product = df.groupby("Produit")["Montant TTC"].sum().idxmax()
    top_revenue = df.groupby("Produit")["Montant TTC"].sum().max()
    st.subheader(f"🥇 Produit le plus rentable : {top_product} (${top_revenue:.2f})")

    fig, ax = plt.subplots()
    df.groupby("Produit")["Montant TTC"].sum().nlargest(10).plot(kind='bar', color='orange', ax=ax)
    ax.set_title("Top 10 Produits les plus rentables")
    st.pyplot(fig)

elif option == "Client ayant effectué le plus d’achats":
    top_client = df.groupby("Client")["Montant TTC"].sum().idxmax()
    st.subheader(f"🥇 Client ayant dépensé le plus : {top_client}")

    fig, ax = plt.subplots()
    df.groupby("Wilaya")["Montant TTC"].sum().nlargest(10).plot(kind='barh', color='green', ax=ax)
    ax.set_title("Top 10 Wilayas avec le plus de dépenses")
    st.pyplot(fig)

elif option == "Ventes quantitatives par mois et année":
    sales_data = df.groupby(['Année', 'Mois', 'Category'])['Quantité'].sum().reset_index()
    st.dataframe(sales_data)

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_data = sales_data.pivot(index=['Année', 'Mois'], columns='Category', values='Quantité')
    pivot_data.plot(kind='line', marker='o', colormap='viridis', ax=ax)
    ax.set_title('Quantité vendue par catégorie et par mois')
    st.pyplot(fig)

elif option == "Meilleur client par mois et catégorie":
    mois = st.selectbox("Sélectionnez le mois", list(range(1, 13)))
    category = st.selectbox("Sélectionnez la catégorie", df["Category"].unique())

    top_client = df[(df["Mois"] == mois) & (df["Category"] == category)]
    if not top_client.empty:
        best_client = top_client.groupby("Client")["Montant TTC"].sum().idxmax()
        st.subheader(f"🥇 Meilleur client pour {category} en {mois}: {best_client}")
    else:
        st.warning("Aucune donnée disponible pour ces critères.")

elif option == "Catégorie de produit la plus rentable":
    top_category = df.groupby("Category")["Montant TTC"].sum().idxmax()
    st.subheader(f"🥇 Catégorie la plus rentable : {top_category}")

    fig, ax = plt.subplots()
    df.groupby("Category")["Montant TTC"].sum().plot(kind='bar', color='purple', ax=ax)
    ax.set_title("Revenu par catégorie de produit")
    st.pyplot(fig)

st.write("---")
st.info("💡 Développé avec Streamlit pour une analyse interactive des ventes.")
