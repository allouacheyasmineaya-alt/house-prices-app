import streamlit as st
import pandas as pd
import joblib

# Configuration de la page
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Chargement des artefacts
@st.cache_resource
def load_assets():
    model = joblib.load("model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    columns = joblib.load("columns.pkl")
    return model, preprocessor, columns

model, preprocessor, columns = load_assets()

# Entête avec style
st.title(" Estimateur de Prix Immobilier")
st.markdown("Estimez la valeur marchande d'une propriété en renseignant ses caractéristiques ci-dessous.")
st.divider()

# Disposition en 2 colonnes principales (Champs à gauche, Résultats à droite)
col_input, col_result = st.columns([2, 1], gap="large")

with col_input:
    # Organisation par onglets pour plus d'informations sans surcharger l'écran
    tab1, tab2, tab3 = st.tabs(["📐 Surfaces & Qualité", "🛏️ Pièces & Équipements", "📍 Emplacement & Âge"])
    
    with tab1:
        st.subheader("Surfaces principales")
        overall_qual = st.select_slider(
            "Qualité Générale (Overall Quality)", 
            options=list(range(1, 11)), 
            value=5,
            help="1: Très mauvaise, 10: Excellente"
        )
        
        c1, c2 = st.columns(2)
        with c1:
            gr_liv_area = st.number_input("Surface habitable (sq ft)", 500, 6000, 1500)
            lot_area = st.number_input("Surface du terrain (Lot Area in sq ft)", 1000, 50000, 9500)
        with c2:
            total_bsmt = st.number_input("Surface du sous-sol (sq ft)", 0, 3000, 800)
            first_flr_sf = st.number_input("Surface 1er étage (1st Flr sq ft)", 300, 4000, 1000)

    with tab2:
        st.subheader("Pièces et Équipements")
        c1, c2 = st.columns(2)
        with c1:
            full_bath = st.slider("Salles de bain complètes", 0, 4, 2)
            totrms_abvgrd = st.slider("Total pièces (hors sous-sol)", 2, 14, 6)
        with c2:
            garage_cars = st.slider("Capacité du garage (véhicules)", 0, 4, 2)
            fireplaces = st.slider("Nombre de cheminées", 0, 3, 1)

    with tab3:
        st.subheader("Détails sur l'immeuble")
        c1, c2 = st.columns(2)
        with c1:
            year_built = st.number_input("Année de construction", 1900, 2026, 2000)
            year_remod = st.number_input("Année de rénovation", 1900, 2026, 2005)
        with c2:
            neighborhood = st.selectbox(
                "Quartier (Neighborhood)",
                ["CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", "Somerst", "NridgHt", "OldTown"]
            )

with col_result:
    st.subheader("📊 Prediction")
    st.info("Cliquez sur le bouton ci-dessous pour lancer le calcul du modèle.")
    
    predict_btn = st.button("🚀 Calculer l'estimation", use_container_width=True, type="primary")
    
    if predict_btn:
        # Construction du DataFrame avec valeurs par défaut
        input_df = pd.DataFrame([{col: None for col in columns}])
        
        # Injection des variables utilisateur
        input_df["OverallQual"] = overall_qual
        input_df["GrLivArea"] = gr_liv_area
        input_df["TotalBsmtSF"] = total_bsmt
        input_df["GarageCars"] = garage_cars
        input_df["YearBuilt"] = year_built
        input_df["LotArea"] = lot_area
        input_df["1stFlrSF"] = first_flr_sf
        input_df["FullBath"] = full_bath
        input_df["TotRmsAbvGrd"] = totrms_abvgrd
        input_df["Fireplaces"] = fireplaces
        input_df["YearRemodAdd"] = year_remod
        input_df["Neighborhood"] = neighborhood
        
        # Features calculées
        input_df["YrSold"] = 2010
        input_df["TotalSF"] = total_bsmt + gr_liv_area
        input_df["HouseAge"] = 2010 - year_built

        # Transformation et Prédiction
        processed = preprocessor.transform(input_df)
        price = model.predict(processed)[0]
        
        # Affichage élégant de la métrique
        st.metric(label="Prix Estimé du Bien", value=f"${price:,.0f}")
        st.success("Prédiction effectuée avec succès !")