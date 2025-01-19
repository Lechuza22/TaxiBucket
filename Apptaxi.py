import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Configuración de la página con el logo como ícono
st.set_page_config(
    page_title="TaxiCom2.0", 
    page_icon="Logo.png",  
    layout="wide"
)
# Colores de la paleta
PRIMARY_COLOR = "#008080"  # Verde azulado del logo
SECONDARY_COLOR = "#444444"  # Gris oscuro
BACKGROUND_COLOR = "#F4F4F4"  # Fondo claro

# Estilo general
st.markdown(
    f"""
    <style>
        .css-18e3th9 {{
            background-color: {BACKGROUND_COLOR};
        }}
        .stButton > button {{
            background-color: {PRIMARY_COLOR};
            color: white;
            border-radius: 5px;
        }}
        h1 {{
            color: {PRIMARY_COLOR};
        }}
        .stSidebar {{
            background-color: {SECONDARY_COLOR};
            color: white;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Título y logo
st.sidebar.image("Logo.png", use_container_width=True)
st.sidebar.title("TaxiCom2.0")

# Opciones del menú
menu_option = st.sidebar.radio(
    "Seleccione una sección:",
    ("Comparación Marcas y Modelos", "Recomendaciones", "Predicción amortización")
)

# Cargar datos
@st.cache_data
def load_data():
    file_path = 'ElectricCarData.csv'
    return pd.read_csv(file_path)

data = load_data()

def load_taxi_data():
    taxi_trip_path = 'green_tripdata_2024-10_reducido.csv'
    return pd.read_csv(taxi_trip_path)

taxi_trip_data = load_taxi_data()

if menu_option == "Comparación Marcas y Modelos":
    st.header("Comparación Marcas y Modelos")
    st.subheader ("Marcas(Brands) y modelos")
    st.text ("Para comparar primero selecciona las marcas y modelos, luego selecciona las variables que quieras incluir en la comparación")
    # Selección de marcas para comparación
    brands = data["brand"].unique()
    col1, col2 = st.columns(2)

    with col1:
        brand1 = st.selectbox("Seleccione la primera marca", brands, key="brand1")
    with col2:
        brand2 = st.selectbox("Seleccione la segunda marca", brands, key="brand2")

    # Filtrar modelos por marca
    models_brand1 = data[data["brand"] == brand1]
    models_brand2 = data[data["brand"] == brand2]

    # Selección de modelo dentro de cada marca
    col1, col2 = st.columns(2)
    
    with col1:
        model1 = st.selectbox("Seleccione el modelo de la primera marca", models_brand1["model"].unique(), key="model1")
    with col2:
        model2 = st.selectbox("Seleccione el modelo de la segunda marca", models_brand2["model"].unique(), key="model2")

    # Mostrar variables seleccionables
    variables = ["accel", "topspeed", "range", "efficiency", "priceusd"]
    selected_variables = st.multiselect("Seleccione las variables a comparar", variables, default=variables)

    # Filtrar datos de los modelos seleccionados
    data_model1 = data[(data["brand"] == brand1) & (data["model"] == model1)][selected_variables]
    data_model2 = data[(data["brand"] == brand2) & (data["model"] == model2)][selected_variables]

    # Mostrar tablas comparativas
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Datos de {model1}")
        st.write(data_model1)

    with col2:
        st.subheader(f"Datos de {model2}")
        st.write(data_model2)

    # Graficar comparación
    if selected_variables:
        st.subheader("Gráficos comparativos por variable")
        for variable in selected_variables:
            fig, ax = plt.subplots()
    
            # Obtener valores para la variable seleccionada
            value1 = data_model1[variable].values[0]
            value2 = data_model2[variable].values[0]
    
            # Crear gráfico de barras para la variable actual
            ax.bar([model1, model2], [value1, value2], color=["#107D74", "#02163F"])
            ax.set_title(f"Comparación de {variable.capitalize()}")
            ax.set_ylabel(variable.capitalize())
            ax.set_xlabel("Modelos")
    
            # Mostrar el gráfico
            st.pyplot(fig)

elif menu_option == "Recomendaciones":
    st.header("Recomendaciones")
    st.text("Para encontrar recomendaciones selecciona la marca, modelo y las variables de interés.")

    # Selección de marca y modelo
    selected_brand = st.selectbox("Seleccione una marca", data["brand"].unique(), key="reco_brand")
    models = data[data["brand"] == selected_brand]["model"].unique()
    selected_model = st.selectbox("Seleccione un modelo", models, key="reco_model")

    # Selección de variables
    variables = ["accel", "topspeed", "range", "efficiency", "priceusd"]
    selected_variables = st.multiselect("Seleccione las variables para la recomendación", variables, default=variables)

    if st.button("Recomendación"):
        if selected_variables:
            model_data = data[(data["brand"] == selected_brand) & (data["model"] == selected_model)]
            feature_data = data[selected_variables]

            # Crear y entrenar el modelo DBSCAN
            dbscan = DBSCAN(eps=50, min_samples=2, metric="euclidean")
            clusters = dbscan.fit_predict(feature_data)
            data["cluster"] = clusters

            # Encontrar el clúster del modelo seleccionado
            selected_cluster = data.loc[data["model"] == selected_model, "cluster"].values[0]

            # Filtrar modelos del mismo clúster
            recommended_models = data[data["cluster"] == selected_cluster]

            # Mostrar resultados
            st.subheader("Modelos recomendados")
            st.write(recommended_models[["brand", "model"] + selected_variables])
        else:
            st.warning("Por favor, seleccione al menos una variable para realizar la recomendación.")

elif menu_option == "Predicción amortización":
    st.header("Predicción de Amortización")
    st.write("Seleccione un vehículo eléctrico para predecir el tiempo estimado de amortización basado en el precio y las ganancias diarias promedio.")

    # Selección de marca y modelo
    selected_brand = st.selectbox("Seleccione una marca", data["brand"].unique(), key="amort_brand")
    filtered_models = data[data["brand"] == selected_brand]["model"].unique()
    selected_model = st.selectbox("Seleccione un modelo", filtered_models, key="amort_model")

    # Filtrar datos del modelo seleccionado
    selected_car_data = data[(data["brand"] == selected_brand) & (data["model"] == selected_model)].iloc[0]

    # Mostrar información del modelo seleccionado
    st.write(f"**Precio del vehículo (USD):** {selected_car_data['priceusd']:.2f}")

    # Preparar datos para el modelo de predicción
    taxi_trip_data_filtered = taxi_trip_data[["total_amount"]].dropna()
    X = taxi_trip_data_filtered.index.values.reshape(-1, 1)
    y = taxi_trip_data_filtered["total_amount"].values

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar modelo de RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predicción del ingreso diario promedio
    avg_total_amount_per_trip = model.predict([[len(taxi_trip_data) // 2]])[0]
    daily_trips_per_car = 15
    daily_revenue = daily_trips_per_car * avg_total_amount_per_trip

    # Ajustar la ganancia neta considerando el 65%
    net_daily_revenue = daily_revenue * 0.18
    st.write(f"**Ganancia neta promedio diaria estimada (USD):** {net_daily_revenue:.2f}")

    # Predicción del tiempo de amortización
    if st.button("Predecir Amortización"):
        car_price = selected_car_data["priceusd"]
        months_to_amortize = car_price / (net_daily_revenue * 30)

        # Convertir a años y meses
        years = int(months_to_amortize // 12)
        months = int(months_to_amortize % 12)

        if years > 0:
            st.success(f"El vehículo se amortizará en aproximadamente **{years} años y {months} meses**.")
        else:
            st.success(f"El vehículo se amortizará en aproximadamente **{months} meses**.")
