import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

# =========================
# CONFIGURACIÓN
# =========================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
FEATURE_COLUMNS = ["QUANTITYORDERED", "PRICEEACH", "MSRP", "YEAR_ID"]

st.title("Sistema Inteligente de Predicción de Ventas")
st.write("Ingrese los datos del pedido para predecir las ventas de chocolate.")

if not MODEL_PATH.exists():
    st.error("No se encontró el modelo. Ejecute train_model.py primero para crear model.pkl.")
else:
    model = joblib.load(MODEL_PATH)

    cantidad = st.number_input("Cantidad ordenada", min_value=1, step=1, value=10)
    precio_unitario = st.number_input("Precio por unidad", min_value=0.0, step=0.01, value=50.0)
    msrp = st.number_input("MSRP", min_value=0.0, step=0.01, value=95.0)
    año = st.number_input("Año", min_value=2000, max_value=2100, step=1, value=2003)

    if st.button("Predecir"):
        datos = pd.DataFrame(
            [
                [cantidad, precio_unitario, msrp, año]
            ],
            columns=FEATURE_COLUMNS
        )

        prediccion = model.predict(datos)
        st.success(f"Ventas estimadas: {prediccion[0]:,.2f}")
        st.write("Modelo cargado desde:", MODEL_PATH)
        st.write(datos)
