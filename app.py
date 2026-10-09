"""
Aplicación Streamlit: cargar una imagen y clasificarla con la CNN entrenada.

Ejecutar desde la raíz del repositorio:

    streamlit run app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent))
from src.predict import cargar_modelo, predecir  # noqa: E402

st.set_page_config(page_title="Clasificador CIFAR-10", page_icon="🖼️")


@st.cache_resource
def obtener_modelo():
    # Se carga una sola vez, no en cada interacción con la página
    return cargar_modelo()


st.title("Clasificador de imágenes con CNN")
st.write(
    "Sube una imagen y la red neuronal te dice a cuál de estas 10 clases se parece más: "
    "avión, automóvil, pájaro, gato, ciervo, perro, rana, caballo, barco o camión.")
st.caption(
    "El modelo se entrenó con fotos de 32x32 píxeles. Funciona mejor con imágenes donde "
    "el objeto ocupe casi todo el encuadre. Si subes algo que no es de estas clases, "
    "igual va a elegir una de las 10.")

try:
    modelo, clases = obtener_modelo()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

archivo = st.file_uploader("Elige una imagen", type=["jpg", "jpeg", "png", "webp", "bmp"])

if archivo is not None:
    imagen = Image.open(archivo)
    st.image(imagen, caption="Imagen cargada", width="stretch")

    clase, confianza, todas = predecir(modelo, clases, imagen)

    st.subheader(f"Predicción: {clase}")
    st.metric("Confianza", f"{confianza:.1%}")
    st.progress(confianza)

    if confianza < 0.5:
        st.warning("La confianza es baja. El modelo no está muy seguro de esta predicción.")

    tabla = (pd.DataFrame({"Clase": list(todas.keys()), "Probabilidad": list(todas.values())})
             .sort_values("Probabilidad", ascending=False)
             .reset_index(drop=True))
    st.write("Probabilidad para cada clase:")
    st.bar_chart(tabla, x="Clase", y="Probabilidad", horizontal=True, sort="-Probabilidad")
    st.dataframe(tabla.style.format({"Probabilidad": "{:.2%}"}), hide_index=True)
