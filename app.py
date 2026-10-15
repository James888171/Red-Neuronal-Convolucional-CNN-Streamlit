"""Aplicacion Streamlit para predecir imagenes con la CNN entrenada."""
import os
import sys
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

# Asegurar imports del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import CLASS_NAMES, IMG_SIZE, model_path

st.set_page_config(page_title="CNN CIFAR-10 - Clasificador de imagenes", layout="centered")

st.title("Clasificador de imagenes con CNN")
st.markdown("Sube una imagen y el modelo entrenado predecira su clase junto con la confianza.")

@st.cache_resource
def load_model_cached():
    mp = model_path(BASE_DIR)
    if not os.path.exists(mp):
        st.error(f"No se encontro el modelo en {mp}. Entrena primero con train_cnn.py")
        return None
    return tf.keras.models.load_model(mp)

model = load_model_cached()

uploaded = st.file_uploader("Sube una imagen (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded is not None and model is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Imagen cargada", width=250)

    # Preprocesar
    img_rgb = img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img_rgb, dtype="float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    with st.spinner("Prediciendo..."):
        probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    cls = CLASS_NAMES[idx]
    conf = float(probs[idx])

    st.success(f"Clase predicha: **{cls}**  |  Confianza: **{conf*100:.2f}%**")

    # Grafico de probabilidades
    fig, ax = plt.subplots(figsize=(8, 4))
    order = np.argsort(probs)[::-1]
    ax.barh([CLASS_NAMES[i] for i in order], probs[order], color="steelblue")
    ax.invert_yaxis()
    ax.set_xlabel("Probabilidad")
    ax.set_title("Distribucion de probabilidades")
    st.pyplot(fig)

    st.subheader("Top-3 predicciones")
    for i in order[:3]:
        st.write(f"- {CLASS_NAMES[i]}: {probs[i]*100:.2f}%")
