"""Prediccion sobre una imagen nueva usando el modelo entrenado.

Uso:
    python predict.py ruta/imagen.jpg
"""
import sys
import os
import numpy as np
from PIL import Image
import tensorflow as tf

try:
    from src.config import CLASS_NAMES, IMG_SIZE, model_path
except ImportError:
    from config import CLASS_NAMES, IMG_SIZE, model_path


def load_model(path=None):
    if path is None:
        base = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
        path = model_path(base)
    return tf.keras.models.load_model(path)


def prepare_image(img, target_size=IMG_SIZE):
    img = img.convert("RGB").resize(target_size)
    arr = np.asarray(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)


def predict_image(model, img):
    arr = prepare_image(img)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs


def main():
    if len(sys.argv) < 2:
        print("Uso: python predict.py <ruta_imagen>")
        sys.exit(1)
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print("No existe la imagen:", img_path)
        sys.exit(1)
    model = load_model()
    img = Image.open(img_path)
    cls, conf, probs = predict_image(model, img)
    print(f"Clase predicha: {cls} (confianza: {conf:.4f})")
    for name, p in zip(CLASS_NAMES, probs):
        print(f"  {name}: {p:.4f}")


if __name__ == "__main__":
    main()
