"""Configuracion global del proyecto."""
import os

# Clases de CIFAR-10 (dataset elegido)
CLASS_NAMES = [
    "avion", "automovil", "pajaro", "gato", "ciervo",
    "perro", "rana", "caballo", "barco", "camion",
]

# Nombres en ingles para mostrar en la app
CLASS_NAMES_EN = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

IMG_SIZE = (32, 32)
NUM_CLASSES = len(CLASS_NAMES)
MODEL_FILENAME = "cnn_cifar10.keras"
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"


def model_path(base: str = ".") -> str:
    return os.path.join(base, MODEL_DIR, MODEL_FILENAME)


def output_path(base: str = ".", filename: str = "") -> str:
    d = os.path.join(base, OUTPUT_DIR)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, filename) if filename else d
