"""Carga y preprocesamiento del dataset CIFAR-10."""
import numpy as np
from tensorflow import keras
from src.config import CLASS_NAMES, NUM_CLASSES, IMG_SIZE


def load_data():
    """Carga CIFAR-10 y devuelve (x_train, y_train), (x_test, y_test)."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    return (x_train, y_train), (x_test, y_test)


def preprocess(x, y=None):
    """Normaliza pixeles a [0,1] y convierte etiquetas a one-hot si se pide."""
    x = x.astype("float32") / 255.0
    return x


def split_train_val(x_train, y_train, val_size=0.1, seed=42):
    """Separa conjunto de validacion del entrenamiento."""
    from sklearn.model_selection import train_test_split
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train, test_size=val_size, random_state=seed, stratify=y_train
    )
    return x_tr, x_val, y_tr, y_val
