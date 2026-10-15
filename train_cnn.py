"""Entrenamiento de la CNN sobre CIFAR-10.

Ejecutar:
    python train_cnn.py
Funciona igual en Google Colab (ejecutar las celdas o este script).
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

import tensorflow as tf
from tensorflow import keras

# Permitir ejecucion tanto como script como importado
try:
    from src.config import CLASS_NAMES, output_path, model_path
    from src.data import load_data, preprocess, split_train_val
    from src.model import build_cnn
except ImportError:
    from config import CLASS_NAMES, output_path, model_path
    from data import load_data, preprocess, split_train_val
    from model import build_cnn

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def plot_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoca")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoca")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print("Guardado:", save_path)


def plot_confusion(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de confusion")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print("Guardado:", save_path)


def plot_predictions(x, y_true, y_pred, probs, save_path, n=10):
    plt.figure(figsize=(15, 6))
    for i in range(n):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x[i])
        color = "green" if y_true[i] == y_pred[i] else "red"
        plt.title(f"Real: {CLASS_NAMES[y_true[i]]}\nPred: {CLASS_NAMES[y_pred[i]]}\n{probs[i]:.2f}",
                  color=color, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print("Guardado:", save_path)



def plot_class_distribution(y, save_path):
    unique, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(10, 5))
    plt.bar([CLASS_NAMES[i] for i in unique], counts, color="steelblue")
    plt.xticks(rotation=45, ha="right")
    plt.title("Distribucion de clases (CIFAR-10)")
    plt.xlabel("Clase")
    plt.ylabel("Cantidad")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print("Guardado:", save_path)


def plot_eda_samples(x, y, save_path, n=25):
    plt.figure(figsize=(10, 10))
    idx = np.random.choice(len(x), size=n, replace=False)
    for i in range(n):
        plt.subplot(5, 5, i + 1)
        plt.imshow(x[idx[i]])
        plt.title(CLASS_NAMES[y[idx[i]]], fontsize=8)
        plt.axis("off")
    plt.suptitle("Muestras del dataset", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print("Guardado:", save_path)


def main():
    base = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
    # Colab: trabajar en /content o cwd
    out_dir = output_path(base)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(base, "models"), exist_ok=True)

    print("=== Cargando dataset CIFAR-10 ===")
    (x_train, y_train), (x_test, y_test) = load_data()
    print("Train:", x_train.shape, y_train.shape)
    print("Test :", x_test.shape, y_test.shape)

    # EDA basica
    print("Clases:", CLASS_NAMES)
    print("Rango pixeles:", x_train.min(), x_train.max())
    unique, counts = np.unique(y_train, return_counts=True)
    print("Distribucion train:", dict(zip(unique.tolist(), counts.tolist())))

    # Visualizaciones EDA guardadas
    plot_class_distribution(y_train, os.path.join(out_dir, "class_distribution.png"))
    plot_eda_samples(x_train, y_train, os.path.join(out_dir, "eda_samples.png"))

    # Preprocesamiento
    x_train = preprocess(x_train)
    x_test = preprocess(x_test)

    # Split train/val
    x_tr, x_val, y_tr, y_val = split_train_val(x_train, y_train, val_size=0.1, seed=SEED)
    print("Train split:", x_tr.shape, "Val:", x_val.shape)

    # Modelo
    model = build_cnn()
    model.summary()

    mp = model_path(base)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
        keras.callbacks.ModelCheckpoint(mp, monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    print("=== Entrenando ===")
    history = model.fit(
        x_tr, y_tr,
        validation_data=(x_val, y_val),
        epochs=20,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    # Guardar modelo final (mejor ya guardado por checkpoint)
    model.save(mp)
    print("Modelo guardado en", mp)

    # Historia
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)
    plot_history(history, os.path.join(out_dir, "training_history.png"))

    # Evaluacion en test
    print("=== Evaluacion en test ===")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    print(report)
    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # Confusion matrix
    plot_confusion(y_test, y_pred, os.path.join(out_dir, "confusion_matrix.png"))

    # 10 predicciones
    idx = np.random.choice(len(x_test), size=10, replace=False)
    plot_predictions(
        x_test[idx], y_test[idx], y_pred[idx],
        np.max(y_pred_probs[idx], axis=1),
        os.path.join(out_dir, "predictions.png"),
        n=10,
    )

    # Metricas resumen
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metricas:", metrics)


if __name__ == "__main__":
    main()
