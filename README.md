# Red Neuronal Convolucional (CNN) + Streamlit

Proyecto de Deep Learning para clasificacion de imagenes usando una CNN sobre CIFAR-10, con aplicacion web en Streamlit.

## Dataset

Se utiliza **CIFAR-10**, incluido en TensorFlow/Keras (`keras.datasets.cifar10`). Contiene 60.000 imagenes a color de 32x32 pixeles distribuidas en 10 clases:

`avion, automovil, pajaro, gato, ciervo, perro, rana, caballo, barco, camion`.

Division oficial: 50.000 imagenes de entrenamiento y 10.000 de prueba. En este proyecto se separa un 10% del entrenamiento para validacion.

## Estructura

- `notebook/cnn_cifar10.ipynb`: notebook de experimentacion.
- `train_cnn.py`: codigo de entrenamiento.
- `predict.py`: prediccion sobre una imagen.
- `app.py`: aplicacion Streamlit.
- `src/`: modulos reutilizables (config, data, model).
- `models/cnn_cifar10.keras`: modelo entrenado.
- `outputs/`: graficas y metricas.
- `requirements.txt`: dependencias.

## Instalacion

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

## Entrenamiento

```bash
python train_cnn.py
```

En Google Colab se puede ejecutar el notebook `notebook/cnn_cifar10.ipynb` celda por celda.

## Prediccion por CLI

```bash
python predict.py ruta/a/imagen.jpg
```

## Ejecutar Streamlit

```bash
streamlit run app.py
```

Carga el modelo `models/cnn_cifar10.keras` y muestra la clase predicha con confianza.

## Resultados

- Test accuracy: ~0.82
- Se generan `outputs/training_history.png`, `outputs/confusion_matrix.png` y `outputs/predictions.png`.

## Notas para Colab

Clona el repositorio y ejecuta las celdas del notebook. Los paths son relativos, por lo que funciona en Colab sin cambios.
