# Trabajo Práctico Final - Clasificación de Autos con Deep Learning

Este repositorio contiene el código, notebooks y utilidades para comparar y analizar modelos de clasificación de autos usando PyTorch y Keras, aplicando técnicas de transfer learning, fine-tuning, visualización de métricas, Grad-CAM y reducción de dimensionalidad (t-SNE/PCA).

## Estructura principal

- `analisis/`: Notebooks de análisis y comparación de modelos (ResNet, AlexNet, SimpleCNN, etc.) y utilidades de análisis.
- `models/`: Definiciones de arquitecturas y notebooks de entrenamiento de modelos en PyTorch.
- `data/stanford_cars/`: Dataset y etiquetas en CSV (train, val, test, names.csv).
- `utils_dataset.py`, `data_loader.py`: Utilidades para cargar datasets y DataLoaders personalizados.
- `prueba.ipynb`: Entrenamiento y pruebas rápidas de la CNN propia.
- `pesos/`: Pesos entrenados de los modelos (por ejemplo, `resnet18_best.pth`, `simplecnn_best.pth`).

## ¿Qué incluye?

- Entrenamiento y análisis de modelos (ResNet18, ResNet50, AlexNet, SimpleCNN) en PyTorch y Keras.
- Visualización de métricas, ejemplos de aciertos/errores, Grad-CAM y t-SNE.
- Guardado de predicciones, labels y features en `.npz` para análisis cruzado.
- Comparación visual y cuantitativa de todos los modelos.

## Requisitos

- Python 3.8+
- PyTorch, torchvision
- TensorFlow/Keras
- scikit-learn, matplotlib, seaborn, pandas, numpy, tqdm

Instala los requisitos con:

```bash
pip install torch torchvision tensorflow keras scikit-learn matplotlib seaborn pandas tqdm