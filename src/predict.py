"""
Predicción con el modelo ya entrenado.

Uso desde la terminal:

    python src/predict.py ruta/a/imagen.jpg

También lo usa la aplicación de Streamlit (app.py), así que el preprocesamiento
de una imagen nueva vive en un solo lugar.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow import keras

RAIZ = Path(__file__).resolve().parent.parent
RUTA_MODELO = RAIZ / "models" / "cnn_cifar10.keras"
RUTA_CLASES = RAIZ / "models" / "clases.json"
TAMANO = (32, 32)  # CIFAR-10 trabaja con imágenes de 32x32 píxeles


def cargar_modelo(ruta_modelo=RUTA_MODELO, ruta_clases=RUTA_CLASES):
    if not Path(ruta_modelo).exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {ruta_modelo}. "
            "Entrénalo primero con el notebook o con src/train.py.")
    modelo = keras.models.load_model(ruta_modelo)
    clases = json.loads(Path(ruta_clases).read_text(encoding="utf-8"))
    return modelo, clases


def preprocesar_imagen(imagen):
    """Recibe una imagen PIL de cualquier tamaño y la deja igual que las de entrenamiento."""
    imagen = imagen.convert("RGB").resize(TAMANO, Image.LANCZOS)
    arreglo = np.asarray(imagen, dtype="float32") / 255.0
    return np.expand_dims(arreglo, axis=0)  # forma (1, 32, 32, 3)


def predecir(modelo, clases, imagen):
    """Devuelve la clase predicha, su probabilidad y las probabilidades de todas las clases."""
    probabilidades = modelo.predict(preprocesar_imagen(imagen), verbose=0)[0]
    indice = int(np.argmax(probabilidades))
    return clases[indice], float(probabilidades[indice]), dict(zip(clases, probabilidades.tolist()))


def main():
    if len(sys.argv) != 2:
        print("Uso: python src/predict.py ruta/a/imagen.jpg")
        sys.exit(1)

    modelo, clases = cargar_modelo()
    clase, confianza, todas = predecir(modelo, clases, Image.open(sys.argv[1]))

    print(f"Clase predicha: {clase} ({confianza:.1%})")
    print("\nTodas las clases:")
    for nombre, prob in sorted(todas.items(), key=lambda par: par[1], reverse=True):
        print(f"  {nombre:<10} {prob:.1%}")


if __name__ == "__main__":
    main()
