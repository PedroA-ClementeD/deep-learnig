"""
Entrenamiento de la CNN sobre CIFAR-10.

Es la versión en script de lo que hace el notebook. Sirve para entrenar sin abrir Colab:

    python src/train.py
    python src/train.py --epocas 40

Deja el modelo en models/ y las gráficas y métricas en reports/.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # permite guardar gráficas sin pantalla
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_MODELOS = RAIZ / "models"
CARPETA_REPORTES = RAIZ / "reports"
RUTA_MODELO = CARPETA_MODELOS / "cnn_cifar10.keras"
RUTA_CLASES = CARPETA_MODELOS / "clases.json"

CLASES = ["avión", "automóvil", "pájaro", "gato", "ciervo",
          "perro", "rana", "caballo", "barco", "camión"]
SEMILLA = 42


def cargar_datos():
    """Carga CIFAR-10, normaliza a [0, 1] y separa entrenamiento, validación y prueba."""
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train_full = y_train_full.flatten()
    y_test = y_test.flatten()

    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # 10% del entrenamiento pasa a validación, manteniendo la proporción de cada clase
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.10,
        stratify=y_train_full, random_state=SEMILLA)

    return x_train, y_train, x_val, y_val, x_test, y_test


def construir_modelo(num_clases=10):
    """CNN de tres bloques convolucionales con Data Augmentation al inicio."""
    aumento = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ], name="data_augmentation")

    modelo = keras.Sequential([
        keras.Input(shape=(32, 32, 3)),
        aumento,

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.4),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_clases, activation="softmax"),
    ], name="cnn_cifar10")

    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    return modelo


def graficar_entrenamiento(historial, ruta):
    h = historial.history
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
    ejes[0].plot(h["accuracy"], label="Entrenamiento")
    ejes[0].plot(h["val_accuracy"], label="Validación")
    ejes[0].set_title("Accuracy por época")
    ejes[0].set_xlabel("Época")
    ejes[0].legend()
    ejes[1].plot(h["loss"], label="Entrenamiento")
    ejes[1].plot(h["val_loss"], label="Validación")
    ejes[1].set_title("Pérdida (loss) por época")
    ejes[1].set_xlabel("Época")
    ejes[1].legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def graficar_matriz(y_real, y_pred, ruta):
    matriz = confusion_matrix(y_real, y_pred)
    fig = plt.figure(figsize=(9, 7))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASES, yticklabels=CLASES)
    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")
    plt.title("Matriz de confusión (conjunto de prueba)")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Entrena la CNN sobre CIFAR-10")
    parser.add_argument("--epocas", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    keras.utils.set_random_seed(SEMILLA)
    CARPETA_MODELOS.mkdir(exist_ok=True)
    CARPETA_REPORTES.mkdir(exist_ok=True)

    x_train, y_train, x_val, y_val, x_test, y_test = cargar_datos()
    print(f"Entrenamiento: {len(x_train)} | Validación: {len(x_val)} | Prueba: {len(x_test)}")

    modelo = construir_modelo(len(CLASES))
    modelo.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=3, min_lr=1e-5),
    ]
    historial = modelo.fit(x_train, y_train,
                           validation_data=(x_val, y_val),
                           epochs=args.epocas, batch_size=args.batch,
                           callbacks=callbacks, verbose=2)

    perdida, accuracy = modelo.evaluate(x_test, y_test, verbose=0)
    y_pred = np.argmax(modelo.predict(x_test, verbose=0), axis=1)
    reporte = classification_report(y_test, y_pred, labels=list(range(len(CLASES))), target_names=CLASES, digits=3, zero_division=0)
    print(f"\nAccuracy en prueba: {accuracy:.4f} | Pérdida en prueba: {perdida:.4f}\n")
    print(reporte)

    graficar_entrenamiento(historial, CARPETA_REPORTES / "entrenamiento.png")
    graficar_matriz(y_test, y_pred, CARPETA_REPORTES / "matriz_confusion.png")
    (CARPETA_REPORTES / "metricas.txt").write_text(
        f"Accuracy prueba: {accuracy:.4f}\nPérdida prueba: {perdida:.4f}\n\n{reporte}",
        encoding="utf-8")

    modelo.save(RUTA_MODELO)
    RUTA_CLASES.write_text(json.dumps(CLASES, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Modelo guardado en {RUTA_MODELO}")


if __name__ == "__main__":
    main()
