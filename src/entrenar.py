"""Entrena el modelo final: CNN base con tasa de aprendizaje 1e-4.

Decisión tomada con los experimentos de la Unidad 2: el base con lr 1e-4 tiene el
mayor F1-macro de validación (0,8168 ± 0,0055 con tres semillas) y ninguna
modificación lo supera.

En la Unidad 3 se evaluaron dos mejoras adicionales definidas al cierre de la Unidad 2:
- Más épocas y paciencia en EarlyStopping: en esa corrida, más paciencia no mejoró el resultado. El
  entrenamiento se detuvo en la época 15, así que el máximo de 40 épocas no llegó a usarse.
- Transfer learning con MobileNetV2: mejor desempeño; queda como arquitectura candidata para una
  siguiente iteración.
El modelo aprobado sigue siendo este.
Detalle en notebooks/experimentos_u3.ipynb y artefactos/unidad3/.

Uso desde la raíz del repositorio:
    python -m src.entrenar

Genera en artefactos/:
    modelo_final.keras     pesos del mejor modelo (restaurados por EarlyStopping)
    clases.json            nombres de las 40 clases, en el orden de la salida del modelo
    historial.json         pérdida y accuracy por época
    metricas_validacion.json
"""

import json
import os
import random

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping

from src.datos import (IMG_SIZE, SEED, cargar_dataset_modelo, dividir,
                       generador_entrenamiento, generador_evaluacion)
from src.modelo import construir_modelo

EPOCAS = 15
PACIENCIA = 3
LEARNING_RATE = 1e-4
CARPETA_ARTEFACTOS = "artefactos"


def fijar_semillas():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def entrenar():
    fijar_semillas()
    os.makedirs(CARPETA_ARTEFACTOS, exist_ok=True)

    df_modelo, _ = cargar_dataset_modelo()
    print(f"Imágenes para modelar: {len(df_modelo)}")

    train_df, val_df, _ = dividir(df_modelo)
    train_gen = generador_entrenamiento(train_df)
    val_gen = generador_evaluacion(val_df)

    nombres_clases = sorted(train_gen.class_indices, key=train_gen.class_indices.get)

    modelo = construir_modelo(input_shape=(*IMG_SIZE, 3), num_clases=len(nombres_clases))
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    parada = EarlyStopping(monitor="val_loss", patience=PACIENCIA,
                           restore_best_weights=True)
    historia = modelo.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCAS,
        callbacks=[parada],
        verbose=2,
    )

    # Métricas de validación con el mejor modelo (la que se usa para decidir)
    y_val = val_gen.classes
    y_pred_val = modelo.predict(val_gen, verbose=0).argmax(axis=1)
    metricas_val = {
        "accuracy": accuracy_score(y_val, y_pred_val),
        "f1_macro": f1_score(y_val, y_pred_val, average="macro"),
        "f1_weighted": f1_score(y_val, y_pred_val, average="weighted"),
    }

    modelo.save(os.path.join(CARPETA_ARTEFACTOS, "modelo_final.keras"))

    with open(os.path.join(CARPETA_ARTEFACTOS, "clases.json"), "w", encoding="utf-8") as f:
        json.dump(nombres_clases, f, ensure_ascii=False, indent=2)

    historial = {k: [float(x) for x in v] for k, v in historia.history.items()}
    with open(os.path.join(CARPETA_ARTEFACTOS, "historial.json"), "w", encoding="utf-8") as f:
        json.dump(historial, f, indent=2)

    with open(os.path.join(CARPETA_ARTEFACTOS, "metricas_validacion.json"),
              "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metricas_val.items()}, f, indent=2)

    print("Métricas de validación:", metricas_val)
    print(f"Artefactos guardados en {CARPETA_ARTEFACTOS}/")


if __name__ == "__main__":
    entrenar()