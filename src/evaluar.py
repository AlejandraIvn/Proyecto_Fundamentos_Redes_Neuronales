"""Evalúa el modelo final en el conjunto de prueba.

Uso desde la raíz del repositorio (después de entrenar):
    python -m src.evaluar

El conjunto de prueba se usa solo aquí, una vez. No se usa para elegir nada.

Genera en artefactos/:
    metricas.json            accuracy, F1-macro y F1-weighted en prueba
    reporte_por_clase.csv    precision, recall y F1 por clase
    matriz_confusion.csv     matriz de confusión (filas: clase real, columnas: predicha)
"""

import json
import os

import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.datos import cargar_dataset_modelo, dividir, generador_evaluacion

CARPETA_ARTEFACTOS = "artefactos"


def evaluar():
    with open(os.path.join(CARPETA_ARTEFACTOS, "clases.json"), encoding="utf-8") as f:
        nombres_clases = json.load(f)

    df_modelo, _ = cargar_dataset_modelo()
    _, _, test_df = dividir(df_modelo)
    test_gen = generador_evaluacion(test_df)

    orden_generador = sorted(test_gen.class_indices, key=test_gen.class_indices.get)
    assert orden_generador == nombres_clases, "El orden de clases no coincide con clases.json"

    modelo = tf.keras.models.load_model(os.path.join(CARPETA_ARTEFACTOS, "modelo_final.keras"))
    y_true = test_gen.classes
    y_pred = modelo.predict(test_gen, verbose=0).argmax(axis=1)

    metricas = {
        "n_prueba": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }
    with open(os.path.join(CARPETA_ARTEFACTOS, "metricas.json"), "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2)

    reporte = classification_report(
        y_true, y_pred, labels=range(len(nombres_clases)),
        target_names=nombres_clases, output_dict=True, zero_division=0)
    pd.DataFrame(reporte).T.to_csv(os.path.join(CARPETA_ARTEFACTOS, "reporte_por_clase.csv"))

    matriz = confusion_matrix(y_true, y_pred, labels=range(len(nombres_clases)))
    pd.DataFrame(matriz, index=nombres_clases, columns=nombres_clases).to_csv(
        os.path.join(CARPETA_ARTEFACTOS, "matriz_confusion.csv"))

    print("Métricas en prueba:", metricas)
    print("Referencia del notebook: accuracy 0,8578, F1-macro 0,8005, F1-weighted 0,8528")


if __name__ == "__main__":
    evaluar()