"""Arquitectura de la CNN del modelo final (configuración base de la Unidad 2)."""

from tensorflow.keras import layers, models


def construir_modelo(input_shape=(80, 60, 3), num_clases=40,
                     filtros=(32, 64, 128), unidades_densa=128, dropout=0.4):
    """CNN de bloques Conv2D + BatchNormalization + MaxPooling, seguida de una capa densa.

    Con los valores por defecto reproduce el modelo base del notebook (1.246.312 parámetros).
    """
    capas = [layers.Input(shape=input_shape)]
    for f in filtros:
        capas += [
            layers.Conv2D(f, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
        ]
    capas += [
        layers.Flatten(),
        layers.Dense(unidades_densa, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(num_clases, activation="softmax"),
    ]
    return models.Sequential(capas)