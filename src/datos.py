"""Carga, depuración y partición del dataset Fashion Product Images (Small).

Replica las decisiones de la Unidad 1 y 2 (ver notebooks/experimentos.ipynb):
- se descartan los registros sin imagen en disco;
- se eliminan los grupos de copias exactas con etiquetas contradictorias;
- se conserva una sola imagen por huella MD5;
- articleType se restringe a las clases con al menos 200 imágenes únicas;
- partición estratificada 70/15/15 con semilla 42.
"""

import hashlib
import os

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET = "paramaggarwal/fashion-product-images-small"
COLUMNAS = [
    "id", "gender", "masterCategory", "subCategory", "articleType",
    "baseColour", "season", "year", "usage", "productDisplayName",
]
UMBRAL_CLASE = 200
SEED = 42

IMG_SIZE = (80, 60)  # (alto, ancho), igual que en el notebook
BATCH_SIZE = 64


def descargar_dataset():
    """Descarga el dataset desde Kaggle (usa la caché si ya existe) y devuelve su ruta."""
    return kagglehub.dataset_download(DATASET)


def _encontrar_archivo(raiz, nombre):
    """Busca un archivo en la raíz del dataset o dentro de myntradataset/."""
    for candidata in (os.path.join(raiz, nombre),
                      os.path.join(raiz, "myntradataset", nombre)):
        if os.path.exists(candidata):
            return candidata
    raise FileNotFoundError(f"No se encontró {nombre} en {raiz}")


def _encontrar_carpeta_imagenes(raiz):
    for candidata in (os.path.join(raiz, "images"),
                      os.path.join(raiz, "myntradataset", "images")):
        if os.path.isdir(candidata):
            return candidata
    raise FileNotFoundError(f"No se encontró la carpeta de imágenes en {raiz}")


def cargar_styles(raiz):
    """Lee styles.csv. Las filas cuyo nombre de producto tiene comas se reparan."""
    ruta = _encontrar_archivo(raiz, "styles.csv")
    filas = []
    with open(ruta, "r", encoding="utf-8") as f:
        next(f)  # encabezado
        for linea in f:
            partes = linea.rstrip("\n\r").split(",")
            if len(partes) == 10:
                filas.append(partes)
            elif len(partes) > 10:
                filas.append(partes[:9] + [",".join(partes[9:])])
            # con menos de 10 campos la fila se descarta, como en el notebook
    return pd.DataFrame(filas, columns=COLUMNAS)


def agregar_rutas(df, carpeta_imagenes):
    """Asocia cada producto con su imagen y descarta los que no tienen archivo."""
    df = df.copy()
    df["image_path"] = df["id"].apply(
        lambda x: os.path.join(carpeta_imagenes, f"{x}.jpg"))
    df["image_exists"] = df["image_path"].apply(os.path.exists)
    return df[df["image_exists"]].reset_index(drop=True)


def _md5_archivo(ruta):
    with open(ruta, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def depurar(df):
    """Aplica la depuración de duplicados y la restricción de clases.

    Devuelve el DataFrame listo para modelar y la lista ordenada de clases.
    """
    df = df.copy()
    df["md5"] = df["image_path"].map(_md5_archivo)

    etiquetas_por_grupo = df.groupby("md5")["articleType"].nunique()
    md5_conflicto = etiquetas_por_grupo[etiquetas_por_grupo > 1].index

    unicas = df[~df["md5"].isin(md5_conflicto)].drop_duplicates("md5")

    conteo = unicas["articleType"].value_counts()
    clases = sorted(conteo[conteo >= UMBRAL_CLASE].index)

    df_modelo = unicas[unicas["articleType"].isin(clases)].reset_index(drop=True)
    return df_modelo, clases


def cargar_dataset_modelo():
    """Pipeline de datos completo: descarga, lectura, depuración y clases."""
    raiz = descargar_dataset()
    df = cargar_styles(raiz)
    df = agregar_rutas(df, _encontrar_carpeta_imagenes(raiz))
    return depurar(df)


def dividir(df):
    """Partición estratificada 70/15/15."""
    entrenamiento, temporal = train_test_split(
        df, test_size=0.30, stratify=df["articleType"], random_state=SEED)
    validacion, prueba = train_test_split(
        temporal, test_size=0.50, stratify=temporal["articleType"], random_state=SEED)
    return entrenamiento, validacion, prueba


def generador_entrenamiento(df):
    """Generador de entrenamiento: escala a [0, 1] y volteo horizontal, mezclado."""
    datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
    return datagen.flow_from_dataframe(
        dataframe=df, x_col="image_path", y_col="articleType",
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=True, seed=SEED)


def generador_evaluacion(df):
    """Generador para validación y prueba: solo escala, sin mezclar."""
    datagen = ImageDataGenerator(rescale=1. / 255)
    return datagen.flow_from_dataframe(
        dataframe=df, x_col="image_path", y_col="articleType",
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False, seed=SEED)