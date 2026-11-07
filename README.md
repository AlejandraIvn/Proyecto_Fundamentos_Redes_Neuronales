# Clasificación de productos de moda con CNN

Proyecto integrador de Fundamentos de Redes Neuronales. El objetivo es predecir la categoría (`articleType`) de un producto de moda a partir de su imagen, para automatizar la catalogación en una plataforma de e-commerce.

El repositorio tiene tres partes:

1. **Experimentos (evidencias):** el notebook con todas las corridas, comparaciones y el diseño factorial, con sus salidas.
2. **Pipeline final:** un script reproducible que entrena el modelo seleccionado y guarda sus artefactos.
3. **Interfaz:** una aplicación pequeña donde se sube una imagen, se muestra la categoría predicha con sus probabilidades y las métricas del modelo.

## Problema y datos

- **Dataset:** [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small/data), de Kaggle. Tiene 44.446 filas originales.
- **Depuración:** se descartan registros sin imagen, copias exactas con etiquetas contradictorias y se deduplican copias exactas. Quedan 39.437 imágenes.
- **Variable objetivo:** `articleType`, limitada a las 40 clases con al menos 200 imágenes únicas (cubren el 90,3 % del catálogo).
- **Entrada:** imagen convertida a RGB, redimensionada a 80 x 60 y normalizada a [0, 1].
- **Partición:** estratificada 70/15/15 con semilla 42 (entrenamiento 27.605, validación 5.916, prueba 5.916).

## Modelo final

CNN de tres bloques convolucionales (32, 64 y 128 filtros) con BatchNormalization y MaxPooling, seguida de una capa densa de 128 neuronas, Dropout de 0,4 y salida softmax de 40 clases.

**Configuración seleccionada:** Adam con tasa de aprendizaje 1e-4, categorical_crossentropy, lote de 64, máximo 15 épocas y EarlyStopping sobre `val_loss` con paciencia 3 y restauración de los mejores pesos.

La selección se hizo con F1-macro sobre validación, en un protocolo de un cambio a la vez, y se confirmó con tres semillas. El conjunto de prueba se usó solo para la evaluación final.

## Resultados

| Métrica | Valor |
|---|---|
| F1-macro (validación, 3 semillas) | 0,8168 ± 0,0055 |
| F1-macro (prueba) | 0,8005 |
| Accuracy (prueba) | 0,8578 |
| F1-weighted (prueba) | 0,8528 |

Hallazgos principales:

- La tasa de aprendizaje fue el factor que más mejoró el resultado. Con 1e-3 la validación era inestable; con 1e-4 la oscilación del val loss bajó de 0,897 a 0,053.
- Ninguna regularización adicional (class weights, augmentation, Dropout 0,6, GAP, bloque extra) superó al modelo base en validación.
- Las clases más débiles son *Tunics*, *Kurtis*, *Flats*, *Jackets* y *Sweaters*. Se confunden sobre todo con *Kurtas*, *Tshirts*, *Shirts* y *Heels*.

Los detalles completos están en el notebook de experimentos.

## Estructura del repositorio
```
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│ └── experimentos.ipynb # Evidencias: corridas, comparaciones, factorial, matriz de confusión
├── src/
│ ├── datos.py # Limpieza, depuración y partición
│ ├── modelo.py # Arquitectura de la CNN
│ ├── entrenar.py # Entrenamiento del modelo final
│ └── evaluar.py # Métricas en prueba y matriz de confusión
├── app/
│ └── app.py # Interfaz: subir imagen, predecir y mostrar métricas
├── artefactos/ # Modelo entrenado, lista de clases y métricas (JSON)
└── data/ # Vacío. El dataset se descarga desde Kaggle
```
## Cómo reproducirlo

Requiere Python 3.10 o superior.

```bash
pip install -r requirements.txt

# 1. Preparar los datos y entrenar el modelo final
python src/entrenar.py

# 2. Evaluar en prueba (genera artefactos/metricas.json)
python src/evaluar.py

# 3. Abrir la interfaz
python app/app.py
```

El dataset se descarga con `kagglehub` la primera vez que se ejecuta el pipeline.

## Interfaz

La aplicación recibe una imagen y muestra:

- La categoría predicha y las 5 categorías más probables con su probabilidad.
- Las métricas globales del modelo en el conjunto de prueba.
- Una advertencia: la métrica por imagen no existe, porque el modelo no conoce la categoría real de la imagen subida.

## Notebook de experimentos

`notebooks/experimentos.ipynb` contiene todas las corridas de la Unidad 2:

- Experimento base y los 6 experimentos de un solo cambio.
- Diseño factorial de 32 configuraciones.
- Confirmación con semillas adicionales.
- Evaluación en prueba y matriz de confusión.

Se conserva con sus salidas para poder revisar los resultados sin volver a entrenar.

## Limitaciones

- Algunas comparaciones usan una sola semilla. Solo el modelo base y las mejores configuraciones se confirmaron con varias.
- El conjunto de prueba se usó en dos momentos durante la experimentación.
- La resolución de 80 x 60 puede limitar las diferencias finas entre clases parecidas.
- El entrenamiento en GPU no es completamente determinista; las cifras exactas pueden variar ligeramente.
- La auditoría detecta copias exactas, pero no imágenes casi idénticas ni errores de etiquetado aislados.

## Integrantes

- Julian: Ingeniero de datos
- Kinan: Deep Learning Engineer
- Ivanna: Analista y líder del proyecto

## Uso de herramientas de IA

Se usaron herramientas de IA como apoyo para revisar código y redacción. La implementación, los experimentos y la interpretación de resultados fueron realizados por el equipo.
