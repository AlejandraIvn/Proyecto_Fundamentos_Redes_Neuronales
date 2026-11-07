# Clasificación de productos de moda con CNN

Proyecto integrador de Fundamentos de Redes Neuronales. El objetivo es predecir la categoría (`articleType`) de un producto de moda a partir de su imagen, para automatizar la catalogación en una plataforma de e-commerce.

El repositorio tiene tres partes:

1. **Experimentos (evidencias):** el notebook con todas las corridas, comparaciones y el diseño factorial, con sus salidas.
2. **Pipeline final:** scripts reproducibles que entrenan y evalúan el modelo seleccionado y guardan sus artefactos.
3. **Interfaz:** una aplicación pequeña donde se sube una imagen, se muestra la categoría predicha con sus probabilidades y las métricas del modelo.

## Problema y datos

- **Dataset:** [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small/data), de Kaggle. Tiene 44.446 filas originales.
- **Depuración:** se descartan registros sin imagen, grupos de copias exactas con etiquetas contradictorias y se conserva una imagen por huella MD5. Quedan 39.437 imágenes.
- **Variable objetivo:** `articleType`, limitada a las 40 clases con al menos 200 imágenes únicas (cubren el 90,3 % del catálogo).
- **Entrada:** imagen convertida a RGB, redimensionada a 80 x 60 y normalizada a [0, 1].
- **Partición:** estratificada 70/15/15 con semilla 42 (entrenamiento 27.605, validación 5.916, prueba 5.916).

## Modelo final

CNN de tres bloques convolucionales (32, 64 y 128 filtros) con BatchNormalization y MaxPooling, seguida de una capa densa de 128 neuronas, Dropout de 0,4 y salida softmax de 40 clases.

**Configuración seleccionada:** Adam con tasa de aprendizaje 1e-4, categorical_crossentropy, lote de 64, máximo 15 épocas y EarlyStopping sobre `val_loss` con paciencia 3 y restauración de los mejores pesos.

La selección se hizo con F1-macro sobre validación, en un protocolo de un cambio a la vez, y se confirmó con tres semillas. El conjunto de prueba se usó solo para la evaluación final.

## Resultados

Se reportan dos conjuntos de cifras. Ambos usan la misma configuración (CNN base con tasa de aprendizaje 1e-4), pero provienen de corridas distintas.

### 1. Notebook de experimentos (evidencia de la selección)

Corrida de la configuración base con lr 1e-4 y semilla 42, dentro del diseño factorial de 32 configuraciones. Sirve para justificar la selección del modelo. Con tres semillas, el F1-macro de validación fue 0,8168 ± 0,0055.

| Métrica | Validación | Prueba |
|---|---|---|
| F1-macro | 0,8165 | 0,8005 |
| Accuracy | 0,8648 | 0,8578 |
| F1-weighted | 0,8607 | 0,8528 |

### 2. Pipeline final (modelo que usa la interfaz)

Entrenado con `python -m src.entrenar` y evaluado con `python -m src.evaluar`, en una GPU Tesla T4 de Google Colab. Estas son las cifras que corresponden a los archivos de `artefactos/`.

| Métrica | Validación | Prueba |
|---|---|---|
| F1-macro | 0,8085 | 0,7907 |
| Accuracy | 0,8638 | 0,8496 |
| F1-weighted | 0,8579 | 0,8420 |

La diferencia de unos 0,01 en prueba entre ambas tablas se debe a la variabilidad del entrenamiento en GPU, que no es determinista. No corresponde a un cambio de modelo. La evaluación en prueba se ejecutó una sola vez, para no elegir el modelo con base en ese conjunto.

### Hallazgos principales

- La tasa de aprendizaje fue el factor que más mejoró el resultado. Con 1e-3 la validación era inestable; con 1e-4 la oscilación del val loss bajó de 0,897 a 0,053.
- Ninguna regularización adicional (class weights, augmentation, Dropout 0,6, GAP, bloque extra) superó al modelo base en validación.
- Las clases más débiles son *Tunics*, *Kurtis*, *Flats*, *Jackets* y *Sweaters*. Se confunden sobre todo con *Kurtas*, *Tshirts*, *Shirts* y *Heels*.

## Estructura del repositorio

```
Proyecto_Fundamentos_Redes_Neuronales/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── app/
│   └── app.py
├── artefactos/
│   ├── clases.json
│   ├── historial.json
│   ├── matriz_confusion.csv
│   ├── metricas.json
│   ├── metricas_validacion.json
│   ├── modelo_final.keras
│   └── reporte_por_clase.csv
├── data/
│   └── .gitkeep
├── notebooks/
│   └── experimentos.ipynb
└── src/
    ├── __init__.py
    ├── datos.py
    ├── entrenar.py
    ├── evaluar.py
    └── modelo.py
```

## Configuración de credenciales

El pipeline descarga el dataset desde Kaggle, así que necesita tu usuario y tu API key. Crea un archivo `.env` en la raíz del repositorio (Git lo ignora):

```
KAGGLE_USERNAME=tu_usuario
KAGGLE_KEY=tu_api_key
```

El archivo `.env.example` muestra qué variables se necesitan, sin valores.

## Cómo reproducirlo

Requiere Python 3.10 o superior. Ejecuta los comandos desde la raíz del repositorio:

```bash
pip install -r requirements.txt

# 1. Entrenar el modelo final (genera artefactos/)
python -m src.entrenar

# 2. Evaluar en prueba (una sola vez)
python -m src.evaluar

# 3. Abrir la interfaz
python app/app.py
```

El dataset se descarga la primera vez que se ejecuta el pipeline. Para entrenar con GPU, el proyecto se puede correr en Google Colab con GPU activada.

## Interfaz

La aplicación recibe una imagen y muestra:

- La categoría predicha y las 5 categorías más probables con su probabilidad.
- Las métricas globales del modelo en el conjunto de prueba, leídas de `artefactos/metricas.json`.
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
- El conjunto de prueba se usó en dos momentos durante la experimentación del notebook, y una vez más en el pipeline final.
- La resolución de 80 x 60 puede limitar las diferencias finas entre clases parecidas.
- El entrenamiento en GPU no es completamente determinista; las cifras exactas pueden variar ligeramente entre corridas.
- La auditoría detecta copias exactas, pero no imágenes casi idénticas ni errores de etiquetado aislados.

## Integrantes

- Julian: Ingeniero de datos
- Kinan: Deep Learning Engineer
- Ivanna: Analista y líder del proyecto

## Uso de herramientas de IA

Se usaron herramientas de IA como apoyo para revisar código y redacción. La implementación, los experimentos y la interpretación de resultados fueron realizados por el equipo.