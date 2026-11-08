# Clasificación de productos de moda con CNN

Proyecto integrador de Fundamentos de Redes Neuronales. El objetivo es predecir la categoría (`articleType`) de un producto de moda a partir de su imagen, para automatizar la catalogación en una plataforma de e-commerce.

El repositorio tiene cuatro partes:

1. **Experimentos (evidencias):** los notebooks con las corridas y sus salidas. `experimentos.ipynb` cubre las Unidades 1 y 2 (calidad de datos, comparaciones, diseño factorial). `experimentos_u3.ipynb` cubre la Unidad 3 (evaluación de mejoras adicionales).
2. **Pipeline final:** scripts reproducibles que entrenan y evalúan el modelo seleccionado y guardan sus artefactos.
3. **Página web (Vercel):** el modelo corre en el navegador con TensorFlow.js. Clasifica de 1 a 40 imágenes con descarga de CSV, trae imágenes de ejemplo e incluye la sección **Dentro del modelo**, que explica de forma visual y no técnica cómo llega el modelo a su respuesta.
4. **App Gradio:** la misma clasificación (1 a 40 imágenes y CSV), servida con Python.

## Problema y datos

- **Dataset:** [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small/data), de Kaggle. Tiene 44.446 filas originales.
- **Depuración:** se descartan registros sin imagen, grupos de copias exactas con etiquetas contradictorias y se conserva una imagen por huella MD5. Quedan 39.437 imágenes.
- **Variable objetivo:** `articleType`, limitada a las 40 clases con al menos 200 imágenes únicas (cubren el 90,3 % del catálogo).
- **Entrada:** imagen convertida a RGB, redimensionada a 80 x 60 y normalizada a [0, 1].
- **Partición:** estratificada 70/15/15 con semilla 42 (entrenamiento 27.605, validación 5.916, prueba 5.916).

## Modelo seleccionado (Unidad 2)

CNN de tres bloques convolucionales (32, 64 y 128 filtros) con BatchNormalization y MaxPooling, seguida de una capa densa de 128 neuronas, Dropout de 0,4 y salida softmax de 40 clases (1.246.312 parámetros).

**Configuración seleccionada:** Adam con tasa de aprendizaje 1e-4, `categorical_crossentropy`, lote de 64, máximo 15 épocas y EarlyStopping sobre `val_loss` con paciencia 3 y restauración de los mejores pesos.

La selección se hizo con F1-macro sobre validación, en un protocolo de un cambio a la vez y un diseño factorial de 32 configuraciones, y se confirmó con tres semillas. Es el modelo aprobado: el que usan la app, la página web y el pipeline.

## Resultados

### Unidad 2: modelo seleccionado

Se reportan dos conjuntos de cifras de la misma configuración (CNN base con lr 1e-4), provenientes de corridas distintas.

| Métrica | Notebook, validación | Notebook, prueba | Pipeline (artefactos), validación | Pipeline (artefactos), prueba |
|---|---|---|---|---|
| F1-macro | 0,8165 (0,8168 ± 0,0055 con 3 semillas) | 0,8005 | 0,8085 | 0,7907 |
| Accuracy | 0,8648 | 0,8578 | 0,8638 | 0,8496 |
| F1-weighted | 0,8607 | 0,8528 | 0,8579 | 0,8420 |

La diferencia de unos 0,01 entre ambas corridas se debe a la variabilidad del entrenamiento en GPU, que no es determinista; no corresponde a un cambio de modelo. Las cifras del pipeline son las de los archivos de `artefactos/` y las que muestra la página.

### Unidad 3: evaluación de dos mejoras adicionales

Al cierre de la Unidad 2 se definieron mejoras para la siguiente iteración. En la Unidad 3 se evaluaron dos de ellas frente al modelo seleccionado, con la misma partición, los mismos generadores y la misma semilla:

1. **Más épocas y más paciencia en EarlyStopping:** 40 épocas y paciencia 6, en lugar de 15 y 3.
2. **Transfer learning con un modelo preentrenado oficial:** MobileNetV2 con pesos de ImageNet (`tf.keras.applications`), ajustada a las 40 clases en dos fases. En la fase 1 se entrena solo la nueva capa de salida; en la fase 2 se ajustan los bloques superiores con lr 1e-4.

El modelo de la Unidad 2 se volvió a entrenar en la misma sesión como referencia, para que la comparación sea justa.

**Validación:**

| Modelo | F1-macro | Accuracy | Épocas (mejor) | Δ F1 vs. modelo U2 |
|---|---|---|---|---|
| Modelo seleccionado en la U2 (CNN, lr 1e-4, 15 épocas, paciencia 3) | 0,8112 | 0,8621 | 14 (11) | — |
| Mejora 1: misma CNN con 40 épocas y paciencia 6 | 0,8084 | 0,8605 | 15 (9) | −0,003 |
| Mejora 2: MobileNetV2 preentrenada y ajustada | **0,8519** | **0,8916** | 9 (3) | **+0,041** |

**Prueba:**

| Métrica | Modelo seleccionado en la U2 | Mejora 2: MobileNetV2 | Diferencia |
|---|---|---|---|
| F1-macro | 0,8005 | 0,8402 | +0,040 |
| Accuracy | 0,8578 | 0,8861 | +0,028 |
| F1-weighted | 0,8528 | 0,8830 | +0,030 |

### Hallazgos

- **La mejora 1 (más épocas y paciencia) no mejoró al modelo seleccionado en esta corrida.**
  - Con paciencia 6, EarlyStopping detuvo la CNN en la época 15 (mejor época: la 9), así que el máximo de 40 épocas nunca se usó. En la práctica, la corrida evaluó la paciencia, no el número de épocas.
  - La diferencia (−0,003) es menor que la variación entre semillas (±0,0055).
  - La conclusión válida es acotada: con lr 1e-4 y una semilla, esperar más épocas sin mejora no ayudó. No permite afirmar que entrenar más épocas no ayude.
- **La mejora 2 (MobileNetV2) tiene un desempeño claramente superior.**
  - Sube +0,041 de F1-macro en validación (unas siete veces la variación entre semillas) y +0,040 en prueba.
  - La ganancia viene sobre todo del ajuste fino: con la red congelada, MobileNetV2 iguala a la CNN (0,804 frente a 0,811).
  - El ajuste fino alcanza su mejor época muy pronto (la 3), así que lo importante es restaurar la mejor época, no entrenar más.
- **Errores que persisten en ambos modelos:**
  - las clases más débiles son *Tunics*, *Flats* y *Kurtis*;
  - las confusiones más frecuentes son Casual Shoes ↔ Sports Shoes, Tshirts ↔ Tops y Flats → Heels.
- **Unidad 2:** la tasa de aprendizaje fue el factor que más mejoró el resultado (con 1e-4 la oscilación del val loss bajó de 0,897 a 0,053). Ninguna regularización adicional superó al modelo base.

### Decisión y trabajo futuro

El modelo seleccionado en la Unidad 2 (CNN con lr 1e-4) **se mantiene como modelo aprobado**. La app, la página web y el pipeline lo usan.

MobileNetV2 tuvo mejor desempeño que el modelo aprobado en validación y en prueba. En una siguiente iteración se puede evaluar como la **arquitectura preferida** y llegar a ser el modelo predefinido. Antes de ese cambio conviene:

- confirmar el resultado con semillas adicionales (en la Unidad 3 se usó una);
- medir su costo en la aplicación (2,3 millones de parámetros y unos 9 MB de pesos, frente a 1,2 millones y 5 MB de la CNN);
- revisar con negocio las clases que se siguen confundiendo.

La infraestructura ya está lista para ese cambio:
- `artefactos/unidad3/` guarda el modelo MobileNetV2 entrenado y sus métricas;
- el conversor de la página ya está verificado con esa arquitectura (ver más abajo).

## Estructura del repositorio

```
Proyecto_Fundamentos_Redes_Neuronales/
├── .gitignore
├── README.md
├── requirements.txt
├── app/
│   └── app.py                  interfaz Gradio (1 a 40 imágenes + CSV)
├── artefactos/                 modelo aprobado (CNN de la Unidad 2)
│   ├── modelo_final.keras
│   ├── clases.json
│   ├── historial.json
│   ├── matriz_confusion.csv
│   ├── metricas.json
│   ├── metricas_validacion.json
│   ├── reporte_por_clase.csv
│   └── unidad3/                evaluación de las mejoras de la Unidad 3
│       ├── experimentos_u3.csv
│       ├── historiales_u3.json
│       ├── figuras/curvas_validacion_u3.png
│       ├── mobilenetv2.keras   modelo candidato (no es el aprobado)
│       ├── metricas_prueba_mobilenetv2.json
│       ├── metricas_validacion_mobilenetv2.json
│       ├── reporte_por_clase_mobilenetv2.csv
│       └── matriz_confusion_mobilenetv2.csv
├── data/
│   └── .gitkeep
├── notebooks/
│   ├── experimentos.ipynb      Unidades 1 y 2
│   └── experimentos_u3.ipynb   Unidad 3 (ejecutado en Colab, con salidas)
├── src/
│   ├── __init__.py
│   ├── datos.py                carga, depuración, partición y generadores
│   ├── modelo.py               arquitectura de la CNN
│   ├── entrenar.py             entrenamiento del modelo aprobado
│   └── evaluar.py              evaluación en prueba
└── web/                        página para Vercel (TensorFlow.js)
    ├── package.json, vercel.json
    ├── scripts/convertir_modelo.mjs   .keras (Keras 3) -> TensorFlow.js
    └── public/
        ├── index.html, styles.css
        ├── app.js              clasificación de 1 a 40 imágenes, CSV, ejemplos
        ├── dentro.js           sección "Dentro del modelo" (Grad-CAM y más)
        ├── ejemplos/           imágenes de ejemplo del conjunto de prueba
        ├── model/              modelo convertido
        └── data/               clases, métricas y configuración
```

## Credenciales de Kaggle

El dataset es público y `kagglehub` lo descarga sin credenciales. Si Kaggle las pidiera, `kagglehub` las busca solo en este orden:

- **Google Colab:** panel *Secretos* (ícono de llave) con `KAGGLE_USERNAME` y `KAGGLE_KEY`.
- **Variables de entorno:** `KAGGLE_USERNAME` y `KAGGLE_KEY`.
- **Archivo de Kaggle:** `~/.kaggle/kaggle.json` (Kaggle → Settings → API → Create New Token).

Las credenciales nunca se suben al repositorio: `.gitignore` excluye `.env` y `kaggle.json`.

## Cómo reproducirlo

### Notebook en Google Colab

1. Abrir `notebooks/experimentos_u3.ipynb` en Colab y activar la GPU T4.
2. Ejecutarlo completo. Es independiente de las Unidades 1 y 2, no usa Google Drive y no requiere subir archivos: la primera celda clona este repositorio para usar `src/`.
3. La última celda guarda los resultados en `artefactos/unidad3/` y los descarga como `resultados_u3.zip`: el modelo del mejor experimento, sus métricas, las curvas y la tabla de experimentos. No toca los archivos del modelo aprobado. Al descomprimir el zip en la raíz del repositorio, cada archivo queda en `artefactos/unidad3/`.

### Pipeline (modelo aprobado)

Requiere Python 3.10–3.12 (las versiones que soporta TensorFlow). Desde la raíz del repositorio:

```bash
pip install -r requirements.txt
python -m src.entrenar      # entrena el modelo aprobado y escribe artefactos/
python -m src.evaluar       # evalúa en prueba (una sola vez)
python app/app.py           # interfaz Gradio (agregar --share para un enlace público)
```

## Página web

### Clasificación

Una sola sección recibe **una o varias imágenes (hasta 40)**:

- **Con una imagen**, muestra la categoría predicha y las 5 más probables.
- **Con varias**, muestra un resumen y una tabla (miniatura, categoría, probabilidad, segunda opción y estado).
- **CSV:** en ambos casos genera uno descargable con `nombre_imagen`, `categoria_predicha`, `probabilidad`, `segunda_categoria`, `probabilidad_segunda`, `requiere_revision` y `estado`.
- **Revisión manual:** las predicciones con probabilidad menor a 50 % se marcan para revisión.
- **Imágenes de ejemplo:** el botón *Probar con imágenes de ejemplo* carga 12 fotos del conjunto de prueba, que el modelo no vio al entrenar. Incluye un caso para revisión manual y uno que el modelo falla (Flats → Heels).

### Dentro del modelo

Recorrido interactivo en cuatro pasos sobre una imagen de ejemplo o una propia:

1. **Mira la foto:** la cuadrícula de 80 x 60 que recibe el modelo. Al pasar el cursor se ven los valores de rojo, verde y azul de cada punto.
2. **Detecta patrones:** las activaciones de una capa al inicio, a la mitad y al final de la red, de bordes simples a partes del producto.
3. **Se enfoca:** el mapa de calor **Grad-CAM** de la última capa convolucional, que muestra en qué zonas se apoyó el modelo para cada una de sus tres categorías más probables.
4. **Pruébalo:** tapar partes de la foto y ver en vivo cómo cambia la respuesta (oclusión).

Todo se calcula en el navegador; las imágenes no salen del equipo del usuario. La sección funciona con la CNN aprobada y también con MobileNetV2.

### Despliegue en Vercel

1. En [vercel.com](https://vercel.com): **Add New… → Project** e importar el repositorio.
2. **Root Directory:** `web`. El resto lo toma de `web/vercel.json`.
3. **Deploy.** En cada despliegue, Vercel convierte `artefactos/modelo_final.keras` al formato de TensorFlow.js. Más detalles en [web/README.md](web/README.md).

**Verificación de la conversión:** sobre las 5.916 imágenes de prueba, el modelo en TensorFlow.js reproduce exactamente las métricas de Keras:

| Modelo | Accuracy | F1-macro |
|---|---|---|
| CNN aprobada | 0,849561 | 0,790679 |
| MobileNetV2 | 0,886072 | 0,840151 |

## Limitaciones

- Los experimentos de la Unidad 3 se corrieron con una sola semilla. La ventaja de MobileNetV2 (+0,041) es mucho mayor que la variación entre semillas, pero conviene confirmarla antes de cambiar el modelo predefinido.
- La mejora de más épocas no llegó a usar su presupuesto de 40. Para evaluarla de verdad haría falta otra corrida, por ejemplo bajando la tasa de aprendizaje cuando el val loss se estanca (`ReduceLROnPlateau`).
- El conjunto de prueba se usó dos veces en la Unidad 2 y una vez en la Unidad 3, para comparar el mejor experimento con el modelo aprobado.
- La resolución de 80 x 60 limita las diferencias finas entre clases parecidas.
- El modelo se entrenó con fotos de catálogo (producto centrado sobre fondo blanco). Con otro tipo de fotos las predicciones son menos confiables.
- El entrenamiento en GPU no es completamente determinista; las cifras exactas pueden variar ligeramente entre corridas.
- La auditoría detecta copias exactas, pero no imágenes casi idénticas ni errores de etiquetado aislados.

## Integrantes

- Julian: Ingeniero de datos
- Kinan: Deep Learning Engineer
- Ivanna: Analista y líder del proyecto

## Uso de herramientas de IA

Se usaron herramientas de IA como apoyo para revisar código y redacción. La implementación, los experimentos y la interpretación de resultados fueron realizados por el equipo.
