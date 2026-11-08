# FashionVision · versión web (Vercel)

La misma interfaz de `app/app.py` (Gradio), pero como página estática: el modelo aprobado (la CNN de la Unidad 2, `artefactos/modelo_final.keras`) corre **en el navegador** con TensorFlow.js. No necesita servidor de Python ni GPU, así que se puede desplegar gratis en Vercel.

- Una sola sección para clasificar **una o varias imágenes (hasta 40)**. Con una imagen muestra las 5 categorías más probables; con varias, una tabla del lote con resumen.
- En ambos casos genera un **CSV** con `nombre_imagen`, `categoria_predicha`, `probabilidad`, `segunda_categoria`, `probabilidad_segunda`, `requiere_revision` y `estado`.
- Marca para revisión manual las predicciones por debajo del umbral de `config_final.json` (50 % si no existe).
- Muestra las métricas globales en prueba (`metricas.json`).
- Las imágenes no salen del navegador del usuario.
- Botón **Probar con imágenes de ejemplo**: 12 fotos del conjunto de prueba (`public/ejemplos/`), que el modelo no vio al entrenar. Incluye un caso para revisión manual y uno que el modelo falla (Flats → Heels).
- Sección **Dentro del modelo** (`public/dentro.js`): recorrido visual y no técnico en cuatro pasos. Muestra la cuadrícula de 80 × 60 que recibe el modelo y las activaciones al inicio, a la mitad y al final de la red. Luego el mapa de calor Grad-CAM por categoría y, por último, la oclusión interactiva (tapar partes de la foto y ver cómo cambia la respuesta).

## Cómo funciona

```
artefactos/modelo_final.keras  ──(npm run build: scripts/convertir_modelo.mjs)──►  public/model/model.json + *.bin
artefactos/clases.json, metricas.json, config_final.json  ──────────────────────►  public/data/
public/index.html + styles.css + app.js  ──►  TensorFlow.js carga el modelo y clasifica en el navegador
```

- El conversor lee el archivo `.keras` de Keras 3 (un zip con `config.json` y `model.weights.h5`) y escribe el formato *layers-model* de TensorFlow.js. Soporta la CNN del proyecto y el modelo de transfer learning con MobileNetV2.
- Keras 3 guarda los pesos con el nombre de la clase de cada capa en snake_case y un sufijo por orden (`layers/functional/layers/batch_normalization_10/vars`), no con el nombre de la capa. El conversor reconstruye esas rutas recorriendo la configuración en el mismo orden.
- **Verificación de la conversión:** evaluados en TensorFlow.js sobre las 5.916 imágenes de prueba (misma partición y preprocesamiento), los dos modelos reproducen exactamente las métricas de Keras:
  - **CNN aprobada:** accuracy 0,849561 y F1-macro 0,790679.
  - **MobileNetV2**, el candidato evaluado en la Unidad 3: accuracy 0,886072 y F1-macro 0,840151.

  Para usar MobileNetV2 en la página basta con copiar `artefactos/unidad3/mobilenetv2.keras` como `artefactos/modelo_final.keras`, junto con sus métricas.
- **Preprocesamiento:** es el mismo de la app Gradio: RGB, relleno blanco hasta la proporción 3:4, 80 × 60 con vecino más cercano y escala a [0, 1]. En las pruebas, ambas apps dan píxeles idénticos.
- `public/model/` y `public/data/` ya están generados con el modelo actual. La página funciona aunque el *build* no encuentre los artefactos.

## Desplegar en Vercel (desde GitHub)

1. Sube el repositorio a GitHub con esta carpeta `web/` incluida.
2. En [vercel.com](https://vercel.com), entra con tu cuenta de GitHub → **Add New… → Project** → importa el repositorio.
3. En **Root Directory** elige `web`. El resto lo toma de `vercel.json` (framework *Other*, `npm run build`, salida `public`).
4. Pulsa **Deploy**. Obtendrás una URL del tipo `https://<proyecto>.vercel.app`.

Cada `git push` vuelve a desplegar. En el *build*, Vercel instala dos dependencias pequeñas (`fflate` y `h5wasm`) en sus servidores y convierte `../artefactos/modelo_final.keras`. En tu computador no se instala nada.

> Para que el *build* pueda leer `../artefactos`, la opción **Settings → Build and Deployment → Root Directory → "Include files outside the root directory in the Build Step"** debe estar activada (lo está por defecto). Si no lo está, el *build* avisa en el log y publica el modelo ya convertido que está en `public/model/`.

### Cambiar el modelo de la página

1. Reemplaza `artefactos/modelo_final.keras`, `clases.json` y `metricas.json`. Opcionalmente, agrega `config_final.json` con `umbral_revision`.
2. Haz `git push`. Vercel convierte el modelo nuevo y publica la página actualizada. La sección *Dentro del modelo* se adapta sola a la arquitectura: CNN o MobileNetV2.

### Si prefieres un repositorio solo para la web

Copia la carpeta `web/` como raíz de un repositorio nuevo y pon dentro una carpeta `artefactos/` con los archivos del modelo. El conversor busca primero `../artefactos` y luego `./artefactos`. En Vercel deja **Root Directory** vacío.

## Estructura

```
web/
├── package.json             build: node scripts/convertir_modelo.mjs
├── package-lock.json
├── vercel.json              framework Other, salida public/
├── scripts/
│   └── convertir_modelo.mjs  .keras (Keras 3) -> TensorFlow.js
└── public/                  lo que se publica
    ├── index.html
    ├── styles.css
    ├── app.js               clasificación, CSV y ejemplos
    ├── dentro.js            sección "Dentro del modelo" (activaciones, Grad-CAM, oclusión)
    ├── ejemplos/            12 imágenes del conjunto de prueba + ejemplos.json
    ├── data/                clases.json, metricas.json, config.json
    └── model/               model.json + group1-shard*.bin
```

## Notas

- TensorFlow.js (versión 4.22.0) se carga desde el CDN de jsDelivr. La primera visita descarga unos 5 MB de pesos (CNN), que luego quedan en caché. Con MobileNetV2 serían unos 9 MB.
- El modelo se entrenó con fotos de catálogo (producto centrado sobre fondo blanco). Con fotos de otro estilo (personas, fondos con objetos, ángulos raros) las predicciones son menos confiables; para eso está la marca de revisión manual.
