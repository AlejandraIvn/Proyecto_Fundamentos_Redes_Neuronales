"""Interfaz: subir una o varias imágenes (hasta 40) en la misma sección, predecir y mostrar métricas.

Uso desde la raíz del repositorio:
    python app/app.py            # local
    python app/app.py --share    # enlace público temporal (útil en Google Colab)

Lee los artefactos generados por el pipeline (src.entrenar y src.evaluar) o por el notebook:
    artefactos/modelo_final.keras   modelo entrenado
    artefactos/clases.json          orden de las 40 clases de salida
    artefactos/metricas.json        métricas globales en el conjunto de prueba
    artefactos/config_final.json    arquitectura elegida y umbral de revisión manual (opcional)
"""

import base64
import csv
import html
import io
import json
import sys
import tempfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, UnidentifiedImageError

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

CARPETA_ARTEFACTOS = RAIZ / "artefactos"
RUTA_MODELO = CARPETA_ARTEFACTOS / "modelo_final.keras"
RUTA_CLASES = CARPETA_ARTEFACTOS / "clases.json"
RUTA_METRICAS = CARPETA_ARTEFACTOS / "metricas.json"
RUTA_CONFIG = CARPETA_ARTEFACTOS / "config_final.json"

try:  # mismo tamaño que el pipeline; valor de respaldo si TensorFlow no carga
    from src.datos import IMG_SIZE
except Exception:  # noqa: BLE001
    IMG_SIZE = (80, 60)  # (alto, ancho)

TOP_K = 5
MAX_LOTE = 40              # máximo de imágenes por lote en la clasificación masiva
UMBRAL_POR_DEFECTO = 0.5   # se usa si config_final.json no define umbral_revision


def cargar_config():
    """Configuración del modelo final (la escribe el notebook); vacía si no existe."""
    try:
        with open(RUTA_CONFIG, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


CONFIG = cargar_config()
UMBRAL_REVISION = float(CONFIG.get("umbral_revision", UMBRAL_POR_DEFECTO))
ES_TRANSFER = CONFIG.get("arquitectura") == "mobilenetv2"

# Nombres y descripciones legibles para las claves que escribe src/evaluar.py.
# Las claves desconocidas se muestran igual, con un nombre derivado de la clave.
DESCRIPCION_METRICAS = {
    "accuracy": ("Accuracy", "Porcentaje de imágenes de prueba cuya categoría se predijo correctamente.", True),
    "f1_macro": ("F1-macro", "Promedio del F1 de cada clase, con el mismo peso para todas. Fue el criterio de selección del modelo.", True),
    "f1_weighted": ("F1-weighted", "Promedio del F1 por clase ponderado según cuántas imágenes tiene cada clase.", True),
    "n_prueba": ("Imágenes de prueba", "Imágenes del conjunto de prueba, que el modelo no vio durante el entrenamiento.", False),
}


# --------------------------------------------------------------------------- #
# Lógica: artefactos e inferencia
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def cargar_modelo():
    import tensorflow as tf  # import diferido: la página carga aunque TF tarde
    return tf.keras.models.load_model(RUTA_MODELO)


@lru_cache(maxsize=1)
def cargar_clases():
    with open(RUTA_CLASES, encoding="utf-8") as f:
        return json.load(f)


def cargar_metricas():
    with open(RUTA_METRICAS, encoding="utf-8") as f:
        return json.load(f)


def preprocesar(ruta):
    """Replica el preprocesamiento de evaluación (generador_evaluacion en src/datos.py):
    RGB, redimensionado a 80 x 60 con interpolación 'nearest' (la de Keras por defecto)
    y escalado a [0, 1].

    Para fotos que no vienen del catálogo: las zonas transparentes se ponen en blanco y, si la foto
    no tiene la proporción 3:4 del dataset (60 x 80), se rellena con blanco en lugar de deformarla.
    Las imágenes del dataset ya tienen esa proporción, así que para ellas no cambia nada.
    """
    with Image.open(ruta) as img:
        img.load()
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            img = Image.new("RGB", rgba.size, "white")
            img.paste(rgba, mask=rgba.split()[3])
        else:
            img = img.convert("RGB")
    alto, ancho = IMG_SIZE
    w, h = img.size
    if w * alto != h * ancho:
        lienzo = (w, round(w * alto / ancho)) if w * alto > h * ancho else (round(h * ancho / alto), h)
        fondo = Image.new("RGB", lienzo, "white")
        fondo.paste(img, ((lienzo[0] - w) // 2, (lienzo[1] - h) // 2))
        img = fondo
    img = img.resize((ancho, alto), Image.NEAREST)
    arreglo = np.asarray(img, dtype=np.float32) / 255.0
    return arreglo[np.newaxis, ...]


def predecir_lote(rutas):
    """Clasifica varias imágenes en una sola pasada del modelo.

    Devuelve una fila por imagen, en el orden recibido. Las imágenes que no se pueden leer quedan
    con estado de error y no detienen el resto del lote.
    """
    filas, validas, tensores = [], [], []
    for ruta in rutas:
        nombre = Path(ruta).name
        if Path(ruta).suffix.lower() not in EXTENSIONES:
            filas.append({"nombre_imagen": nombre, "estado": "formato no admitido"})
            continue
        try:
            tensores.append(preprocesar(ruta))
            validas.append(len(filas))
            filas.append({"nombre_imagen": nombre, "ruta": ruta, "estado": "ok"})
        except (UnidentifiedImageError, OSError):
            filas.append({"nombre_imagen": nombre, "estado": "imagen no válida"})
    if tensores:
        probs = cargar_modelo().predict(np.concatenate(tensores), verbose=0)
        clases = cargar_clases()
        if probs.shape[1] != len(clases):
            raise ValueError("La salida del modelo no coincide con clases.json.")
        for k, p in zip(validas, probs):
            orden = np.argsort(p)[::-1]
            filas[k].update({
                "categoria_predicha": clases[orden[0]], "probabilidad": float(p[orden[0]]),
                "segunda_categoria": clases[orden[1]], "probabilidad_segunda": float(p[orden[1]]),
                "requiere_revision": bool(p[orden[0]] < UMBRAL_REVISION),
                "top": [(clases[j], float(p[j])) for j in orden[:TOP_K]],
            })
    return filas


def guardar_csv_lote(filas):
    """Escribe el CSV del lote en una carpeta temporal y devuelve su ruta."""
    columnas = ["nombre_imagen", "categoria_predicha", "probabilidad", "segunda_categoria",
                "probabilidad_segunda", "requiere_revision", "estado"]
    sello = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = Path(tempfile.mkdtemp(prefix="lote_")) / f"clasificacion_lote_{sello}.csv"
    with open(ruta, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: Excel respeta las tildes
        escritor = csv.DictWriter(f, fieldnames=columnas, extrasaction="ignore")
        escritor.writeheader()
        for fila in filas:
            salida = dict(fila)
            for clave in ("probabilidad", "probabilidad_segunda"):
                if clave in salida:
                    salida[clave] = f"{salida[clave]:.4f}"
            if "requiere_revision" in salida:
                salida["requiere_revision"] = "si" if salida["requiere_revision"] else "no"
            escritor.writerow(salida)
    return str(ruta)


# --------------------------------------------------------------------------- #
# Fragmentos HTML
# --------------------------------------------------------------------------- #

def _pct(valor):
    return f"{valor * 100:.1f}".replace(".", ",") + " %"


def html_estado_vacio():
    return """
    <div class="res-empty">
      <div class="res-empty-icon">
        <svg viewBox="0 0 24 24" width="34" height="34" fill="none" stroke="currentColor" stroke-width="1.6">
          <path d="M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2"/>
          <circle cx="12" cy="12" r="3.5"/>
        </svg>
      </div>
      <h4>Los resultados aparecerán aquí</h4>
      <p>Carga una o varias fotos (hasta 40) y pulsa <b>Clasificar</b>. Con una imagen verás las cinco
      categorías más probables; con varias, una tabla del lote. En ambos casos podrás descargar el CSV.</p>
    </div>"""


def html_cargando(n=1):
    titulo = "Analizando la imagen…" if n == 1 else f"Clasificando {n} imágenes…"
    return f"""
    <div class="res-empty res-loading">
      <div class="scanner"><span></span></div>
      <h4>{titulo}</h4>
      <p>Redimensionando a 80 x 60 píxeles y calculando las probabilidades de las 40 categorías.</p>
    </div>"""


def html_error(titulo, detalle):
    return f"""
    <div class="res-error" role="alert">
      <div class="res-error-icon">!</div>
      <div><h4>{html.escape(titulo)}</h4><p>{html.escape(detalle)}</p></div>
    </div>"""


def html_resultados(fila):
    """Resultado de una sola imagen: miniatura, categoría predicha y top 5."""
    top = fila["top"]
    clase, prob = top[0]
    mini = _miniatura(fila["ruta"], (84, 112))
    imagen = f'<img src="{mini}" alt="">' if mini else ""
    estado = ('<span class="tag tag-warn">Revisión manual</span>' if fila["requiere_revision"]
              else '<span class="tag tag-ok">Automática</span>')
    filas = []
    for i, (c, p) in enumerate(top):
        filas.append(f"""
        <li class="bar-row{' is-top' if i == 0 else ''}" style="--d:{i * 70}ms">
          <div class="bar-head">
            <span class="bar-rank">{i + 1}</span>
            <span class="bar-name">{html.escape(c)}</span>
            <span class="bar-val">{_pct(p)}</span>
          </div>
          <div class="bar-track"><div class="bar-fill" style="--w:{max(p * 100, 0.6):.2f}%"></div></div>
        </li>""")
    return f"""
    <div class="res-card">
      <div class="res-main">
        {imagen}
        <div>
          <span class="res-label">Categoría predicha</span>
          <div class="res-class">{html.escape(clase)}</div>
          <div class="res-prob"><span>{_pct(prob)}</span> de probabilidad asignada por el modelo</div>
          <span class="res-file">{html.escape(fila["nombre_imagen"])}</span>
          {estado}
        </div>
      </div>
      <div class="res-top">
        <span class="res-label">Top 5 categorías más probables</span>
        <ul class="bars">{''.join(filas)}</ul>
      </div>
      <p class="res-foot">Las probabilidades indican cómo reparte el modelo su salida entre las 40 categorías.
      No son una medida de acierto para esta imagen. Por debajo de {_pct(UMBRAL_REVISION)} la predicción
      se marca para revisión manual.</p>
    </div>"""


def html_metricas():
    try:
        metricas = cargar_metricas()
    except FileNotFoundError:
        return html_error("Métricas no disponibles",
                          "No se encontró artefactos/metricas.json. Ejecuta python -m src.evaluar.")
    except (json.JSONDecodeError, OSError) as e:
        return html_error("No se pudieron leer las métricas", str(e))

    tarjetas = []
    for clave, valor in metricas.items():
        nombre, desc, es_tasa = DESCRIPCION_METRICAS.get(
            clave, (clave.replace("_", " ").capitalize(), "", isinstance(valor, float) and 0 <= valor <= 1))
        if isinstance(valor, (int, float)) and es_tasa:
            texto = _pct(valor)
            medidor = f'<div class="m-gauge"><div style="width:{valor * 100:.2f}%"></div></div>'
        elif isinstance(valor, (int, float)):
            texto = f"{valor:,}".replace(",", ".") if isinstance(valor, int) else f"{valor:.4f}"
            medidor = ""
        else:
            texto, medidor = html.escape(str(valor)), ""
        tarjetas.append(f"""
        <div class="m-card{' m-count' if not es_tasa else ''}">
          <span class="m-name">{html.escape(nombre)}</span>
          <span class="m-value">{texto}</span>
          {medidor}
          <p class="m-desc">{html.escape(desc)}</p>
          <code class="m-key">{html.escape(clave)}</code>
        </div>""")
    return f'<div class="m-grid">{"".join(tarjetas)}</div>'


def _miniatura(ruta, tamano=(72, 96)):
    """Miniatura en base64 (no depende de que Gradio sirva el archivo); fondo blanco si es PNG transparente."""
    try:
        with Image.open(ruta) as img:
            img = img.convert("RGBA")
            fondo = Image.new("RGB", img.size, "white")
            fondo.paste(img, mask=img.split()[3])
            fondo.thumbnail(tamano)
            buffer = io.BytesIO()
            fondo.save(buffer, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    except Exception:  # noqa: BLE001
        return ""


def html_lote(filas):
    ok = [f for f in filas if f["estado"] == "ok"]
    revision = [f for f in ok if f["requiere_revision"]]
    errores = len(filas) - len(ok)
    conteo = {}
    for f in ok:
        conteo[f["categoria_predicha"]] = conteo.get(f["categoria_predicha"], 0) + 1
    top_cats = sorted(conteo.items(), key=lambda kv: -kv[1])[:6]

    tarjetas = [
        ("Imágenes procesadas", f"{len(filas)}", ""),
        ("Clasificadas automáticamente", f"{len(ok) - len(revision)}", "is-ok"),
        ("Para revisión manual", f"{len(revision)}", "is-warn"),
        ("Con error de lectura", f"{errores}", "is-err" if errores else ""),
    ]
    resumen = "".join(f'<div class="lote-stat {c}"><span>{html.escape(t)}</span><b>{v}</b></div>'
                      for t, v, c in tarjetas)
    chips = "".join(f'<span class="chip">{html.escape(c)} <b>{n}</b></span>' for c, n in top_cats)

    filas_html = []
    for i, f in enumerate(filas, start=1):
        mini = _miniatura(f["ruta"]) if f["estado"] == "ok" else ""
        img = f'<img src="{mini}" alt="">' if mini else '<div class="thumb-x">!</div>'
        nombre = (f'<small class="t-name" title="{html.escape(f["nombre_imagen"])}">'
                  f'{html.escape(f["nombre_imagen"])}</small>')
        if f["estado"] != "ok":
            celdas = f'<td class="t-err"><b>{html.escape(f["estado"].capitalize())}</b>{nombre}</td><td></td><td></td>'
        else:
            estado = ('<span class="tag tag-warn">Revisión manual</span>' if f["requiere_revision"]
                      else '<span class="tag tag-ok">Automática</span>')
            celdas = (f'<td><b>{html.escape(f["categoria_predicha"])}</b>{nombre}'
                      f'<small>2.ª opción: {html.escape(f["segunda_categoria"])} · {_pct(f["probabilidad_segunda"])}</small></td>'
                      f'<td class="t-num">{_pct(f["probabilidad"])}'
                      f'<div class="bar-track"><div class="bar-fill" style="--w:{max(f["probabilidad"] * 100, 0.6):.1f}%"></div></div></td>'
                      f'<td>{estado}</td>')
        filas_html.append(f'<tr><td class="t-idx">{i}</td><td class="t-img">{img}</td>{celdas}</tr>')

    return f"""
    <div class="res-card lote-card">
      <div class="lote-stats">{resumen}</div>
      {f'<div class="chips"><span class="res-label">Categorías más frecuentes en el lote</span>{chips}</div>' if chips else ''}
      <div class="tabla-wrap">
        <table class="lote-tabla">
          <thead><tr><th>#</th><th>Imagen</th><th>Categoría predicha</th><th>Probabilidad</th><th>Estado</th></tr></thead>
          <tbody>{''.join(filas_html)}</tbody>
        </table>
      </div>
      <p class="res-foot">Se marca para revisión manual toda predicción con probabilidad menor a
      {_pct(UMBRAL_REVISION)}. El CSV contiene el nombre de cada imagen, la categoría predicha,
      su probabilidad, la segunda opción y si requiere revisión.</p>
    </div>"""


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #

EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def _rutas(archivos):
    """gr.File puede entregar rutas (str) u objetos con .name, según la versión de Gradio."""
    return [a if isinstance(a, str) else getattr(a, "name", str(a)) for a in (archivos or [])]


def al_cambiar_archivos(archivos):
    return html_estado_vacio(), None


def mostrar_cargando(archivos):
    n = len(_rutas(archivos))
    if not n or n > MAX_LOTE:
        return html_estado_vacio()
    return html_cargando(n)


def clasificar(archivos):
    """Clasifica una o varias imágenes (máximo MAX_LOTE) y devuelve el HTML del resultado y el CSV.

    Con una imagen muestra la categoría y el top 5; con varias, el resumen y la tabla del lote.
    """
    rutas = _rutas(archivos)
    if not rutas:
        return html_error("Faltan las imágenes", "Selecciona o arrastra entre 1 y "
                          f"{MAX_LOTE} imágenes antes de clasificar."), None
    if len(rutas) > MAX_LOTE:
        return html_error("Demasiadas imágenes", f"Seleccionaste {len(rutas)} imágenes; el máximo es "
                          f"{MAX_LOTE}. Divide la carga en varios lotes."), None
    try:
        filas = predecir_lote(rutas)
    except ImportError:
        return html_error("TensorFlow no está instalado",
                          "Instala las dependencias con pip install -r requirements.txt."), None
    except Exception as e:  # noqa: BLE001
        if not RUTA_MODELO.exists():
            return html_error("Modelo no encontrado",
                              "No existe artefactos/modelo_final.keras. Ejecuta python -m src.entrenar."), None
        return html_error("Error durante la predicción", str(e)), None
    if len(filas) == 1:
        if filas[0]["estado"] != "ok":
            detalle = ("Usa una imagen JPG, PNG, WEBP, BMP o GIF." if filas[0]["estado"] == "formato no admitido"
                       else "No se pudo leer el archivo como imagen. Prueba con otro archivo.")
            return html_error("No se pudo clasificar la imagen", detalle), None
        return html_resultados(filas[0]), guardar_csv_lote(filas)
    return html_lote(filas), guardar_csv_lote(filas)


# --------------------------------------------------------------------------- #
# Secciones estáticas
# --------------------------------------------------------------------------- #

NAV = """
<header class="nav">
  <a class="brand" href="#inicio">
    <span class="brand-mark">
      <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3"/>
        <path d="M9 9l3-2 3 2v6H9z"/>
      </svg>
    </span>
    Fashion<span>Vision</span>
  </a>
  <nav class="nav-links">
    <a href="#inicio">Inicio</a>
    <a href="#modelo">Conoce el modelo</a>
    <a href="#probar">Probar modelo</a>
    <a href="#metricas">Métricas</a>
  </nav>
  <a class="nav-cta" href="#probar">Probar ahora</a>
</header>
"""

HERO_SVG = """
<svg class="hero-svg" viewBox="0 0 440 400" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <defs>
    <linearGradient id="g1" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#22d3ee"/><stop offset="1" stop-color="#8b5cf6"/>
    </linearGradient>
    <linearGradient id="g2" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#1e293b"/><stop offset="1" stop-color="#0f172a"/>
    </linearGradient>
    <pattern id="grid" width="16" height="16" patternUnits="userSpaceOnUse">
      <path d="M16 0H0V16" fill="none" stroke="#334155" stroke-width=".6"/>
    </pattern>
  </defs>
  <rect x="20" y="20" width="270" height="340" rx="22" fill="url(#g2)" stroke="#334155"/>
  <rect x="38" y="56" width="234" height="250" rx="14" fill="url(#grid)" opacity=".7"/>
  <circle cx="58" cy="38" r="4" fill="#f87171"/><circle cx="72" cy="38" r="4" fill="#fbbf24"/><circle cx="86" cy="38" r="4" fill="#34d399"/>
  <!-- prenda: camiseta -->
  <path d="M110 110l35-22h20a20 12 0 0 0 40 0h20l35 22-18 34-20-10v112h-94V134l-20 10z"
        fill="#e2e8f0" opacity=".92"/>
  <path d="M165 88a20 12 0 0 0 40 0" fill="none" stroke="#94a3b8" stroke-width="2"/>
  <!-- marco de detección -->
  <g stroke="url(#g1)" stroke-width="3" fill="none" stroke-linecap="round">
    <path d="M86 92V76h16M268 92V76h-16M86 262v16h16M268 262v16h-16"/>
  </g>
  <rect class="scanline" x="86" y="80" width="182" height="3" fill="url(#g1)" opacity=".85"/>
  <rect x="38" y="318" width="120" height="10" rx="5" fill="#334155"/>
  <rect x="38" y="318" width="84" height="10" rx="5" fill="url(#g1)"/>
  <!-- tarjeta de predicción -->
  <g class="float-card">
    <rect x="232" y="150" width="190" height="148" rx="16" fill="#0b1224" stroke="#3b4a6b"/>
    <text x="250" y="178" fill="#94a3b8" font-size="11" font-family="Inter, sans-serif" letter-spacing="1">TOP 5 · EJEMPLO</text>
    <rect x="250" y="192" width="154" height="8" rx="4" fill="#1e293b"/><rect x="250" y="192" width="128" height="8" rx="4" fill="url(#g1)"/>
    <rect x="250" y="212" width="154" height="8" rx="4" fill="#1e293b"/><rect x="250" y="212" width="52" height="8" rx="4" fill="#6366f1" opacity=".7"/>
    <rect x="250" y="232" width="154" height="8" rx="4" fill="#1e293b"/><rect x="250" y="232" width="30" height="8" rx="4" fill="#6366f1" opacity=".55"/>
    <rect x="250" y="252" width="154" height="8" rx="4" fill="#1e293b"/><rect x="250" y="252" width="18" height="8" rx="4" fill="#6366f1" opacity=".45"/>
    <rect x="250" y="272" width="154" height="8" rx="4" fill="#1e293b"/><rect x="250" y="272" width="10" height="8" rx="4" fill="#6366f1" opacity=".35"/>
  </g>
  <g class="float-chip">
    <rect x="300" y="66" width="122" height="44" rx="22" fill="#0b1224" stroke="#22d3ee" stroke-opacity=".6"/>
    <circle cx="324" cy="88" r="8" fill="url(#g1)"/>
    <text x="340" y="93" fill="#e2e8f0" font-size="14" font-weight="600" font-family="Inter, sans-serif">Tshirts</text>
  </g>
</svg>
"""

HERO = f"""
<section id="inicio" class="hero">
  <div class="hero-glow"></div>
  <div class="hero-grid">
    <div class="hero-copy">
      <span class="pill"><i></i> Visión artificial para e-commerce de moda</span>
      <h1>Cataloga tu moda <span class="grad">con solo una foto</span></h1>
      <p class="lead">Una red neuronal convolucional que mira la imagen de un producto y predice
      su categoría entre 40 tipos de artículo, para automatizar la catalogación de tu tienda.</p>
      <div class="hero-actions">
        <a class="btn btn-primary" href="#probar">Probar modelo
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M13 6l6 6-6 6"/></svg></a>
        <a class="btn btn-ghost" href="#modelo">Cómo funciona</a>
      </div>
      <div class="hero-stats">
        <div><b>40</b><span>categorías</span></div>
        <div><b>39.437</b><span>imágenes depuradas</span></div>
        <div><b>Top 5</b><span>con probabilidades</span></div>
      </div>
    </div>
    <div class="hero-art">{HERO_SVG}</div>
  </div>
</section>
"""

ICONO = {
    "upload": '<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>',
    "layers": '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>',
    "chart": '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="M8 12l3 3 5-6"/>',
}


def _icono(nombre):
    return (f'<svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{ICONO[nombre]}</svg>')


if ES_TRANSFER:  # el texto sigue al modelo elegido en config_final.json
    PASO_PROCESAMIENTO = ("MobileNetV2, preentrenada en ImageNet y ajustada a este catálogo, reescala la imagen y "
                          "detecta bordes, texturas y formas cada vez más complejas.")
    PASO_CLASIFICACION = ("Un promedio global resume esas características y la salida softmax asigna una "
                          "probabilidad a cada una de las 40 categorías.")
else:
    PASO_PROCESAMIENTO = ("Tres bloques convolucionales (32, 64 y 128 filtros) detectan bordes, texturas y formas "
                          "cada vez más complejas.")
    PASO_CLASIFICACION = ("Una capa densa combina esas características y la salida softmax asigna una "
                          "probabilidad a cada una de las 40 categorías.")

PASOS = [
    ("upload", "Entrada", "Cargas la foto de un producto. Se convierte a RGB, se reduce a 80 x 60 píxeles y sus valores se normalizan entre 0 y 1."),
    ("layers", "Procesamiento", PASO_PROCESAMIENTO),
    ("chart", "Clasificación", PASO_CLASIFICACION),
    ("check", "Resultado", "Se muestra la categoría con mayor probabilidad y las cinco alternativas más probables."),
]

MODELO = f"""
<section id="modelo" class="about">
  <div class="sec-head">
    <span class="eyebrow">Conoce el modelo</span>
    <h2>De la foto a la categoría en cuatro pasos</h2>
    <p>Las tiendas en línea deben etiquetar miles de productos. Este modelo aprende a reconocer el tipo de
    artículo (camisetas, relojes, sandalias, bolsos…) a partir de su imagen, para que la catalogación
    no dependa solo del trabajo manual.</p>
  </div>
  <div class="about-cards">
    <div class="info-card">
      <h3>¿Qué es?</h3>
      <p>Una red neuronal convolucional (CNN) entrenada con el dataset público <i>Fashion Product Images (Small)</i>
      de Kaggle, depurado a 39.437 imágenes sin duplicados.</p>
    </div>
    <div class="info-card">
      <h3>¿Qué resuelve?</h3>
      <p>Predice el <code>articleType</code> de un producto entre las 40 clases con al menos 200 imágenes,
      que cubren el 90,3 % del catálogo original.</p>
    </div>
  </div>
  <ol class="steps">
    {''.join(f'''<li class="step"><span class="step-n">0{i + 1}</span><div class="step-ic">{_icono(ic)}</div>
    <h4>{t}</h4><p>{d}</p></li>''' for i, (ic, t, d) in enumerate(PASOS))}
  </ol>
</section>
"""

PROBAR_HEAD = f"""
<div id="probar" class="sec-head sec-head-center">
  <span class="eyebrow">Probar modelo</span>
  <h2>Clasifica uno o varios productos</h2>
  <p>Sube una foto o un lote de hasta {MAX_LOTE}, como haría un vendedor al publicar. Funciona mejor con fotos de
  catálogo: un solo producto, centrado y con fondo claro, como las imágenes del dataset.</p>
</div>
"""

METRICAS_HEAD = """
<div id="metricas" class="sec-head">
  <span class="eyebrow">Rendimiento del modelo</span>
  <h2>Cómo se desempeña en datos que nunca vio</h2>
  <p>Métricas globales calculadas sobre el conjunto de prueba (15 % de los datos, separado antes del entrenamiento).
  Se leen directamente de <code>artefactos/metricas.json</code>.</p>
</div>
"""

NOTA = """
<div class="note">
  <div class="note-ic">i</div>
  <div>
    <h4>Métricas globales, no por imagen</h4>
    <p>Estas cifras resumen la evaluación del modelo sobre todo el conjunto de prueba. Para la imagen que tú cargas
    no existe una métrica de acierto: el modelo no conoce su categoría real. La probabilidad que ves junto a cada
    predicción es la salida del modelo, no una medida de que la predicción sea correcta.</p>
  </div>
</div>
"""

FOOTER = """
<footer class="foot">
  <div><b>Fashion<span>Vision</span></b> · Proyecto integrador de Fundamentos de Redes Neuronales</div>
  <div>Julian · Kinan · Ivanna</div>
</footer>
"""

# --------------------------------------------------------------------------- #
# Estilos
# --------------------------------------------------------------------------- #

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

:root {
  --bg: #070b18; --bg2: #0c1326; --panel: #111a33; --line: #23304f;
  --text: #e6ebf5; --muted: #94a3b8; --cyan: #22d3ee; --violet: #8b5cf6; --indigo: #6366f1;
  --grad: linear-gradient(120deg, #22d3ee, #6366f1 55%, #a855f7);
  --radius: 18px;
}
body, gradio-app, .gradio-container { background: var(--bg) !important; }
.gradio-container { max-width: 100% !important; padding: 0 !important; margin: 0 !important;
  font-family: 'Inter', system-ui, sans-serif !important; color: var(--text); }
.gradio-container .main, .gradio-container .wrap, .gradio-container > .main > .wrap { padding: 0 !important; }
.gradio-container .contain, .gradio-container .column, .gradio-container .form { gap: 0; }
html { scroll-behavior: smooth; }
footer.svelte-1ax1toq, .gradio-container > footer { display: none !important; }
.prose, .html-container { padding: 0 !important; }
a { text-decoration: none; }

.page { max-width: 1180px; margin: 0 auto; padding: 0 24px; }

/* NAV */
.nav { position: sticky; top: 0; z-index: 50; display: flex; align-items: center; justify-content: space-between;
  gap: 16px; padding: 14px max(24px, calc((100% - 1180px) / 2)); background: rgba(7, 11, 24, .78);
  backdrop-filter: blur(14px); border-bottom: 1px solid rgba(148, 163, 184, .12); }
.brand { display: flex; align-items: center; gap: 10px; color: var(--text) !important; font: 700 19px 'Space Grotesk', sans-serif; }
.brand span { color: var(--cyan); }
.brand-mark { display: grid; place-items: center; width: 32px; height: 32px; border-radius: 10px; background: var(--grad); color: #fff !important; }
.nav-links { display: flex; gap: 6px; }
.nav-links a { color: var(--muted) !important; font-size: 14px; font-weight: 500; padding: 8px 14px; border-radius: 999px; transition: .2s; }
.nav-links a:hover { color: var(--text) !important; background: rgba(148, 163, 184, .1); }
.nav-cta { color: #fff !important; font-size: 14px; font-weight: 600; padding: 9px 18px; border-radius: 999px;
  background: var(--grad); box-shadow: 0 6px 20px -6px rgba(99, 102, 241, .7); transition: .2s; }
.nav-cta:hover { transform: translateY(-1px); box-shadow: 0 10px 26px -6px rgba(99, 102, 241, .9); }

/* HERO */
.hero { position: relative; overflow: hidden; padding: 84px 24px 96px;
  background: radial-gradient(1200px 500px at 85% -10%, rgba(139, 92, 246, .22), transparent 60%),
              radial-gradient(900px 500px at 0% 100%, rgba(34, 211, 238, .14), transparent 60%), var(--bg); }
.hero-glow { position: absolute; inset: 0; background-image: linear-gradient(rgba(148,163,184,.06) 1px, transparent 1px),
  linear-gradient(90deg, rgba(148,163,184,.06) 1px, transparent 1px); background-size: 44px 44px;
  mask-image: radial-gradient(ellipse at 60% 40%, #000 20%, transparent 70%); pointer-events: none; }
.hero-grid { position: relative; max-width: 1180px; margin: 0 auto; display: grid; grid-template-columns: 1.05fr .95fr; gap: 48px; align-items: center; }
.pill { display: inline-flex; align-items: center; gap: 8px; padding: 7px 14px; border-radius: 999px; font-size: 13px; color: #c7d2fe;
  background: rgba(99, 102, 241, .12); border: 1px solid rgba(99, 102, 241, .35); }
.pill i { width: 7px; height: 7px; border-radius: 50%; background: var(--cyan); box-shadow: 0 0 10px var(--cyan); }
.hero h1 { font: 700 clamp(38px, 5.4vw, 64px)/1.04 'Space Grotesk', sans-serif; letter-spacing: -.02em; margin: 22px 0 18px; color: #fff; }
.hero h1 .grad { background-image: linear-gradient(120deg, #22d3ee, #6366f1 55%, #a855f7) !important;
  -webkit-background-clip: text !important; background-clip: text !important; color: transparent !important; -webkit-text-fill-color: transparent; }
.lead { font-size: 18px; line-height: 1.6; color: var(--muted); max-width: 540px; margin: 0; }
.hero-actions { display: flex; flex-wrap: wrap; gap: 12px; margin: 32px 0 40px; }
.btn { display: inline-flex; align-items: center; gap: 8px; padding: 14px 24px; border-radius: 12px; font-weight: 600; font-size: 15px; transition: .2s; }
.btn-primary { color: #fff !important; background: var(--grad); box-shadow: 0 12px 30px -10px rgba(99, 102, 241, .8); }
.btn-primary:hover { transform: translateY(-2px); box-shadow: 0 18px 36px -10px rgba(99, 102, 241, 1); }
.btn-ghost { color: var(--text) !important; border: 1px solid var(--line); background: rgba(17, 26, 51, .6); }
.btn-ghost:hover { border-color: var(--indigo); background: rgba(99, 102, 241, .1); }
.hero-stats { display: flex; gap: 36px; flex-wrap: wrap; padding-top: 28px; border-top: 1px solid rgba(148,163,184,.14); }
.hero-stats b { display: block; font: 700 26px 'Space Grotesk', sans-serif; color: #fff; }
.hero-stats span { font-size: 13px; color: var(--muted); }
.hero-art { position: relative; }
.hero-art::before { content: ""; position: absolute; inset: 10% 5%; background: var(--grad); filter: blur(80px); opacity: .28; border-radius: 50%; }
.hero-svg { position: relative; width: 100%; height: auto; filter: drop-shadow(0 30px 60px rgba(0,0,0,.5)); }
.scanline { animation: scan 3.2s ease-in-out infinite; }
@keyframes scan { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(190px); } }
.float-card { animation: floaty 6s ease-in-out infinite; }
.float-chip { animation: floaty 5s ease-in-out infinite reverse; }
@keyframes floaty { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }

/* SECCIONES */
.sec-head { max-width: 720px; margin-bottom: 44px; }
.sec-head-center { margin-left: auto; margin-right: auto; text-align: center; }
.eyebrow { display: inline-block; font-size: 13px; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; color: var(--cyan); }
.sec-head h2 { font: 700 clamp(28px, 3.6vw, 42px)/1.12 'Space Grotesk', sans-serif; color: #fff; margin: 12px 0 14px; letter-spacing: -.01em; }
.sec-head p { color: var(--muted); font-size: 16.5px; line-height: 1.65; margin: 0; }
code { background: rgba(99,102,241,.14); color: #c7d2fe; padding: 2px 7px; border-radius: 6px; font-size: .9em; }
section, #probar, #metricas { scroll-margin-top: 80px; }

/* CONOCE EL MODELO */
.about { max-width: 1180px; margin: 0 auto; padding: 100px 24px; }
.about-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 28px; }
.info-card { padding: 28px; border-radius: var(--radius); background: linear-gradient(160deg, rgba(34,211,238,.08), rgba(17,26,51,.6) 50%);
  border: 1px solid var(--line); }
.info-card h3 { margin: 0 0 10px; font: 600 20px 'Space Grotesk', sans-serif; color: #fff; }
.info-card p { margin: 0; color: var(--muted); line-height: 1.65; }
.steps { list-style: none; padding: 0; margin: 0; display: grid; grid-template-columns: repeat(4, 1fr); gap: 18px; counter-reset: s; position: relative; }
.step { position: relative; padding: 26px 22px; border-radius: var(--radius); background: var(--panel); border: 1px solid var(--line);
  transition: transform .25s, border-color .25s, box-shadow .25s; }
.step:hover { transform: translateY(-4px); border-color: rgba(99,102,241,.6); box-shadow: 0 20px 40px -20px rgba(99,102,241,.6); }
.step-n { position: absolute; top: 20px; right: 22px; font: 700 14px 'Space Grotesk', sans-serif; color: #475569; }
.step-ic { display: grid; place-items: center; width: 48px; height: 48px; border-radius: 14px; color: #fff; background: var(--grad); margin-bottom: 18px;
  box-shadow: 0 10px 24px -10px rgba(99,102,241,.9); }
.step h4 { margin: 0 0 8px; font-size: 17px; color: #fff; }
.step p { margin: 0; font-size: 14.5px; line-height: 1.6; color: var(--muted); }

/* PROBAR */
#zona-probar { background: linear-gradient(180deg, var(--bg2), var(--bg)); border-top: 1px solid rgba(148,163,184,.08);
  padding: 100px max(24px, calc((100% - 1180px) / 2)) 110px !important; gap: 0 !important; }
#tool { background: rgba(17, 26, 51, .7); border: 1px solid var(--line); border-radius: 24px; padding: 14px !important; gap: 14px !important;
  box-shadow: 0 40px 80px -40px rgba(0,0,0,.8), 0 0 0 1px rgba(99,102,241,.08) inset; flex-wrap: wrap; }
#tool > .column { min-width: min(100%, 320px) !important; }
.tool-bar { display: flex; align-items: center; gap: 8px; padding: 4px 6px 12px; color: var(--muted); font-size: 13px; }
.tool-bar i { width: 9px; height: 9px; border-radius: 50%; background: #334155; }
.tool-bar i:first-child { background: var(--cyan); box-shadow: 0 0 8px var(--cyan); }
.tool-bar span { margin-left: 6px; font-weight: 600; letter-spacing: .04em; }
#panel-carga, #panel-res { background: var(--bg) !important; border: 1px solid var(--line) !important; border-radius: 18px !important; padding: 18px !important; gap: 14px !important; }
#uploader { border-radius: 14px !important; border: 1.5px dashed #3b4a6b !important; background: rgba(99,102,241,.04) !important; min-height: 220px; transition: border-color .2s, background .2s; }
#uploader:hover { border-color: var(--cyan) !important; background: rgba(34,211,238,.05) !important; }
#csv-salida { border-radius: 14px !important; }
#btn-clasificar { background: var(--grad) !important; color: #fff !important; border: none !important; border-radius: 12px !important;
  font-weight: 600 !important; font-size: 16px !important; padding: 14px !important; box-shadow: 0 12px 28px -12px rgba(99,102,241,.9); transition: .2s !important; }
#btn-clasificar:hover { transform: translateY(-1px); filter: brightness(1.08); }
.hint { font-size: 13px; color: var(--muted); display: flex; gap: 8px; align-items: center; }
.hint b { color: var(--text); font-weight: 500; }
.panel-title { font: 600 15px 'Space Grotesk', sans-serif; color: #fff; display: flex; align-items: center; gap: 8px; }
.panel-title em { font-style: normal; font-size: 12px; font-weight: 500; color: var(--cyan); background: rgba(34,211,238,.1);
  border: 1px solid rgba(34,211,238,.3); padding: 2px 9px; border-radius: 999px; }

/* RESULTADOS */
.res-empty { min-height: 380px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 30px; color: var(--muted); }
.res-empty-icon { width: 72px; height: 72px; border-radius: 20px; display: grid; place-items: center; color: var(--cyan);
  background: rgba(34,211,238,.08); border: 1px solid rgba(34,211,238,.25); margin-bottom: 18px; }
.res-empty h4 { color: var(--text); font-size: 17px; margin: 0 0 8px; }
.res-empty p { margin: 0; max-width: 340px; line-height: 1.6; font-size: 14.5px; }
.scanner { position: relative; width: 84px; height: 84px; border-radius: 18px; border: 2px solid rgba(99,102,241,.5); overflow: hidden; margin-bottom: 20px; }
.scanner span { position: absolute; left: 0; right: 0; height: 3px; background: var(--grad); box-shadow: 0 0 14px var(--cyan); animation: scan2 1.3s ease-in-out infinite; }
@keyframes scan2 { 0%, 100% { top: 4px; } 50% { top: 76px; } }
.res-error { display: flex; gap: 14px; align-items: flex-start; padding: 18px; border-radius: 14px; background: rgba(248,113,113,.08); border: 1px solid rgba(248,113,113,.35); }
.res-error-icon { flex: none; width: 30px; height: 30px; border-radius: 50%; display: grid; place-items: center; font-weight: 700; color: #fff; background: #ef4444; }
.res-error h4 { margin: 2px 0 4px; color: #fecaca; font-size: 15.5px; }
.res-error p { margin: 0; color: #fca5a5; font-size: 14px; line-height: 1.5; }
.res-card { animation: fadeUp .45s ease both; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
.res-label { display: block; font-size: 12px; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }
.res-main { display: flex; gap: 18px; align-items: center; padding: 20px; border-radius: 16px; margin-bottom: 22px; position: relative; overflow: hidden;
  background: linear-gradient(135deg, rgba(34,211,238,.16), rgba(139,92,246,.2)); border: 1px solid rgba(139,92,246,.45); }
.res-main::after { content: ""; position: absolute; width: 180px; height: 180px; right: -60px; top: -60px; border-radius: 50%; background: var(--grad); opacity: .25; filter: blur(30px); }
.res-main img { flex: none; width: 84px; height: 112px; object-fit: contain; border-radius: 12px; background: #fff; position: relative; z-index: 1; }
.res-main > div { position: relative; z-index: 1; min-width: 0; }
.res-file { display: block; margin: 6px 0 10px; font-size: 12.5px; color: #94a3b8; font-family: ui-monospace, monospace; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.res-class { font: 700 clamp(30px, 4vw, 40px)/1.1 'Space Grotesk', sans-serif; color: #fff; margin-bottom: 8px; }
.res-prob { color: #cbd5e1; font-size: 14.5px; }
.res-prob span { font-weight: 700; color: var(--cyan); font-size: 16px; }
.bars { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 14px; }
.bar-row { animation: fadeUp .4s ease both; animation-delay: var(--d); }
.bar-head { display: flex; align-items: center; gap: 10px; margin-bottom: 7px; font-size: 14.5px; }
.bar-rank { width: 22px; height: 22px; border-radius: 7px; display: grid; place-items: center; font-size: 12px; font-weight: 700; color: var(--muted); background: #1e293b; }
.bar-name { flex: 1; color: #cbd5e1; }
.bar-val { font-variant-numeric: tabular-nums; color: var(--muted); font-weight: 600; }
.bar-track { height: 9px; border-radius: 999px; background: #1e293b; overflow: hidden; }
.bar-fill { height: 100%; width: var(--w); border-radius: 999px; background: linear-gradient(90deg, #475569, #64748b); animation: grow .8s cubic-bezier(.2,.8,.2,1) both; animation-delay: var(--d); }
@keyframes grow { from { width: 0; } }
.is-top .bar-rank { background: var(--grad); color: #fff; }
.is-top .bar-name { color: #fff; font-weight: 600; }
.is-top .bar-val { color: var(--cyan); }
.is-top .bar-fill { background: var(--grad); box-shadow: 0 0 12px rgba(34,211,238,.5); }
.res-foot { margin: 20px 0 0; padding-top: 14px; border-top: 1px solid var(--line); font-size: 12.5px; color: #64748b; line-height: 1.5; }

/* MÉTRICAS */
#zona-metricas { padding: 100px max(24px, calc((100% - 1180px) / 2)) 90px !important; gap: 0 !important;
  background: radial-gradient(700px 400px at 100% 0%, rgba(99,102,241,.14), transparent 60%), var(--bg); }
.m-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 18px; }
.m-card { position: relative; padding: 26px 24px 22px; border-radius: var(--radius); background: var(--panel); border: 1px solid var(--line);
  display: flex; flex-direction: column; gap: 10px; transition: transform .25s, border-color .25s; overflow: hidden; }
.m-card::before { content: ""; position: absolute; left: 0; top: 0; right: 0; height: 3px; background: var(--grad); }
.m-card:hover { transform: translateY(-4px); border-color: rgba(34,211,238,.45); }
.m-count { background: linear-gradient(160deg, rgba(139,92,246,.16), var(--panel) 60%); }
.m-name { font-size: 14px; font-weight: 600; color: var(--muted); }
.m-value { font: 700 40px/1 'Space Grotesk', sans-serif; color: #fff; letter-spacing: -.01em; }
.m-gauge { height: 6px; border-radius: 999px; background: #1e293b; overflow: hidden; }
.m-gauge div { height: 100%; background: var(--grad); border-radius: 999px; }
.m-desc { margin: 0; font-size: 13.5px; line-height: 1.55; color: var(--muted); flex: 1; }
.m-key { align-self: flex-start; font-size: 11.5px; }
.note { display: flex; gap: 16px; margin-top: 28px; padding: 22px 24px; border-radius: var(--radius);
  background: rgba(251,191,36,.06); border: 1px solid rgba(251,191,36,.3); }
.note-ic { flex: none; width: 34px; height: 34px; border-radius: 50%; display: grid; place-items: center; font: 700 italic 17px Georgia, serif; color: #1f1300; background: #fbbf24; }
.note h4 { margin: 4px 0 6px; color: #fde68a; font-size: 16px; }
.note p { margin: 0; color: #d6c79a; font-size: 14.5px; line-height: 1.65; }

.foot { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px; padding: 28px max(24px, calc((100% - 1180px) / 2));
  border-top: 1px solid rgba(148,163,184,.1); color: #64748b; font-size: 13.5px; background: var(--bg); }
.foot b { color: var(--text); font-family: 'Space Grotesk', sans-serif; }
.foot b span { color: var(--cyan); }

/* LOTE (varias imágenes) */
.lote-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 16px; }
.lote-stat { padding: 14px; border-radius: 14px; background: var(--panel); border: 1px solid var(--line); display: flex; flex-direction: column; gap: 6px; }
.lote-stat span { font-size: 12px; color: var(--muted); }
.lote-stat b { font: 700 26px 'Space Grotesk', sans-serif; color: #fff; }
.lote-stat.is-ok b { color: #34d399; }
.lote-stat.is-warn b { color: #fbbf24; }
.lote-stat.is-err b { color: #f87171; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 16px; }
.chips .res-label { width: 100%; margin-bottom: 2px; }
.chip { font-size: 13px; color: #cbd5e1; padding: 5px 11px; border-radius: 999px; background: rgba(99,102,241,.12); border: 1px solid rgba(99,102,241,.35); }
.chip b { color: var(--cyan); margin-left: 4px; }
.tabla-wrap { max-height: 560px; overflow: auto; border: 1px solid var(--line); border-radius: 14px; scrollbar-color: #334155 transparent; scrollbar-width: thin; }
.lote-tabla { width: 100%; border-collapse: collapse; font-size: 14px; }
.lote-tabla thead th { position: sticky; top: 0; z-index: 1; background: var(--panel); color: var(--muted); font-size: 12px;
  font-weight: 600; letter-spacing: .06em; text-transform: uppercase; text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--line); }
.lote-tabla td { padding: 10px 12px; border-bottom: 1px solid rgba(148,163,184,.08); color: #cbd5e1; vertical-align: middle; }
.lote-tabla tr:hover td { background: rgba(99,102,241,.06); }
.lote-tabla td b { color: #fff; font-weight: 600; }
.lote-tabla small { display: block; color: #64748b; font-size: 12px; margin-top: 3px; }
.t-idx { color: #64748b !important; width: 28px; }
.t-img { width: 64px; min-width: 64px; }
.t-img img, .thumb-x { display: grid; place-items: center; width: 44px !important; max-width: none !important; height: 58px !important; object-fit: contain; border-radius: 8px; background: #fff; }
.thumb-x { background: rgba(248,113,113,.15); color: #f87171; font-weight: 700; }
.lote-tabla small.t-name { max-width: 210px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: ui-monospace, monospace; color: #94a3b8; }
.t-num { min-width: 96px; font-variant-numeric: tabular-nums; }
.t-num .bar-track { height: 6px; margin-top: 6px; }
.t-num .bar-fill { background: var(--grad); animation: none; }
.t-err { color: #fca5a5 !important; }
.tag { display: inline-block; font-size: 12px; font-weight: 600; padding: 4px 10px; border-radius: 999px; white-space: nowrap; }
.tag-ok { color: #6ee7b7; background: rgba(52,211,153,.12); border: 1px solid rgba(52,211,153,.35); }
.tag-warn { color: #fde68a; background: rgba(251,191,36,.12); border: 1px solid rgba(251,191,36,.4); }

/* RESPONSIVE */
@media (max-width: 960px) {
  .hero-grid { grid-template-columns: 1fr; }
  .hero-art { max-width: 480px; margin: 0 auto; }
  .steps { grid-template-columns: 1fr 1fr; }
  .nav-links { display: none; }
}
@media (max-width: 640px) {
  .hero { padding: 56px 16px 64px; }
  .about { padding: 72px 16px; }
  #zona-probar, #zona-metricas { padding-top: 72px !important; padding-bottom: 72px !important; padding-left: 16px !important; padding-right: 16px !important; }
  .lote-stats { grid-template-columns: 1fr 1fr; }
  .res-main { flex-direction: column; align-items: flex-start; }
  .about-cards, .steps { grid-template-columns: 1fr; }
  .hero-stats { gap: 22px; }
  .nav { padding: 12px 16px; }
  .nav-cta { padding: 8px 14px; font-size: 13px; }
  #tool { padding: 8px !important; border-radius: 18px; }
  .m-value { font-size: 34px; }
}
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
"""


# --------------------------------------------------------------------------- #
# Página
# --------------------------------------------------------------------------- #

def construir_interfaz():
    tema = gr.themes.Base(primary_hue="indigo", neutral_hue="slate").set(
        body_background_fill="#070b18", body_background_fill_dark="#070b18",
        body_text_color="#e6ebf5", body_text_color_dark="#e6ebf5",
        block_background_fill="#070b18", block_background_fill_dark="#070b18",
        block_border_width="0px", block_label_text_color="#94a3b8", block_label_text_color_dark="#94a3b8",
        block_label_background_fill="#111a33", block_label_background_fill_dark="#111a33",
        color_accent_soft="#111a33", color_accent_soft_dark="#111a33",
    )
    gradio_6 = int(gr.__version__.split(".")[0]) >= 6
    opciones = {} if gradio_6 else {"css": CSS, "theme": tema}

    with gr.Blocks(title="FashionVision · Clasificador de moda con IA", **opciones) as demo:
        gr.HTML(NAV + HERO + MODELO)

        with gr.Column(elem_id="zona-probar"):
            gr.HTML(PROBAR_HEAD)
            with gr.Row(elem_id="tool", equal_height=False):
                with gr.Column(elem_id="panel-carga", scale=4):
                    gr.HTML(f'<div class="tool-bar"><i></i><i></i><i></i><span>IMÁGENES DE ENTRADA (1 A {MAX_LOTE})</span></div>')
                    entrada = gr.File(file_count="multiple", file_types=["image"], type="filepath",
                                      label=f"Arrastra una o varias imágenes (hasta {MAX_LOTE}) o haz clic para seleccionarlas",
                                      elem_id="uploader")
                    boton = gr.Button("Clasificar", elem_id="btn-clasificar", variant="primary")
                    csv_salida = gr.File(label="CSV con la clasificación (nombre de la imagen y categoría)",
                                         interactive=False, elem_id="csv-salida")
                    gr.HTML(f'<div class="hint">Formatos: <b>JPG, PNG, WEBP</b> · Umbral de revisión manual: '
                            f'<b>{_pct(UMBRAL_REVISION)}</b></div>')
                with gr.Column(elem_id="panel-res", scale=8):
                    gr.HTML('<div class="panel-title">Resultado de la predicción <em>40 categorías</em></div>')
                    salida = gr.HTML(html_estado_vacio())

        with gr.Column(elem_id="zona-metricas"):
            gr.HTML(METRICAS_HEAD)
            gr.HTML(html_metricas)  # se lee de metricas.json en cada carga de página
            gr.HTML(NOTA)

        gr.HTML(FOOTER)

        entrada.change(al_cambiar_archivos, entrada, [salida, csv_salida], queue=False)
        boton.click(mostrar_cargando, entrada, salida, queue=False).then(clasificar, entrada, [salida, csv_salida])

    return demo, (tema if gradio_6 else None)


demo, tema_launch = construir_interfaz()

if __name__ == "__main__":
    kwargs = {"css": CSS, "theme": tema_launch} if tema_launch is not None else {}
    # allowed_paths: el CSV del lote se escribe en la carpeta temporal del sistema
    demo.queue().launch(share="--share" in sys.argv, allowed_paths=[tempfile.gettempdir()], **kwargs)
