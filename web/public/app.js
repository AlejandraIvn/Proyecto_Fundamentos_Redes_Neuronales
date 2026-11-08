/* FashionVision · clasificación de productos de moda en el navegador con TensorFlow.js
 *
 * Replica la interfaz de app/app.py:
 *   - una o varias imágenes (máximo 40) en la misma sección;
 *   - mismo preprocesamiento del pipeline: RGB, 80 x 60 con vecino más cercano y escala a [0, 1]
 *     (con relleno blanco hasta la proporción 3:4 para fotos que no la tienen);
 *   - top 5 para una imagen, tabla + resumen para varias, y CSV descargable en ambos casos;
 *   - métricas globales leídas de data/metricas.json;
 *   - botón con imágenes de ejemplo (ejemplos/ejemplos.json).
 * La sección "Dentro del modelo" está en dentro.js y usa window.FashionVision.
 * El modelo (public/model/) lo genera scripts/convertir_modelo.mjs desde artefactos/modelo_final.keras.
 */
(() => {
  "use strict";

  const MAX_LOTE = 40;
  const TOP_K = 5;
  const EXTENSIONES = /\.(jpe?g|png|webp|bmp|gif)$/i;
  const COLUMNAS_CSV = ["nombre_imagen", "categoria_predicha", "probabilidad", "segunda_categoria",
    "probabilidad_segunda", "requiere_revision", "estado"];

  const DESCRIPCION_METRICAS = {
    accuracy: ["Accuracy", "Porcentaje de imágenes de prueba cuya categoría se predijo correctamente.", true],
    f1_macro: ["F1-macro", "Promedio del F1 de cada clase, con el mismo peso para todas. Fue el criterio de selección del modelo.", true],
    f1_weighted: ["F1-weighted", "Promedio del F1 por clase ponderado según cuántas imágenes tiene cada clase.", true],
    n_prueba: ["Imágenes de prueba", "Imágenes del conjunto de prueba, que el modelo no vio durante el entrenamiento.", false],
  };

  const estado = {
    modelo: null,
    clases: [],
    config: { alto: 80, ancho: 60, umbral_revision: 0.5, arquitectura: "cnn" },
    archivos: [],   // { id, file, url }
    csvUrl: null,
    ocupado: false,
  };

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (v) => `${(v * 100).toFixed(1).replace(".", ",")} %`;
  const miles = (n) => n.toLocaleString("es-CO");

  // ------------------------------------------------------------------------- //
  // Capas propias con la misma semántica de Keras (las usa el modelo MobileNetV2)
  // ------------------------------------------------------------------------- //

  class ResizingHP extends tf.layers.Layer {
    static get className() { return "ResizingHP"; }
    constructor(config) { super(config); this.height = config.height; this.width = config.width; }
    computeOutputShape(s) { return [s[0], this.height, this.width, s[3]]; }
    call(inputs) {
      const x = Array.isArray(inputs) ? inputs[0] : inputs;
      // Keras (TF) redimensiona con half_pixel_centers = true
      return tf.tidy(() => tf.image.resizeBilinear(x, [this.height, this.width], false, true));
    }
    getConfig() { return Object.assign(super.getConfig(), { height: this.height, width: this.width }); }
  }

  class RescalingSimple extends tf.layers.Layer {
    static get className() { return "RescalingSimple"; }
    constructor(config) { super(config); this.scale = config.scale; this.offset = config.offset ?? 0; }
    computeOutputShape(s) { return s; }
    call(inputs) {
      const x = Array.isArray(inputs) ? inputs[0] : inputs;
      return tf.tidy(() => tf.add(tf.mul(x, this.scale), this.offset));
    }
    getConfig() { return Object.assign(super.getConfig(), { scale: this.scale, offset: this.offset }); }
  }

  tf.serialization.registerClass(ResizingHP);
  tf.serialization.registerClass(RescalingSimple);

  // ------------------------------------------------------------------------- //
  // Secciones estáticas
  // ------------------------------------------------------------------------- //

  const ICONOS = {
    upload: '<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>',
    layers: '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>',
    chart: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    check: '<circle cx="12" cy="12" r="9"/><path d="M8 12l3 3 5-6"/>',
  };
  const icono = (n) => `<svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${ICONOS[n]}</svg>`;

  function pintarPasos() {
    const transfer = estado.config.arquitectura === "mobilenetv2";
    const pasos = [
      ["upload", "Entrada", "Cargas la foto de uno o varios productos. Se convierten a RGB, se reducen a 80 x 60 píxeles y sus valores se normalizan entre 0 y 1."],
      ["layers", "Procesamiento", transfer
        ? "MobileNetV2, preentrenada en ImageNet y ajustada a este catálogo, reescala la imagen y detecta bordes, texturas y formas cada vez más complejas."
        : "Tres bloques convolucionales (32, 64 y 128 filtros) detectan bordes, texturas y formas cada vez más complejas."],
      ["chart", "Clasificación", transfer
        ? "Un promedio global resume esas características y la salida softmax asigna una probabilidad a cada una de las 40 categorías."
        : "Una capa densa combina esas características y la salida softmax asigna una probabilidad a cada una de las 40 categorías."],
      ["check", "Resultado", "Se muestra la categoría con mayor probabilidad (y las cinco más probables) o la tabla del lote, con un CSV descargable."],
    ];
    $("pasos").innerHTML = pasos.map(([ic, t, d], i) => `
      <li class="step"><span class="step-n">0${i + 1}</span><div class="step-ic">${icono(ic)}</div>
      <h4>${t}</h4><p>${d}</p></li>`).join("");
    $("que-es").innerHTML = transfer
      ? `Una red neuronal convolucional (MobileNetV2) que primero aprendió a reconocer objetos con 1,2 millones de
        fotos de ImageNet y luego se ajustó con el dataset público <i>Fashion Product Images (Small)</i> de Kaggle,
        depurado a 39.437 imágenes sin duplicados. Aquí corre directamente en tu navegador.`
      : `Una red neuronal convolucional (CNN) entrenada con el dataset público <i>Fashion Product Images (Small)</i>
        de Kaggle, depurado a 39.437 imágenes sin duplicados. Aquí corre directamente en tu navegador.`;
    // Tarjeta "cómo aprendió" de la sección Dentro del modelo
    $("dm-aprendio-t").textContent = transfer ? "Aprendió en dos etapas" : "Aprendió desde cero, con ejemplos";
    $("dm-aprendio").textContent = transfer
      ? "Primero aprendió a reconocer objetos en general con 1,2 millones de fotos (ImageNet). Después se ajustó con 27.605 fotos de este catálogo para distinguir las 40 categorías."
      : "Vio 27.605 fotos de este catálogo, cada una con su categoría. En cada pasada ajustó sus 1,2 millones de parámetros para equivocarse un poco menos, hasta reconocer las 40 categorías.";
  }

  function pintarMetricas(metricas) {
    if (!metricas) {
      $("metricas-grid").innerHTML = htmlError("Métricas no disponibles", "No se encontró data/metricas.json.");
      return;
    }
    $("metricas-grid").innerHTML = Object.entries(metricas).map(([clave, valor]) => {
      const [nombre, desc, esTasa] = DESCRIPCION_METRICAS[clave]
        ?? [clave.replace(/_/g, " "), "", typeof valor === "number" && valor >= 0 && valor <= 1];
      const texto = typeof valor === "number" ? (esTasa ? pct(valor) : miles(valor)) : esc(valor);
      const medidor = esTasa ? `<div class="m-gauge"><div style="width:${(valor * 100).toFixed(2)}%"></div></div>` : "";
      return `<div class="m-card${esTasa ? "" : " m-count"}"><span class="m-name">${esc(nombre)}</span>
        <span class="m-value">${texto}</span>${medidor}<p class="m-desc">${esc(desc)}</p>
        <code class="m-key">${esc(clave)}</code></div>`;
    }).join("");
  }

  // ------------------------------------------------------------------------- //
  // Fragmentos de resultados
  // ------------------------------------------------------------------------- //

  const htmlVacio = () => `
    <div class="res-empty">
      <div class="res-empty-icon"><svg viewBox="0 0 24 24" width="34" height="34" fill="none" stroke="currentColor" stroke-width="1.6">
        <path d="M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2"/><circle cx="12" cy="12" r="3.5"/></svg></div>
      <h4>Los resultados aparecerán aquí</h4>
      <p>Carga una o varias fotos (hasta ${MAX_LOTE}) y pulsa <b>Clasificar</b>. Con una imagen verás las cinco
      categorías más probables; con varias, una tabla del lote. En ambos casos podrás descargar el CSV.</p>
    </div>`;

  const htmlCargando = (n) => `
    <div class="res-empty">
      <div class="scanner"><span></span></div>
      <h4>${n === 1 ? "Analizando la imagen…" : `Clasificando ${n} imágenes…`}</h4>
      <p>Redimensionando a 80 x 60 píxeles y calculando las probabilidades de las 40 categorías.</p>
    </div>`;

  const htmlError = (titulo, detalle) => `
    <div class="res-error" role="alert"><div class="res-error-icon">!</div>
      <div><h4>${esc(titulo)}</h4><p>${esc(detalle)}</p></div></div>`;

  const etiquetaEstado = (f) => (f.requiere_revision
    ? '<span class="tag tag-warn">Revisión manual</span>'
    : '<span class="tag tag-ok">Automática</span>');

  function htmlIndividual(f) {
    const barras = f.top.map(([c, p], i) => `
      <li class="bar-row${i === 0 ? " is-top" : ""}" style="--d:${i * 70}ms">
        <div class="bar-head"><span class="bar-rank">${i + 1}</span><span class="bar-name">${esc(c)}</span>
        <span class="bar-val">${pct(p)}</span></div>
        <div class="bar-track"><div class="bar-fill" style="--w:${Math.max(p * 100, 0.6).toFixed(2)}%"></div></div>
      </li>`).join("");
    return `
      <div class="res-card">
        <div class="res-main">
          <img src="${f.url}" alt="${esc(f.nombre_imagen)}">
          <div>
            <span class="res-label">Categoría predicha</span>
            <div class="res-class">${esc(f.categoria_predicha)}</div>
            <div class="res-prob"><span>${pct(f.probabilidad)}</span> de probabilidad asignada por el modelo</div>
            <span class="res-file">${esc(f.nombre_imagen)}</span>
            ${etiquetaEstado(f)}
          </div>
        </div>
        <div class="res-top"><span class="res-label">Top ${TOP_K} categorías más probables</span><ul class="bars">${barras}</ul></div>
        <p class="res-foot">Las probabilidades indican cómo reparte el modelo su salida entre las 40 categorías.
        No son una medida de acierto para esta imagen. Por debajo de ${pct(estado.config.umbral_revision)}
        la predicción se marca para revisión manual.</p>
      </div>`;
  }

  function htmlLote(filas) {
    const ok = filas.filter((f) => f.estado === "ok");
    const revision = ok.filter((f) => f.requiere_revision);
    const errores = filas.length - ok.length;
    const conteo = {};
    ok.forEach((f) => { conteo[f.categoria_predicha] = (conteo[f.categoria_predicha] ?? 0) + 1; });
    const top = Object.entries(conteo).sort((a, b) => b[1] - a[1]).slice(0, 6);
    const tarjetas = [
      ["Imágenes procesadas", filas.length, ""],
      ["Clasificadas automáticamente", ok.length - revision.length, "is-ok"],
      ["Para revisión manual", revision.length, "is-warn"],
      ["Con error de lectura", errores, errores ? "is-err" : ""],
    ].map(([t, v, c]) => `<div class="lote-stat ${c}"><span>${t}</span><b>${v}</b></div>`).join("");
    const chips = top.map(([c, n]) => `<span class="chip">${esc(c)} <b>${n}</b></span>`).join("");
    const filasHtml = filas.map((f, i) => {
      const img = f.estado === "ok" ? `<img src="${f.url}" alt="">` : '<div class="thumb-x">!</div>';
      const nombre = `<small class="t-name" title="${esc(f.nombre_imagen)}">${esc(f.nombre_imagen)}</small>`;
      const celdas = f.estado !== "ok"
        ? `<td class="t-err"><b>${esc(f.estado.charAt(0).toUpperCase() + f.estado.slice(1))}</b>${nombre}</td><td></td><td></td>`
        : `<td><b>${esc(f.categoria_predicha)}</b>${nombre}<small>2.ª opción: ${esc(f.segunda_categoria)} · ${pct(f.probabilidad_segunda)}</small></td>
           <td class="t-num">${pct(f.probabilidad)}<div class="bar-track"><div class="bar-fill" style="--w:${Math.max(f.probabilidad * 100, 0.6).toFixed(1)}%"></div></div></td>
           <td>${etiquetaEstado(f)}</td>`;
      return `<tr><td class="t-idx">${i + 1}</td><td class="t-img">${img}</td>${celdas}</tr>`;
    }).join("");
    return `
      <div class="res-card">
        <div class="lote-stats">${tarjetas}</div>
        ${chips ? `<div class="chips"><span class="res-label">Categorías más frecuentes en el lote</span>${chips}</div>` : ""}
        <div class="tabla-wrap"><table class="lote-tabla">
          <thead><tr><th>#</th><th>Imagen</th><th>Categoría predicha</th><th>Probabilidad</th><th>Estado</th></tr></thead>
          <tbody>${filasHtml}</tbody></table></div>
        <p class="res-foot">Se marca para revisión manual toda predicción con probabilidad menor a
        ${pct(estado.config.umbral_revision)}. El CSV contiene el nombre de cada imagen, la categoría predicha,
        su probabilidad, la segunda opción y si requiere revisión.</p>
      </div>`;
  }

  // ------------------------------------------------------------------------- //
  // Preprocesamiento: igual que el pipeline (RGB, vecino más cercano, escala a [0, 1])
  // ------------------------------------------------------------------------- //

  async function aPixeles(file) {
    const bmp = await createImageBitmap(file);
    const { alto, ancho } = estado.config;
    // Relleno blanco hasta la proporción 3:4 del dataset (60 x 80) para no deformar fotos apaisadas.
    // Las imágenes del dataset ya tienen esa proporción, así que para ellas no cambia nada.
    let W = bmp.width;
    let H = bmp.height;
    let ox = 0;
    let oy = 0;
    if (W * alto > H * ancho) {
      const nuevoH = Math.round((W * alto) / ancho);
      oy = Math.floor((nuevoH - H) / 2);
      H = nuevoH;
    } else if (W * alto < H * ancho) {
      const nuevoW = Math.round((H * ancho) / alto);
      ox = Math.floor((nuevoW - W) / 2);
      W = nuevoW;
    }
    const canvas = document.createElement("canvas");
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.fillStyle = "#ffffff";           // relleno y zonas transparentes de un PNG en blanco
    ctx.fillRect(0, 0, W, H);
    ctx.drawImage(bmp, ox, oy);
    if (bmp.close) bmp.close();
    const { data } = ctx.getImageData(0, 0, W, H);
    const salida = new Float32Array(alto * ancho * 3);
    for (let y = 0; y < alto; y++) {
      const sy = Math.min(H - 1, Math.floor(((y + 0.5) * H) / alto));
      for (let x = 0; x < ancho; x++) {
        const sx = Math.min(W - 1, Math.floor(((x + 0.5) * W) / ancho));
        const i = (sy * W + sx) * 4;
        const o = (y * ancho + x) * 3;
        salida[o] = data[i] / 255;
        salida[o + 1] = data[i + 1] / 255;
        salida[o + 2] = data[i + 2] / 255;
      }
    }
    return salida;
  }

  // ------------------------------------------------------------------------- //
  // Clasificación
  // ------------------------------------------------------------------------- //

  async function clasificar() {
    if (!estado.archivos.length || estado.ocupado) return;
    if (!estado.modelo) {
      $("resultado").innerHTML = htmlError("El modelo todavía se está cargando", "Espera unos segundos y vuelve a intentarlo.");
      return;
    }
    estado.ocupado = true;
    actualizarControles();
    ocultarCsv();
    $("resultado").innerHTML = htmlCargando(estado.archivos.length);
    await new Promise((r) => setTimeout(r, 40)); // deja pintar el estado de carga

    try {
      const { alto, ancho } = estado.config;
      const filas = [];
      const pixeles = [];
      const indices = [];
      for (const item of estado.archivos) {
        const f = item.file;
        const fila = { nombre_imagen: f.name, url: item.url, estado: "ok" };
        if (!(f.type.startsWith("image/") || EXTENSIONES.test(f.name))) {
          fila.estado = "formato no admitido";
        } else {
          try {
            pixeles.push(await aPixeles(f));
            indices.push(filas.length);
          } catch {
            fila.estado = "imagen no válida";
          }
        }
        filas.push(fila);
      }

      if (pixeles.length) {
        const tam = alto * ancho * 3;
        const x = new Float32Array(pixeles.length * tam);
        pixeles.forEach((p, i) => x.set(p, i * tam));
        const salida = tf.tidy(() => estado.modelo.predict(tf.tensor4d(x, [pixeles.length, alto, ancho, 3])));
        const probs = await salida.data();
        salida.dispose();
        const C = estado.clases.length;
        indices.forEach((k, i) => {
          const p = Array.from(probs.subarray(i * C, (i + 1) * C));
          const orden = p.map((v, j) => j).sort((a, b) => p[b] - p[a]);
          Object.assign(filas[k], {
            categoria_predicha: estado.clases[orden[0]], probabilidad: p[orden[0]],
            segunda_categoria: estado.clases[orden[1]], probabilidad_segunda: p[orden[1]],
            requiere_revision: p[orden[0]] < estado.config.umbral_revision,
            top: orden.slice(0, TOP_K).map((j) => [estado.clases[j], p[j]]),
          });
        });
      }

      if (filas.length === 1 && filas[0].estado !== "ok") {
        $("resultado").innerHTML = htmlError("No se pudo clasificar la imagen",
          filas[0].estado === "formato no admitido" ? "Usa una imagen JPG, PNG, WEBP, BMP o GIF."
            : "No se pudo leer el archivo como imagen. Prueba con otro archivo.");
        return;
      }
      $("resultado").innerHTML = filas.length === 1 ? htmlIndividual(filas[0]) : htmlLote(filas);
      prepararCsv(filas);
    } catch (e) {
      console.error(e);
      $("resultado").innerHTML = htmlError("Error durante la predicción", String(e.message ?? e));
    } finally {
      estado.ocupado = false;
      actualizarControles();
    }
  }

  // ------------------------------------------------------------------------- //
  // CSV
  // ------------------------------------------------------------------------- //

  function prepararCsv(filas) {
    const celda = (v) => {
      const s = v == null ? "" : String(v);
      return /[",;\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const lineas = [COLUMNAS_CSV.join(",")];
    for (const f of filas) {
      const ok = f.estado === "ok";
      lineas.push([
        f.nombre_imagen,
        ok ? f.categoria_predicha : "",
        ok ? f.probabilidad.toFixed(4) : "",
        ok ? f.segunda_categoria : "",
        ok ? f.probabilidad_segunda.toFixed(4) : "",
        ok ? (f.requiere_revision ? "si" : "no") : "",
        f.estado,
      ].map(celda).join(","));
    }
    // BOM UTF-8: Excel respeta las tildes
    const blob = new Blob(["﻿" + lineas.join("\r\n") + "\r\n"], { type: "text/csv;charset=utf-8" });
    if (estado.csvUrl) URL.revokeObjectURL(estado.csvUrl);
    estado.csvUrl = URL.createObjectURL(blob);
    const d = new Date();
    const dos = (n) => String(n).padStart(2, "0");
    const sello = `${d.getFullYear()}${dos(d.getMonth() + 1)}${dos(d.getDate())}_${dos(d.getHours())}${dos(d.getMinutes())}${dos(d.getSeconds())}`;
    const enlace = $("btn-csv");
    enlace.href = estado.csvUrl;
    enlace.download = `clasificacion_lote_${sello}.csv`;
    enlace.querySelector("span").textContent = `Descargar CSV (${filas.length} ${filas.length === 1 ? "imagen" : "imágenes"})`;
    enlace.hidden = false;
  }

  function ocultarCsv() {
    $("btn-csv").hidden = true;
  }

  // ------------------------------------------------------------------------- //
  // Lista de archivos
  // ------------------------------------------------------------------------- //

  function agregar(lista) {
    const nuevos = Array.from(lista ?? []);
    if (!nuevos.length) return;
    const espacio = Math.max(0, MAX_LOTE - estado.archivos.length);
    const aceptados = nuevos.slice(0, espacio);
    for (const file of aceptados) {
      estado.archivos.push({ id: `${Date.now()}-${Math.random()}`, file, url: URL.createObjectURL(file) });
    }
    const ignorados = nuevos.length - aceptados.length;
    mostrarAviso(ignorados > 0
      ? `El máximo es ${MAX_LOTE} imágenes por clasificación: se agregaron ${aceptados.length} y se omitieron ${ignorados}.`
      : "");
    reiniciarResultado();
    pintarLista();
  }

  function quitar(id) {
    const i = estado.archivos.findIndex((a) => a.id === id);
    if (i >= 0) {
      URL.revokeObjectURL(estado.archivos[i].url);
      estado.archivos.splice(i, 1);
    }
    mostrarAviso("");
    reiniciarResultado();
    pintarLista();
  }

  function quitarTodas() {
    estado.archivos.forEach((a) => URL.revokeObjectURL(a.url));
    estado.archivos = [];
    mostrarAviso("");
    reiniciarResultado();
    pintarLista();
  }

  function pintarLista() {
    const n = estado.archivos.length;
    $("lista-head").hidden = n === 0;
    $("contador").textContent = `${n} de ${MAX_LOTE} imágenes`;
    $("lista").innerHTML = estado.archivos.map((a) => `
      <li><img src="${a.url}" alt="" onerror="this.replaceWith(Object.assign(document.createElement('div'), {className: 'thumb-x', textContent: '!'}))">
      <span title="${esc(a.file.name)}">${esc(a.file.name)}</span>
      <button type="button" data-id="${esc(a.id)}" aria-label="Quitar ${esc(a.file.name)}">×</button></li>`).join("");
    actualizarControles();
    document.dispatchEvent(new CustomEvent("fv:lista")); // la sección "Dentro del modelo" muestra estas imágenes
  }

  // ------------------------------------------------------------------------- //
  // Imágenes de ejemplo (conjunto de prueba del dataset; el modelo no las vio al entrenar)
  // ------------------------------------------------------------------------- //

  let promesaEjemplos = null;
  function ejemplos() {
    promesaEjemplos ??= leerJson("ejemplos/ejemplos.json")
      .then((lista) => lista.map((e, i) => ({ id: `ej-${i}`, nombre: e.archivo, url: `ejemplos/${e.archivo}`,
        categoria: e.categoria, producto: e.producto })))
      .catch(() => []);
    return promesaEjemplos;
  }

  async function probarEjemplos() {
    if (estado.ocupado) return;
    const boton = $("btn-ejemplos");
    boton.disabled = true;
    try {
      const lista = await ejemplos();
      const archivos = await Promise.all(lista.map(async (e) => {
        const blob = await (await fetch(e.url)).blob();
        return new File([blob], e.nombre, { type: blob.type || "image/jpeg" });
      }));
      quitarTodas();
      agregar(archivos);
      await clasificar();
    } catch (e) {
      console.error(e);
      mostrarAviso("No se pudieron cargar las imágenes de ejemplo.");
    } finally {
      boton.disabled = false;
    }
  }

  function reiniciarResultado() {
    ocultarCsv();
    $("resultado").innerHTML = htmlVacio();
  }

  function mostrarAviso(texto) {
    $("aviso").textContent = texto;
    $("aviso").hidden = !texto;
  }

  function actualizarControles() {
    const n = estado.archivos.length;
    const boton = $("btn-clasificar");
    boton.disabled = n === 0 || estado.ocupado;
    boton.textContent = estado.ocupado ? "Clasificando…"
      : n > 1 ? `Clasificar ${n} imágenes` : "Clasificar";
  }

  function estadoModelo(clase, texto) {
    const el = $("estado-modelo");
    el.className = `estado-modelo ${clase}`;
    el.innerHTML = `<i></i> ${esc(texto)}`;
  }

  // ------------------------------------------------------------------------- //
  // Inicio
  // ------------------------------------------------------------------------- //

  async function leerJson(ruta) {
    const r = await fetch(ruta, { cache: "no-cache" });
    if (!r.ok) throw new Error(`${ruta}: ${r.status}`);
    return r.json();
  }

  function eventos() {
    const zona = $("dropzone");
    $("entrada").addEventListener("change", (e) => { agregar(e.target.files); e.target.value = ""; });
    ["dragenter", "dragover"].forEach((t) => zona.addEventListener(t, (e) => { e.preventDefault(); zona.classList.add("arrastrando"); }));
    ["dragleave", "drop"].forEach((t) => zona.addEventListener(t, (e) => { e.preventDefault(); zona.classList.remove("arrastrando"); }));
    zona.addEventListener("drop", (e) => agregar(e.dataTransfer.files));
    document.addEventListener("paste", (e) => {
      const archivos = Array.from(e.clipboardData?.files ?? []).filter((f) => f.type.startsWith("image/"));
      if (archivos.length) agregar(archivos);
    });
    $("lista").addEventListener("click", (e) => {
      const boton = e.target.closest("button[data-id]");
      if (boton) quitar(boton.dataset.id);
    });
    $("btn-limpiar").addEventListener("click", quitarTodas);
    $("btn-clasificar").addEventListener("click", clasificar);
    $("btn-ejemplos").addEventListener("click", probarEjemplos);
  }

  async function iniciar() {
    eventos();
    pintarPasos();
    reiniciarResultado();
    actualizarControles();

    const [clases, metricas, config] = await Promise.all([
      leerJson("data/clases.json").catch(() => null),
      leerJson("data/metricas.json").catch(() => null),
      leerJson("data/config.json").catch(() => ({})),
    ]);
    Object.assign(estado.config, config);
    $("umbral").textContent = pct(estado.config.umbral_revision);
    pintarPasos();
    pintarMetricas(metricas);
    if (!clases) {
      estadoModelo("error", "No se encontró data/clases.json");
      return;
    }
    estado.clases = clases;
    ejemplos().then((lista) => {
      if (!lista.length) return;
      $("btn-ejemplos").querySelector("span").textContent = `Probar con ${lista.length} imágenes de ejemplo`;
      $("btn-ejemplos").hidden = false;
    });

    try {
      await tf.ready();
      const modelo = await tf.loadLayersModel("model/model.json");
      const forma = modelo.inputs[0].shape; // [null, alto, ancho, 3]
      estado.config.alto = forma[1];
      estado.config.ancho = forma[2];
      const salidas = modelo.outputs[0].shape[1];
      if (salidas !== clases.length) throw new Error("La salida del modelo no coincide con clases.json.");
      tf.tidy(() => modelo.predict(tf.zeros([1, forma[1], forma[2], 3]))); // calentamiento
      estado.modelo = modelo;
      estadoModelo("listo", `Modelo listo · ${tf.getBackend() === "webgl" ? "GPU (WebGL)" : tf.getBackend().toUpperCase()}`);
      document.dispatchEvent(new CustomEvent("fv:modelo"));
    } catch (e) {
      console.error(e);
      estadoModelo("error", "No se pudo cargar el modelo");
      $("resultado").innerHTML = htmlError("No se pudo cargar el modelo", String(e.message ?? e));
    }
    actualizarControles();
  }

  // Lo usan dentro.js (sección "Dentro del modelo") y la consola para depurar
  window.FashionVision = { agregar, clasificar, estado, aPixeles, ejemplos };
  iniciar();
})();
