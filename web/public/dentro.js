/* FashionVision · "Dentro del modelo": recorrido visual y no técnico de lo que hace la red con una imagen.
 *
 * Usa el modelo que ya cargó app.js (window.FashionVision) y muestra, para la imagen elegida:
 *   1. la cuadrícula de 80 x 60 que recibe el modelo;
 *   2. mapas de activación de una capa al inicio, a la mitad y al final de la red;
 *   3. Grad-CAM: en qué zonas se apoyó para cada categoría (gradientes de la última capa convolucional);
 *   4. oclusión interactiva: tapar partes de la foto y ver cómo cambia la respuesta en vivo.
 */
(() => {
  "use strict";

  const FV = window.FashionVision;
  if (!FV || !document.getElementById("dentro")) return;

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (v) => `${(v * 100).toFixed(1).replace(".", ",")} %`;
  const miles = (n) => n.toLocaleString("es-CO");

  // Capas que se muestran en el paso 2 (MobileNetV2). Si el modelo es otro, se eligen solas.
  const PROFUNDIDADES = [
    { capa: "Conv1_relu", titulo: "Inicio de la red",
      texto: (c) => `Las primeras capas funcionan como un juego de lupas. Cada una se “enciende” (zonas claras)
        cuando encuentra algo muy simple: un borde, un cambio de color o una zona oscura. Aquí ves 6 de sus
        ${c} lupas: las que más reaccionaron a esta foto.` },
    { capa: "block_6_expand_relu", titulo: "Mitad",
      texto: (c) => `A mitad del camino ya no mira bordes sueltos: los combina en texturas y piezas, como una
        costura, una suela o el contorno de una manga. La imagen se ve más borrosa porque ahora importa menos el
        detalle y más <i>qué hay</i> y <i>dónde</i>. Esta capa tiene ${c} lupas.` },
    { capa: "block_13_expand_relu", titulo: "Final",
      texto: (c) => `Cerca del final, cada lupa responde a ideas más completas, como “parte superior de una
        prenda” o “punta de un zapato”. Ya no se parece a la foto: es un resumen de dónde aparece cada idea.
        Esta capa tiene ${c} lupas.` },
  ];
  const N_MAPAS = 6;
  const PASOS = [
    { titulo: "Mira", sub: "la foto" },
    { titulo: "Detecta", sub: "patrones" },
    { titulo: "Se enfoca", sub: "en lo importante" },
    { titulo: "Pruébalo", sub: "tapa una parte" },
  ];
  // Rampa secuencial para activaciones (oscuro = nada, claro = mucho) y rampa cálida para el mapa de calor
  const RAMPA_ACTIVACION = [[11, 18, 36], [30, 41, 99], [79, 70, 229], [34, 211, 238], [236, 254, 255]];
  const RAMPA_CALOR = [[253, 224, 71], [249, 115, 22], [220, 38, 38]];

  const st = {
    listo: false,
    explorador: null,   // modelo con salidas intermedias
    capasVis: [],       // [{ capa, titulo, texto }]
    cabeza: [],         // capas entre la última capa convolucional y la salida (sin la última Dense)
    W: null, b: null,   // pesos de la última Dense (para el logit sin softmax)
    ejemplos: [],
    item: null,
    pixeles: null,
    mapas: [], A: null, probs: null, top: [],
    cacheMapas: new Map(), cacheCam: new Map(),
    paso: 0, profundidad: 0, claseCam: null, opacidad: 0.65,
    mascara: null, pincel: 5, probsTapado: null, ocupado: false, repetir: false,
    token: 0,
  };

  // ------------------------------------------------------------------------- //
  // Preparación del modelo explorador
  // ------------------------------------------------------------------------- //

  const esRango4 = (capa) => {
    try { return capa.outputShape.length === 4; } catch { return false; }
  };

  function prepararExplorador(modelo) {
    const capas = modelo.layers;
    let vis = PROFUNDIDADES.map((p) => ({ ...p, capa: capas.find((c) => c.name === p.capa) }));
    if (vis.some((v) => !v.capa)) {
      // Modelo distinto de MobileNetV2: tres capas convolucionales repartidas a lo largo de la red
      const candidatas = capas.filter((c) => esRango4(c) && /conv|relu|activation/i.test(c.getClassName()));
      const elegir = (f) => candidatas[Math.min(candidatas.length - 1, Math.floor(f * (candidatas.length - 1)))];
      vis = PROFUNDIDADES.map((p, i) => ({ ...p, capa: elegir([0, 0.5, 1][i]) }));
    }
    let iCam = -1;
    capas.forEach((c, i) => { if (esRango4(c) && i < capas.length - 1) iCam = i; });
    const resto = capas.slice(iCam + 1);
    const ultima = resto[resto.length - 1];
    if (iCam < 0 || ultima.getClassName() !== "Dense") throw new Error("Estructura de modelo no soportada para Grad-CAM.");
    st.cabeza = resto.slice(0, -1);
    [st.W, st.b] = ultima.getWeights();
    st.capasVis = vis;
    st.explorador = tf.model({
      inputs: modelo.inputs,
      outputs: [...vis.map((v) => v.capa.output), capas[iCam].output, modelo.outputs[0]],
    });
  }

  // ------------------------------------------------------------------------- //
  // Dibujo
  // ------------------------------------------------------------------------- //

  const dpr = () => Math.min(2, window.devicePixelRatio || 1);

  function lienzo(cssW, cssH, clase = "") {
    const c = document.createElement("canvas");
    c.width = Math.round(cssW * dpr());
    c.height = Math.round(cssH * dpr());
    c.style.width = `${cssW}px`;
    c.style.height = `${cssH}px`;
    if (clase) c.className = clase;
    return c;
  }

  function desdeRGBA(rgba, w, h) {
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    c.getContext("2d").putImageData(new ImageData(rgba, w, h), 0, 0);
    return c;
  }

  function dibujar(canvas, fuente, suave, limpiar = true) {
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = suave;
    ctx.imageSmoothingQuality = "high";
    if (limpiar) ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(fuente, 0, 0, canvas.width, canvas.height);
  }

  function color(t, rampa) {
    const s = Math.max(0, Math.min(1, t)) * (rampa.length - 1);
    const i = Math.min(rampa.length - 2, Math.floor(s));
    const f = s - i;
    return rampa[i].map((v, k) => v + (rampa[i + 1][k] - v) * f);
  }

  /** La entrada del modelo como imagen; con máscara, lo tapado se tiñe de rosa (al modelo le llega blanco). */
  function imagenEntrada(conMascara = false) {
    const { alto, ancho } = FV.estado.config;
    const rgba = new Uint8ClampedArray(alto * ancho * 4);
    for (let i = 0; i < alto * ancho; i++) {
      const tapado = conMascara && st.mascara[i];
      rgba[i * 4] = tapado ? 251 : st.pixeles[i * 3] * 255;
      rgba[i * 4 + 1] = tapado ? 113 : st.pixeles[i * 3 + 1] * 255;
      rgba[i * 4 + 2] = tapado ? 133 : st.pixeles[i * 3 + 2] * 255;
      rgba[i * 4 + 3] = 255;
    }
    return desdeRGBA(rgba, ancho, alto);
  }

  function imagenMapa(valores, h, w) {
    let min = Infinity;
    let max = -Infinity;
    for (const v of valores) { if (v < min) min = v; if (v > max) max = v; }
    const rango = max - min || 1;
    const rgba = new Uint8ClampedArray(h * w * 4);
    valores.forEach((v, i) => {
      const [r, g, b] = color((v - min) / rango, RAMPA_ACTIVACION);
      rgba.set([r, g, b, 255], i * 4);
    });
    return desdeRGBA(rgba, w, h);
  }

  function imagenCalor(cam, h, w) {
    const rgba = new Uint8ClampedArray(h * w * 4);
    cam.forEach((t, i) => {
      const [r, g, b] = color(t, RAMPA_CALOR);
      const a = st.opacidad * Math.min(1, Math.max(0, (t - 0.12) / 0.6)) * 255;
      rgba.set([r, g, b, a], i * 4);
    });
    return desdeRGBA(rgba, w, h);
  }

  // ------------------------------------------------------------------------- //
  // Cálculos
  // ------------------------------------------------------------------------- //

  const ordenar = (p) => Array.from(p, (v, j) => j).sort((a, b) => p[b] - p[a]);

  /** Ejecuta fn dentro de tf.tidy, lee el tensor resultante y lo libera. */
  function leer(fn) {
    const t = tf.tidy(fn);
    const datos = t.dataSync();
    t.dispose();
    return datos;
  }

  function mapasDe(d) {
    if (st.cacheMapas.has(d)) return st.cacheMapas.get(d);
    const t = st.mapas[d];
    const [, h, w, c] = t.shape;
    // Las lupas (canales) con más contraste en esta imagen
    const canales = leer(() => tf.topk(tf.moments(t, [0, 1, 2]).variance, Math.min(N_MAPAS, c)).indices);
    const mapas = Array.from(canales, (k) => leer(() => t.gather([k], 3)));
    const res = { h, w, c, mapas };
    st.cacheMapas.set(d, res);
    return res;
  }

  /** Grad-CAM de una clase: gradiente del logit respecto de la última capa convolucional, a 2x la entrada. */
  function camDe(clase) {
    if (st.cacheCam.has(clase)) return st.cacheCam.get(clase);
    const { alto, ancho } = FV.estado.config;
    const logit = (a) => {
      let x = a;
      for (const capa of st.cabeza) x = capa.apply(x);
      return tf.add(tf.matMul(x, st.W), st.b).gather([clase], 1).sum();
    };
    const datos = leer(() => {
      const g = tf.grad(logit)(st.A);
      const pesos = tf.mean(g, [1, 2], true);
      const cam = tf.relu(tf.sum(tf.mul(st.A, pesos), -1, true));
      const grande = tf.image.resizeBilinear(cam, [alto * 2, ancho * 2], false, true);
      return tf.div(grande, tf.maximum(tf.max(grande), 1e-6)).squeeze();
    });
    st.cacheCam.set(clase, datos);
    return datos;
  }

  function entradaTapada() {
    const x = Float32Array.from(st.pixeles);
    st.mascara.forEach((m, i) => { if (m) x.fill(1, i * 3, i * 3 + 3); });
    return x;
  }

  async function predecirTapado() {
    if (st.ocupado) { st.repetir = true; return; }
    st.ocupado = true;
    const { alto, ancho } = FV.estado.config;
    const token = st.token;
    const salida = tf.tidy(() => FV.estado.modelo.predict(tf.tensor4d(entradaTapada(), [1, alto, ancho, 3])));
    const p = await salida.data();
    salida.dispose();
    st.ocupado = false;
    if (token !== st.token) return;
    st.probsTapado = st.mascara.some(Boolean) ? p : null;
    pintarDecision();
    if (st.repetir) { st.repetir = false; predecirTapado(); }
  }

  // ------------------------------------------------------------------------- //
  // Selección de imagen
  // ------------------------------------------------------------------------- //

  function liberar() {
    [...st.mapas, st.A].forEach((t) => t && t.dispose());
    st.mapas = [];
    st.A = null;
    st.cacheMapas.clear();
    st.cacheCam.clear();
  }

  async function seleccionar(item) {
    if (!st.listo) return;
    st.item = item;
    const token = ++st.token;
    pintarSelector();
    $("dm-visual").innerHTML = '<div class="dm-cargando"><div class="scanner"><span></span></div>Analizando la imagen…</div>';
    try {
      const blob = item.file ?? await (await fetch(item.url)).blob();
      const pixeles = await FV.aPixeles(blob);
      if (token !== st.token) return;
      const { alto, ancho } = FV.estado.config;
      const salidas = tf.tidy(() => st.explorador.predict(tf.tensor4d(pixeles, [1, alto, ancho, 3])));
      const probsT = salidas.pop();
      const probs = await probsT.data();
      probsT.dispose();
      if (token !== st.token) { salidas.forEach((t) => t.dispose()); return; }
      liberar();
      st.pixeles = pixeles;
      st.A = salidas.pop();
      st.mapas = salidas;
      st.probs = probs;
      st.top = ordenar(probs);
      st.claseCam = st.top[0];
      st.mascara = new Uint8Array(alto * ancho);
      st.probsTapado = null;
      pintarPaso();
      pintarDecision();
    } catch (e) {
      console.error(e);
      $("dm-visual").innerHTML = '<div class="dm-cargando">No se pudo leer esta imagen. Prueba con otra.</div>';
    }
  }

  // ------------------------------------------------------------------------- //
  // Interfaz
  // ------------------------------------------------------------------------- //

  function itemsSelector() {
    const propios = FV.estado.archivos
      .filter((a) => !st.ejemplos.some((e) => e.nombre === a.file.name))   // los ejemplos ya están en la fila
      .map((a) => ({ id: `tu-${a.id}`, nombre: a.file.name, url: a.url, file: a.file }));
    return { ejemplos: st.ejemplos, propios };
  }

  function pintarSelector() {
    const { ejemplos, propios } = itemsSelector();
    const boton = (it) => `
      <button type="button" class="dm-thumb${st.item?.id === it.id ? " activo" : ""}" data-id="${esc(it.id)}"
        aria-pressed="${st.item?.id === it.id}" title="${esc(it.producto ?? it.nombre)}">
        <img src="${esc(it.url)}" alt="${esc(it.categoria ?? it.nombre)}" loading="lazy"></button>`;
    $("dm-thumbs").innerHTML = ejemplos.map(boton).join("")
      + (propios.length ? `<span class="dm-sep">Tus imágenes</span>${propios.map(boton).join("")}` : "");
  }

  function pintarTabs() {
    $("dm-tabs").innerHTML = PASOS.map((p, i) => `
      <button type="button" role="tab" class="dm-tab${i === st.paso ? " activo" : ""}" data-paso="${i}"
        aria-selected="${i === st.paso}"><span class="dm-tab-n">${i + 1}</span>
        <span><b>${p.titulo}</b><small>${p.sub}</small></span></button>`).join("");
  }

  function navegacion() {
    return `<div class="dm-nav">
      <button type="button" class="link-btn" data-ir="${st.paso - 1}" ${st.paso === 0 ? "disabled" : ""}>← Anterior</button>
      <button type="button" class="dm-sig" data-ir="${st.paso + 1}" ${st.paso === PASOS.length - 1 ? "disabled" : ""}>Siguiente paso →</button>
    </div>`;
  }

  function pintarPaso() {
    pintarTabs();
    if (!st.pixeles) return;
    const visual = $("dm-visual");
    const texto = $("dm-texto");
    const { alto, ancho } = FV.estado.config;
    const clases = FV.estado.clases;
    visual.innerHTML = "";
    visual.dataset.paso = st.paso;

    if (st.paso === 0) {
      const foto = document.createElement("figure");
      foto.innerHTML = `<img src="${esc(st.item.url)}" alt=""><figcaption>Lo que ves tú</figcaption>`;
      const fig = document.createElement("figure");
      const c = lienzo(180, 240, "dm-pixel");
      dibujar(c, imagenEntrada(), false);
      fig.append(c);
      fig.insertAdjacentHTML("beforeend", "<figcaption>Lo que recibe el modelo</figcaption>");
      visual.append(foto, fig);
      c.addEventListener("pointermove", (e) => {
        const r = c.getBoundingClientRect();
        const x = Math.min(ancho - 1, Math.floor(((e.clientX - r.left) / r.width) * ancho));
        const y = Math.min(alto - 1, Math.floor(((e.clientY - r.top) / r.height) * alto));
        const i = (y * ancho + x) * 3;
        const [R, G, B] = [0, 1, 2].map((k) => Math.round(st.pixeles[i + k] * 255));
        $("dm-lectura").innerHTML = `<i style="background:rgb(${R},${G},${B})"></i>
          Fila ${y + 1}, columna ${x + 1}: <b>rojo ${R}</b> · <b>verde ${G}</b> · <b>azul ${B}</b>`;
      });
      texto.innerHTML = `
        <h3>Mira la foto… en baja resolución</h3>
        <p>Para el modelo, una foto es una cuadrícula de <b>${alto} × ${ancho}</b> cuadritos. Cada cuadrito son tres
        números: cuánto rojo, verde y azul tiene. En total, <b>${miles(alto * ancho * 3)} números</b>: eso es todo lo
        que recibe. No lee el nombre del producto ni su descripción.</p>
        <p class="dm-lectura" id="dm-lectura">Pasa el cursor (o el dedo) por la cuadrícula para ver los números de cada cuadrito.</p>
        ${navegacion()}`;
    }

    if (st.paso === 1) {
      const d = st.profundidad;
      const { h, w, c, mapas } = mapasDe(d);
      const grid = document.createElement("div");
      grid.className = "dm-mapas";
      mapas.forEach((m, i) => {
        const fig = document.createElement("figure");
        const cv = lienzo(78, 104);
        dibujar(cv, imagenMapa(m, h, w), true);
        fig.append(cv);
        fig.insertAdjacentHTML("beforeend", `<figcaption>Lupa ${i + 1}</figcaption>`);
        grid.append(fig);
      });
      visual.append(grid);
      const segmentos = st.capasVis.map((v, i) => `
        <button type="button" class="dm-seg${i === d ? " activo" : ""}" data-prof="${i}" aria-pressed="${i === d}">${v.titulo}</button>`).join("");
      texto.innerHTML = `
        <h3>Detecta patrones, de lo simple a lo complejo</h3>
        <div class="dm-segs" role="group" aria-label="Profundidad en la red">${segmentos}</div>
        <p>${st.capasVis[d].texto(miles(c))}</p>
        <p class="dm-meta">Cada lupa es una cuadrícula de ${h} × ${w}.</p>
        <div class="dm-leyenda"><span>No se activa</span><i class="dm-rampa-act"></i><span>Se activa mucho</span></div>
        ${navegacion()}`;
    }

    if (st.paso === 2) {
      const c = lienzo(210, 280, "dm-calor");
      const dibujarCalor = () => {
        dibujar(c, imagenEntrada(), true);
        dibujar(c, imagenCalor(camDe(st.claseCam), alto * 2, ancho * 2), true, false);
      };
      dibujarCalor();
      visual.append(c);
      const chips = st.top.slice(0, 3).map((k) => `
        <button type="button" class="dm-chip${k === st.claseCam ? " activo" : ""}" data-cam="${k}" aria-pressed="${k === st.claseCam}">
        ${esc(clases[k])} <b>${pct(st.probs[k])}</b></button>`).join("");
      texto.innerHTML = `
        <h3>Se enfoca en lo que importa</h3>
        <p>Este mapa de calor muestra en qué partes de la foto se apoyó el modelo para responder: las zonas
        <b class="dm-rojo">rojas</b> pesaron mucho y las transparentes casi nada. Toca otra categoría para ver qué
        partes de la foto lo harían pensar en ella.</p>
        <div class="dm-chips">${chips}</div>
        <label class="dm-rango">Intensidad del mapa
          <input type="range" id="dm-opacidad" min="0" max="1" step="0.05" value="${st.opacidad}"></label>
        <div class="dm-leyenda"><span>Poca influencia</span><i class="dm-rampa-calor"></i><span>Mucha influencia</span></div>
        <p class="dm-meta">Técnica: Grad-CAM sobre la última capa convolucional (mapa de ${st.A.shape[1]} × ${st.A.shape[2]}).</p>
        ${navegacion()}`;
      $("dm-opacidad").addEventListener("input", (e) => { st.opacidad = Number(e.target.value); dibujarCalor(); });
    }

    if (st.paso === 3) {
      const c = lienzo(210, 280, "dm-tapar");
      const redibujar = () => dibujar(c, imagenEntrada(true), false);
      redibujar();
      const fig = document.createElement("figure");
      fig.append(c);
      fig.insertAdjacentHTML("beforeend", '<figcaption class="dm-vivo" id="dm-vivo" aria-live="polite"></figcaption>');
      visual.append(fig);
      let pintando = false;
      const pintarEn = (e) => {
        const r = c.getBoundingClientRect();
        const cx = ((e.clientX - r.left) / r.width) * ancho;
        const cy = ((e.clientY - r.top) / r.height) * alto;
        const rad = st.pincel;
        for (let y = Math.max(0, Math.floor(cy - rad)); y <= Math.min(alto - 1, Math.ceil(cy + rad)); y++) {
          for (let x = Math.max(0, Math.floor(cx - rad)); x <= Math.min(ancho - 1, Math.ceil(cx + rad)); x++) {
            if ((x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2 <= rad * rad) st.mascara[y * ancho + x] = 1;
          }
        }
        redibujar();
        predecirTapado();
      };
      c.addEventListener("pointerdown", (e) => { pintando = true; c.setPointerCapture(e.pointerId); pintarEn(e); });
      c.addEventListener("pointermove", (e) => { if (pintando) pintarEn(e); });
      ["pointerup", "pointercancel"].forEach((t) => c.addEventListener(t, () => { pintando = false; }));
      texto.innerHTML = `
        <h3>Pruébalo: tapa una parte de la foto</h3>
        <p>Arrastra el cursor (o el dedo) sobre la foto para taparla y mira cómo cambia la respuesta en vivo.
        Si tapas las zonas rojas del paso anterior, la confianza suele bajar; si tapas el fondo, casi no cambia.</p>
        <label class="dm-rango">Tamaño del pincel
          <input type="range" id="dm-pincel" min="2" max="12" step="1" value="${st.pincel}"></label>
        <div class="dm-acciones">
          <button type="button" class="dm-btn" id="dm-auto">Tapar la zona más importante</button>
          <button type="button" class="link-btn" id="dm-restaurar">Restaurar la foto</button>
        </div>
        <p class="dm-meta">Lo tapado (en rosa) le llega al modelo como fondo blanco.</p>
        ${navegacion()}`;
      $("dm-pincel").addEventListener("input", (e) => { st.pincel = Number(e.target.value); });
      $("dm-restaurar").addEventListener("click", () => { st.mascara.fill(0); redibujar(); st.probsTapado = null; pintarDecision(); });
      $("dm-auto").addEventListener("click", () => {
        const cam = camDe(st.top[0]);
        for (let y = 0; y < alto; y++) {
          for (let x = 0; x < ancho; x++) {
            const v = Math.max(cam[(2 * y) * 2 * ancho + 2 * x], cam[(2 * y + 1) * 2 * ancho + 2 * x + 1]);
            if (v >= 0.55) st.mascara[y * ancho + x] = 1;
          }
        }
        redibujar();
        predecirTapado();
      });
    }
  }

  function pintarDecision() {
    const panel = $("dm-decision");
    if (!st.probs) return;
    const clases = FV.estado.clases;
    const p = st.paso === 3 && st.probsTapado ? st.probsTapado : st.probs;
    const tapado = p !== st.probs;
    const orden = ordenar(p);
    const k = orden[0];
    const umbral = FV.estado.config.umbral_revision;
    const barras = orden.slice(0, 5).map((j, i) => `
      <li class="bar-row${i === 0 ? " is-top" : ""}">
        <div class="bar-head"><span class="bar-rank">${i + 1}</span><span class="bar-name">${esc(clases[j])}</span>
        <span class="bar-val">${pct(p[j])}</span></div>
        <div class="bar-track"><div class="bar-fill dm-fill" style="width:${Math.max(p[j] * 100, 0.6).toFixed(2)}%"></div></div>
      </li>`).join("");
    const real = st.item?.categoria;
    const acierto = real && clases[k] === real;
    const antes = st.top[0];
    panel.innerHTML = `
      <span class="res-label">${tapado ? "Respuesta con partes tapadas" : "Respuesta del modelo"}</span>
      <div class="dm-clase">${esc(clases[k])}</div>
      <div class="dm-conf"><b>${pct(p[k])}</b> de confianza
        ${p[k] < umbral ? '<span class="tag tag-warn">Revisión manual</span>' : '<span class="tag tag-ok">Automática</span>'}</div>
      ${tapado ? `<p class="dm-antes">Sin tapar: ${esc(clases[antes])} con ${pct(st.probs[antes])}.
        Ahora le da ${pct(p[antes])}.</p>` : ""}
      ${real ? `<p class="dm-real ${acierto ? "ok" : "no"}">Categoría real en el catálogo: <b>${esc(real)}</b>
        <span>${acierto ? "✓ Coincide" : "✗ No coincide"}</span></p>` : ""}
      <ul class="bars dm-bars">${barras}</ul>
      <p class="dm-nota">Con menos de ${pct(umbral)} de confianza, la imagen se enviaría a revisión manual.</p>`;
    const vivo = $("dm-vivo");
    if (vivo) vivo.innerHTML = `Ahora: <b>${esc(clases[k])}</b> · ${pct(p[k])}`;
  }

  function eventos() {
    $("dm-thumbs").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-id]");
      if (!b) return;
      const { ejemplos, propios } = itemsSelector();
      const item = [...ejemplos, ...propios].find((it) => it.id === b.dataset.id);
      if (item) seleccionar(item);
    });
    $("dm-tabs").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-paso]");
      if (b) irA(Number(b.dataset.paso));
    });
    $("dm-texto").addEventListener("click", (e) => {
      const ir = e.target.closest("button[data-ir]");
      if (ir) irA(Number(ir.dataset.ir));
      const prof = e.target.closest("button[data-prof]");
      if (prof) { st.profundidad = Number(prof.dataset.prof); pintarPaso(); }
      const cam = e.target.closest("button[data-cam]");
      if (cam) { st.claseCam = Number(cam.dataset.cam); pintarPaso(); }
    });
    document.addEventListener("fv:lista", () => { if (st.listo) pintarSelector(); });
  }

  function irA(paso) {
    st.paso = Math.max(0, Math.min(PASOS.length - 1, paso));
    pintarPaso();
    pintarDecision();
  }

  async function activar() {
    if (st.listo || !FV.estado.modelo) return;
    try {
      prepararExplorador(FV.estado.modelo);
      st.ejemplos = await FV.ejemplos();
      st.listo = true;
      pintarTabs();
      pintarSelector();
      if (st.ejemplos.length) seleccionar(st.ejemplos[0]);
      else $("dm-visual").innerHTML = '<div class="dm-cargando">Sube una imagen en “Probar modelo” para explorarla aquí.</div>';
    } catch (e) {
      console.error(e);
      $("dm-visual").innerHTML = `<div class="dm-cargando">Esta sección no está disponible para este modelo (${esc(e.message ?? e)}).</div>`;
    }
  }

  eventos();
  pintarTabs();
  document.addEventListener("fv:modelo", activar);
  activar();
})();
