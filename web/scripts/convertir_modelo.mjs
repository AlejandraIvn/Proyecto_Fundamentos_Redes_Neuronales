#!/usr/bin/env node
/**
 * Convierte artefactos/modelo_final.keras (Keras 3) al formato "layers-model" de TensorFlow.js
 * y copia clases, métricas y configuración a public/data/.
 *
 * Uso:  npm run build      (Vercel lo ejecuta en cada despliegue)
 *
 * Busca los artefactos en este orden:
 *   1. la variable de entorno ARTEFACTOS_DIR
 *   2. ../artefactos   (carpeta del repositorio principal, si web/ vive dentro de él)
 *   3. ./artefactos    (copia dentro de web/, si web/ se publica como repositorio aparte)
 * Si no encuentra modelo_final.keras pero ya existe public/model/model.json, conserva ese modelo.
 *
 * Soporta los modelos del proyecto: la CNN secuencial (Conv2D, BatchNormalization, MaxPooling2D,
 * Flatten, Dense, Dropout) y el modelo funcional de transfer learning con MobileNetV2 anidada
 * (Resizing, Rescaling, DepthwiseConv2D, ReLU, Add, ZeroPadding2D, GlobalAveragePooling2D), que se
 * "aplana" en un solo grafo para que TensorFlow.js lo cargue sin submodelos.
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { strFromU8, unzipSync } from "fflate";
import h5wasm from "h5wasm/node";

const RAIZ_WEB = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SALIDA_MODELO = path.join(RAIZ_WEB, "public", "model");
const SALIDA_DATOS = path.join(RAIZ_WEB, "public", "data");
const TAM_FRAGMENTO = 4 * 1024 * 1024; // 4 MB por archivo .bin, como el conversor oficial

const SOPORTADAS = new Set([
  "InputLayer", "Conv2D", "DepthwiseConv2D", "BatchNormalization", "MaxPooling2D", "AveragePooling2D",
  "Flatten", "Dense", "Dropout", "ReLU", "Activation", "Add", "Concatenate", "ZeroPadding2D",
  "GlobalAveragePooling2D", "Resizing", "Rescaling",
]);
// Capas que se reemplazan por versiones propias registradas en public/app.js (misma semántica que Keras)
const RENOMBRAR = { Resizing: "ResizingHP", Rescaling: "RescalingSimple" };
// Claves de Keras 3 que TensorFlow.js no necesita para inferencia
const QUITAR = new Set([
  "module", "registered_name", "build_config", "compile_config", "shared_object_id", "groups",
  "synchronized", "quantization_config", "lora_rank", "lora_alpha", "sparse", "ragged", "optional",
  "seed", "noise_shape", "negative_slope", "threshold", "fill_mode", "fill_value", "antialias",
  "pad_to_aspect_ratio", "crop_to_aspect_ratio", "interpolation", "activity_regularizer",
]);

const log = (...m) => console.log("[convertir_modelo]", ...m);

function buscarArtefactos() {
  const candidatos = [process.env.ARTEFACTOS_DIR, path.join(RAIZ_WEB, "..", "artefactos"),
    path.join(RAIZ_WEB, "artefactos")].filter(Boolean);
  return candidatos.find((c) => fs.existsSync(path.join(c, "modelo_final.keras")));
}

// --------------------------------------------------------------------------- //
// Topología: configuración de Keras 3 -> configuración que entiende TensorFlow.js
// --------------------------------------------------------------------------- //

function limpiarConfig(claseOriginal, config) {
  const limpia = {};
  for (const [clave, valor] of Object.entries(config)) {
    if (QUITAR.has(clave) || /_(initializer|regularizer|constraint)$/.test(clave)) continue;
    limpia[clave] = valor;
  }
  limpia.dtype = "float32";
  if (claseOriginal === "GlobalAveragePooling2D") {
    if (config.keepdims) throw new Error("GlobalAveragePooling2D con keepdims=True no está soportado.");
    delete limpia.keepdims;
  }
  if (claseOriginal === "Resizing" && config.interpolation && config.interpolation !== "bilinear") {
    throw new Error(`Resizing con interpolación '${config.interpolation}' no está soportado.`);
  }
  if (claseOriginal === "ReLU" && ((config.negative_slope ?? 0) !== 0 || (config.threshold ?? 0) !== 0)) {
    throw new Error("ReLU con negative_slope o threshold distintos de 0 no está soportado.");
  }
  return limpia;
}

function convertirCapa(capa) {
  const clase = capa.class_name;
  if (!SOPORTADAS.has(clase)) {
    throw new Error(`La capa '${capa.config?.name}' (${clase}) no está soportada por el conversor.`);
  }
  return { class_name: RENOMBRAR[clase] ?? clase, config: limpiarConfig(clase, capa.config) };
}

/** Referencias [capa, nodo, tensor] de los argumentos de un nodo de Keras 3. */
function referencias(nodo) {
  const refs = [];
  (function visitar(x) {
    if (Array.isArray(x)) x.forEach(visitar);
    else if (x && typeof x === "object") {
      if (x.class_name === "__keras_tensor__") refs.push([...x.config.keras_history]);
      else Object.values(x).forEach(visitar);
    }
  })(nodo.args ?? nodo);
  return refs;
}

const normalizarRefs = (lista) => (typeof lista?.[0] === "string" ? [lista] : lista);

/** Aplana un modelo funcional (con submodelos funcionales) en una lista de capas con sus entradas. */
function aplanarFuncional(cfg) {
  const capas = [];
  const salidasSub = new Map(); // "submodelo:tensor" -> referencia de su salida interna
  for (const capa of cfg.layers) {
    const nodos = (capa.inbound_nodes ?? []).map(referencias);
    if (capa.class_name === "Functional" || capa.class_name === "Model") {
      if (nodos.length !== 1) throw new Error(`El submodelo ${capa.config.name} se usa más de una vez.`);
      const sub = aplanarFuncional(capa.config);
      const entradasExternas = nodos[0];
      const mapaEntradas = new Map(sub.entradas.map((ref, i) => [ref[0], entradasExternas[i]]));
      for (const interna of sub.capas) {
        interna.nodos = interna.nodos.map((nodo) => nodo.map((ref) => mapaEntradas.get(ref[0]) ?? ref));
        capas.push(interna);
      }
      sub.salidas.forEach((ref, i) => salidasSub.set(`${capa.config.name}:${i}`, ref));
      continue;
    }
    capas.push({ original: capa, nombre: capa.config.name, nodos });
  }
  const resolver = (ref) => salidasSub.get(`${ref[0]}:${ref[2]}`) ?? ref;
  for (const c of capas) c.nodos = c.nodos.map((nodo) => nodo.map(resolver));
  const entradas = normalizarRefs(cfg.input_layers);
  const salidas = normalizarRefs(cfg.output_layers).map(resolver);
  // Las capas de entrada de submodelos ya no se usan: se descartan
  const usadas = new Set(capas.flatMap((c) => c.nodos.flat().map((r) => r[0])).concat(salidas.map((r) => r[0])));
  const nombresEntrada = new Set(entradas.map((r) => r[0]));
  const finales = capas.filter((c) => c.original.class_name !== "InputLayer"
    || nombresEntrada.has(c.nombre) || usadas.has(c.nombre));
  return { capas: finales, entradas, salidas };
}

function convertirTopologia(modelo) {
  if (modelo.class_name === "Sequential") {
    const capas = modelo.config.layers;
    const entrada = capas.find((c) => c.class_name === "InputLayer");
    const forma = entrada?.config.batch_shape ?? modelo.config.build_input_shape;
    const resto = capas.filter((c) => c.class_name !== "InputLayer").map(convertirCapa);
    resto[0].config.batch_input_shape = forma;
    return {
      topologia: { class_name: "Sequential", config: { name: modelo.config.name, layers: resto } },
      capas: resto.map((c) => ({ clase: c.class_name, config: c.config })),
      formaEntrada: forma,
    };
  }
  if (modelo.class_name === "Functional" || modelo.class_name === "Model") {
    const { capas, entradas, salidas } = aplanarFuncional(modelo.config);
    if (entradas.length !== 1) throw new Error("La web solo admite modelos con una entrada (imagen).");
    const capasTfjs = capas.map((c) => {
      const conv = c.original.class_name === "InputLayer"
        ? { class_name: "InputLayer", config: { name: c.nombre, dtype: "float32",
            batch_input_shape: c.original.config.batch_shape } }
        : convertirCapa(c.original);
      return { ...conv, name: c.nombre, inbound_nodes: c.nodos.map((nodo) => nodo.map((r) => [r[0], r[1], r[2], {}])) };
    });
    const entradaCfg = capas.find((c) => c.nombre === entradas[0][0]).original.config;
    return {
      topologia: { class_name: "Model", config: { name: modelo.config.name, layers: capasTfjs,
        input_layers: entradas, output_layers: salidas } },
      capas: capasTfjs.map((c) => ({ clase: c.class_name, config: c.config })),
      formaEntrada: entradaCfg.batch_shape,
    };
  }
  throw new Error(`Tipo de modelo no soportado: ${modelo.class_name}`);
}

// --------------------------------------------------------------------------- //
// Pesos: model.weights.h5 (Keras 3) -> fragmentos .bin + manifiesto
// --------------------------------------------------------------------------- //

function nombresDePesos(clase, config) {
  switch (clase) {
    case "Conv2D":
    case "Dense":
      return config.use_bias === false ? ["kernel"] : ["kernel", "bias"];
    case "DepthwiseConv2D":
      return config.use_bias === false ? ["depthwise_kernel"] : ["depthwise_kernel", "bias"];
    case "BatchNormalization":
      return [...(config.scale !== false ? ["gamma"] : []), ...(config.center !== false ? ["beta"] : []),
        "moving_mean", "moving_variance"];
    default:
      return [];
  }
}

/** Igual que keras.src.utils.naming.to_snake_case (Conv2D -> conv2d, ReLU -> re_lu). */
function aSnakeCase(nombre) {
  return nombre.replace(/\W+/g, "").replace(/(.)([A-Z][a-z]+)/g, "$1_$2")
    .replace(/([a-z])([A-Z])/g, "$1_$2").toLowerCase();
}

/**
 * Ruta de cada capa dentro de model.weights.h5. Keras 3 no guarda los pesos con el nombre de la capa,
 * sino con el nombre de su clase en snake_case y un sufijo por orden de aparición
 * (layers/conv2d, layers/conv2d_1, layers/functional/layers/batch_normalization...).
 */
function rutasDePesos(modelo, prefijo = "", mapa = new Map()) {
  const usados = {};
  let capas = modelo.config.layers;
  if (modelo.class_name === "Sequential") capas = capas.filter((c) => c.class_name !== "InputLayer");
  for (const capa of capas) {
    let nombre = aSnakeCase(capa.class_name);
    if (nombre in usados) nombre = `${nombre}_${++usados[nombre]}`;
    else usados[nombre] = 0;
    const ruta = `${prefijo}/layers/${nombre}`;
    if (["Functional", "Model", "Sequential"].includes(capa.class_name)) rutasDePesos(capa, ruta, mapa);
    else mapa.set(capa.config.name, `${ruta}/vars`);
  }
  return mapa;
}

function extraerPesos(rutaH5, capas, modeloKeras) {
  const archivo = new h5wasm.File(rutaH5, "r");
  try {
    const rutas = rutasDePesos(modeloKeras);
    const especificaciones = [];
    const datos = [];
    for (const { clase, config } of capas) {
      const nombres = nombresDePesos(clase, config);
      if (!nombres.length) continue;
      const grupo = rutas.has(config.name) ? archivo.get(rutas.get(config.name)) : null;
      if (!(grupo instanceof h5wasm.Group)) throw new Error(`No se encontraron los pesos de la capa ${config.name}.`);
      const claves = [...grupo.keys()].sort((a, b) => Number(a) - Number(b));
      if (claves.length !== nombres.length) {
        throw new Error(`La capa ${config.name} tiene ${claves.length} variables; se esperaban ${nombres.length}.`);
      }
      claves.forEach((clave, i) => {
        const ds = grupo.get(clave);
        const valores = Float32Array.from(ds.value);
        especificaciones.push({ name: `${config.name}/${nombres[i]}`, shape: ds.shape, dtype: "float32" });
        datos.push(valores);
      });
    }
    return { especificaciones, datos };
  } finally {
    archivo.close();
  }
}

function escribirModelo(topologia, especificaciones, datos, version) {
  fs.rmSync(SALIDA_MODELO, { recursive: true, force: true });
  fs.mkdirSync(SALIDA_MODELO, { recursive: true });
  const total = datos.reduce((s, d) => s + d.byteLength, 0);
  const buffer = Buffer.alloc(total);
  let desplazamiento = 0;
  for (const d of datos) {
    Buffer.from(d.buffer, d.byteOffset, d.byteLength).copy(buffer, desplazamiento);
    desplazamiento += d.byteLength;
  }
  const n = Math.max(1, Math.ceil(total / TAM_FRAGMENTO));
  const rutas = [];
  for (let i = 0; i < n; i++) {
    const nombre = `group1-shard${i + 1}of${n}.bin`;
    fs.writeFileSync(path.join(SALIDA_MODELO, nombre), buffer.subarray(i * TAM_FRAGMENTO, (i + 1) * TAM_FRAGMENTO));
    rutas.push(nombre);
  }
  const modeloJson = {
    format: "layers-model",
    generatedBy: `keras ${version}`,
    convertedBy: "web/scripts/convertir_modelo.mjs",
    modelTopology: { ...topologia, keras_version: version, backend: "tensorflow" },
    weightsManifest: [{ paths: rutas, weights: especificaciones }],
  };
  fs.writeFileSync(path.join(SALIDA_MODELO, "model.json"), JSON.stringify(modeloJson));
  return { total, n };
}

function copiarDatos(dirArtefactos, formaEntrada, arquitecturaDetectada) {
  fs.mkdirSync(SALIDA_DATOS, { recursive: true });
  for (const nombre of ["clases.json", "metricas.json"]) {
    const origen = path.join(dirArtefactos, nombre);
    if (!fs.existsSync(origen)) throw new Error(`Falta ${nombre} en ${dirArtefactos}.`);
    fs.copyFileSync(origen, path.join(SALIDA_DATOS, nombre));
  }
  let final = {};
  const rutaConfig = path.join(dirArtefactos, "config_final.json");
  if (fs.existsSync(rutaConfig)) final = JSON.parse(fs.readFileSync(rutaConfig, "utf8"));
  const config = {
    nombre: final.nombre ?? "Modelo final",
    arquitectura: final.arquitectura ?? arquitecturaDetectada,
    umbral_revision: final.umbral_revision ?? 0.5,
    alto: formaEntrada[1],
    ancho: formaEntrada[2],
    convertido: new Date().toISOString(),
  };
  fs.writeFileSync(path.join(SALIDA_DATOS, "config.json"), JSON.stringify(config, null, 2));
  return config;
}

async function main() {
  const dirArtefactos = buscarArtefactos();
  if (!dirArtefactos) {
    if (fs.existsSync(path.join(SALIDA_MODELO, "model.json"))) {
      log("AVISO: no se encontró modelo_final.keras; se publica el modelo ya convertido en public/model/.");
      return;
    }
    throw new Error("No se encontró artefactos/modelo_final.keras (ni ../artefactos ni ./artefactos).");
  }
  log("Artefactos:", dirArtefactos);
  await h5wasm.ready;

  const zip = unzipSync(new Uint8Array(fs.readFileSync(path.join(dirArtefactos, "modelo_final.keras"))));
  const config = JSON.parse(strFromU8(zip["config.json"]));
  const version = zip["metadata.json"] ? JSON.parse(strFromU8(zip["metadata.json"])).keras_version : "3";
  const rutaH5 = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "keras-")), "model.weights.h5");
  fs.writeFileSync(rutaH5, zip["model.weights.h5"]);

  const { topologia, capas, formaEntrada } = convertirTopologia(config);
  const { especificaciones, datos } = extraerPesos(rutaH5, capas, config);
  const { total, n } = escribirModelo(topologia, especificaciones, datos, version);
  const arquitectura = capas.some((c) => c.clase === "DepthwiseConv2D") ? "mobilenetv2" : "cnn";
  const cfg = copiarDatos(dirArtefactos, formaEntrada, arquitectura);

  const parametros = especificaciones.reduce((s, e) => s + e.shape.reduce((a, b) => a * b, 1), 0);
  log(`Modelo ${config.class_name} (Keras ${version}): ${capas.length} capas, ${especificaciones.length} tensores, `
    + `${parametros.toLocaleString("es-CO")} parámetros, ${(total / 1e6).toFixed(1)} MB en ${n} archivo(s).`);
  log(`Entrada ${cfg.alto} x ${cfg.ancho} | arquitectura ${cfg.arquitectura} | umbral de revisión ${cfg.umbral_revision}`);
}

main().catch((error) => {
  console.error("[convertir_modelo] ERROR:", error.message);
  process.exit(1);
});
