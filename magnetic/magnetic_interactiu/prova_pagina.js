/* prova_pagina.js -- fer correr la pagina sencera dins de node, sense navegador.
 *
 * PER QUE EXISTEIX
 * ----------------
 * compara.py demostra que motor.js diu el mateix que el Python. Aixo NO
 * demostra que la pagina en tregui el mapa bo: entremig hi ha la reticula de
 * tres nivells, el repartiment de files, i la pintura, on la y del mapa va cap
 * amunt i la del llenc cap avall. Un signe canviat alla no el veuria ningu si
 * el mapa no es compares amb res.
 *
 * La pagina, al capdavall, no es res mes que JavaScript que crida motor.js i
 * pinta. Posant-li un DOM de mentida a sota corre igual dins de node, i llavors
 * el mapa que en surt es pot comparar amb el del Python.
 *
 * EL RESULTAT ES LLEGEIX PEL LLENC, no per cap variable de dins. Es a posta:
 * aixi la prova cobreix tambe la pintura. Com que del llenc en surten colors i
 * no codis, es tornen a passar a codi amb la paleta, que te els cinc destins
 * ben separats i no admet confusio.
 *
 * ES PASSA DUES VEGADES, una per cada cami que la pagina pot agafar:
 *   - sense --fils: no hi ha Worker, la pagina se n'adona i calcula al fil
 *     principal a trossos. Es el que passa si el navegador bloqueja els
 *     treballadors obrint un fitxer local.
 *   - amb --fils: hi ha un Worker de mentida que executa el mateix codi font del
 *     treballador amb el seu propi "self". No hi ha dos fils de debo -- els
 *     missatges van per setTimeout -- pero si que hi ha la glu del treballador i
 *     el protocol de missatges, que es el que es pot trencar. Si la glu es
 *     dolenta, les files no arriben mai i la pagina es queda al 0 %.
 *
 * EL QUE NO COBREIX, i val mes dir-ho: el TEMPS de caiguda. Del llenc en surt el
 * color, i el color diu el desti i prou. El temps ja el comprova l'autotest
 * incrustat, punt per punt i exacte.
 *
 * PER QUE eval I NO EL MODUL vm. Mesurat en aquesta maquina: el mateix bucle
 * numeric va 59 vegades mes a poc a poc dins d'un vm.createContext que amb un
 * eval global, perque V8 no optimitza igual a traves de la frontera del
 * context. Amb el vm, aquesta prova trigava vuit minuts.
 *
 *   node prova_pagina.js <cami de l'html> <finestra json en metres> [--fils]
 *   node prova_pagina.js <cami de l'html> <finestra> --vista N
 */

"use strict";

const fs = require("fs");

const CAMI = process.argv[2];
const FINESTRA = JSON.parse(process.argv[3] || "[-0.12,0.12,-0.12,0.12]");
// --vista N: la pagina s'obre amb la vista incrustada N (0 = la general) i no
// calcula res. Es la prova que aquella vista es descomprimeix i es col.loca be.
const iVista = process.argv.indexOf("--vista");
const VISTA = iVista >= 0 ? Number(process.argv[iVista + 1]) : null;
if (!CAMI) {
  process.stderr.write("falta el cami de mapa-interactiu.html\n");
  process.exit(2);
}

// --- treure els quatre <script> de la pagina ------------------------------
// L'ordre es motor.js, analisi.js, les constants i pagina.js. Si algun dia
// se n'hi posa un altre, aixo peta amb un missatge clar en comptes de
// carregar-ne un de dolent i donar un resultat mut.
const html = fs.readFileSync(CAMI, "utf8");
const blocs = [];
for (let a = 0; ; ) {
  const i = html.indexOf("<script>", a);
  if (i < 0) break;
  const j = html.indexOf("</script>", i);
  blocs.push(html.slice(i + 8, j));
  a = j + 9;
}
if (blocs.length !== 4) {
  process.stderr.write(`esperava 4 blocs de <script> a la pagina i n'hi ha ${blocs.length}\n`);
  process.exit(2);
}

// --- el DOM de mentida ----------------------------------------------------
// Nomes hi ha el que pagina.js toca de debo. Si algun dia en toca mes, aixo
// petara amb un missatge clar en comptes de donar un resultat mut i dolent.
let imatge = null;
const elements = {};

function element(id) {
  return {
    id,
    style: {}, dataset: {}, classList: { add() {}, remove() {}, toggle() {} },
    textContent: "", innerHTML: "", className: "", hidden: false, disabled: false,
    addEventListener() {}, setAttribute() {}, setPointerCapture() {},
    // La galeria en demana els marcs de figura per enganxar-hi el clic. Aqui no
    // hi ha cap DOM de debo, o sigui que no n'hi ha cap: retornar una llista
    // buida es el que toca i deixa que la pagina continui.
    querySelectorAll: () => [],
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 601, height: 601 }),
    getContext: () => ({
      createImageData(w, h) {
        imatge = { data: new Uint8ClampedArray(w * h * 4), width: w, height: h };
        return imatge;
      },
      putImageData() {},
      // Els anells dels imants es dibuixen en un llenc a part. Aqui no hi ha
      // cap dibuix de debo, pero els metodes han d'existir: si no, la pagina
      // peta dins de node i la prova no diria si el MAPA es bo.
      clearRect() {}, beginPath() {}, arc() {}, stroke() {},
    }),
  };
}

globalThis.document = {
  getElementById(id) { return (elements[id] = elements[id] || element(id)); },
  querySelectorAll(sel) {
    const clau = sel.indexOf("resolucio") >= 0 ? "nivell" : "vista";
    const valors = clau === "nivell" ? ["0", "1", "2"] : ["conques", "temps"];
    return valors.map((v) => { const e = element(sel + v); e.dataset[clau] = v; return e; });
  },
};
// node porta un navigator propi que nomes te getter, o sigui que no s'hi pot
// assignar a sobre; s'ha de redefinir la propietat. El mateix per si algun dia
// n'hi afegeix mes que xoquin amb els noms del navegador.
function global(nom, valor) {
  Object.defineProperty(globalThis, nom, { value: valor, writable: true, configurable: true });
}
global("navigator", { hardwareConcurrency: 4 });
global("requestAnimationFrame", (f) => setTimeout(f, 0));
global("self", globalThis);
global("window", globalThis);

// ELS DOS CAMINS DE LA PAGINA. Sense Worker ni Blob, provaTreballadors() falla i
// la pagina se'n va sola al fil principal. Amb --fils s'hi posa un Worker de
// mentida que executa el MATEIX codi font del treballador -- el que surt de
// FONT_MOTOR mes la glu -- amb el seu propi "self", i que passa els missatges
// per setTimeout. No hi ha dos fils de debo, pero si que hi ha les dues coses
// que es poden trencar: la glu del treballador i el protocol de missatges.
if (process.argv.includes("--fils")) {
  const blobs = new Map();
  let seguent = 0;
  global("Blob", function (trossos) { this.text = trossos.join(""); });
  global("URL", {
    createObjectURL(b) { const k = "blob:" + seguent++; blobs.set(k, b.text); return k; },
  });
  global("Worker", function (url) {
    const jo = this;
    const ambit = {};
    ambit.self = ambit;
    ambit.postMessage = (m) => setTimeout(() => { if (jo.onmessage) jo.onmessage({ data: m }); }, 0);
    new Function("self", blobs.get(url))(ambit);
    jo.postMessage = (m) => setTimeout(() => { if (ambit.onmessage) ambit.onmessage({ data: m }); }, 0);
    jo.terminate = () => { ambit.onmessage = null; };
  });
}

const executa = (0, eval);                 // eval INDIRECTE: corre a l'abast global
executa(blocs[0]);                         // motor.js
executa(blocs[1]);                         // analisi.js
executa(blocs[2]);                         // les constants

// Amb --vista N es prova NOMES que la pagina sapiga descomprimir i col.locar
// aquella vista incrustada: no calcula res. Es una prova a part perque un error
// alla no seria de calcul sino de zlib, de base64 o d'indexs. Es fa que la
// pagina s'obri amb ella, que es el cami de qualsevol vista: el boto i
// l'engegada criden el mateix posaVista().
//
// Sense --vista, la finestra es una altra que la de la vista general, les
// vistes incrustades s'anul.len i la pagina ho ha de calcular tot, que es el
// cas normal.
if (VISTA !== null) {
  if (!(VISTA >= 0 && VISTA < VISTES.length)) {
    process.stderr.write(`la pagina porta ${VISTES.length} vistes i s'ha demanat la ${VISTA}\n`);
    process.exit(2);
  }
  executa("VISTES = [VISTES[" + VISTA + "]]; FINESTRA_INICIAL = VISTES[0].finestra.slice();");
} else {
  executa("FINESTRA_INICIAL = " + JSON.stringify(FINESTRA) + "; VISTES = [];");
  // A 601 serien 361.201 punts amb un sol fil: mitja hora llarga. L'arnes es
  // queda a 151, que ja cobreix la reticula, el repartiment de files i la
  // pintura, que es el que aquesta prova ha de vigilar.
  executa("OBJECTIU_PER_DEFECTE = 1;");
}
// Des del 28/10 la pagina ja no refina de fons: s'atura al nivell demanat, o
// sigui que quan la barra arriba al 100 % ja es pot mirar el llenc.
executa(blocs[3]);                         // pagina.js

// --- esperar que acabi i llegir el llenc ----------------------------------
const colorACodi = new Map();
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)).join(",");
colorACodi.set(hex(COLORS.no_convergit), -1);
colorACodi.set(hex(COLORS.centre), 0);
COLORS.imants.forEach((h, i) => colorACodi.set(hex(h), i + 1));

const t0 = Date.now();
let quiet = 0;
const rellotge = setInterval(() => {
  const barra = elements["barra"];
  if (!barra || barra.style.width !== "100.0%") { quiet = 0; return; }
  if (++quiet < 3) return;
  clearInterval(rellotge);

  // Es llegeix sempre a salt 4, o sigui 151 x 151 dels 601 de la reticula.
  // Amb --vista el mapa incrustat es mes fi, pero llegir-ne una part ja diu si
  // s'ha descomprimit i col.locat be, que es el que alla es vol comprovar; i
  // comparar 361.201 punts amb el Python costaria molts minuts de mes.
  const N = imatge.width;
  const salt = 4;
  const codis = [];
  for (let k = 0; k <= N - 1; k += salt) {         // k = fila del MAPA, y cap amunt
    const fila = [];
    for (let i = 0; i <= N - 1; i += salt) {
      const o = 4 * ((N - 1 - k) * N + i);         // a fila del LLENC, y cap avall
      const clau = `${imatge.data[o]},${imatge.data[o + 1]},${imatge.data[o + 2]}`;
      fila.push(colorACodi.has(clau) ? colorACodi.get(clau) : -99);
    }
    codis.push(fila);
  }
  process.stdout.write(JSON.stringify({
    finestra: VISTA !== null ? VISTES[0].finestra : FINESTRA,
    vista: VISTA !== null ? VISTES[0].etiqueta : null,
    // El que la pagina diu de la resolucio: en una vista incrustada ha de ser
    // "Ja calculada a 601 x 601", que es el que es veu.
    estat_calcul: elements["estat-calcul"] ? elements["estat-calcul"].textContent : null,
    salt: salt,
    punts: codis.length,
    codis: codis,
    segons: (Date.now() - t0) / 1000,
    com_va: elements["com-va"].textContent,
    autotest: elements["autotest"].textContent,
  }));
}, 100);
