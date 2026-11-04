/* prova.js -- l'arnes que deixa cridar motor.js des del Python.
 *
 * No te cap fisica a dins. Nomes llegeix feines per l'entrada estandard,
 * les passa a motor.js i escriu els resultats per la sortida estandard. Qui
 * decideix quines feines s'han de fer i qui compara els numeros es
 * magnetic/magnetic_interactiu/compara.py.
 *
 *     node prova.js < feines.json > resultats.json
 *
 * PER QUE JSON I NO UN TEXT AMB DECIMALS. Un double escrit amb %.17g i tornat
 * a llegir hauria de sobreviure, pero depen del format que triis i d'un
 * arrodoniment que no controles. En canvi tant el json de Python com el
 * JSON.stringify de JavaScript escriuen els numeros amb els digits justos per
 * poder-los recuperar EXACTAMENT, i totes dues bandes els tornen a llegir amb
 * arrodoniment correcte. O sigui que el double que surt d'una banda es el
 * mateix bit per bit a l'altra, i qualsevol diferencia que trobi compara.py es
 * del calcul i no del transport. Aixo es la condicio per poder parlar d'ulps.
 *
 * Tot el que no siguin resultats va per l'error estandard, perque la sortida
 * estandard ha de ser JSON i prou.
 */

"use strict";

const fs = require("fs");
const path = require("path");
const motor = require(path.join(__dirname, "motor.js"));
const analisi = require(path.join(__dirname, "analisi.js"));

// Els parametres arriben amb els noms del Python (im_x, im_y, im_K) i motor.js
// els vol amb els seus (imX, imY, imK). La conversio es aqui i no al motor
// perque el motor no ha de saber res del Python.
function parametres(d) {
  return {
    L: d.L,
    g: d.g,
    b: d.b,
    n: d.n,
    d0: d.d0,
    imX: Float64Array.from(d.im_x),
    imY: Float64Array.from(d.im_y),
    imK: Float64Array.from(d.im_K),
  };
}

// Un numero que no es finit no cap al JSON (Infinity hi surt com a null i NaN
// no hi cap). Es converteix a null a posta i compara.py ja sap llegir-ho.
function jsonNumero(v) {
  return Number.isFinite(v) ? v : null;
}

// --- les tres feines ------------------------------------------------------

// 1. Trajectoria a curt termini: N passos de RK4 des d'un estat.
//    Es el que ha de coincidir AL BIT: no hi ha cap classificacio pel mig,
//    nomes les equacions i l'integrador.
function feinaPassos(f) {
  const p = parametres(f.parametres);
  const estat = Float64Array.from(f.estat);
  for (let k = 0; k < f.passos; k++) {
    motor.passRk4(estat, f.dt, p.L, p.g, p.b, p.n, p.d0, p.imX, p.imY, p.imK);
  }
  return { estat: Array.from(estat) };
}

// 2. Desenllacos d'una graella sencera.
//    Les taules de destins arriben JA CALCULADES des del Python, a posta: aixi
//    l'unica cosa que es compara es la integracio. Que els punts de repos que
//    calcula el JavaScript pel seu compte siguin bons o no es la feina 3, i
//    barrejar les dues coses faria que no se sabes d'on ve una diferencia.
//
//    L'ordre de recorregut es el mateix que el de magnetic.motor.graella_nucli:
//    j sobre y0 (les files) i i sobre x0 (les columnes).
function feinaDesenllac(f) {
  const p = parametres(f.parametres);
  const destX = Float64Array.from(f.dest_x);
  const destY = Float64Array.from(f.dest_y);
  const destCodi = Int32Array.from(f.dest_codi);
  const destRadi2 = Float64Array.from(f.dest_radi2);
  const x0 = f.x0, y0 = f.y0;

  const codi = new Array(x0.length * y0.length);
  const temps = new Array(x0.length * y0.length);
  let a = 0;
  for (let j = 0; j < y0.length; j++) {
    for (let i = 0; i < x0.length; i++) {
      const r = motor.desenllacNucli(
        x0[i], y0[j], p.L, p.g, p.b, p.n, p.d0, p.imX, p.imY, p.imK,
        destX, destY, destCodi, destRadi2,
        f.dt, f.passos, f.passos_confirma, f.passos_confirma_centre);
      codi[a] = r[0];
      temps[a] = r[1];
      a++;
    }
    if ((j + 1) % 10 === 0) {
      process.stderr.write(`      node: fila ${j + 1}/${y0.length}\n`);
    }
  }
  return { codi: codi, temps: temps };
}

// 3. Punts de repos i estabilitat del centre, calculats pel JavaScript tot sol
//    (biseccio pel radi i formula tancada del 2x2), per comparar-los amb els
//    que el Python treu amb fsolve i numpy.
function feinaRepos(f) {
  const p = parametres(f.parametres);
  const r = motor.puntsDeRepos(p);
  const rig = motor.rigidesaAlCentre(p);
  return {
    punts: r.punts.map((q) =>
      q === null ? null : { x: q.x, y: q.y, r: q.r, residu: q.residu }),
    centre_estable: r.centreEstable,
    valors_propis_J: motor.valorsPropis(rig.J),
    factor_critic: jsonNumero(motor.factorCritic(p)),
  };
}

// 4. Nomes delta^(n+1), sense res mes al voltant.
//    Es la sospitosa numero u de qualsevol diferencia entre les dues
//    implementacions, i val mes tenir-la mesurada tota sola que deduir-la de
//    les altres proves. Aixi la frase "pow no arrodoneix igual a Python i a V8"
//    deixa de ser una excusa i passa a ser un numero.
function feinaPotencia(f) {
  return { valors: f.bases.map((d) => motor.potencia(d, f.exponent)) };
}

// 5. Frontera i dimensio fractal d'una matriu de destins ja calculada.
//    La matriu arriba FETA des del Python, a posta: aixi l'unica cosa que es
//    compara es el compte, i no s'hi barreja cap diferencia d'integracio.
//    Es la prova que analisi.js es pot fer servir per als numeros en viu de
//    la pagina.
function feinaAnalisi(f) {
  const r = analisi.analitza(f.codis, f.files, f.columnes);
  return {
    percentatge_de_frontera: r.percentatgeDeFrontera,
    pixels_de_frontera: r.pixelsDeFrontera,
    dimensio: jsonNumero(r.dimensio),
    interval: r.interval === null ? null : r.interval.map(jsonNumero),
    costats: r.costats,
    comptes: r.comptes,
  };
}

const FEINES = {
  passos: feinaPassos,
  desenllac: feinaDesenllac,
  repos: feinaRepos,
  potencia: feinaPotencia,
  analisi: feinaAnalisi,
};

// --- el programa ----------------------------------------------------------

function main() {
  const entrada = JSON.parse(fs.readFileSync(0, "utf8"));
  process.stderr.write("      node " + process.version + ": " +
    motor.configura({ potenciaRapida: !!entrada.potencia_rapida }) + "\n");

  const resultats = entrada.feines.map((f) => {
    const fer = FEINES[f.tipus];
    if (!fer) throw new Error(`feina desconeguda: ${f.tipus}`);
    return fer(f);
  });

  process.stdout.write(JSON.stringify({ resultats: resultats }));
}

main();
