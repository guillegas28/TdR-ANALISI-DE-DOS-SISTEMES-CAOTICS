/* pagina.js -- el llenc, els treballadors i els controls.
 *
 * AQUI NO HI HA FISICA. Tota la fisica es a motor.js, que es la copia validada
 * del nucli de Python. Aquest fitxer nomes decideix QUINS punts s'han de
 * calcular, els reparteix entre fils i pinta el resultat. Si algun dia un mapa
 * surt malament, la pregunta "es de la fisica o de la pintura?" es contesta
 * sabent que aquesta ratlla separa les dues coses.
 *
 * construeix.py hi incrusta les constants de dalt (PARAMETRES, CRITERI, DT,
 * COLORS, MAGMA, AUTOTEST, VISTES, FONT_MOTOR).
 *
 * LES CINC DECISIONS QUE EXPLIQUEN EL FITXER
 * ------------------------------------------
 * 1. GRAELLES QUE ENCAIXEN. Els quatre nivells son de 76, 151, 301 i 601 punts.
 *    Els salts son 8, 4, 2 i 1, i tots divideixen 600, o sigui que la separacio
 *    es parteix exactament per la meitat a cada nivell i un de cada dos punts
 *    del nivell fi cau A SOBRE d'un que ja tens calculat. El refinament
 *    progressiu surt llavors GRATIS:
 *        5.776 + 17.025 + 67.800 + 270.600 = 361.201 = 601^2,
 *    que es exactament el que costaria calcular directament el nivell fi. La
 *    diferencia es que la primera imatge la veus en un parell de segons.
 *
 *    Perque encaixin de debo, les coordenades NO es calculen per nivell: hi ha
 *    una sola reticula de 601 punts i cada nivell la recorre a salts. Aixi el
 *    punt del nivell gruixut i el del fi son bit a bit el mateix numero.
 *
 * 2. RES ES CALCULA DUES VEGADES. La pila d'ampliacions guarda LES DADES, no
 *    nomes la finestra: tornar enrere es restaurar i repintar, no recalcular.
 *    Hi ha pila endavant tambe, perque desfer i refer siguin simetrics. Cada
 *    entrada ocupa uns 2,2 MB (601^2 x 6 bytes) i se'n guarden 25 com a molt.
 *
 * 3. S'ATURA A LA RESOLUCIO TRIADA, I LES VISTES DE LA FIGURA JA VENEN FETES.
 *    Fins al 27/10, en acabar el nivell demanat la pagina continuava refinant
 *    sola fins a 601 ("calcul de fons"). Amb la llei mesurada una finestra de
 *    la frontera triga molt mes, i allo volia dir fins a mitja hora amb el
 *    processador al 80 %. Ara (decisio d'en Guillem, 27/10) s'atura al nivell
 *    del boto, que comença a 301. Les quatre finestres de l'autosemblanca
 *    (general, x4, x16 i x64) venen calculades a 601 dins del fitxer:
 *    s'obren a l'instant i, mentre ets en una d'elles, el boto marca 601.
 *
 * 4. QUANTS TREBALLADORS, I QUE NO ES FACIN I ES DESFACIN. Mesurat en aquesta
 *    maquina (12 nuclis logics, 6 fisics): l'acceleracio se satura als 10 fils
 *    (5,67x) i del 10 al 12 no es guanya res. Per aixo s'engeguen
 *    hardwareConcurrency-2 i no mes.
 *
 *    I es fan UN COP i es reaprofiten. Matar-los i tornar-los a fer a cada
 *    nivell voldria dir tornar a analitzar 19 kB de motor.js deu vegades per
 *    ampliacio i estrenar el JIT cada cop. Es cancel.la per jobId, i les files
 *    es reparteixen de poques en poques perque en cancel.lar quedi poca cosa
 *    començada.
 *
 *    El resultat NO depen de quants fils facis servir. Cada punt escriu nomes a
 *    la seva cel.la i no hi ha cap suma acumulada, o sigui que no hi ha res que
 *    pugui dependre de l'ordre. Comprovat al Python amb 1, 2, 3, 4, 6, 8, 10 i
 *    12 fils: la matriu surt identica.
 *
 * 5. SI EL NAVEGADOR NO DEIXA FER TREBALLADORS. Obrint un fitxer local, algun
 *    navegador els bloqueja. No es dona per fet que funcionin: se'n fa un i se
 *    li demana que digui "soc aqui"; si no contesta a temps, es passa sol a un
 *    bucle a trossos al fil principal. Es mes lent, pero funciona i s'avisa.
 */

(function () {
  "use strict";

  var motor = window.MotorMagnetic;
  // La copia d'analisi.py, per als numeros de la finestra que s'esta mirant.
  // Ha passat la prova 5 de compara: els mateixos numeros que el Python sobre
  // la mateixa matriu.
  var analisi = window.AnalisiMagnetica;

  // ------------------------------------------------------------------
  // La reticula
  // ------------------------------------------------------------------
  var FI = 600;                        // index maxim: 601 punts, de 0 a 600
  var NIVELLS = [
    { punts: 76, salt: 8 },
    { punts: 151, salt: 4 },
    { punts: 301, salt: 2 },
    { punts: 601, salt: 1 },
  ];
  var CEL = (FI + 1) * (FI + 1);
  var MAX_HISTORIA = 25;               // ~55 MB de dades guardades

  var parametres = {
    L: PARAMETRES.L, g: PARAMETRES.g, b: PARAMETRES.b,
    n: PARAMETRES.n, d0: PARAMETRES.d0,
    imX: Float64Array.from(PARAMETRES.im_x),
    imY: Float64Array.from(PARAMETRES.im_y),
    imK: Float64Array.from(PARAMETRES.im_K),
  };
  var criteri = {
    rCaptura: CRITERI.r_captura_m,
    rCentre: CRITERI.r_centre_m,
    tConfirma: CRITERI.t_confirma_s,
    tConfirmaCentre: CRITERI.t_confirma_centre_s,
    tMax: CRITERI.t_max_s,
  };
  var dest = motor.destins(parametres, criteri);
  var PASSOS = Math.round(criteri.tMax / DT);
  var PASSOS_CONFIRMA = Math.round(criteri.tConfirma / DT);
  // El centre es mira mes estona que els imants: el seu pou es molt mes pla
  // (motor.Criteri.t_confirma_centre, 25/10/2026).
  var PASSOS_CONFIRMA_CENTRE = Math.round(criteri.tConfirmaCentre / DT);

  // ------------------------------------------------------------------
  // Estat
  // ------------------------------------------------------------------
  var estat = {
    finestra: FINESTRA_INICIAL.slice(),   // [xMin, xMax, yMin, yMax] en METRES
    // DUES RESOLUCIONS, i no es redundant. objectiuTriat es la del boto que
    // has premut (o la d'arrencada, que ve de construeix.py; prova_pagina.js
    // l'abaixa perque l'arnes no hagi de calcular 361.201 punts amb un sol fil).
    // objectiu es amb la que s'ensenya el que tens davant: en una vista
    // incrustada es 601, perque ja esta feta, i en ampliar des d'alla es torna
    // a la triada. Vegeu objectiuPer().
    objectiuTriat: typeof OBJECTIU_PER_DEFECTE !== "undefined" ? OBJECTIU_PER_DEFECTE : 1,
    objectiu: typeof OBJECTIU_PER_DEFECTE !== "undefined" ? OBJECTIU_PER_DEFECTE : 1,
    vista: "conques",
    enrere: [],                           // {finestra, dades, rangTemps}
    endavant: [],
    dades: null,
    jobId: 0,
    calculant: false,
    rangTemps: null,
  };

  // Una vista incrustada s'ensenya sencera; qualsevol altra, a la resolucio
  // triada.
  function objectiuPer(d) {
    return d.incrustada ? d.nivellFet : estat.objectiuTriat;
  }

  function dadesNoves() {
    return {
      codis: new Int8Array(CEL),
      temps: new Float32Array(CEL),
      fet: new Uint8Array(CEL),
      nivellFet: -1,
    };
  }

  function pas() { return (estat.finestra[1] - estat.finestra[0]) / FI; }
  function coordX(k) { return estat.finestra[0] + k * pas(); }
  function coordY(k) { return estat.finestra[2] + k * pas(); }

  // Les columnes que toca calcular d'una fila. Si la fila ja existia al nivell
  // gruixut, els punts parells ja hi son i nomes calen els senars.
  function columnes(salt, nomesSenars) {
    var fora = [];
    for (var i = nomesSenars ? salt : 0; i <= FI; i += nomesSenars ? 2 * salt : salt) {
      fora.push(i);
    }
    return fora;
  }

  // ------------------------------------------------------------------
  // La bassa de treballadors
  // ------------------------------------------------------------------
  var GLU_TREBALLADOR = [
    "var comu = null;",
    "self.onmessage = function (e) {",
    "  var m = e.data;",
    "  if (m.tipus === 'ping') { self.postMessage({tipus:'pong'}); return; }",
    "  if (m.tipus === 'prepara') { comu = m; return; }",
    "  if (m.tipus !== 'fila') return;",
    "  var c = comu, y = c.yMin + m.k * c.pas;",
    "  var is = [], i;",
    "  for (i = m.nomesSenars ? m.salt : 0; i <= c.fi; i += m.nomesSenars ? 2*m.salt : m.salt) is.push(i);",
    "  var codis = new Int8Array(is.length), temps = new Float32Array(is.length);",
    "  for (var a = 0; a < is.length; a++) {",
    "    var r = MotorMagnetic.desenllacNucli(",
    "      c.xMin + is[a]*c.pas, y,",
    "      c.p.L, c.p.g, c.p.b, c.p.n, c.p.d0, c.p.imX, c.p.imY, c.p.imK,",
    "      c.destX, c.destY, c.destCodi, c.destRadi2,",
    "      c.dt, c.passos, c.passosConfirma, c.passosConfirmaCentre);",
    "    codis[a] = r[0]; temps[a] = r[1];",
    "  }",
    "  self.postMessage({tipus:'fila', jobId: m.jobId, k: m.k, salt: m.salt,",
    "    nomesSenars: m.nomesSenars, codis: codis, temps: temps},",
    "    [codis.buffer, temps.buffer]);",
    "};",
  ].join("\n");

  var bassa = [];               // els treballadors, fets un sol cop
  var potTreballadors = null;
  var enCurs = null;            // la passada de calcul que s'esta fent

  function quantsFils() {
    var hc = navigator.hardwareConcurrency || 4;
    return Math.max(1, Math.min(12, hc - 2));
  }

  function fontTreballador() {
    return URL.createObjectURL(
      new Blob([FONT_MOTOR + "\n" + GLU_TREBALLADOR], { type: "text/javascript" }));
  }

  // No es dona per fet que els treballadors funcionin: se'n fa un i se li
  // demana que contesti. Si no ho fa en mig segon, es que el navegador els ha
  // bloquejat (passa obrint un fitxer local) i es tira pel fil principal.
  function provaTreballadors() {
    return new Promise(function (resol) {
      var t;
      try { t = new Worker(fontTreballador()); } catch (e) { resol(false); return; }
      var rellotge = setTimeout(function () { t.terminate(); resol(false); }, 500);
      t.onmessage = function () { clearTimeout(rellotge); t.terminate(); resol(true); };
      t.onerror = function () { clearTimeout(rellotge); t.terminate(); resol(false); };
      t.postMessage({ tipus: "ping" });
    });
  }

  function obreBassa() {
    var url = fontTreballador();
    for (var w = 0; w < quantsFils(); w++) {
      var t = new Worker(url);
      t.onmessage = function (e) {
        if (e.data.tipus === "fila") rebFila(e.data, this);
      }.bind(t);
      bassa.push(t);
    }
  }

  // ------------------------------------------------------------------
  // Una passada de calcul
  // ------------------------------------------------------------------
  // Cancel.lar es incrementar jobId: el que arribi amb un jobId vell s'ignora i
  // la cua del que encara no s'ha repartit es llenca. Els treballadors NO es
  // maten: acabar la fila que tenen entre mans costa menys que tornar-los a fer.
  function rebFila(m, treballador) {
    var feina = enCurs;
    if (!feina || m.jobId !== estat.jobId) return;

    var is = columnes(m.salt, m.nomesSenars);
    var d = feina.dades;
    for (var a = 0; a < is.length; a++) {
      var idx = m.k * (FI + 1) + is[a];
      d.codis[idx] = m.codis[a];
      d.temps[idx] = m.temps[a];
      d.fet[idx] = 1;
    }
    // El que es pinta no passa mai de la resolucio demanada: si abaixes el boto
    // a mig calcul, el nivell que s'estava fent s'acaba pero no canvia la
    // imatge sota els peus. Aixi el boto continua volent dir alguna cosa.
    if (d === estat.dades && feina.nivell <= estat.objectiu) dibuixaFila(m.k, m.salt, is);
    feina.fetes++;
    posaBarra(feina.fetes / feina.total);

    if (feina.cua.length && treballador) reparteix(treballador, feina);
    if (feina.fetes === feina.total) {
      d.nivellFet = Math.max(d.nivellFet, feina.nivell);
      var acabar = feina.alAcabar;
      enCurs = null;
      acabar();
    }
  }

  function reparteix(treballador, feina) {
    var f = feina.cua.shift();
    treballador.postMessage({
      tipus: "fila", jobId: feina.jobId,
      k: f.k, salt: f.salt, nomesSenars: f.nomesSenars,
    });
  }

  function calcula(nivell, alAcabar) {
    var salt = NIVELLS[nivell].salt;
    var cua = [];
    for (var k = 0; k <= FI; k += salt) {
      cua.push({ k: k, salt: salt, nomesSenars: nivell > 0 && k % (2 * salt) === 0 });
    }

    var jobId = ++estat.jobId;
    var feina = {
      jobId: jobId, nivell: nivell, dades: estat.dades,
      cua: cua, total: cua.length, fetes: 0, alAcabar: alAcabar,
    };
    enCurs = feina;

    if (potTreballadors) {
      var comu = {
        tipus: "prepara", fi: FI,
        xMin: estat.finestra[0], yMin: estat.finestra[2], pas: pas(),
        p: parametres, dt: DT, passos: PASSOS, passosConfirma: PASSOS_CONFIRMA,
        passosConfirmaCentre: PASSOS_CONFIRMA_CENTRE,
        destX: dest.destX, destY: dest.destY,
        destCodi: dest.destCodi, destRadi2: dest.destRadi2,
      };
      // Dues files per treballador d'entrada: prou perque no quedi parat entre
      // fila i fila, i prou poques perque cancel.lar deixi poca cosa començada.
      bassa.forEach(function (t) { t.postMessage(comu); });
      bassa.forEach(function (t) {
        for (var q = 0; q < 2 && feina.cua.length; q++) reparteix(t, feina);
      });
    } else {
      // Sense treballadors: el mateix calcul a trossos, tornant el control al
      // navegador entre fila i fila perque la pagina no es quedi morta.
      (function seguent() {
        if (jobId !== estat.jobId || !feina.cua.length) return;
        var f = feina.cua.shift();
        var is = columnes(f.salt, f.nomesSenars);
        var codis = new Int8Array(is.length), temps = new Float32Array(is.length);
        var y = coordY(f.k);
        for (var q = 0; q < is.length; q++) {
          var r = motor.desenllacNucli(
            coordX(is[q]), y,
            parametres.L, parametres.g, parametres.b, parametres.n, parametres.d0,
            parametres.imX, parametres.imY, parametres.imK,
            dest.destX, dest.destY, dest.destCodi, dest.destRadi2,
            DT, PASSOS, PASSOS_CONFIRMA, PASSOS_CONFIRMA_CENTRE);
          codis[q] = r[0]; temps[q] = r[1];
        }
        rebFila({ jobId: jobId, k: f.k, salt: f.salt, nomesSenars: f.nomesSenars,
                  codis: codis, temps: temps }, null);
        if (feina.cua.length) setTimeout(seguent, 0);
      })();
    }
  }

  // Demana els nivells l'un darrere l'altre i S'ATURA a l'objectiu. L'objectiu
  // es mira a cada nivell, o sigui que pujar el boto a mig calcul el fa seguir.
  function refina(desDe) {
    var nivell = desDe;
    (function seguent() {
      if (nivell > estat.objectiu) {
        estat.calculant = false;
        posaBarra(1); posaEstat();
        return;
      }
      var aquest = nivell++;
      estat.calculant = true;
      posaEstat();
      calcula(aquest, function () {
        // En acabar cada nivell hi ha temps nous i el rang canvia: es refa i es
        // repinta, o el que ja hi ha quedaria amb l'escala de color vella.
        if (estat.vista === "temps") { calculaRangTemps(); repintaTot(); }
        actualitzaLectura();
        setTimeout(seguent, 0);
      });
    })();
  }

  function nouCalcul() {
    estat.jobId++;
    enCurs = null;
    estat.dades = dadesNoves();
    estat.rangTemps = null;
    estat.objectiu = estat.objectiuTriat;
    netejaLlenc();
    actualitzaLectura();
    refina(0);
  }

  // Ensenya les dades que ja hi ha a estat.dades i, si no arriben a l'objectiu,
  // calcula el que falta. Es el final comu de tornar enrere, reiniciar i obrir
  // una vista incrustada.
  function ensenyaDades() {
    estat.objectiu = objectiuPer(estat.dades);
    netejaLlenc();
    repintaTot();
    actualitzaLectura();
    if (estat.dades.nivellFet < estat.objectiu) refina(estat.dades.nivellFet + 1);
    else { estat.calculant = false; posaBarra(1); posaEstat(); }
  }

  // Anar a una finestra nova guardant d'on venies, AMB LES DADES.
  function vesA(finestra) {
    empeny(estat.enrere);
    estat.endavant.length = 0;
    estat.finestra = finestra;
    nouCalcul();
  }

  function empeny(pila) {
    if (!estat.dades) return;
    pila.push({ finestra: estat.finestra.slice(), dades: estat.dades,
                rangTemps: estat.rangTemps });
    while (pila.length > MAX_HISTORIA) pila.shift();
  }

  // Tornar enrere o endavant: NO es recalcula res. Es restauren les dades que
  // ja hi eren i es repinta. Si aquella finestra s'havia quedat a mig calcular,
  // es continua des d'on era fins a la resolucio triada.
  function salta(dEs, aOn) {
    if (!dEs.length) return;
    var q = dEs.pop();
    empeny(aOn);
    estat.jobId++;
    enCurs = null;
    estat.finestra = q.finestra;
    estat.dades = q.dades;
    estat.rangTemps = q.rangTemps;
    ensenyaDades();
  }

  // ------------------------------------------------------------------
  // Pintar
  // ------------------------------------------------------------------
  var llenc = document.getElementById("mapa");
  var ctx = llenc.getContext("2d");
  var imatge = ctx.createImageData(FI + 1, FI + 1);
  var pendentDePintar = false;

  var RGB = {};
  RGB[-1] = hexARgb(COLORS.no_convergit);
  RGB[0] = hexARgb(COLORS.centre);
  COLORS.imants.forEach(function (h, i) { RGB[i + 1] = hexARgb(h); });

  function hexARgb(h) {
    return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
  }

  function netejaLlenc() {
    var g = hexARgb(COLORS.graella);
    for (var a = 0; a < imatge.data.length; a += 4) {
      imatge.data[a] = g[0]; imatge.data[a + 1] = g[1];
      imatge.data[a + 2] = g[2]; imatge.data[a + 3] = 255;
    }
    ctx.putImageData(imatge, 0, 0);
  }

  // EL COLOR DEL TEMPS, EN UN SOL LLOC. El pinten dos: els pixels del mapa i la
  // barra de la llegenda. Si cadascu es fes el seu, algun dia la llegenda diria
  // un color i el mapa un altre -- que es exactament la mena d'error que ja va
  // passar amb el rang de temps quan es mantenia a ma des de quatre llocs.
  // f va de 0 (el mes rapid) a 1 (el mes lent).
  function colorDeTemps(f) {
    return MAGMA[Math.max(0, Math.min(255, Math.round(255 * (1 - f))))];
  }

  function colorDe(idx) {
    if (estat.vista === "conques") return RGB[estat.dades.codis[idx]];
    if (estat.dades.codis[idx] === motor.NO_CLASSIFICAT) return RGB[-1];
    var r = estat.rangTemps;
    if (!r) return RGB[-1];
    var f = r[1] > r[0] ? (estat.dades.temps[idx] - r[0]) / (r[1] - r[0]) : 0;
    return colorDeTemps(f);
  }

  function bloc(k, i, salt, mig) {
    var c = colorDe(k * (FI + 1) + i);
    for (var dy = -mig; dy < salt - mig; dy++) {
      var py = FI - (k + dy);                 // la y del mapa va cap amunt
      if (py < 0 || py > FI) continue;
      for (var dx = -mig; dx < salt - mig; dx++) {
        var px = i + dx;
        if (px < 0 || px > FI) continue;
        var o = 4 * (py * (FI + 1) + px);
        imatge.data[o] = c[0]; imatge.data[o + 1] = c[1];
        imatge.data[o + 2] = c[2]; imatge.data[o + 3] = 255;
      }
    }
  }

  // Cada punt calculat pinta un bloc de salt x salt centrat a sobre seu: als
  // nivells gruixuts es veu quadriculat, i al nivell fi cada bloc es un pixel.
  // Mai s'interpola: un color que no s'ha calculat no s'ensenya.
  function dibuixaFila(k, salt, is) {
    var mig = salt >> 1;
    for (var a = 0; a < is.length; a++) bloc(k, is[a], salt, mig);
    demanaPintada();
  }

  function demanaPintada() {
    if (pendentDePintar) return;
    pendentDePintar = true;
    requestAnimationFrame(function () {
      pendentDePintar = false;
      ctx.putImageData(imatge, 0, 0);
    });
  }

  // Repintar-ho tot de nou: es fa nivell per nivell, del gruixut al fi, perque
  // els blocs grossos omplin els forats i els fins els tapin a sobre.
  function nivellVisible() {
    return estat.dades ? Math.min(estat.dades.nivellFet, estat.objectiu) : -1;
  }

  function repintaTot() {
    // EL RANG DE TEMPS ES DERIVAT, NO ESTAT. Es va tenir guardat i mantingut a
    // ma des de quatre llocs, i dos se n'oblidaven: tornant enrere a una
    // finestra que s'havia calculat mirant conques, el rang valia null i el mapa
    // es pintava TOT GRIS sense recuperar-se mai. Ara es demana aqui, que es per
    // on passa qualsevol repintat sencer, i per tant no pot quedar desaparellat.
    assegurarRangTemps();
    var d = estat.dades;
    for (var nv = 0; nv <= nivellVisible(); nv++) {
      var salt = NIVELLS[nv].salt, mig = salt >> 1;
      for (var k = 0; k <= FI; k += salt) {
        for (var i = 0; i <= FI; i += salt) {
          if (d.fet[k * (FI + 1) + i]) bloc(k, i, salt, mig);
        }
      }
    }
    demanaPintada();
  }

  function assegurarRangTemps() {
    if (estat.vista === "temps" && !estat.rangTemps) calculaRangTemps();
  }

  // El minim i el maxim del temps de caiguda d'AQUESTA finestra, que son els que
  // fan l'escala de color. Es per finestra i no global a posta: aixi cada
  // ampliacio fa servir tota la paleta i s'hi veu l'estructura.
  function calculaRangTemps() {
    var d = estat.dades, lo = Infinity, hi = -Infinity;
    for (var a = 0; a < CEL; a++) {
      if (!d.fet[a] || d.codis[a] === motor.NO_CLASSIFICAT) continue;
      if (d.temps[a] < lo) lo = d.temps[a];
      if (d.temps[a] > hi) hi = d.temps[a];
    }
    estat.rangTemps = lo <= hi ? [lo, hi] : null;
  }

  // ------------------------------------------------------------------
  // Lectures
  // ------------------------------------------------------------------
  function mesura(m) {
    if (m >= 1e-2) return (m * 100).toFixed(3) + " cm";
    if (m >= 1e-3) return (m * 1e3).toFixed(3) + " mm";
    if (m >= 1e-6) return (m * 1e6).toFixed(3) + " µm";
    if (m >= 1e-9) return (m * 1e9).toFixed(3) + " nm";
    return (m * 1e12).toFixed(3) + " pm";
  }

  function fila(t, valor) { return "<tr><td>" + t + "</td><td>" + valor + "</td></tr>"; }

  function actualitzaLectura() {
    var f = estat.finestra;
    var ample = f[1] - f[0];
    var zoom = (FINESTRA_INICIAL[1] - FINESTRA_INICIAL[0]) / ample;
    var nivell = Math.max(0, nivellVisible());
    document.getElementById("lectura").innerHTML =
      fila("x", (f[0] * 100).toFixed(4) + " a " + (f[1] * 100).toFixed(4) + " cm") +
      fila("y", (f[2] * 100).toFixed(4) + " a " + (f[3] * 100).toFixed(4) + " cm") +
      fila("ample", mesura(ample)) +
      fila("ampliació", "×" + (zoom < 10 ? zoom.toFixed(2) : Math.round(zoom).toLocaleString("ca"))) +
      fila("píxel", mesura(ample / FI)) +
      fila("graella", NIVELLS[nivell].punts + " × " + NIVELLS[nivell].punts) +
      (estat.vista === "temps" && estat.rangTemps
        ? fila("temps", estat.rangTemps[0].toFixed(2) + " a " + estat.rangTemps[1].toFixed(2) + " s")
        : "");
    comptaDestins();
    pintaLlegenda();
    numerosEnViu();
    pintaImants();
    posaEstat();
  }

  // ------------------------------------------------------------------
  // Els anells dels imants
  // ------------------------------------------------------------------
  // ON SON I DE QUINA MIDA. 5 mm de radi es LA MIDA REAL DEL DISC: els imants
  // del muntatge son de 10 x 3 mm. No es una mida decorativa, i per aixo
  // l'anell ensenya de passada un limit del model que ja esta declarat -- que
  // els tracta com a PUNTS, quan a 2 cm de distancia un disc de 10 mm no ho es
  // gens.
  //
  // I NO ES ON REPOSA LA MASSA. El punt de repos queda 9,34 mm CAP
  // ENDINS de l'imant (a 4,07 cm en comptes de 5,00; eren 5,85 mm abans del
  // 25/10, sense l'estirada vertical dels imants), perque alla la gravetat
  // i l'imant s'igualen, i el criteri d'aturada mesura des d'alla i no des
  // d'aqui. L'anell diu on hi ha l'imant, no on acaba la massa.
  var RADI_IMANT_M = 0.005;

  var llencImants = document.getElementById("imants");
  var ctxImants = llencImants.getContext("2d");

  // El color de l'imant, tirat cap al blanc. L'anell ha de dir de qui es sense
  // competir amb el mapa: es una referencia geometrica, no una dada.
  function aclarit(c, quant) {
    return "rgba(" + c.map(function (v) {
      return Math.round(v + (255 - v) * quant);
    }).join(",") + ",0.95)";
  }

  function pintaImants() {
    ctxImants.clearRect(0, 0, FI + 1, FI + 1);
    var p = pas();
    var r = RADI_IMANT_M / p;
    for (var i = 0; i < PARAMETRES.im_x.length; i++) {
      var px = (PARAMETRES.im_x[i] - estat.finestra[0]) / p;
      // La y del mapa va cap amunt i la del llenc cap avall, com a bloc().
      var py = FI - (PARAMETRES.im_y[i] - estat.finestra[2]) / p;
      // Ampliant molt, l'imant queda fora de la finestra: llavors no es dibuixa.
      if (px + r < 0 || px - r > FI || py + r < 0 || py - r > FI) continue;

      // Dos tracos: un de blanc translucid a sota i el color a sobre. Amb un de
      // sol, l'anell desapareixeria damunt de la regio negra del centre o es
      // perdria damunt de les conques clares, segons el color que se li posi.
      ctxImants.beginPath();
      ctxImants.arc(px, py, r, 0, 2 * Math.PI);
      ctxImants.lineWidth = 3;
      ctxImants.strokeStyle = "rgba(255,255,255,0.7)";
      ctxImants.stroke();
      ctxImants.lineWidth = 1.5;
      ctxImants.strokeStyle = aclarit(RGB[i + 1], 0.45);
      ctxImants.stroke();
    }
  }

  // LA LLEGENDA DE COLORS SEGUEIX LA VISTA. Amb el mapa pintat per temps, els
  // cinc destins ja no son el que hi ha al llenc: ensenyar-los seria una
  // llegenda que no correspon al dibuix. Es deriva de estat.vista i es repinta
  // des de actualitzaLectura(), igual que assegurarRangTemps(); no s'hi guarda
  // cap estat que es pugui desaparellar.
  function pintaLlegenda() {
    var e = document.getElementById("llegenda-color");
    if (estat.vista === "conques") { e.innerHTML = ""; return; }
    var r = estat.rangTemps;
    if (!r) { e.innerHTML = '<p class="nota">Encara no hi ha cap temps calculat.</p>'; return; }
    // El degradat es mostreja de la MATEIXA funcio que pinta els pixels.
    var parades = [];
    for (var k = 0; k <= 16; k++) {
      var c = colorDeTemps(k / 16);
      parades.push("rgb(" + c.join(",") + ") " + (100 * k / 16).toFixed(1) + "%");
    }
    var mig = (r[0] + r[1]) / 2;
    e.innerHTML =
      '<div class="barra-temps" style="background:linear-gradient(to right,'
      + parades.join(",") + ')"></div>' +
      '<div class="extrems-temps"><span>' + r[0].toFixed(2) + ' s</span>' +
      '<span>' + mig.toFixed(2) + '</span>' +
      '<span>' + r[1].toFixed(2) + ' s</span></div>';
  }

  // ------------------------------------------------------------------
  // Els numeros d'aquesta finestra
  // ------------------------------------------------------------------
  // Es calculen AQUI, sobre el que s'esta veient, perque el Python no ho pot
  // saber per endavant: la finestra la tries tu. Fan servir analisi.js, que es
  // la copia d'analisi.py i que ha passat la prova 5 de compara.
  //
  // NO son els mateixos numeros que els de "ja calculat" ni ho han de ser: allo
  // es va calcular sobre una graella de 600x600 feta amb linspace i aixo es una
  // reticula de 601 recorreguda a salts. Son dues graelles del mateix mapa.
  //
  // LA DIMENSIO VA AMB DOS DECIMALS I EL SEU INTERVAL (31/10): el minim i el
  // maxim dels pendents d'escala a escala, que diuen quant canvia segons
  // l'escala. El tercer decimal no volia dir res.
  function textDimensio(d, interval) {
    if (d === null || d === undefined) return "—";
    return d.toFixed(2) + (interval ? " (" + interval[0].toFixed(2) + "–" +
                                      interval[1].toFixed(2) + ")" : "");
  }

  function numerosEnViu() {
    var e = document.getElementById("viu");
    var nv = nivellVisible();
    if (!estat.dades || nv < 0) { e.innerHTML = ""; return; }

    var salt = NIVELLS[nv].salt, punts = NIVELLS[nv].punts;
    var m = new Int8Array(punts * punts), a = 0;
    for (var k = 0; k <= FI; k += salt) {
      for (var i = 0; i <= FI; i += salt) m[a++] = estat.dades.codis[k * (FI + 1) + i];
    }
    var r = analisi.analitza(m, punts, punts);
    e.innerHTML =
      fila("graella", punts + " × " + punts) +
      fila("frontera", r.percentatgeDeFrontera.toFixed(2) + " %") +
      fila("dimensió", textDimensio(r.dimensio, r.interval));
  }

  // ------------------------------------------------------------------
  // Els parametres amb que s'ha calculat tot
  // ------------------------------------------------------------------
  // CAP NUMERO ESCRIT A MA. Tots surten de PARAMETRES, que es la mateixa
  // constant que llegeix el motor: construint amb --b/--n/--K, aquest bloc ha
  // de dir els valors amb que s'ha fet el mapa, no els del parametres.json.
  //
  // I L'ASTERISC SURT DE PARAMETRES.inventats, no d'una llista escrita aqui.
  // Marcar b, n i K a ma voldria dir que el dia que es mesurin i surtin de
  // _valors_inventats, la pagina els continuaria dient inventats. La marca ha
  // de poder desapareixer sola, igual que ja fa l'avis de dalt de tot.
  //
  // El que SI que va escrit aqui es QUE VOL DIR cada simbol. Els textos del
  // parametres.json fan de 200 a 500 lletres -- el de n tot sol ompliria la
  // columna -- i en 275 px no s'hi poden posar. Es una duplicacio i val mes
  // dir-ho que amagar-la, pero el que es duplica es el SIGNIFICAT d'un simbol,
  // que no canvia el dia que se'n mesuri el valor; el numero i la marca, que si
  // que canvien, no van a ma.
  //
  // I no es cap estat de la interficie mantingut a ma, que es el parany de
  // sempre: son constants fixades en construir i no hi ha res que es pugui
  // desaparellar.
  var XIFRES_ALTES = "⁰¹²³⁴⁵⁶⁷⁸⁹";

  function exponentAlt(k) {
    var s = String(Math.abs(k)).split("").map(function (d) {
      return XIFRES_ALTES.charAt(+d);
    }).join("");
    return (k < 0 ? "⁻" : "") + s;
  }

  function cientific(v) {
    var t = v.toExponential(1).split("e");
    return t[0] + "·10" + exponentAlt(parseInt(t[1], 10));
  }

  function esEnter(v) { return Math.abs(v - Math.round(v)) < 1e-12; }

  function pintaParametres() {
    var P = PARAMETRES;
    var inv = P.inventats || [];
    var K = P.im_K;
    var iguals = K.every(function (k) { return k === K[0]; });

    // LA UNITAT DE K DEPEN DE n: es m^(n+1)/s^2. Amb n = 3 son m^4/s^2, pero n
    // encara s'ha de mesurar i el dia que surti 2,7 aixo deixaria de ser cert.
    var e = P.n + 1;
    var unitatK = "m" + (esEnter(e) ? exponentAlt(Math.round(e)) : "^" + e) + "/s²";

    var files = [
      ["L", P.L.toFixed(3) + " m",
       "llargada del fil, del punt de gir al centre de masses de la massa"],
      ["R", (P.R * 100).toFixed(2) + " cm",
       "radi de la circumferència on hi ha els tres imants"],
      ["d₀", (P.d0 * 100).toFixed(2) + " cm",
       "alçada de la massa sobre el pla dels imants, en repòs al centre"],
      ["g", P.g.toFixed(3) + " m/s²",
       "gravetat, el valor a Catalunya"],
      // Tres decimals: la b mesurada es 0,039, i amb dos sortia "0,04".
      ["b", P.b.toFixed(3) + " s⁻¹",
       "fricció, per unitat de massa" +
       (P.b > 0 ? ". L'amplitud cau a 1/e en 2/b = " + (2 / P.b).toFixed(0) + " s" : "")],
      ["n", esEnter(P.n) ? String(Math.round(P.n)) : P.n.toFixed(2),
       "exponent de la llei dels imants: la força cau com K/Δⁿ. " +
       "És el que més canvia la forma del fractal"],
      ["K", (iguals ? cientific(K[0]) : K.map(cientific).join(" / ")) + " " + unitatK,
       "força dels imants, per unitat de massa" +
       (iguals ? ", igual a tots tres" : ", una per imant")],
      ["m", (P.m * 1000).toFixed(2) + " g",
       "NO surt a les equacions: el mapa no depèn de la massa. Només " +
       "serveix per convertir una força mesurada al laboratori en K"],
    ];

    document.getElementById("parametres").innerHTML = files.map(function (f) {
      var clau = f[0] === "d₀" ? "d0" : f[0];
      var marca = inv.indexOf(clau) >= 0 ? ' <span class="inventat">*</span>' : "";
      return '<dt><span class="simbol">' + f[0] + marca + "</span>" +
             '<span class="valor">' + f[1] + "</span></dt>" +
             "<dd>" + f[2] + "</dd>";
    }).join("");

    document.getElementById("nota-parametres").innerHTML = inv.length
      ? '<span class="inventat">*</span> inventat: no és cap mesura del ' +
        "muntatge. Vegeu l'avís de dalt de tot."
      : "";
  }

  // ------------------------------------------------------------------
  // Els numeros ja calculats i les figures
  // ------------------------------------------------------------------
  // Tots dos surten de GALERIA, que construeix.py omple llegint els JSON de
  // magnetic/figures. Aqui no hi ha ni un numero escrit a ma.
  var galeria = typeof GALERIA !== "undefined" ? GALERIA : null;

  function descriuMuntatge(m) {
    if (!m) return "un altre muntatge";
    var k = m.K && m.K.length ? m.K[0] : null;
    return "b = " + m.b + " · n = " + m.n +
      (k === null ? "" : " · K = " + k.toExponential(2));
  }

  function pintaReferencia() {
    var e = document.getElementById("referencia");
    var nota = document.getElementById("nota-referencia");
    if (!galeria || !galeria.referencia.length) {
      e.innerHTML = "";
      nota.textContent = "No hi ha cap JSON de mètriques: torna a fer els mapes.";
      return;
    }
    var html = '<tr><td></td><td>frontera</td><td>dimensió</td></tr>';
    galeria.referencia.forEach(function (r) {
      html += "<tr><td>" + r.etiqueta + (r.mateix_muntatge ? "" : " *") +
        "</td><td>" + r.frontera.toFixed(2) + " %</td><td>" +
        textDimensio(r.dimensio, r.interval) + "</td></tr>";
    });
    e.className = "lectura tres";
    e.innerHTML = html;

    // Nomes la marca de muntatge, que no es cap explicacio: sense ella les
    // files amb * no voldrien dir res.
    nota.textContent = galeria.referencia.some(function (r) { return !r.mateix_muntatge; })
      ? "Les files amb * són d'un altre muntatge que el mapa de dalt."
      : "";
  }

  function pintaGaleria() {
    var e = document.getElementById("galeria");
    if (!galeria) return;
    var html = "";
    if (galeria.figures.length) html += "<h2>Més informació sobre el mapa</h2>";
    galeria.figures.forEach(function (q) {
      html += '<figure class="figura">' +
        "<h3>" + q.titol + "</h3>" +
        '<div class="marc-figura"><img src="' + q.dades + '" alt="' + q.titol + '"></div>' +
        '<p class="peu-figura">' + q.peu + "</p>" +
        (q.mateix_muntatge === false
          ? '<p class="altre-muntatge">Aquesta figura és d\'un muntatge diferent (' +
            descriuMuntatge(q.muntatge) + ") que el mapa de dalt. Hi és per " +
            "comparar-la, no per llegir-la com si fos del mateix.</p>"
          : "") +
        '<p class="ordre">' + q.ordre + "</p>" +
        "</figure>";
    });
    if (galeria.falten.length) {
      html += '<p class="falta">Falten figures o números, i per això no hi són:\n' +
        galeria.falten.join("\n") +
        "\n\nmagnetic/figures i magnetic/dades són al .gitignore: en un " +
        "repositori acabat de clonar no hi ha res fins que es torna a calcular.</p>";
    }
    e.innerHTML = html;

    // Clicar una figura la posa a mida completa dins d'un marc que es pot
    // desplacar. No s'obre en una pestanya nova: des d'un fitxer local, obrir
    // un data: sovint el bloqueja el navegador.
    e.querySelectorAll(".marc-figura").forEach(function (marc) {
      marc.addEventListener("click", function () { marc.classList.toggle("plena"); });
    });
  }

  function comptaDestins() {
    var d = estat.dades;
    if (!d) return;
    var compte = {}, total = 0;
    for (var a = 0; a < CEL; a++) {
      if (!d.fet[a]) continue;
      compte[d.codis[a]] = (compte[d.codis[a]] || 0) + 1;
      total++;
    }
    if (!total) return;
    var noms = { "-1": "no convergit", "0": "el centre" };
    PARAMETRES.noms.forEach(function (nm, i) { noms[String(i + 1)] = "imant " + nm; });
    var conques = estat.vista === "conques";
    var html = "";
    [1, 2, 3, 0, -1].forEach(function (codi) {
      var n = compte[codi] || 0;
      if (!n && codi !== -1 && codi !== 0) return;
      var c = RGB[codi];
      // La pastilla de color nomes surt quan el mapa esta pintat aixi. A la
      // vista de temps els percentatges continuen sent certs -- son del mateix
      // calcul -- pero el color ja no els representa.
      html += fila(
        (conques
          ? '<span class="pastilla" style="background:rgb(' + c.join(",") + ')"></span>'
          : "") +
        (noms[String(codi)] || "imant " + codi),
        (100 * n / total).toFixed(2) + " %");
    });
    document.getElementById("destins").innerHTML = html;

    var gris = 100 * (compte["-1"] || 0) / total;
    var nota = document.getElementById("nota-gris");
    if (gris > 0.5) {
      nota.className = "nota malament";
      nota.textContent = "El " + gris.toFixed(2) + " % no ha convergit en " +
        criteri.tMax + " s. El gris NO és cap resultat: és un avís que el temps " +
        "màxim s'ha quedat curt i que aquest mapa no està acabat.";
    } else {
      nota.className = "nota";
      nota.textContent = "";
    }
  }

  function posaBarra(f) {
    var b = document.getElementById("barra");
    b.style.width = (100 * Math.min(1, f)).toFixed(1) + "%";
  }

  function posaEstat() {
    document.querySelectorAll("#botons-resolucio button").forEach(function (b) {
      b.setAttribute("aria-pressed", String(Number(b.dataset.nivell) === estat.objectiu));
    });
    document.querySelectorAll("#botons-vista button").forEach(function (b) {
      b.setAttribute("aria-pressed", String(b.dataset.vista === estat.vista));
    });
    // El boto de la vista incrustada que tens davant, si n'es una.
    document.getElementById("botons-vistes").querySelectorAll("button").forEach(function (b) {
      var q = vistes[Number(b.dataset.vistaIncrustada)];
      b.setAttribute("aria-pressed",
        String(!!(q && estat.dades && estat.dades.incrustada
                  && mateixaFinestra(q.v.finestra, estat.finestra))));
    });
    document.getElementById("boto-enrere").disabled = !estat.enrere.length;
    document.getElementById("boto-endavant").disabled = !estat.endavant.length;

    var fet = estat.dades ? estat.dades.nivellFet : -1;
    var e = document.getElementById("estat-calcul");
    if (!estat.calculant) {
      e.textContent = fet >= 0
        ? (estat.dades.incrustada ? "Ja calculada a " : "Calculat a ") +
          NIVELLS[fet].punts + " × " + NIVELLS[fet].punts + "."
        : "";
    } else {
      e.textContent = "Calculant…";
    }
  }

  // ------------------------------------------------------------------
  // Controls
  // ------------------------------------------------------------------
  // Tota ampliacio deixa la finestra QUADRADA: el mapa te la mateixa escala als
  // dos eixos i una finestra rectangular deformaria el fractal.
  function finestraDe(cx, cy, ample) {
    var mig = ample / 2;
    var maxim = PARAMETRES.L * 0.98;
    if (Math.hypot(Math.abs(cx) + mig, Math.abs(cy) + mig) > maxim) return null;
    return [cx - mig, cx + mig, cy - mig, cy + mig];
  }

  var requadre = document.getElementById("requadre");
  var arrossega = null;

  function aMapa(ev) {
    var r = llenc.getBoundingClientRect();
    return { u: (ev.clientX - r.left) / r.width, v: (ev.clientY - r.top) / r.height, r: r };
  }

  llenc.addEventListener("pointerdown", function (ev) {
    var p = aMapa(ev);
    arrossega = { x0: ev.clientX, y0: ev.clientY, u0: p.u, v0: p.v, r: p.r };
    llenc.setPointerCapture(ev.pointerId);
  });

  llenc.addEventListener("pointermove", function (ev) {
    if (!arrossega) return;
    var costat = Math.min(Math.abs(ev.clientX - arrossega.x0), Math.abs(ev.clientY - arrossega.y0));
    var x = Math.min(arrossega.x0, arrossega.x0 + Math.sign(ev.clientX - arrossega.x0) * costat);
    var y = Math.min(arrossega.y0, arrossega.y0 + Math.sign(ev.clientY - arrossega.y0) * costat);
    requadre.hidden = costat < 4;
    requadre.style.left = (x - arrossega.r.left) + "px";
    requadre.style.top = (y - arrossega.r.top) + "px";
    requadre.style.width = costat + "px";
    requadre.style.height = costat + "px";
  });

  llenc.addEventListener("pointerup", function (ev) {
    if (!arrossega) return;
    var a = arrossega; arrossega = null; requadre.hidden = true;
    var costat = Math.min(Math.abs(ev.clientX - a.x0), Math.abs(ev.clientY - a.y0));
    if (costat < 4) return;                       // un clic, no una ampliacio
    var p = aMapa(ev);
    var f = estat.finestra, ample = f[1] - f[0];
    var cx = f[0] + (a.u0 + p.u) / 2 * ample;
    var cy = f[3] - (a.v0 + p.v) / 2 * ample;     // la v del llenc va cap avall
    var nova = finestraDe(cx, cy, ample * costat / a.r.width);
    if (nova) vesA(nova);
  });

  var esperaRoda = null;
  llenc.addEventListener("wheel", function (ev) {
    ev.preventDefault();
    var p = aMapa(ev);
    var f = estat.finestra, ample = f[1] - f[0];
    var factor = ev.deltaY < 0 ? 1 / 1.6 : 1.6;
    // S'amplia al voltant del cursor: el punt de sota el ratoli no es mou.
    var cx = f[0] + p.u * ample, cy = f[3] - p.v * ample;
    var nova = finestraDe(cx + (f[0] + ample / 2 - cx) * factor,
                          cy + (f[2] + ample / 2 - cy) * factor, ample * factor);
    clearTimeout(esperaRoda);
    esperaRoda = setTimeout(function () { if (nova) vesA(nova); }, 150);
  }, { passive: false });

  llenc.addEventListener("dblclick", function (ev) {
    var p = aMapa(ev);
    var f = estat.finestra, ample = f[1] - f[0];
    var nova = finestraDe(f[0] + p.u * ample, f[3] - p.v * ample, ample);
    if (nova) vesA(nova);
  });

  document.getElementById("boto-enrere").addEventListener("click", function () {
    salta(estat.enrere, estat.endavant);
  });
  document.getElementById("boto-endavant").addEventListener("click", function () {
    salta(estat.endavant, estat.enrere);
  });
  document.getElementById("boto-reinicia").addEventListener("click", vesAlaInicial);

  document.querySelectorAll("#botons-resolucio button").forEach(function (b) {
    b.addEventListener("click", function () {
      estat.objectiu = estat.objectiuTriat = Number(b.dataset.nivell);
      // Si ja esta calculat, nomes cal repintar.
      if (estat.dades && estat.dades.nivellFet >= estat.objectiu) {
        repintaTot(); actualitzaLectura(); return;
      }
      if (!estat.calculant) refina(Math.max(0, estat.dades.nivellFet + 1));
      posaEstat();
    });
  });

  document.querySelectorAll("#botons-vista button").forEach(function (b) {
    b.addEventListener("click", function () {
      if (estat.vista === b.dataset.vista) return;
      estat.vista = b.dataset.vista;
      repintaTot();          // ja s'ocupa del rang de temps si en fa falta
      actualitzaLectura();
    });
  });

  // ------------------------------------------------------------------
  // L'autotest
  // ------------------------------------------------------------------
  // compara.py demostra que el motor es bo EN AQUELLA MAQUINA. Aixo demostra
  // que tambe ho es al navegador de qui obri el fitxer, que no es el mateix:
  // un altre V8, un altre sistema, un altre pow.
  //
  // Els punts son de DINS de conca, triats per construeix.py comprovant que el
  // desti no canvia movent-los una desena de mil.limetre. A la frontera no
  // tindria sentit demanar coincidencia exacta.
  function autotest() {
    var malament = 0, pitjorTemps = 0;
    AUTOTEST.forEach(function (c) {
      var r = motor.desenllacNucli(
        c.x, c.y,
        parametres.L, parametres.g, parametres.b, parametres.n, parametres.d0,
        parametres.imX, parametres.imY, parametres.imK,
        dest.destX, dest.destY, dest.destCodi, dest.destRadi2,
        DT, PASSOS, PASSOS_CONFIRMA, PASSOS_CONFIRMA_CENTRE);
      if (r[0] !== c.codi) malament++;
      pitjorTemps = Math.max(pitjorTemps, Math.abs(r[1] - c.temps));
    });
    var e = document.getElementById("autotest");
    if (malament === 0 && pitjorTemps === 0) {
      e.className = "nota be";
      e.textContent = "OK — els " + AUTOTEST.length + " punts de referència donen " +
        "exactament el mateix destí i el mateix temps que el Python.";
    } else {
      e.className = "nota malament";
      e.textContent = "ATENCIÓ — " + malament + " de " + AUTOTEST.length +
        " punts donen un destí diferent del que va calcular el Python" +
        (pitjorTemps ? " i el temps s'aparta fins a " + pitjorTemps.toFixed(3) + " s" : "") +
        ". No et creguis aquest mapa: el motor no fa el mateix en aquest navegador.";
    }
  }

  // ------------------------------------------------------------------
  // Les vistes ja calculades
  // ------------------------------------------------------------------
  // Venen JA CALCULADES dins de l'HTML, fetes amb el Python: la general, amb
  // la qual s'obre la pagina, i les tres ampliacions de la figura
  // d'autosemblanca. No es cap drecera ni cap imatge estirada: son els mateixos
  // desenllacos que la pagina calcularia, calculats pel nucli que mana, a la
  // mateixa reticula. compara.py --pagina comprova cada una contra el Python.
  //
  // COMPRIMIDES amb zlib (28/10): sense comprimir, quatre vistes farien una
  // pagina de 10 MB. Es desfan un sol cop en obrir, amb DecompressionStream,
  // que es asincron; per aixo l'engegada espera aquesta promesa.
  var vistes = [];                       // {v: VISTES[i], codis, passos}

  function deBase64(text) {
    return Uint8Array.from(atob(text), function (c) { return c.charCodeAt(0); });
  }

  function descomprimeix(text) {
    var lector = new ReadableStream({
      start: function (c) { c.enqueue(deBase64(text)); c.close(); },
    }).pipeThrough(new DecompressionStream("deflate")).getReader();
    var trossos = [], total = 0;
    return (function llegeix() {
      return lector.read().then(function (r) {
        if (!r.done) { trossos.push(r.value); total += r.value.length; return llegeix(); }
        var tot = new Uint8Array(total), o = 0;
        trossos.forEach(function (t) { tot.set(t, o); o += t.length; });
        return tot;
      });
    })();
  }

  function preparaVistes() {
    var totes = typeof VISTES !== "undefined" && VISTES ? VISTES : [];
    return Promise.all(totes.map(function (v) {
      // QUANTS BYTES OCUPA CADA COMPTE ES LLEGEIX, NO SE SUPOSA. Amb el t_max a
      // 60 s sempre eren dos; amb 200 s la trajectoria mes lenta en demana mes
      // de 65.535. construeix.py tria l'amplada mirant les dades i la diu aqui.
      if (v.bytes_del_temps !== 2 && v.bytes_del_temps !== 4) {
        throw new Error("bytes_del_temps ha de ser 2 o 4, i a " + v.nom + " es " + v.bytes_del_temps);
      }
      if (NIVELLS[v.nivell].salt !== v.salt) {
        throw new Error("la vista " + v.nom + " no es de la reticula de la pagina");
      }
      return Promise.all([descomprimeix(v.codis), descomprimeix(v.passos)]).then(function (b) {
        var codis = new Int8Array(b[0].buffer);
        var passos = v.bytes_del_temps === 2 ? new Uint16Array(b[1].buffer) : new Uint32Array(b[1].buffer);
        if (codis.length !== v.punts * v.punts || passos.length !== v.punts * v.punts) {
          throw new Error("la vista " + v.nom + " no te " + v.punts + " x " + v.punts + " punts");
        }
        return { v: v, codis: codis, passos: passos };
      });
    }));
  }

  // Dades NOVES a cada obertura, no compartides: aixi res del que es faci
  // despres no pot tocar la vista desada.
  function dadesDeVista(q) {
    var d = dadesNoves();
    var salt = q.v.salt, a = 0;
    for (var k = 0; k <= FI; k += salt) {
      for (var i = 0; i <= FI; i += salt) {
        var idx = k * (FI + 1) + i;
        d.codis[idx] = q.codis[a];
        // El temps ve comptat en PASSOS, no en segons: sempre es k*dt amb k
        // sencer, i refent-ho aqui amb els mateixos dos doubles surt bit a bit
        // el mateix numero que al Python. Un float32 hi hauria perdut xifres.
        d.temps[idx] = q.passos[a] * DT;
        d.fet[idx] = 1;
        a++;
      }
    }
    d.nivellFet = q.v.nivell;
    d.incrustada = true;
    return d;
  }

  function posaVista(q) {
    estat.jobId++;
    enCurs = null;
    estat.finestra = q.v.finestra.slice();
    estat.dades = dadesDeVista(q);
    // Dades noves vol dir rang de temps nou. Sense aixo es quedava el de la
    // finestra d'on venies i els colors sortien equivocats -- que es pitjor que
    // sortir grisos, perque sembla que funcioni.
    estat.rangTemps = null;
    ensenyaDades();
  }

  // Obrir una vista des del seu boto es com ampliar: queda a la pila d'enrere.
  function obreVista(i) {
    var q = vistes[i];
    if (!q) return;
    if (estat.dades && estat.dades.incrustada && mateixaFinestra(q.v.finestra, estat.finestra)) return;
    empeny(estat.enrere);
    estat.endavant.length = 0;
    posaVista(q);
  }

  function pintaBotonsVistes() {
    var e = document.getElementById("botons-vistes");
    e.innerHTML = vistes.map(function (q, i) {
      return '<button type="button" data-vista-incrustada="' + i + '">' + q.v.etiqueta + "</button>";
    }).join("");
    e.querySelectorAll("button").forEach(function (b) {
      b.addEventListener("click", function () { obreVista(Number(b.dataset.vistaIncrustada)); });
    });
    document.getElementById("bloc-vistes").hidden = !vistes.length;
  }

  function mateixaFinestra(a, b) {
    return a[0] === b[0] && a[1] === b[1] && a[2] === b[2] && a[3] === b[3];
  }

  function vistaDe(finestra) {
    for (var i = 0; i < vistes.length; i++) {
      if (mateixaFinestra(vistes[i].v.finestra, finestra)) return vistes[i];
    }
    return null;
  }

  // ------------------------------------------------------------------
  // Engegada
  // ------------------------------------------------------------------
  // Reiniciar ha de fer EXACTAMENT el mateix que obrir el fitxer: si recarregar
  // la pagina amb Ctrl+R fos mes rapid que el boto de reiniciar, el boto estaria
  // mal fet. Per tant no recalcula res.
  //
  // Si la pagina no porta la vista general incrustada (--sense-mapa) pero ja
  // l'has calculada i la tens a la pila, es reaprofita aquella: reiniciar no ha
  // de fer perdre feina que ja esta feta.
  function vesAlaInicial() {
    var guardada = null;
    estat.enrere.concat(estat.endavant).forEach(function (q) {
      if (mateixaFinestra(q.finestra, FINESTRA_INICIAL)
          && (!guardada || q.dades.nivellFet > guardada.dades.nivellFet)) {
        guardada = q;
      }
    });
    estat.enrere.length = 0;
    estat.endavant.length = 0;

    var incrustada = vistaDe(FINESTRA_INICIAL);
    if (incrustada && (!guardada || incrustada.v.nivell >= guardada.dades.nivellFet)) {
      posaVista(incrustada);
      return;
    }
    estat.jobId++;
    enCurs = null;
    estat.finestra = FINESTRA_INICIAL.slice();
    if (!guardada) {
      nouCalcul();               // la pagina no en porta cap: s'ha de calcular
      return;
    }
    estat.dades = guardada.dades;
    estat.rangTemps = guardada.rangTemps;
    ensenyaDades();
  }

  provaTreballadors().then(function (pot) {
    potTreballadors = pot;
    if (pot) obreBassa();
    document.getElementById("com-va").textContent =
      (pot ? bassa.length + " treballadors web (de " + (navigator.hardwareConcurrency || "?") +
             " nuclis; se'n deixen dos lliures perquè els últims no aporten res " +
             "i la pàgina ha de continuar responent)"
           : "SENSE treballadors: aquest navegador els bloqueja obrint un fitxer " +
             "local, i es calcula a trossos al fil principal. Funciona igual, " +
             "però va tants cops més lent com nuclis té la màquina.") +
      " · " + motor.comVa();
    pintaParametres();
    pintaReferencia();
    pintaGaleria();
    autotest();

    // Un navegador antic sense DecompressionStream no pot llegir les vistes: no
    // es cap error del mapa, i la pagina les calcula com si no n'hi hagues.
    if (typeof DecompressionStream === "undefined") {
      throw new Error("aquest navegador no sap descomprimir (no te DecompressionStream)");
    }
    return preparaVistes();
  }).then(function (llestes) {
    vistes = llestes;
  }, function (e) {
    vistes = [];
    document.getElementById("com-va").textContent +=
      " · NO s'han pogut llegir les vistes ja calculades (" + e.message + "): es calcularan aquí.";
  }).then(function () {
    pintaBotonsVistes();
    // Engegar es exactament el mateix que reiniciar, i no per estalviar linies:
    // aixi no hi ha manera que el boto de reiniciar faci una altra cosa que
    // obrir el fitxer.
    vesAlaInicial();
  });
})();
