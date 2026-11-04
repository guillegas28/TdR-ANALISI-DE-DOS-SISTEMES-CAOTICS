/* analisi.js -- frontera i dimensio fractal, traduit a JavaScript.
 *
 * ES UNA COPIA, NO LA FONT DE VERITAT, igual que motor.js. L'original es
 * magnetic/analisi.py i es el que mana; d'aquell surten els numeros que van a
 * la memoria del treball. Aquest fitxer nomes existeix perque la pagina pugui
 * dir el percentatge de frontera i la dimensio DE LA FINESTRA QUE ESTAS
 * MIRANT, que el Python no pot saber per endavant.
 *
 * Si algun dia els dos fitxers no diuen el mateix, el que s'arregla es AQUEST.
 * Ho comprova la prova 5 de magnetic/magnetic_interactiu/compara.py, que passa
 * matrius fetes amb el Python per totes dues implementacions i exigeix que en
 * surtin els mateixos numeros.
 *
 * PER QUE UN FITXER A PART I NO DINS DE motor.js. motor.js es la copia de
 * fisica.py + motor.py, i aquella correspondencia d'un a un val la pena
 * conservar-la: quan es busca on es tradueix una funcio del Python, no s'ha
 * d'endevinar. Aixo es la copia d'analisi.py i te el seu fitxer.
 *
 * AIXO NO ES FISICA. No hi ha cap equacio ni cap integracio: nomes es mira la
 * matriu de destins ja calculada i es compta. Per aixo pot viure fora del
 * motor sense trencar la ratlla que separa la fisica de la pintura.
 *
 * Es un script pla que deixa un sol objecte global, AnalisiMagnetica, pel
 * mateix motiu que motor.js: res de moduls ES.
 */

(function (arrel) {
  "use strict";

  // Els vuit veins, en el mateix ordre que VUIT_VEINS del Python. L'ordre no
  // canvia el resultat -- es un O logic -- pero es manté igual per no haver de
  // pensar-hi mai.
  var VUIT_VEINS = [
    [-1, -1], [-1, 0], [-1, 1],
    [0, -1], [0, 1],
    [1, -1], [1, 0], [1, 1],
  ];

  // ------------------------------------------------------------------
  // La mascara de frontera -- traduccio de analisi.mascara_de_frontera
  // ------------------------------------------------------------------
  // Un pixel es de frontera si TE ALGUN VEI D'UN ALTRE COLOR.
  //
  // El Python ho fa amb vuit comparacions de matrius desplacades, i cada
  // desplacament compara nomes la part que se solapa; aixi les vores queden
  // ben tractades sense cap cas especial. Aqui es fa amb un bucle per pixel
  // que salta els veins de fora, que dona EXACTAMENT el mateix conjunt: un
  // pixel de la vora simplement te menys veins amb qui comparar-se.
  //
  // No es una traduccio literal linia a linia perque no cal que ho sigui: aqui
  // no hi ha coma flotant, nomes comparacions de sencers. La regla de l'ordre
  // sagrat de les operacions es de motor.js, on una suma canviada mou l'ultim
  // bit; comptar veins diferents no te ultim bit.
  //
  // codis es una matriu de fila x columna aplanada, com la de la pagina.
  function mascaraDeFrontera(codis, files, columnes) {
    var fora = new Uint8Array(files * columnes);
    for (var j = 0; j < files; j++) {
      for (var i = 0; i < columnes; i++) {
        var idx = j * columnes + i;
        var meu = codis[idx];
        for (var v = 0; v < VUIT_VEINS.length; v++) {
          var jj = j + VUIT_VEINS[v][0];
          var ii = i + VUIT_VEINS[v][1];
          if (jj < 0 || jj >= files || ii < 0 || ii >= columnes) continue;
          if (codis[jj * columnes + ii] !== meu) { fora[idx] = 1; break; }
        }
      }
    }
    return fora;
  }

  function percentatgeDeFrontera(codis, files, columnes) {
    var mascara = mascaraDeFrontera(codis, files, columnes);
    var n = 0;
    for (var a = 0; a < mascara.length; a++) n += mascara[a];
    return 100.0 * n / (files * columnes);
  }

  // ------------------------------------------------------------------
  // Box-counting -- traduccio de analisi.compta_capses i
  // analisi.dimensio_per_box_counting
  // ------------------------------------------------------------------
  // Quantes capses de costat x costat contenen algun pixel de la mascara,
  // dins del rectangle regio = {j0, i0, hh, ww}. Sense regio, el de la matriu
  // sencera.
  //
  // EL RETALL ES IMPORTANT. El Python talla la matriu a un multiple del costat
  // (hh = (h // costat) * costat) i llenca la resta. Aqui es fa igual, i no per
  // comoditat: si una banda comptes la franja sobrant i l'altra no, els comptes
  // no coincidirien i la prova 5 fallaria per un motiu que no te res a veure
  // amb la frontera. dimensioPerBoxCounting no la hi deixa arribar mai: li
  // passa la regio comuna, que totes les capses divideixen.
  function comptaCapses(mascara, files, columnes, costat, regio) {
    if (!regio) regio = { j0: 0, i0: 0, hh: files, ww: columnes };
    var hh = Math.floor(regio.hh / costat) * costat;
    var ww = Math.floor(regio.ww / costat) * costat;
    if (hh === 0 || ww === 0) return 0;
    var n = 0;
    for (var j = regio.j0; j < regio.j0 + hh; j += costat) {
      for (var i = regio.i0; i < regio.i0 + ww; i += costat) {
        var hi = false;
        for (var dj = 0; dj < costat && !hi; dj++) {
          for (var di = 0; di < costat; di++) {
            if (mascara[(j + dj) * columnes + (i + di)]) { hi = true; break; }
          }
        }
        if (hi) n++;
      }
    }
    return n;
  }

  // Els costats de capsa -- traduccio de analisi.escales: 1, 2, 4, 8... fins a
  // max(4, min(files, columnes) / 8), i mai mes grans que la matriu.
  function escales(files, columnes, costatMin, costatMax) {
    costatMin = costatMin || 1;
    if (costatMax === undefined || costatMax === null) {
      costatMax = Math.max(4, Math.floor(Math.min(files, columnes) / 8));
    }
    var fora = [];
    for (var s = costatMin; s <= costatMax && s <= Math.min(files, columnes); s *= 2) {
      fora.push(s);
    }
    return fora;
  }

  // TOTES LES ESCALES DAMUNT DE LA MATEIXA AREA -- traduccio de
  // analisi.regio_comuna. El rectangle central amb els costats multiples de la
  // capsa mes gran. Fins al 31/10 cada escala retallava la seva franja, i un
  // quadrat ple de 601 donava D = 2,023 en comptes de 2.
  function regioComuna(files, columnes, costats) {
    var gran = Math.max.apply(null, costats);
    var hh = Math.floor(files / gran) * gran;
    var ww = Math.floor(columnes / gran) * gran;
    return { j0: Math.floor((files - hh) / 2), i0: Math.floor((columnes - ww) / 2),
             hh: hh, ww: ww };
  }

  // Pendent de log N contra log(1/s) pel metode dels minims quadrats. El
  // Python fa np.polyfit(x, y, 1) i en pren el pendent; per a una recta aixo es
  // la formula tancada de sota, i no cal cap biblioteca.
  //
  // Amb menys de tres escales no hi ha res a ajustar: alla peta amb un missatge
  // i aqui es torna null, que la pagina ja sap ensenyar.
  //
  // L'INTERVAL es el minim i el maxim dels pendents de cada parell d'escales
  // consecutives (analisi.pendents_locals). La dimensio sempre hi cau a dins.
  function dimensioPerBoxCounting(mascara, files, columnes, costatMin, costatMax) {
    var totes = escales(files, columnes, costatMin, costatMax);
    var costats = [], comptes = [];
    if (totes.length < 3) {
      return { dimensio: null, interval: null, costats: costats, comptes: comptes };
    }
    var regio = regioComuna(files, columnes, totes);
    for (var t = 0; t < totes.length; t++) {
      var n = comptaCapses(mascara, files, columnes, totes[t], regio);
      if (n > 0) { costats.push(totes[t]); comptes.push(n); }
    }
    if (costats.length < 3) {
      return { dimensio: null, interval: null, costats: costats, comptes: comptes };
    }

    var m = costats.length, sx = 0, sy = 0;
    var xs = [], ys = [];
    for (var k = 0; k < m; k++) {
      xs.push(Math.log(1.0 / costats[k]));
      ys.push(Math.log(comptes[k]));
      sx += xs[k];
      sy += ys[k];
    }
    var mx = sx / m, my = sy / m;
    var num = 0.0, den = 0.0;
    for (var q = 0; q < m; q++) {
      num += (xs[q] - mx) * (ys[q] - my);
      den += (xs[q] - mx) * (xs[q] - mx);
    }
    var locals = [];
    for (var r = 1; r < m; r++) {
      locals.push((ys[r] - ys[r - 1]) / (xs[r] - xs[r - 1]));
    }
    return {
      dimensio: den === 0.0 ? null : num / den,
      interval: [Math.min.apply(null, locals), Math.max.apply(null, locals)],
      costats: costats,
      comptes: comptes,
    };
  }

  // El que la pagina crida: els dos numeros d'una tirada.
  function analitza(codis, files, columnes) {
    var mascara = mascaraDeFrontera(codis, files, columnes);
    var n = 0;
    for (var a = 0; a < mascara.length; a++) n += mascara[a];
    var box = dimensioPerBoxCounting(mascara, files, columnes);
    return {
      percentatgeDeFrontera: 100.0 * n / (files * columnes),
      pixelsDeFrontera: n,
      dimensio: box.dimensio,
      interval: box.interval,
      costats: box.costats,
      comptes: box.comptes,
    };
  }

  var AnalisiMagnetica = {
    VUIT_VEINS: VUIT_VEINS,
    mascaraDeFrontera: mascaraDeFrontera,
    percentatgeDeFrontera: percentatgeDeFrontera,
    comptaCapses: comptaCapses,
    escales: escales,
    regioComuna: regioComuna,
    dimensioPerBoxCounting: dimensioPerBoxCounting,
    analitza: analitza,
  };

  arrel.AnalisiMagnetica = AnalisiMagnetica;
  if (typeof module !== "undefined" && module.exports) module.exports = AnalisiMagnetica;
})(typeof self !== "undefined" ? self : this);
