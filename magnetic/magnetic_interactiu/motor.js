/* motor.js -- el nucli del pendol magnetic, traduit a JavaScript.
 *
 * ES UNA COPIA, NO LA FONT DE VERITAT. L'original es magnetic/motor.py i
 * magnetic/fisica.py, i es el que mana. Aquest fitxer nomes existeix perque un
 * navegador pugui recalcular el mapa en directe quan s'amplia un requadre.
 *
 * Si algun dia els dos fitxers no diuen el mateix, el que s'arregla es AQUEST.
 * Ho comprova magnetic/magnetic_interactiu/compara.py, que s'ha de tornar a
 * passar cada cop que es toqui qualsevol dels dos costats.
 *
 * TRES COSES QUE NO SON TRANSLITERACIO DIRECTA, I PER QUE
 * ------------------------------------------------------
 * 1. Els punts de repos. El Python els busca amb el fsolve de scipy, que aqui
 *    no hi es. Es fan per BISECCIO al llarg del radi que va de l'origen a
 *    l'imant. Vegeu puntsDeRepos(): hi ha explicat per que la solucio hi es, al
 *    radi, i que passa quan les K no son totes iguals.
 * 2. L'estabilitat del centre. El Python demana els valors propis a numpy;
 *    aqui es fa amb la formula tancada del 2x2 simetric, que per a una matriu
 *    de dos per dos es exacta i cap.
 * 3. delta^(n+1). El nucli crida Math.pow i punt. Hi ha un cami rapid per a n
 *    sencer, pero NO dona el mateix numero i el nucli NO el fa servir: viu
 *    apartat, a potencia(), nomes perque la prova 0 de compara.py el mesuri.
 *
 * COM S'USA
 * ---------
 * Es un script pla que deixa un sol objecte global, MotorMagnetic. Aixi val
 * igual per a node (require) que per a la pagina, on construeix.py l'incrusta
 * dins d'un <script> i dins dels treballadors web. Res de moduls ES: un
 * treballador web engegat des d'un fitxer local no els sap carregar.
 *
 * Els "parametres" son sempre un objecte {L, g, b, n, d0, imX, imY, imK}, amb
 * els tres ultims camps vectors paral.lels, com els im_x/im_y/im_K del Python.
 */

(function (arrel) {
  "use strict";

  // Els quatre destins, tal com els fixa magnetic/CLAUDE.md. Han de valer el
  // mateix que a magnetic/motor.py: son els codis que acaben pintats al mapa.
  var CENTRE = 0;
  var NO_CLASSIFICAT = -1;

  // ------------------------------------------------------------------
  // delta^(n+1): l'operacio que decideix si les dues implementacions
  // coincideixen bit a bit
  // ------------------------------------------------------------------
  // El Python fa delta ** (n + 1.0), que amb n float acaba cridant el pow de
  // la biblioteca de C. El JavaScript fa Math.pow, que a V8 es una altra
  // implementacio. Cap de les dues no esta OBLIGADA a arrodonir igual, i per
  // tant una diferencia a l'ultim bit es inevitable.
  //
  // En un sistema caotic, aquest ultim bit canvia el desti d'alguns punts de
  // FRONTERA. No es cap defecte: es el mateix fenomen que magnetic/converge.py
  // ja mesura amb dt/2. Per aixo compara.py separa "dins de conca" (ha de
  // coincidir el 100 %) de "frontera" (hi haura diferencies, i es diu quantes).
  //
  // EL CAMI RAPID, MESURAT I DESCARTAT. Amb n sencer, delta^4 es pot fer amb
  // dues multiplicacions en comptes d'una crida a pow. Pero NO dona el mateix
  // numero: pow arrodoneix un sol cop i les multiplicacions dues vegades, i la
  // primera es paga doble perque elevar al quadrat multiplica per dos l'error
  // relatiu que ja portaves. Mesurat amb 20.001 distancies del rang del mapa,
  // pow(d, 4.0) i (d*d)*(d*d) difereixen en el 50 % dels casos i fins a 2 ulp,
  // mentre que Math.pow contra el pow de Python difereix en el 0,05 % i 1 ulp.
  //
  // Per aixo el nucli NO el fa servir: acceleracioNucli() i rigidesaAlCentre()
  // criden Math.pow directament i punt. potencia() i configura() nomes existeixen
  // perque la prova 0 de compara.py pugui mesurar la diferencia i deixar-la
  // escrita; res del que hi facin no pot arribar a cap mapa.
  //
  // Treure l'embolcall del bucle no es cosmetica: mesurat en aquesta maquina,
  // cridar Math.pow directament en comptes de passar per potencia() fa
  // l'acceleracio un 10,5 % mes rapida, i l'acceleracio es on va tot el temps.
  var potenciaRapida = false;

  function potenciaEntera(base, exponent) {
    if (!Number.isInteger(exponent) || exponent < 0 || exponent > 64) {
      return Math.pow(base, exponent);
    }
    // Exponenciacio per quadrats. Amb exponent 4 son exactament les mateixes
    // operacions que (d*d)*(d*d): el resultat *= 1.0 inicial es exacte.
    var r = 1.0;
    var b = base;
    var k = exponent;
    while (k > 0) {
      if (k & 1) r *= b;
      b *= b;
      k >>= 1;
    }
    return r;
  }

  function potencia(base, exponent) {
    return potenciaRapida ? potenciaEntera(base, exponent) : Math.pow(base, exponent);
  }

  function configura(opcions) {
    if (opcions && typeof opcions.potenciaRapida === "boolean") {
      potenciaRapida = opcions.potenciaRapida;
    }
    return comVa();
  }

  // Que fa el NUCLI, que es sempre el mateix. El que digui configura() no hi
  // entra: nomes canvia el que mesura la prova 0 de compara.py.
  function comVa() {
    return "delta^(n+1) amb Math.pow, el camí fidel al Python";
  }

  // ------------------------------------------------------------------
  // Les equacions -- traduccio de fisica.acceleracio_nucli
  // ------------------------------------------------------------------
  // L'ORDRE DE LES OPERACIONS ES SAGRAT. En coma flotant la suma no es
  // associativa: canviar l'ordre canvia l'ultim bit, i l'ultim bit canvia
  // pixels. Cada linia d'aqui sota esta escrita amb el mateix ordre que la
  // linia equivalent del Python, encara que n'hi hagi alguna que "es podria
  // simplificar". No se simplifica.
  //
  // El resultat va a un vector de dos que es passa de fora, en comptes de
  // retornar-lo: aquesta funcio es crida quatre vegades per pas de RK4 i
  // milers de vegades per pixel, i reservar-hi memoria cada cop es nota.
  function acceleracioNucli(x, y, vx, vy, L, g, b, n, d0, imX, imY, imK, fora) {
    var s = Math.sqrt(L * L - x * x - y * y);
    var d = d0 + L - s;

    // Primer els imants, perque la seva estirada vertical entra a la gravetat
    // (25/10/2026: l'imant fa de gravetat de mes, g -> g + fz). Vegeu el
    // docstring de magnetic/fisica.py.
    var mx = 0.0;
    var my = 0.0;
    var gv = g;
    var exponent = n + 1.0;
    for (var i = 0; i < imX.length; i++) {
      var dx = imX[i] - x;
      var dy = imY[i] - y;
      var delta = Math.sqrt(dx * dx + dy * dy + d * d);
      var forca = imK[i] / Math.pow(delta, exponent);
      mx += forca * dx;
      my += forca * dy;
      gv += forca * d;
    }

    fora[0] = -gv * x / s - b * vx + mx;
    fora[1] = -gv * y / s - b * vy + my;
    return fora;
  }

  // Vector de treball per a l'acceleracio dins del pas de RK4. Es de modul, i
  // aixo vol dir que passRk4 NO es reentrant: el valor s'ha de copiar a
  // variables locals immediatament despres de cada crida, que es exactament el
  // que fa el codi de sota. En JavaScript no hi ha dos fils dins d'un mateix
  // treballador, o sigui que no hi ha cap altre risc.
  var _a = new Float64Array(2);

  // ------------------------------------------------------------------
  // Un pas de RK4 -- traduccio de motor.pas_rk4
  // ------------------------------------------------------------------
  // L'estat entra i surt pel mateix vector de quatre [x, y, vx, vy], pel mateix
  // motiu que abans: no reservar memoria dins del bucle.
  function passRk4(estat, dt, L, g, b, n, d0, imX, imY, imK) {
    var x = estat[0], y = estat[1], vx = estat[2], vy = estat[3];

    acceleracioNucli(x, y, vx, vy, L, g, b, n, d0, imX, imY, imK, _a);
    var a1x = _a[0], a1y = _a[1];
    var k1x = vx, k1y = vy;

    var x2 = x + 0.5 * dt * k1x, y2 = y + 0.5 * dt * k1y;
    var vx2 = vx + 0.5 * dt * a1x, vy2 = vy + 0.5 * dt * a1y;
    acceleracioNucli(x2, y2, vx2, vy2, L, g, b, n, d0, imX, imY, imK, _a);
    var a2x = _a[0], a2y = _a[1];

    var x3 = x + 0.5 * dt * vx2, y3 = y + 0.5 * dt * vy2;
    var vx3 = vx + 0.5 * dt * a2x, vy3 = vy + 0.5 * dt * a2y;
    acceleracioNucli(x3, y3, vx3, vy3, L, g, b, n, d0, imX, imY, imK, _a);
    var a3x = _a[0], a3y = _a[1];

    var x4 = x + dt * vx3, y4 = y + dt * vy3;
    var vx4 = vx + dt * a3x, vy4 = vy + dt * a3y;
    acceleracioNucli(x4, y4, vx4, vy4, L, g, b, n, d0, imX, imY, imK, _a);
    var a4x = _a[0], a4y = _a[1];

    var sis = dt / 6.0;
    estat[0] = x + sis * (k1x + 2.0 * vx2 + 2.0 * vx3 + vx4);
    estat[1] = y + sis * (k1y + 2.0 * vy2 + 2.0 * vy3 + vy4);
    estat[2] = vx + sis * (a1x + 2.0 * a2x + 2.0 * a3x + a4x);
    estat[3] = vy + sis * (a1y + 2.0 * a2y + 2.0 * a3y + a4y);
    return estat;
  }

  // ------------------------------------------------------------------
  // El criteri d'aturada -- traduccio de motor.classifica
  // ------------------------------------------------------------------
  // Nomes mira la POSICIO. Que la massa s'hi hagi quedat o nomes hi passi ho
  // decideix el temps que hi aguanta, no aquesta funcio.
  function classifica(x, y, destX, destY, destCodi, destRadi2) {
    for (var i = 0; i < destX.length; i++) {
      var ex = x - destX[i];
      var ey = y - destY[i];
      if (ex * ex + ey * ey < destRadi2[i]) return destCodi[i];
    }
    return NO_CLASSIFICAT;
  }

  // ------------------------------------------------------------------
  // El desenllac d'un pixel -- traduccio de motor.desenllac_nucli
  // ------------------------------------------------------------------
  // La massa es deixa anar EN REPOS. Retorna [codi, temps de caiguda].
  //
  // SORTIDA ANTICIPADA: es para en el moment que la trajectoria queda
  // classificada. Es el que fa viable el mapa; sense aixo cada pixel costaria
  // el temps maxim sencer.
  //
  // El temps que es retorna es el de l'ENTRADA a la rodona definitiva, no el de
  // la confirmacio: els segons de mirar-la-hi no son temps de caiguda.
  var _estat = new Float64Array(4);

  // passosConfirma val per als imants i passosConfirmaCentre per al centre, com
  // a motor.desenllac_nucli: el pou del centre es molt mes pla (25/10/2026).
  function desenllacNucli(x0, y0, L, g, b, n, d0, imX, imY, imK,
                          destX, destY, destCodi, destRadi2,
                          dt, passos, passosConfirma, passosConfirmaCentre) {
    _estat[0] = x0;
    _estat[1] = y0;
    _estat[2] = 0.0;
    _estat[3] = 0.0;

    var candidat = NO_CLASSIFICAT;
    var seguits = 0;
    var kEntrada = 0;

    for (var k = 0; k < passos; k++) {
      passRk4(_estat, dt, L, g, b, n, d0, imX, imY, imK);
      var x = _estat[0], y = _estat[1];
      if (x * x + y * y >= L * L) return [NO_CLASSIFICAT, passos * dt];

      var codi = classifica(x, y, destX, destY, destCodi, destRadi2);
      if (codi === NO_CLASSIFICAT || codi !== candidat) {
        candidat = codi;
        seguits = codi !== NO_CLASSIFICAT ? 1 : 0;
        kEntrada = k + 1;
      } else {
        seguits += 1;
        if (seguits >= (codi === CENTRE ? passosConfirmaCentre : passosConfirma)) {
          return [codi, kEntrada * dt];
        }
      }
    }

    return [NO_CLASSIFICAT, passos * dt];
  }

  // ------------------------------------------------------------------
  // El centre: atractor o no? -- traduccio de fisica.rigidesa_al_centre
  // ------------------------------------------------------------------
  // J = -(g/L)*I + M , amb
  //     M = SUM_i K_i*[ -(1 + d0/L)*I/D^(n+1) + (n+1)*p*p^T/D^(n+3) ]
  // i D_i la distancia de la massa EN REPOS AL CENTRE a l'imant i, que no es
  // R sino sqrt(R^2 + d0^2): la massa penja mes amunt que el pla dels imants.
  // El d0/L es l'estirada vertical dels imants (25/10/2026), que al centre fa
  // de gravetat de mes; va dins de M perque M sigui tot el que fan els imants.
  //
  // La gravetat estira cap ENDINS i els imants, sumats, cap ENFORA. El centre
  // es atractor si els dos valors propis de J son negatius, i que ho sigui o no
  // es un RESULTAT del treball, no un detall: si guanyen els imants, la regio
  // negra del mapa no existeix.
  //
  // Les matrius son simetriques de 2x2 i es guarden com [xx, xy, yy].
  function rigidesaAlCentre(p) {
    var mxx = 0.0, mxy = 0.0, myy = 0.0;
    for (var i = 0; i < p.imK.length; i++) {
      var px = p.imX[i], py = p.imY[i];
      var D = Math.sqrt(px * px + py * py + p.d0 * p.d0);
      var unitat = (1.0 + p.d0 / p.L) / Math.pow(D, p.n + 1.0);
      var tensor = (p.n + 1.0) / Math.pow(D, p.n + 3.0);
      mxx += p.imK[i] * (-unitat + tensor * px * px);
      mxy += p.imK[i] * (tensor * px * py);
      myy += p.imK[i] * (-unitat + tensor * py * py);
    }
    var gl = p.g / p.L;
    return { M: [mxx, mxy, myy], J: [-gl + mxx, mxy, -gl + myy] };
  }

  // Valors propis d'una simetrica de 2x2, ordenats de petit a gran. Amb dues
  // dimensions la formula tancada es exacta i estalvia portar-se una
  // biblioteca d'algebra al navegador.
  function valorsPropis(sim) {
    var a = sim[0], b = sim[1], c = sim[2];
    var mitjana = 0.5 * (a + c);
    var arrel = Math.sqrt(0.25 * (a - c) * (a - c) + b * b);
    return [mitjana - arrel, mitjana + arrel];
  }

  function centreEsEstable(p) {
    return valorsPropis(rigidesaAlCentre(p).J)[1] < 0.0;
  }

  // Per quant es podrien multiplicar TOTES les K abans que el centre deixes de
  // ser atractor. Menys d'1 vol dir que ja no ho es.
  //
  // Amb els valors de serie surt 1,419, o sigui K critica = 1,42e-4, des del
  // 25/10/2026 (amb l'estirada vertical dels imants al model). Abans sortia
  // 1,286 (K critica 1,29e-4), i aquell numero es va COMPROVAR contra el mapa
  // amb les banderes --K de magnetic.mapa; el nou encara no s'hi ha comprovat.
  function factorCritic(p) {
    var mu = valorsPropis(rigidesaAlCentre(p).M)[1];
    return mu <= 0.0 ? Infinity : (p.g / p.L) / mu;
  }

  // ------------------------------------------------------------------
  // Els punts de repos -- l'equivalent de fisica.punts_dequilibri
  // ------------------------------------------------------------------
  // LA MASSA NO REPOSA A SOBRE DE L'IMANT. Encara que hi sigui a tocar, la
  // gravetat continua estirant cap al centre, i el punt de repos queda uns
  // mil.limetres CAP ENDINS. Amb els valors d'ara son 9,34 mm (5,85 abans del
  // 25/10, sense l'estirada vertical dels imants). El criteri
  // d'aturada mesura distancies A AQUESTS PUNTS i no als imants.
  //
  // PER QUE ES POT FER PER BISECCIO EN COMPTES D'AMB UN fsolve EN DUES
  // DIMENSIONS. Si les tres K son iguals, el muntatge te la simetria de 120
  // graus i el punt de repos de cada imant cau EXACTAMENT sobre el radi que va
  // de l'origen a l'imant: fora del radi, la component transversal de les
  // forces no s'anul.la. Aixo redueix el problema a una equacio d'una sola
  // variable, i llavors la biseccio es preferible a qualsevol Newton: no
  // necessita derivades i no pot divergir.
  //
  // SI LES K NO SON IGUALS aquesta hipotesi ja no val i el punt trobat es una
  // aproximacio. Per aixo el resultat porta un camp "residu" amb el modul de
  // l'acceleracio que hi queda de debo: si les K son iguals surt del nivell del
  // soroll de la coma flotant, i si no, es veu de seguida. compara.py el mira.
  //
  // La funcio que s'anul.la es la component RADIAL de l'acceleracio al llarg
  // del radi. Val la pena saber quina forma te, perque explica el codi:
  //
  //   r -> 0      negativa: la gravetat guanya i torna la massa al centre
  //               (nomes si el centre es estable; si no, ja es positiva)
  //   r mitja     positiva: passat el coll, l'imant ja mana i estira cap enfora
  //   r = R       negativa altre cop: just a sobre de l'imant la seva estirada
  //               es tota vertical i no queda res que compensi la gravetat
  //
  // O sigui que hi ha DUES arrels, i la que interessa es la de mes enfora, la
  // de + a -, que es la estable. La de - a + es el coll entre la conca del
  // centre i la de l'imant, i no serveix de desti.
  //
  // Si l'imant es massa fluix la corba no arriba a ser positiva enlloc, no hi
  // ha cap arrel i aquell imant NO RETE LA MASSA: el desti desapareix del mapa.
  // Es el mateix que al Python, on aquell cas fa petar punts_dequilibri.
  var _acc = new Float64Array(2);

  function acceleracioRadial(r, ux, uy, p) {
    acceleracioNucli(r * ux, r * uy, 0.0, 0.0,
                     p.L, p.g, p.b, p.n, p.d0, p.imX, p.imY, p.imK, _acc);
    return _acc[0] * ux + _acc[1] * uy;
  }

  function puntDeReposDeLimant(i, p, mostres) {
    if (p.imK[i] === 0.0) return null; // imant apagat: no te punt de repos

    var R = Math.hypot(p.imX[i], p.imY[i]);
    if (R === 0.0) return null;
    var ux = p.imX[i] / R, uy = p.imY[i] / R;

    // Escombrada gruixuda buscant l'ultim canvi de signe de + a -. Es comenca
    // una mica enfora de l'origen perque a r = 0 l'acceleracio val zero
    // exactament i el signe no vol dir res.
    mostres = mostres || 512;
    var lo = null, hi = null;
    var rAnterior = R * 1e-3;
    var fAnterior = acceleracioRadial(rAnterior, ux, uy, p);
    for (var k = 1; k <= mostres; k++) {
      var r = R * 1e-3 + (R - R * 1e-3) * (k / mostres);
      var f = acceleracioRadial(r, ux, uy, p);
      if (fAnterior > 0.0 && f <= 0.0) {
        lo = rAnterior;
        hi = r;
      }
      rAnterior = r;
      fAnterior = f;
    }
    if (lo === null) return null; // aquesta K no rete la massa

    // Biseccio fins que no es pot afinar mes: el bucle s'atura tot sol quan el
    // punt mig ja coincideix amb un dels dos extrems, o sigui a l'ultim bit
    // del double. Son unes 60 voltes i es fa un sol cop per imant.
    for (var v = 0; v < 200; v++) {
      var mig = 0.5 * (lo + hi);
      if (mig === lo || mig === hi) break;
      if (acceleracioRadial(mig, ux, uy, p) > 0.0) lo = mig; else hi = mig;
    }

    var rEq = 0.5 * (lo + hi);
    var x = rEq * ux, y = rEq * uy;
    acceleracioNucli(x, y, 0.0, 0.0, p.L, p.g, p.b, p.n, p.d0, p.imX, p.imY, p.imK, _acc);
    return { x: x, y: y, r: rEq, residu: Math.hypot(_acc[0], _acc[1]) };
  }

  // Tots els punts de repos del muntatge. L'ULTIM ES SEMPRE EL CENTRE, com al
  // Python. Els imants que no reten la massa hi surten com a null.
  function puntsDeRepos(p) {
    var punts = [];
    for (var i = 0; i < p.imK.length; i++) punts.push(puntDeReposDeLimant(i, p));
    var estable = centreEsEstable(p);
    // El centre es un punt d'equilibri EXACTE per simetria (amb K iguals els
    // tres imants s'anul.len entre ells), o sigui que no cal buscar-lo.
    punts.push({ x: 0.0, y: 0.0, r: 0.0, residu: 0.0 });
    return { punts: punts, centreEstable: estable };
  }

  // ------------------------------------------------------------------
  // Les taules de destins -- traduccio de motor.destins
  // ------------------------------------------------------------------
  // Es calculen UN COP, abans de la graella. A dins del bucle dels pixels
  // nomes hi entren aquests quatre vectors de numeros.
  //
  // Si el centre es inestable no hi entra: aquell desti no existeix i al mapa
  // no hi haura regio negra. Es un resultat, no un cas particular del codi.
  function destins(p, criteri) {
    var repos = puntsDeRepos(p);
    var dx = [], dy = [], codi = [], radi2 = [];
    for (var i = 0; i < p.imK.length; i++) {
      var q = repos.punts[i];
      if (q === null) continue; // aquest imant no rete la massa
      dx.push(q.x);
      dy.push(q.y);
      codi.push(i + 1); // els imants es numeren des de 1
      radi2.push(criteri.rCaptura * criteri.rCaptura);
    }
    if (repos.centreEstable) {
      var c = repos.punts[p.imK.length];
      dx.push(c.x);
      dy.push(c.y);
      codi.push(CENTRE);
      radi2.push(criteri.rCentre * criteri.rCentre);
    }
    return {
      destX: Float64Array.from(dx),
      destY: Float64Array.from(dy),
      destCodi: Int32Array.from(codi),
      destRadi2: Float64Array.from(radi2),
      centreEstable: repos.centreEstable,
    };
  }

  var MotorMagnetic = {
    CENTRE: CENTRE,
    NO_CLASSIFICAT: NO_CLASSIFICAT,
    configura: configura,
    comVa: comVa,
    potencia: potencia,
    acceleracioNucli: acceleracioNucli,
    passRk4: passRk4,
    classifica: classifica,
    desenllacNucli: desenllacNucli,
    rigidesaAlCentre: rigidesaAlCentre,
    valorsPropis: valorsPropis,
    centreEsEstable: centreEsEstable,
    factorCritic: factorCritic,
    acceleracioRadial: acceleracioRadial,
    puntsDeRepos: puntsDeRepos,
    destins: destins,
  };

  arrel.MotorMagnetic = MotorMagnetic;
  if (typeof module !== "undefined" && module.exports) module.exports = MotorMagnetic;
})(typeof self !== "undefined" ? self : this);
