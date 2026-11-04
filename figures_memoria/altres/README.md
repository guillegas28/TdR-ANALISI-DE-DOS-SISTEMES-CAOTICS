# Altres figures

Figures vigents del treball que no surten a la memòria ni als annexos: les de `../` són les que sí que hi surten.

## El pèndol doble

| Fitxer | Què ensenya | D'on surt |
|---|---|---|
| [poincare_E0_94.png](doble/poincare_E0_94.png) | Secció de Poincaré sense fricció a l'energia dels 45° (0,94 E_volteig,2): moviment regular | `eines/poincare.py` |
| [poincare_E3_44.png](doble/poincare_E3_44.png) | Secció de Poincaré sense fricció a l'energia dels 90° (3,44 E_volteig,2): caos amb illes regulars | `eines/poincare.py` |
| [poincare_E6_23.png](doble/poincare_E6_23.png) | Secció de Poincaré sense fricció a l'energia dels 150° (6,23 E_volteig,2) | `eines/poincare.py` |
| [poincare_real_energia_corregit.png](doble/poincare_real_energia_corregit.png) | Secció de Poincaré de les preses reals, pintada per energia | `eines/figures_reals.py poincare_energia --corregit --memoria` |
| [espaguetis_reals_corregit.png](doble/espaguetis_reals_corregit.png) | Gràfica espagueti: θ1(t) i θ2(t) de tots els tirs reals de cada grup | `eines/figures_reals.py espaguetis --corregit --memoria` |
| [divergencia_simple_doble.png](doble/divergencia_simple_doble.png) | Com se separen 12 tirs del pèndol simple i 12 del doble (només simulació) | `eines/control_simple.py divergencia --memoria` |
| [friccio_sec.png](doble/friccio_sec.png) | El fregament sec suavitzat amb tanh(ω/ε) (esquema) | `eines/figures_metodologia.py friccio_sec` |
| [tres_trams.png](doble/tres_trams.png) | Els tres trams de ⟨ln(d/d0)⟩ i la recta de la λ (esquema, corba inventada) | `eines/figures_metodologia.py tres_trams` |

## El pèndol magnètic

| Fitxer | Què ensenya | D'on surt |
|---|---|---|
| [mapa_general.png](magnetic/mapa_general.png) | El mapa de conques sencer, de −12 a 12 cm | `-m magnetic.mapa` |
| [mapa_general_temps.png](magnetic/mapa_general_temps.png) | El temps que triga la massa a aturar-se, des de cada punt | `-m magnetic.mapa` |
| [mapa_general_boxcounting.png](magnetic/mapa_general_boxcounting.png) | El box-counting de la frontera del mapa sencer | `-m magnetic.analisi` |
| [regio_negra.png](magnetic/regio_negra.png) | Quina part del mapa acaba al centre i quina és frontera, segons la fricció | `-m magnetic.regio_negra` |
| [regio_negra_nucli.png](magnetic/regio_negra_nucli.png) | La regió negra (el centre), de prop | `-m magnetic.regio_negra` |
| [K_sota_critica.png](magnetic/K_sota_critica.png) | El mapa amb la K un 3 % per sota de la crítica | `-m magnetic.mapa --K` |
| [K_sota_critica_temps.png](magnetic/K_sota_critica_temps.png) | El seu temps de caiguda | `-m magnetic.mapa --K` |
| [K_sobre_critica.png](magnetic/K_sobre_critica.png) | El mapa amb la K un 3 % per sobre de la crítica (el centre deixa de ser estable) | `-m magnetic.mapa --K` |
| [K_sobre_critica_temps.png](magnetic/K_sobre_critica_temps.png) | El seu temps de caiguda | `-m magnetic.mapa --K` |
| [sens_b0030.png](magnetic/sens_b0030.png) | Sensibilitat a la fricció: b = 0,030 s⁻¹ (l'extrem baix de la mesura) | `-m magnetic.mapa --b` |
| [sens_b0039.png](magnetic/sens_b0039.png) | Sensibilitat a la fricció: b = 0,039 s⁻¹ (la mesurada) | `-m magnetic.mapa --b` |
| [sens_b0050.png](magnetic/sens_b0050.png) | Sensibilitat a la fricció: b = 0,050 s⁻¹ (l'extrem alt) | `-m magnetic.mapa --b` |
| [sensin_b0030.png](magnetic/sensin_b0030.png) | El mateix, a ±6 cm: b = 0,030 s⁻¹ | `-m magnetic.mapa --b` |
| [sensin_b0039.png](magnetic/sensin_b0039.png) | El mateix, a ±6 cm: b = 0,039 s⁻¹ | `-m magnetic.mapa --b` |
| [sensin_b0050.png](magnetic/sensin_b0050.png) | El mateix, a ±6 cm: b = 0,050 s⁻¹ | `-m magnetic.mapa --b` |
| [friccio_08.png](magnetic/friccio_08.png) | El mapa amb molta més fricció, b = 0,8 s⁻¹ (exploració, no és el pèndol real) | `-m magnetic.mapa --b 0.8` |
| [friccio_08_temps.png](magnetic/friccio_08_temps.png) | El seu temps de caiguda | `-m magnetic.mapa --b 0.8` |
| [friccio_13.png](magnetic/friccio_13.png) | El mapa amb b = 1,3 s⁻¹ (exploració) | `-m magnetic.mapa --b 1.3` |
| [friccio_13_temps.png](magnetic/friccio_13_temps.png) | El seu temps de caiguda | `-m magnetic.mapa --b 1.3` |
| [llei_de_forca.png](magnetic/llei_de_forca.png) | La llei de força dels imants: l'ajust de la n i la K | `-m magnetic.llei_de_forca` |
| [friccio_3072.png](magnetic/friccio_3072.png) | La fricció del pèndol magnètic: el decaiment de l'amplitud | `-m magnetic.decrement dades/net/magnetic_3072.csv --nom friccio_3072` |
| [quadricula_comparacio.png](magnetic/quadricula_comparacio.png) | La comparació amb el pèndol real, cel·la a cel·la (figura de treball) | `-m magnetic.quadricula compara` |

## La portada

| Fitxer | Què ensenya | D'on surt |
|---|---|---|
| [portada_150_3_0_5s.png](portada/portada_150_3_0_5s.png) | La presa 150_3 a 0,5 s de l'amollada, amb la simulació a sobre (taronja, l'extrem real; blau, el simulat) | `esborranys/_memoria/portada.py (no publicat)` |
| [portada_150_3_0_8s.png](portada/portada_150_3_0_8s.png) | La presa 150_3 a 0,8 s de l'amollada, amb la simulació a sobre (taronja, l'extrem real; blau, el simulat) | `esborranys/_memoria/portada.py (no publicat)` |
| [portada_150_3_1_2s.png](portada/portada_150_3_1_2s.png) | La presa 150_3 a 1,2 s de l'amollada, amb la simulació a sobre (taronja, l'extrem real; blau, el simulat) | `esborranys/_memoria/portada.py (no publicat)` |
