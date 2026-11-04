# Les figures del treball

Totes les figures de la memòria i dels annexos del treball de recerca sobre el caos al pèndol doble i al pèndol magnètic, a resolució completa, amb el número que porten a la memòria al nom del fitxer. Les que surten d'un programa són en PDF (vectorials); les del cos de la memòria i les de l'Annex F, també en PNG.

Aquesta carpeta la fa `eines/figures_memoria.py`, que copia cada figura de l'eina que la fa i comprova que la còpia és idèntica a l'original. Per tornar a fer una figura, l'ordre és la de la columna «Eina», des de l'arrel del repositori i amb el Python de l'entorn (`.venv\Scripts\python.exe`); les opcions completes són a la capçalera de cada programa i a `documentacio/index-figures.md`. Els JSON de `figures/` no es versionen: els refà la mateixa eina. Els de `dades/`, sí.

## La memòria

Les figures 1–9 en PDF i en PNG.

| Figura | Fitxer | Què ensenya | Eina | Dades |
|---|---|---|---|---|
| Figura 1 | [PDF](cos/Figura_01_cicle_del_metode.pdf) · [PNG](cos/Figura_01_cicle_del_metode.png) | Diagrama del cicle del mètode | dibuix fet al document de la memòria, exportat del Google Docs (el PNG, retallat al dibuix) | — |
| Figura 2 | [PDF](cos/Figura_02_esquema_doble.pdf) · [PNG](cos/Figura_02_esquema_doble.png) | Esquema a escala del pèndol doble | `eines/figures_metodologia.py esquema_doble` | `parametres.json` |
| Figura 3.1 | [PDF](cos/Figura_03-1_magnetic_costat.pdf) · [PNG](cos/Figura_03-1_magnetic_costat.png) | El pèndol magnètic, vist de costat | `-m magnetic.figures_metodologia costat` | `magnetic/parametres.json` |
| Figura 3.2 | [PDF](cos/Figura_03-2_magnetic_dalt.pdf) · [PNG](cos/Figura_03-2_magnetic_dalt.png) | El pèndol magnètic, vist de dalt | `-m magnetic.figures_metodologia dalt` | `magnetic/parametres.json` |
| Figura 4 | [PDF](cos/Figura_04_cami_dades.pdf) · [PNG](cos/Figura_04_cami_dades.png) | El camí de les dades del pèndol doble, del vídeo al format canònic | `eines/figures_metodologia.py cami_dades` | les metadades de les preses (`dades/net/*.json`) |
| Figura 5 | [PDF](cos/Figura_05_conques.pdf) · [PNG](cos/Figura_05_conques.png) | Com es construeix el mapa de conques | `-m magnetic.figures_metodologia conques` | esquema, sense càlcul |
| Figura 6 | [PDF](cos/Figura_06_feix_tir.pdf) · [PNG](cos/Figura_06_feix_tir.png) | Experiment 1: el feix de 200 simulacions i el tir real de la mediana | `eines/experiment1.py figura` | `dades/experiment1/y_2026-10-27.json`, `figures/experiment1_figures.json` |
| Figura 7 | [PDF](cos/Figura_07_poincare_friccio.pdf) · [PNG](cos/Figura_07_poincare_friccio.png) | Experiment 2: la secció de Poincaré del model amb fricció i del pèndol real | `eines/figures_experiment2.py poincare_friccio --memoria` | `figures/poincare_friccio.json` |
| Figura 8 | [PDF](cos/Figura_08_lyapunov_veins.pdf) · [PNG](cos/Figura_08_lyapunov_veins.png) | Experiment 4: com se separen els veïns, i la λ | `eines/figures_lyapunov_real.py veins --corregit --memoria` | `dades/divergencia/lyapunov_real_dues_parts_2026-10-27.json` |
| Figura 9 | [PDF](cos/Figura_09_autosemblanca.pdf) · [PNG](cos/Figura_09_autosemblanca.png) | Experiment 3: el mapa de conques i tres ampliacions | `-m magnetic.autosemblanca (els quatre mapes)` | `magnetic/dades/mapa_general.npz` i `zoom1`–`zoom3.npz` (es refan amb `-m magnetic.mapa`) |

## Annex A. Marc teòric

| Figura | Fitxer | Què ensenya | Eina | Dades |
|---|---|---|---|---|
| Figura A.1 | [Figura_A1_esquema_doble.png](annex_A/Figura_A1_esquema_doble.png) | Esquema del pèndol doble de barres | imatge del document dels annexos | — |
| Figura A.2 | [Figura_A2_retrat_de_fases_simple.png](annex_A/Figura_A2_retrat_de_fases_simple.png) | Retrat de fases del pèndol simple | imatge del document dels annexos | — |
| Figura A.3 | [Figura_A3_dues_simulacions.png](annex_A/Figura_A3_dues_simulacions.png) | Dues simulacions que se separen | imatge del document dels annexos | — |

## Annex B. Resums, dissenys 3D i fotos

| Figura | Fitxer | Què ensenya | Eina | Dades |
|---|---|---|---|---|
| B.1 | [B1_double_pendulum_experiment_in_chaos_1.jpg](annex_B/B1_double_pendulum_experiment_in_chaos_1.jpg) | Resum de «Double pendulum: An experiment in chaos» (1 de 2) | foto del resum escrit a mà | — |
| B.1 | [B1_double_pendulum_experiment_in_chaos_2.jpg](annex_B/B1_double_pendulum_experiment_in_chaos_2.jpg) | Resum de «Double pendulum: An experiment in chaos» (2 de 2) | foto del resum escrit a mà | — |
| B.1 | [B1_deterministic_chaos_theory_1.jpg](annex_B/B1_deterministic_chaos_theory_1.jpg) | Resum de «Deterministic Chaos Theory: Basic Concepts» (1 de 3) | foto del resum escrit a mà | — |
| B.1 | [B1_deterministic_chaos_theory_2.jpg](annex_B/B1_deterministic_chaos_theory_2.jpg) | Resum de «Deterministic Chaos Theory: Basic Concepts» (2 de 3) | foto del resum escrit a mà | — |
| B.1 | [B1_deterministic_chaos_theory_3.jpg](annex_B/B1_deterministic_chaos_theory_3.jpg) | Resum de «Deterministic Chaos Theory: Basic Concepts» (3 de 3) | foto del resum escrit a mà | — |
| B.1 | [B1_matematicas_y_fisica_del_caos_1.jpg](annex_B/B1_matematicas_y_fisica_del_caos_1.jpg) | Resum de «Las matemáticas y la física del caos» (1 de 2) | foto del resum escrit a mà | — |
| B.1 | [B1_matematicas_y_fisica_del_caos_2.jpg](annex_B/B1_matematicas_y_fisica_del_caos_2.jpg) | Resum de «Las matemáticas y la física del caos» (2 de 2) | foto del resum escrit a mà | — |
| B.1 | [B1_caos_orden_y_otras_movidas_1.jpg](annex_B/B1_caos_orden_y_otras_movidas_1.jpg) | Resum de «Caos, orden y otras movidas del universo» (1 de 2) | foto del resum escrit a mà | — |
| B.1 | [B1_caos_orden_y_otras_movidas_2.jpg](annex_B/B1_caos_orden_y_otras_movidas_2.jpg) | Resum de «Caos, orden y otras movidas del universo» (2 de 2) | foto del resum escrit a mà | — |
| B.1 | [B1_chaos_theory_applications_1.jpg](annex_B/B1_chaos_theory_applications_1.jpg) | Resum de «Chaos theory and its applications in our real life» (1 de 2) | foto del resum escrit a mà | — |
| B.1 | [B1_chaos_theory_applications_2.jpg](annex_B/B1_chaos_theory_applications_2.jpg) | Resum de «Chaos theory and its applications in our real life» (2 de 2) | foto del resum escrit a mà | — |
| Figura B.2a | [Figura_B2a_magnetic_perspectiva.png](annex_B/Figura_B2a_magnetic_perspectiva.png) | Pèndol magnètic, en perspectiva | disseny 3D | — |
| Figura B.2b | [Figura_B2b_magnetic_alcat.png](annex_B/Figura_B2b_magnetic_alcat.png) | Pèndol magnètic, alçat | disseny 3D | — |
| Figura B.2c | [Figura_B2c_magnetic_planta.png](annex_B/Figura_B2c_magnetic_planta.png) | Pèndol magnètic, planta | disseny 3D | — |
| Figura B.2d | [Figura_B2d_magnetic_perfil.png](annex_B/Figura_B2d_magnetic_perfil.png) | Pèndol magnètic, perfil | disseny 3D | — |
| Figura B.2e | [Figura_B2e_doble_perspectiva.png](annex_B/Figura_B2e_doble_perspectiva.png) | Pèndol doble, en perspectiva | disseny 3D | — |
| Figura B.2f | [Figura_B2f_doble_alcat.png](annex_B/Figura_B2f_doble_alcat.png) | Pèndol doble, alçat | disseny 3D | — |
| Figura B.2g | [Figura_B2g_doble_planta.png](annex_B/Figura_B2g_doble_planta.png) | Pèndol doble, planta | disseny 3D | — |
| Figura B.2h | [Figura_B2h_doble_perfil.png](annex_B/Figura_B2h_doble_perfil.png) | Pèndol doble, perfil | disseny 3D | — |
| Figura B.2i | [Figura_B2i_peca_de_l_iman.png](annex_B/Figura_B2i_peca_de_l_iman.png) | La peça que aguanta l'imant | disseny 3D | — |
| Figura B.3a | [Figura_B3a_muntatge_magnetic.png](annex_B/Figura_B3a_muntatge_magnetic.png) | El muntatge del pèndol magnètic | foto | — |
| Figura B.3b | [Figura_B3b_doble_finalitzat.png](annex_B/Figura_B3b_doble_finalitzat.png) | El pèndol doble acabat | foto | — |
| Figura B.3c | [Figura_B3c_magnetic_finalitzat.jpg](annex_B/Figura_B3c_magnetic_finalitzat.jpg) | El pèndol magnètic acabat | foto | — |

## Annex E. Explicacions

En PDF.

| Figura | Fitxer | Què ensenya | Eina | Dades |
|---|---|---|---|---|
| Figura E.1 | [PDF](annex_E/Figura_E1_angles_desembolicats.pdf) | θ2 de la presa 150_3, plegat i desembolicat | `eines/figures_metodologia.py angles_desembolicats` | la presa 150_3 |
| Figura E.3 | [PDF](annex_E/Figura_E3_derivada_vs_filtre.pdf) | ω1 de la 150_3: restant fotogrames i amb Savitzky-Golay | `eines/figures_metodologia.py derivada_vs_filtre` | la presa 150_3 i el seu `.trk` |
| Figura E.6 | [PDF](annex_E/Figura_E6_criteri_aturada.pdf) | El criteri d'aturada del pèndol magnètic | `-m magnetic.figures_metodologia aturada` | esquema |
| Figura E.7 | [PDF](annex_E/Figura_E7_box_counting.pdf) | El mètode del box-counting | `-m magnetic.figures_metodologia box_counting` | esquema |
| Figura E.8 | [PDF](annex_E/Figura_E8_assajos_friccio.pdf) | Els dos assajos de fricció | `eines/figures_metodologia.py assajos_friccio` | `parametres.json` |
| Figura E.10 | [PDF](annex_E/Figura_E10_poincare.pdf) | Una òrbita projectada a l'espai de fases i tallada (Poincaré) | `eines/figures_metodologia.py poincare` | només simulació |
| Figura E.11 | [PDF](annex_E/Figura_E11_feix.pdf) | Un feix de simulacions i l'horitzó | `eines/figures_metodologia.py feix` | només simulació |

## Annex F. Resultats complementaris

En PDF i en PNG.

| Figura | Fitxer | Què ensenya | Eina | Dades |
|---|---|---|---|---|
| Figura F.1a | [PDF](annex_F/Figura_F1a_sortides.pdf) · [PNG](annex_F/Figura_F1a_sortides.png) | Quan surt del feix cada tir dels 45° i dels 90° | `eines/experiment1.py figura` | `figures/experiment1_figures.json` |
| Figura F.1b | [PDF](annex_F/Figura_F1b_theta2_amollada.pdf) · [PNG](annex_F/Figura_F1b_theta2_amollada.png) | θ2 real menys el model els primers 0,15 s | `eines/centre_automatic.py diagnostic` | `dades/experiment1/centre_automatic_diagnostic_2026-10-29.json` |
| Figura F.1c | [PDF](annex_F/Figura_F1c_angle_entre_barres.pdf) · [PNG](annex_F/Figura_F1c_angle_entre_barres.png) | L'angle entre les barres en amollar (la placa) | `eines/angle_entre_barres.py` | `dades/experiment1/angle_entre_barres_2026-10-29.json` |
| Figura F.1d | [PDF](annex_F/Figura_F1d_model_reiniciat.pdf) · [PNG](annex_F/Figura_F1d_model_reiniciat.png) | El model reiniciat després d'amollar | `eines/model_reiniciat.py` | `dades/experiment1/model_reiniciat_2026-10-29.json` |
| Figura F.2a | [PDF](annex_F/Figura_F2a_lambda_E.pdf) · [PNG](annex_F/Figura_F2a_lambda_E.png) | La λ del model contra l'energia | `eines/lyapunov.py --nomes-resum --memoria` | `dades/lyapunov/lambda_E_2026-10-01/` |
| Figura F.2b | [PDF](annex_F/Figura_F2b_poincare_real.pdf) · [PNG](annex_F/Figura_F2b_poincare_real.png) | La secció de Poincaré del pèndol real, per franges d'energia | `eines/figures_reals.py poincare --corregit --memoria` | les preses reals (`figures/poincare_real_corregit.json`) |
| Figura F.2c | [PDF](annex_F/Figura_F2c_fases_45.pdf) · [PNG](annex_F/Figura_F2c_fases_45.png) | L'espai de fases dels tirs dels 45° | `eines/figures_reals.py fases --corregit --memoria` | les preses reals |
| Figura F.2d | [PDF](annex_F/Figura_F2d_fases_90.pdf) · [PNG](annex_F/Figura_F2d_fases_90.png) | L'espai de fases dels tirs dels 90° | `eines/figures_reals.py fases --corregit --memoria` | les preses reals |
| Figura F.2e | [PDF](annex_F/Figura_F2e_fases_150.pdf) · [PNG](annex_F/Figura_F2e_fases_150.png) | L'espai de fases dels tirs dels 150° | `eines/figures_reals.py fases --corregit --memoria` | les preses reals |
| Figura F.2f | [PDF](annex_F/Figura_F2f_cinc_corbes.pdf) · [PNG](annex_F/Figura_F2f_cinc_corbes.png) | Les cinc corbes de la presa 150_3 | `eines/figures_experiment2.py cinc_corbes --memoria` | `figures/cinc_corbes_150_3.json` |
| Figura F.2g | [PDF](annex_F/Figura_F2g_simple.pdf) · [PNG](annex_F/Figura_F2g_simple.png) | El pèndol simple, el control | `eines/control_simple.py retrat --memoria` | només simulació |
| Figura F.3 | [PDF](annex_F/Figura_F3_box_counting.pdf) · [PNG](annex_F/Figura_F3_box_counting.png) | El box-counting de la frontera del mapa sencer | `-m magnetic.analisi magnetic/dades/mapa_general.npz` | `magnetic/figures/mapa_general_metriques.json` |
| Figura F.4a | [PDF](annex_F/Figura_F4a_validacio_veins.pdf) · [PNG](annex_F/Figura_F4a_validacio_veins.png) | El mètode dels veïns contra la resposta coneguda | `eines/figures_lyapunov_real.py validacio --memoria` | `dades/divergencia/lyapunov_real_validacio2_2026-10-17.json` |
| Figura F.4b | [PDF](annex_F/Figura_F4b_lambda_150.pdf) · [PNG](annex_F/Figura_F4b_lambda_150.png) | El ritme de separació del primer segon als 150° | `eines/figures_lyapunov_real.py llindar_150 --corregit --memoria` | `dades/divergencia/lyapunov_real_dues_parts_2026-10-27.json` |
| Figura F.4c | [PDF](annex_F/Figura_F4c_z.pdf) · [PNG](annex_F/Figura_F4c_z.png) | La Z: la λ* real contra la del model | `eines/figures_lyapunov_real.py z --memoria` | `dades/divergencia/lyapunov_real_z_r2_2026-10-27.json`, `dades/divergencia/lyapunov_real_z_diagnostic_2026-10-27.json` |
| Figura F.5a | [PDF](annex_F/Figura_F5a_fitxa_45_6.pdf) · [PNG](annex_F/Figura_F5a_fitxa_45_6.png) | La fitxa de la presa 45_6 | `eines/fitxes_preses.py --memoria` | la presa 45_6 |
| Figura F.5b | [PDF](annex_F/Figura_F5b_fitxa_90_11.pdf) · [PNG](annex_F/Figura_F5b_fitxa_90_11.png) | La fitxa de la presa 90_11 | `eines/fitxes_preses.py --memoria` | la presa 90_11 |
| Figura F.5c | [PDF](annex_F/Figura_F5c_fitxa_150_3.pdf) · [PNG](annex_F/Figura_F5c_fitxa_150_3.png) | La fitxa de la presa 150_3 | `eines/fitxes_preses.py --memoria` | la presa 150_3 |

## Les fitxes de les 35 preses

Una fitxa per presa real del pèndol doble, amb els angles corregits: θ1 i θ2 al llarg del tir, l'energia, la secció de Poincaré de la presa i l'espai de fases en quatre trams. A l'annex només hi ha les de la 45_6, la 90_11 i la 150_3 (Figura F.5). Les fa `eines/fitxes_preses.py --memoria`, a partir de les preses reals.

- [45_1](fitxes/fitxa_45_1_corregit.pdf)
- [45_2](fitxes/fitxa_45_2_corregit.pdf)
- [45_3](fitxes/fitxa_45_3_corregit.pdf)
- [45_4](fitxes/fitxa_45_4_corregit.pdf)
- [45_5](fitxes/fitxa_45_5_corregit.pdf)
- [45_6](fitxes/fitxa_45_6_corregit.pdf)
- [45_8](fitxes/fitxa_45_8_corregit.pdf)
- [45_9](fitxes/fitxa_45_9_corregit.pdf)
- [45_10](fitxes/fitxa_45_10_corregit.pdf)
- [45_11](fitxes/fitxa_45_11_corregit.pdf)
- [45_12](fitxes/fitxa_45_12_corregit.pdf)
- [45_13](fitxes/fitxa_45_13_corregit.pdf)
- [90_1](fitxes/fitxa_90_1_corregit.pdf)
- [90_2](fitxes/fitxa_90_2_corregit.pdf)
- [90_3](fitxes/fitxa_90_3_corregit.pdf)
- [90_4](fitxes/fitxa_90_4_corregit.pdf)
- [90_5](fitxes/fitxa_90_5_corregit.pdf)
- [90_6](fitxes/fitxa_90_6_corregit.pdf)
- [90_7](fitxes/fitxa_90_7_corregit.pdf)
- [90_8](fitxes/fitxa_90_8_corregit.pdf)
- [90_9](fitxes/fitxa_90_9_corregit.pdf)
- [90_10](fitxes/fitxa_90_10_corregit.pdf)
- [90_11](fitxes/fitxa_90_11_corregit.pdf)
- [150_1](fitxes/fitxa_150_1_corregit.pdf)
- [150_2](fitxes/fitxa_150_2_corregit.pdf)
- [150_3](fitxes/fitxa_150_3_corregit.pdf)
- [150_4](fitxes/fitxa_150_4_corregit.pdf)
- [150_5](fitxes/fitxa_150_5_corregit.pdf)
- [150_6](fitxes/fitxa_150_6_corregit.pdf)
- [150_7](fitxes/fitxa_150_7_corregit.pdf)
- [150_8](fitxes/fitxa_150_8_corregit.pdf)
- [150_9](fitxes/fitxa_150_9_corregit.pdf)
- [150_10](fitxes/fitxa_150_10_corregit.pdf)
- [150_11](fitxes/fitxa_150_11_corregit.pdf)
- [150_12](fitxes/fitxa_150_12_corregit.pdf)
