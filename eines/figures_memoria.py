r"""Les figures de la memòria i dels annexos, per al repositori (2/11).

Copia a figures_memoria/ cada figura que surt d'una eina, amb el número que porta a la memòria al nom del fitxer, i n'escriu
l'índex, figures_memoria/README.md: per a cada figura, què ensenya, amb quina eina es fa i de quines dades surt. No refà cap
figura: copia la que ja ha fet la seva eina (a figures/, figures/memoria/ o magnetic/figures/, que no es versionen), i comprova
que la còpia és idèntica byte a byte a l'original.

Les imatges que no surten de cap eina (les del marc, els resums de la bibliografia, els dissenys 3D i les fotos del muntatge) es
van treure una sola vegada del document dels annexos, i aquí només es comprova que hi són. La Figura 1 (el diagrama del mètode)
és un dibuix fet al document, exportat del Google Docs per en Guillem.

Si una eina torna a fer una figura, es torna a passar aquest programa i la carpeta queda al dia. Amb --comprova no copia res:
només diu si la carpeta és igual que les figures de les eines.

Formats: el PDF de cada figura d'eina (vectorial), i a més el PNG de les del cos i de l'Annex F, per veure-les al navegador.

    .venv\Scripts\python.exe eines\figures_memoria.py
    .venv\Scripts\python.exe eines\figures_memoria.py --comprova
"""
from __future__ import annotations

import argparse
import filecmp
import re
import shutil
import sys
from pathlib import Path

ARREL = Path(__file__).resolve().parent.parent
SORTIDA = ARREL / 'figures_memoria'
MD_RESULTATS = ARREL / 'memòria escrita' / 'nova versió' / 'capitol-resultats.md'

MET = 'eines/figures_metodologia.py'
MAG = '-m magnetic.figures_metodologia'
FL = 'eines/figures_lyapunov_real.py'
FR = 'eines/figures_reals.py'
FE2 = 'eines/figures_experiment2.py'
DIV = 'dades/divergencia/'
EXP1 = 'dades/experiment1/'

# (carpeta, nom, número, què ensenya, font sense extensió, formats, eina i ordre, dades)
D_EINA = [
    ('cos', 'Figura_02_esquema_doble', 'Figura 2', 'Esquema a escala del pèndol doble',
     'figures/metodologia/esquema_doble', ('pdf', 'png'), f'{MET} esquema_doble', '`parametres.json`'),
    ('cos', 'Figura_03-1_magnetic_costat', 'Figura 3.1', 'El pèndol magnètic, vist de costat',
     'magnetic/figures/metodologia/esquema_costat', ('pdf', 'png'), f'{MAG} costat', '`magnetic/parametres.json`'),
    ('cos', 'Figura_03-2_magnetic_dalt', 'Figura 3.2', 'El pèndol magnètic, vist de dalt',
     'magnetic/figures/metodologia/esquema_dalt', ('pdf', 'png'), f'{MAG} dalt', '`magnetic/parametres.json`'),
    ('cos', 'Figura_04_cami_dades', 'Figura 4', 'El camí de les dades del pèndol doble, del vídeo al format canònic',
     'figures/metodologia/cami_dades', ('pdf', 'png'), f'{MET} cami_dades', 'les metadades de les preses (`dades/net/*.json`)'),
    ('cos', 'Figura_05_conques', 'Figura 5', 'Com es construeix el mapa de conques',
     'magnetic/figures/metodologia/esquema_conques', ('pdf', 'png'), f'{MAG} conques', 'esquema, sense càlcul'),
    ('cos', 'Figura_06_feix_tir', 'Figura 6', 'Experiment 1: el feix de 200 simulacions i el tir real de la mediana',
     'figures/experiment1_feix_tir', ('pdf', 'png'), 'eines/experiment1.py figura',
     f'`{EXP1}y_2026-10-27.json`, `figures/experiment1_figures.json`'),
    ('cos', 'Figura_07_poincare_friccio', 'Figura 7', 'Experiment 2: la secció de Poincaré del model amb fricció i del pèndol real',
     'figures/memoria/poincare_friccio', ('pdf', 'png'), f'{FE2} poincare_friccio --memoria', '`figures/poincare_friccio.json`'),
    ('cos', 'Figura_08_lyapunov_veins', 'Figura 8', 'Experiment 4: com se separen els veïns, i la λ',
     'figures/memoria/lyapunov_real_veins_corregit', ('pdf', 'png'), f'{FL} veins --corregit --memoria',
     f'`{DIV}lyapunov_real_dues_parts_2026-10-27.json`'),
    ('cos', 'Figura_09_autosemblanca', 'Figura 9', 'Experiment 3: el mapa de conques i tres ampliacions',
     'magnetic/figures/autosemblanca', ('pdf', 'png'), '-m magnetic.autosemblanca (els quatre mapes)',
     '`magnetic/dades/mapa_general.npz` i `zoom1`–`zoom3.npz` (es refan amb `-m magnetic.mapa`)'),
    ('annex_E', 'Figura_E1_angles_desembolicats', 'Figura E.1', 'θ2 de la presa 150_3, plegat i desembolicat',
     'figures/metodologia/angles_desembolicats', ('pdf',), f'{MET} angles_desembolicats', 'la presa 150_3'),
    ('annex_E', 'Figura_E3_derivada_vs_filtre', 'Figura E.3', 'ω1 de la 150_3: restant fotogrames i amb Savitzky-Golay',
     'figures/metodologia/derivada_vs_filtre', ('pdf',), f'{MET} derivada_vs_filtre',
     'la presa 150_3 i el seu `.trk`'),
    ('annex_E', 'Figura_E6_criteri_aturada', 'Figura E.6', 'El criteri d\'aturada del pèndol magnètic',
     'magnetic/figures/metodologia/esquema_aturada', ('pdf',), f'{MAG} aturada', 'esquema'),
    ('annex_E', 'Figura_E7_box_counting', 'Figura E.7', 'El mètode del box-counting',
     'magnetic/figures/metodologia/esquema_box_counting', ('pdf',), f'{MAG} box_counting', 'esquema'),
    ('annex_E', 'Figura_E8_assajos_friccio', 'Figura E.8', 'Els dos assajos de fricció',
     'figures/metodologia/assajos_friccio', ('pdf',), f'{MET} assajos_friccio', '`parametres.json`'),
    ('annex_E', 'Figura_E10_poincare', 'Figura E.10', 'Una òrbita projectada a l\'espai de fases i tallada (Poincaré)',
     'figures/metodologia/poincare', ('pdf',), f'{MET} poincare', 'només simulació'),
    ('annex_E', 'Figura_E11_feix', 'Figura E.11', 'Un feix de simulacions i l\'horitzó',
     'figures/metodologia/feix', ('pdf',), f'{MET} feix', 'només simulació'),
    ('annex_F', 'Figura_F1a_sortides', 'Figura F.1a', 'Quan surt del feix cada tir dels 45° i dels 90°',
     'figures/experiment1_sortides', ('pdf', 'png'), 'eines/experiment1.py figura', '`figures/experiment1_figures.json`'),
    ('annex_F', 'Figura_F1b_theta2_amollada', 'Figura F.1b', 'θ2 real menys el model els primers 0,15 s',
     'figures/experiment1_theta2_amollada', ('pdf', 'png'), 'eines/centre_automatic.py diagnostic',
     f'`{EXP1}centre_automatic_diagnostic_2026-10-29.json`'),
    ('annex_F', 'Figura_F1c_angle_entre_barres', 'Figura F.1c', 'L\'angle entre les barres en amollar (la placa)',
     'figures/experiment1_angle_entre_barres', ('pdf', 'png'), 'eines/angle_entre_barres.py',
     f'`{EXP1}angle_entre_barres_2026-10-29.json`'),
    ('annex_F', 'Figura_F1d_model_reiniciat', 'Figura F.1d', 'El model reiniciat després d\'amollar',
     'figures/experiment1_model_reiniciat', ('pdf', 'png'), 'eines/model_reiniciat.py',
     f'`{EXP1}model_reiniciat_2026-10-29.json`'),
    ('annex_F', 'Figura_F2a_lambda_E', 'Figura F.2a', 'La λ del model contra l\'energia',
     'figures/memoria/lambda_E', ('pdf', 'png'), 'eines/lyapunov.py --nomes-resum --memoria',
     '`dades/lyapunov/lambda_E_2026-10-01/`'),
    ('annex_F', 'Figura_F2b_poincare_real', 'Figura F.2b', 'La secció de Poincaré del pèndol real, per franges d\'energia',
     'figures/memoria/poincare_real_corregit', ('pdf', 'png'), f'{FR} poincare --corregit --memoria',
     'les preses reals (`figures/poincare_real_corregit.json`)'),
    ('annex_F', 'Figura_F2c_fases_45', 'Figura F.2c', 'L\'espai de fases dels tirs dels 45°',
     'figures/memoria/fases_reals_45_corregit', ('pdf', 'png'), f'{FR} fases --corregit --memoria', 'les preses reals'),
    ('annex_F', 'Figura_F2d_fases_90', 'Figura F.2d', 'L\'espai de fases dels tirs dels 90°',
     'figures/memoria/fases_reals_90_corregit', ('pdf', 'png'), f'{FR} fases --corregit --memoria', 'les preses reals'),
    ('annex_F', 'Figura_F2e_fases_150', 'Figura F.2e', 'L\'espai de fases dels tirs dels 150°',
     'figures/memoria/fases_reals_150_corregit', ('pdf', 'png'), f'{FR} fases --corregit --memoria', 'les preses reals'),
    ('annex_F', 'Figura_F2f_cinc_corbes', 'Figura F.2f', 'Les cinc corbes de la presa 150_3',
     'figures/memoria/cinc_corbes_150_3', ('pdf', 'png'), f'{FE2} cinc_corbes --memoria', '`figures/cinc_corbes_150_3.json`'),
    ('annex_F', 'Figura_F2g_simple', 'Figura F.2g', 'El pèndol simple, el control',
     'figures/memoria/simple_retrat', ('pdf', 'png'), 'eines/control_simple.py retrat --memoria', 'només simulació'),
    ('annex_F', 'Figura_F3_box_counting', 'Figura F.3', 'El box-counting de la frontera del mapa sencer',
     'magnetic/figures/mapa_general_boxcounting', ('pdf', 'png'), '-m magnetic.analisi magnetic/dades/mapa_general.npz',
     '`magnetic/figures/mapa_general_metriques.json`'),
    ('annex_F', 'Figura_F4a_validacio_veins', 'Figura F.4a', 'El mètode dels veïns contra la resposta coneguda',
     'figures/memoria/lyapunov_real_validacio', ('pdf', 'png'), f'{FL} validacio --memoria',
     f'`{DIV}lyapunov_real_validacio2_2026-10-17.json`'),
    ('annex_F', 'Figura_F4b_lambda_150', 'Figura F.4b', 'El ritme de separació del primer segon als 150°',
     'figures/memoria/lyapunov_real_llindar_150_corregit', ('pdf', 'png'), f'{FL} llindar_150 --corregit --memoria',
     f'`{DIV}lyapunov_real_dues_parts_2026-10-27.json`'),
    ('annex_F', 'Figura_F4c_z', 'Figura F.4c', 'La Z: la λ* real contra la del model',
     'figures/memoria/lyapunov_real_z', ('pdf', 'png'), f'{FL} z --memoria',
     f'`{DIV}lyapunov_real_z_r2_2026-10-27.json`, `{DIV}lyapunov_real_z_diagnostic_2026-10-27.json`'),
    ('annex_F', 'Figura_F5a_fitxa_45_6', 'Figura F.5a', 'La fitxa de la presa 45_6',
     'figures/memoria/per_presa/fitxa_45_6_corregit', ('pdf', 'png'), 'eines/fitxes_preses.py --memoria', 'la presa 45_6'),
    ('annex_F', 'Figura_F5b_fitxa_90_11', 'Figura F.5b', 'La fitxa de la presa 90_11',
     'figures/memoria/per_presa/fitxa_90_11_corregit', ('pdf', 'png'), 'eines/fitxes_preses.py --memoria', 'la presa 90_11'),
    ('annex_F', 'Figura_F5c_fitxa_150_3', 'Figura F.5c', 'La fitxa de la presa 150_3',
     'figures/memoria/per_presa/fitxa_150_3_corregit', ('pdf', 'png'), 'eines/fitxes_preses.py --memoria', 'la presa 150_3'),
]

# les que no surten de cap eina: (carpeta, fitxer, número, què és, d'on surt)
DOC = 'imatge del document dels annexos'
FIXES = [
    ('cos', ('Figura_01_cicle_del_metode.pdf', 'Figura_01_cicle_del_metode.png'), 'Figura 1', 'Diagrama del cicle del mètode',
     'dibuix fet al document de la memòria, exportat del Google Docs (el PNG, retallat al dibuix)'),
    ('annex_A', 'Figura_A1_esquema_doble.png', 'Figura A.1', 'Esquema del pèndol doble de barres', DOC),
    ('annex_A', 'Figura_A2_retrat_de_fases_simple.png', 'Figura A.2', 'Retrat de fases del pèndol simple', DOC),
    ('annex_A', 'Figura_A3_dues_simulacions.png', 'Figura A.3', 'Dues simulacions que se separen', DOC),
] + [
    ('annex_B', f'B1_{nom}_{i}.jpg', 'B.1', f'Resum de «{titol}» ({i} de {n})', 'foto del resum escrit a mà')
    for nom, titol, n in (('double_pendulum_experiment_in_chaos', 'Double pendulum: An experiment in chaos', 2),
                          ('deterministic_chaos_theory', 'Deterministic Chaos Theory: Basic Concepts', 3),
                          ('matematicas_y_fisica_del_caos', 'Las matemáticas y la física del caos', 2),
                          ('caos_orden_y_otras_movidas', 'Caos, orden y otras movidas del universo', 2),
                          ('chaos_theory_applications', 'Chaos theory and its applications in our real life', 2))
    for i in range(1, n + 1)
] + [
    ('annex_B', f'Figura_B2{lletra}_{nom}.png', f'Figura B.2{lletra}', que, 'disseny 3D')
    for lletra, nom, que in (('a', 'magnetic_perspectiva', 'Pèndol magnètic, en perspectiva'),
                             ('b', 'magnetic_alcat', 'Pèndol magnètic, alçat'),
                             ('c', 'magnetic_planta', 'Pèndol magnètic, planta'),
                             ('d', 'magnetic_perfil', 'Pèndol magnètic, perfil'),
                             ('e', 'doble_perspectiva', 'Pèndol doble, en perspectiva'),
                             ('f', 'doble_alcat', 'Pèndol doble, alçat'),
                             ('g', 'doble_planta', 'Pèndol doble, planta'),
                             ('h', 'doble_perfil', 'Pèndol doble, perfil'),
                             ('i', 'peca_de_l_iman', 'La peça que aguanta l\'imant'))
] + [
    ('annex_B', 'Figura_B3a_muntatge_magnetic.png', 'Figura B.3a', 'El muntatge del pèndol magnètic', 'foto'),
    ('annex_B', 'Figura_B3b_doble_finalitzat.png', 'Figura B.3b', 'El pèndol doble acabat', 'foto'),
    ('annex_B', 'Figura_B3c_magnetic_finalitzat.jpg', 'Figura B.3c', 'El pèndol magnètic acabat', 'foto'),
]

FITXES = ARREL / 'figures' / 'memoria' / 'per_presa'
APARTATS = [
    ('cos', 'La memòria', 'Les figures 1–9 en PDF i en PNG.'),
    ('annex_A', 'Annex A. Marc teòric', ''),
    ('annex_B', 'Annex B. Resums, dissenys 3D i fotos', ''),
    ('annex_E', 'Annex E. Explicacions', 'En PDF.'),
    ('annex_F', 'Annex F. Resultats complementaris', 'En PDF i en PNG.'),
]


def copies():
    """(font, destí) de cada fitxer que surt d'una eina."""
    out = []
    for carpeta, nom, _n, _q, font, formats, _e, _d in D_EINA:
        for ext in formats:
            out.append((ARREL / f'{font}.{ext}', SORTIDA / carpeta / f'{nom}.{ext}'))
    for f in sorted(FITXES.glob('fitxa_*_corregit.pdf')):
        out.append((f, SORTIDA / 'fitxes' / f.name))
    return out


def fitxes_ordenades():
    def clau(f):
        g, i = re.match(r'fitxa_(\d+)_(\d+)_corregit', f.stem).groups()
        return int(g), int(i)
    return sorted(FITXES.glob('fitxa_*_corregit.pdf'), key=clau)


def readme():
    ll = ['# Les figures del treball',
          '',
          'Totes les figures de la memòria i dels annexos del treball de recerca sobre el caos al pèndol doble i al pèndol '
          'magnètic, a resolució completa, amb el número que porten a la memòria al nom del fitxer. Les que surten d\'un '
          'programa són en PDF (vectorials); les del cos de la memòria i les de l\'Annex F, també en PNG.',
          '',
          'Aquesta carpeta la fa `eines/figures_memoria.py`, que copia cada figura de l\'eina que la fa i comprova que la '
          'còpia és idèntica a l\'original. Per tornar a fer una figura, l\'ordre és la de la columna «Eina», des de l\'arrel '
          'del repositori i amb el Python de l\'entorn (`.venv\\Scripts\\python.exe`); les opcions completes són a la '
          'capçalera de cada programa i a `documentacio/index-figures.md`. Els JSON de `figures/` no es versionen: els refà '
          'la mateixa eina. Els de `dades/`, sí.',
          '']
    for carpeta, titol, nota in APARTATS:
        ll += [f'## {titol}', '']
        if nota:
            ll += [nota, '']
        ll += ['| Figura | Fitxer | Què ensenya | Eina | Dades |', '|---|---|---|---|---|']
        files = []
        for c, fitxers, num, que, origen in FIXES:
            if c == carpeta:
                if isinstance(fitxers, tuple):
                    enll = ' · '.join(f'[{Path(f).suffix[1:].upper()}]({carpeta}/{f})' for f in fitxers)
                else:
                    enll = f'[{fitxers}]({carpeta}/{fitxers})'
                files.append((num, enll, que, origen, '—'))
        for c, nom, num, que, _font, formats, eina, dades in D_EINA:
            if c == carpeta:
                enll = ' · '.join(f'[{ext.upper()}]({carpeta}/{nom}.{ext})' for ext in formats)
                files.append((num, enll, que, f'`{eina}`', dades))
        def clau(f):
            m = re.match(r'(?:Figura )?([A-F]?)\.?(\d+)(?:[.-](\d+))?([a-z]?)', f[0])
            return (m.group(1), int(m.group(2)), int(m.group(3) or 0), m.group(4)) if m else ('', 0, 0, '')
        for f in sorted(files, key=clau):
            ll.append('| ' + ' | '.join(f) + ' |')
        ll.append('')
    ll += ['## Les fitxes de les 35 preses', '',
           'Una fitxa per presa real del pèndol doble, amb els angles corregits: θ1 i θ2 al llarg del tir, l\'energia, la '
           'secció de Poincaré de la presa i l\'espai de fases en quatre trams. A l\'annex només hi ha les de la 45_6, la '
           '90_11 i la 150_3 (Figura F.5). Les fa `eines/fitxes_preses.py --memoria`, a partir de les preses reals.', '']
    for f in fitxes_ordenades():
        presa = f.stem.replace('fitxa_', '').replace('_corregit', '')
        ll.append(f'- [{presa}](fitxes/{f.name})')
    ll.append('')
    return '\n'.join(ll)


def figures_del_md():
    """Les figures que cita el text dels resultats, si el text és en aquest ordinador (no es versiona)."""
    if not MD_RESULTATS.exists():
        return None
    s = MD_RESULTATS.read_text(encoding='utf-8')
    return set(re.findall(r'\]\(((?:figures|magnetic/figures)/[^)]+)\.png\)', s))


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--comprova', action='store_true', help='no copia res: només comprova')
    args = ap.parse_args()

    errors = []
    parells = copies()
    falten = [str(font.relative_to(ARREL)) for font, _ in parells if not font.exists()]
    if falten:
        errors += [f'falta la figura de l\'eina: {f}' for f in falten]
    if len(fitxes_ordenades()) != 35:
        errors.append(f'hi ha d\'haver 35 fitxes, i n\'hi ha {len(fitxes_ordenades())}')
    md = figures_del_md()
    if md is not None:
        fonts = {fila[4] for fila in D_EINA}
        for f in sorted(md - fonts):
            errors.append(f'el text cita {f} i no és a la llista')
    if errors:
        for e in errors:
            print('ERROR:', e)
        sys.exit(1)

    copiades = iguals = diferents = 0
    for font, desti in parells:
        if desti.exists() and filecmp.cmp(font, desti, shallow=False):
            iguals += 1
            continue
        if args.comprova:
            diferents += 1
            print('diferent o sense copiar:', desti.relative_to(ARREL))
            continue
        desti.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(font, desti)
        assert filecmp.cmp(font, desti, shallow=False), desti
        copiades += 1

    fixos = [(c, f) for c, fitxers, *_r in FIXES for f in (fitxers if isinstance(fitxers, tuple) else (fitxers,))]
    sense = [f'{c}/{f}' for c, f in fixos if not (SORTIDA / c / f).exists()]
    esperats = {d for _, d in parells} | {SORTIDA / c / f for c, f in fixos} | {SORTIDA / 'README.md'}
    sobren = [p for p in SORTIDA.rglob('*') if p.is_file() and p not in esperats] if SORTIDA.exists() else []

    text = readme()
    cami = SORTIDA / 'README.md'
    if args.comprova:
        if not cami.exists() or cami.read_text(encoding='utf-8') != text:
            diferents += 1
            print('el README no és al dia')
    else:
        cami.write_text(text, encoding='utf-8', newline='\n')

    print(f'figures d\'eina: {len(parells)} fitxers ({copiades} copiats, {iguals} ja iguals)')
    print(f'figures sense eina: {len(fixos) - len(sense)} fitxers de {len(fixos)}')
    for s in sense:
        print('  falta (no surt de cap eina):', s)
    for p in sobren:
        print('  sobra (no és a la llista):', p.relative_to(ARREL))
    if args.comprova:
        print('OK: la carpeta és al dia' if diferents == 0 and not sobren else f'NO és al dia: {diferents} diferències')
        sys.exit(0 if diferents == 0 and not sobren else 1)


if __name__ == '__main__':
    main()
