# Anàlisi de dos sistemes caòtics, el pèndol doble i el pèndol magnètic, i la seva extrapolació a la realitat

Treball de recerca de batxillerat de Guillem sobre sistemes caòtics, amb dos muntatges reals: un **pèndol doble** de barres i
un **pèndol magnètic** amb tres imants. La pregunta és fins on es pot predir un sistema caòtic real. La memòria es lliura en
paper; aquí hi ha el codi, les dades i les figures d'on surten els seus resultats.

## Què hi ha

| Carpeta | Què hi ha |
|---|---|
| [`figures_memoria/`](figures_memoria/) | **totes les figures de la memòria i dels annexos**, a resolució completa, amb un índex que diu d'on surt cadascuna; les fitxes de les 35 preses; i, a [`altres/`](figures_memoria/altres/), més figures vigents del treball (seccions de Poincaré, mapes del magnètic, la portada) |
| [`pendol/`](pendol/) | el pèndol doble: el model físic, el simulador, la lectura de les preses del Tracker, les comparacions i la simulació interactiva |
| [`magnetic/`](magnetic/) | el pèndol magnètic: el model, el mapa de conques, la seva anàlisi i la pàgina interactiva |
| [`eines/`](eines/) | els programes d'anàlisi i de figures; cadascun explica a la capçalera què fa i com es fa servir |
| [`dades/`](dades/) | les preses del Tracker (els projectes `.trk`, amb totes les marques), les preses convertides (`dades/net/`, un CSV `t,theta1,theta2,omega1,omega2` i un JSON per presa) i el resultat de cada càlcul |

No hi són els vídeos, per la mida, ni el que es torna a fer amb una ordre (les figures de treball, `figures/`, i les simulacions
grosses, `dades/sim/`). Els comentaris del codi citen de vegades documents de treball interns (`documentacio/`, `esborranys/`)
que no es publiquen.

## Com es fa servir

Amb Python 3.13, des d'aquesta carpeta (a Windows; en altres sistemes, `.venv/bin/python`):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Les simulacions interactives:**

```powershell
# el pèndol doble i un clon que surt d'un angle lleugerament diferent, separant-se en directe
.\.venv\Scripts\python.exe -m pendol.interactiu
# la mateixa finestra, amb la fricció mesurada i el botó per importar-hi una presa real o una simulació
.\.venv\Scripts\python.exe -m pendol.interactiu_d3
```

El mapa de conques interactiu del pèndol magnètic és una pàgina web: n'hi ha prou d'obrir
[`magnetic/magnetic_interactiu/mapa-interactiu.html`](magnetic/magnetic_interactiu/mapa-interactiu.html) amb el navegador. Es torna a
construir amb `-m magnetic.magnetic_interactiu.construeix`.

**Algunes ordres del pèndol doble:**

```powershell
.\.venv\Scripts\python.exe eines\converteix_preses.py                                    # els .trk, al format canònic
.\.venv\Scripts\python.exe eines\balanc_energia.py dades/net/90_*.csv dades/net/150_*.csv  # el balanç d'energia
.\.venv\Scripts\python.exe -m pendol.simulador --theta1 150 --temps 15 --sortida dades/sim/tir.csv
.\.venv\Scripts\python.exe eines\deriva_equacions.py                                      # les equacions, comprovades
```

La majoria d'eines tenen `--autoprova`, que en comprova el funcionament amb casos de resposta coneguda.

## Com s'ha treballat

El codi s'ha escrit amb l'ajuda de Claude Code, sota la supervisió de l'autor, que ha decidit els criteris i els resultats. Els
criteris per comparar el model amb el pèndol real es van escriure abans de mirar cap comparació (Taula C.4 dels annexos).
