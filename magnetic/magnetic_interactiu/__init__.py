"""Mapa de conques interactiu: la copia del nucli per al navegador.

QUE HI HA AQUI I QUE NO
-----------------------
Aquesta carpeta NO conte fisica nova. Conte una TRADUCCIO a JavaScript del
nucli que ja hi ha a magnetic/motor.py i magnetic/fisica.py, perque un
navegador pugui recalcular el mapa quan l'usuari amplia un requadre.

EL PYTHON MANA. El nucli de Python continua sent el que es valida i el que
genera els mapes que van a la memoria del treball. El JavaScript es una copia,
i una copia nomes val si es pot demostrar que diu el mateix: aixo ho fa
magnetic/magnetic_interactiu/compara.py, i s'ha de tornar a passar cada cop que
es toqui cap dels dos costats.

Es exactament la mateixa disciplina que el bloc 8 amb Numba (vegeu
magnetic/compila.py): alla es comparava el mateix programa compilat i sense
compilar; aqui es comparen dues implementacions, i per aixo la comparacio es
MES exigent, no menys.

Del codi ja validat aqui nomes se'n LLEGEIX. No s'hi toca res.

    .venv\\Scripts\\python.exe -m magnetic.magnetic_interactiu.compara
"""
