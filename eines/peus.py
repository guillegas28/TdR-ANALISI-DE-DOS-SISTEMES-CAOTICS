"""Els peus tecnics de les figures de l'analisi: dins de la imatge (les de treball) o a part (les de la memoria).

Les figures de treball porten a sota un paragraf tecnic (la font, el filtre, l'estat, noms de funcions). A la memoria, aixo va
al peu del document i no dins de la imatge (2/11; documentacio/pla-analisi-caos.md, «2/11 · Les figures de la memoria»).

  peu(fig, x, y, text, **kw)   sense mode memoria, fig.text(x, y, text, **kw), exactament com abans; amb mode memoria, no
                               el dibuixa. En tots dos casos, el guarda a la figura
  desa(fig, nom, directori)    estil.desa, i apunta el peu de la figura; amb mode memoria, s'atura si encara hi queda un
                               paragraf (MF2) o si la figura no en te cap (MF4)
  escriu(directori)            peus.json: figura -> text, afegit al que ja hi hagi. Les de treball, a figures/peus.json, i
                               les de memoria, a figures/memoria/peus.json: comparant-los es comprova la MF4

Les eines hi entren amb --memoria, que crida memoria() i desa a figures/memoria/.
"""

from __future__ import annotations

import json
from pathlib import Path

MEMORIA = False
LLARGADA_PARAGRAF = 200       # MF2: els peus en fan mes de 300; el titol mes llarg (les fitxes), uns 120
_PEUS: dict[str, str] = {}


def memoria() -> None:
    global MEMORIA
    MEMORIA = True


def peu(fig, x, y, text, **kw):
    fig._peu_tecnic = text
    if MEMORIA:
        return None
    return fig.text(x, y, text, **kw)


def desa(fig, nom, directori="figures") -> None:
    from pendol import estil

    text = getattr(fig, "_peu_tecnic", None)
    if MEMORIA:
        llargs = [t.get_text() for t in fig.texts if len(t.get_text()) > LLARGADA_PARAGRAF]
        if llargs:
            raise SystemExit(f"MF2: la figura {nom} encara porta un paragraf: {llargs[0][:80]}...")
        if text is None:
            raise SystemExit(f"MF4: la figura {nom} no te peu tecnic per apuntar")
    if text is not None:
        _PEUS[nom] = text
    estil.desa(fig, nom, directori)


def escriu(directori) -> None:
    if not _PEUS:
        return
    cami = Path(directori) / "peus.json"
    tots = json.loads(cami.read_text(encoding="utf-8")) if cami.exists() else {}
    tots.update(_PEUS)
    cami.parent.mkdir(parents=True, exist_ok=True)
    cami.write_text(json.dumps(dict(sorted(tots.items())), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  peus: {cami} ({len(_PEUS)} d'aquesta passada, {len(tots)} en total)")
