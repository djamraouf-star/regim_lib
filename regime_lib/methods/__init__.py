"""

regime_lib — bibliothèque de détection de régimes de marché.

Contenu exposé
--------------
- `RegimeDetector` : contrat de base pour toute méthode.
- `REQUIRED_COLUMNS`, `OPTIONAL_COLUMNS` : format uniforme de sortie.
- Méthodes concrètes : enregistrées dans le registre via l'import de
  `regime_lib.methods` (effet de bord).
- Contexte temporel : `enrichir_contexte` et fonctions associées
  (voir `regime_lib.context`).
- Module d'étude : `Study` pour tester la valeur prédictive des régimes
  (voir `regime_lib.study`).
- `__version__` : version de la bibliothèque.

Familles de méthodes
--------------------
- `volatility` : atr_volatility
- `trend` : er_kaufman, adx
- `statistical` : hmm_gaussian
- `price` : price_action, mss_choch
- `misc` : minkowski_causal
- `entropy` : shannon
- `vector` : pca_axis
- `volume` : ofi, divergence_pv

Note sur les imports
--------------------
`import regime_lib` déclenche l'enregistrement de toutes les méthodes
dans `regime_lib.core.registry.METHOD_REGISTRY`. C'est un **effet de
bord volontaire** : le registre est peuplé au chargement du package.

Le CLI (`regime_lib.cli`) s'appuie sur ce mécanisme, mais un import
direct dans un script utilisateur produit le même effet.
"""

# Import each family so its concrete methods register with METHOD_REGISTRY.
from regime_lib.methods import entropy  # noqa: F401
from regime_lib.methods import misc  # noqa: F401
from regime_lib.methods import price  # noqa: F401
from regime_lib.methods import statistical  # noqa: F401
from regime_lib.methods import trend  # noqa: F401
from regime_lib.methods import vector  # noqa: F401
from regime_lib.methods import volatility  # noqa: F401
from regime_lib.methods import volume  # noqa: F401

from regime_lib.core.base import RegimeDetector
from regime_lib.core.output import OPTIONAL_COLUMNS, REQUIRED_COLUMNS

# Effet de bord : peuplement du registre des méthodes.
import regime_lib.methods  # noqa: F401

# Contexte temporel (sessions, calendrier, événements)
from regime_lib.context import enrichir_contexte  # noqa: F401

# Module d'étude (évaluation de la valeur prédictive)
from regime_lib.study import Study  # noqa: F401


__version__ = "0.1.0"

__all__ = [
    "RegimeDetector",
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
    "Study",
    "enrichir_contexte",
    "__version__",
]
