# Décisions de conception

Ce document reprend le tableau de décisions d'architecture (v1) qui fait foi pour l'implémentation de la bibliothèque.

## Tableau des décisions

| # | Point | Décision |
|---|-------|----------|
| 1 | `timestamp` index vs colonne | Entrée : `DatetimeIndex` nommé `timestamp`. Sortie : colonne `timestamp` via `reset_index()`. |
| 2 | `regime` ↔ `regime_id` | `REGIME_MAP: dict[str, int]` obligatoire, stable, bijectif. |
| 3 | `confidence` non probabiliste | `1.0` par défaut, heuristique documentée autorisée, `NaN` interdit. |
| 4 | `asset` | CLI `--asset` explicite, fallback sur nom de fichier, erreur sinon. |
| 5 | Timeframes | `label='right'`, `closed='right'`, colonne `is_partial` ajoutée. |
| 6 | Lookahead | Interdit par défaut + override explicite par méthode, erreur si requis et absent. |
| 7 | Sortie CLI | Un parquet long unique + sidecar `run_meta.json`. |
| 8 | Reproductibilité | Colonne `params_hash` (SHA256 court) + sidecar JSON. |

## Périmètre d'utilisation

La bibliothèque soutient la recherche exploratoire : analyse des régimes de
marché et exploration d'hypothèses de stratégie. Elle n'est ni conçue ni
validée pour un usage en production de trading, la génération d'ordres ou
l'exécution automatisée. Les sorties ne doivent pas être considérées comme des
signaux de trading validés.

## Contrat du détecteur

Toute méthode hérite de `regime_lib.core.base.RegimeDetector` et implémente `fit_predict(df) -> pd.DataFrame`.
Voir la documentation détaillée dans [`adding_method.md`](./adding_method.md).

## Méthodes avec lookahead structurel

Certaines méthodes ont un lookahead **par construction** — tenter de les rendre causales dénaturerait l'algorithme :
- `price_action` : fractales strictes nécessitant d'observer les $N$ barres futures ;
- `hmm_gaussian` : normalisation z-score globale et apprentissage HMM offline sur l'ensemble de la série.

Ces méthodes :
- définissent explicitement l'attribut de classe `requires_lookahead = True` ;
- refusent de s'instancier sans `--allow-lookahead-override <nom>:true` (ou `allow_lookahead=True` en Python) ;
- apparaissent dans `run_meta.json` sous la clé `methodes_avec_lookahead` et marquent le run comme `run_descriptif: true` ;
- sont destinées à l'analyse descriptive a posteriori, à l'annotation de datasets pour le ML supervisé et aux comparaisons d'écart avec le causal.
