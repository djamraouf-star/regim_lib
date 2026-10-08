# Compléments techniques de l'audit — 8 octobre 2026

## Variantes causales

`price_action_causal` publie une fractale de rayon `n_fractale` à la clôture
de la barre `i + n_fractale`. `pivot_observed_at` désigne la barre extrême,
`pivot_confirmed_at` la publication. Les deux dates ne sont renseignées que
sur les événements acceptés. Une barre simultanément sommet et creux stricts
est ambiguë et ignorée. Le remplacement d'un pivot de même type plus ancien
est signalé par `pivot_replaces_previous` **à la date du nouvel événement** ;
aucune sortie passée n'est modifiée. Le régime persiste entre événements,
et reste inconnu tant que deux sommets et deux creux ne sont pas connus.
Les fenêtres contenant des barres invalides ne confirment aucun pivot.
La tolérance utilise l'ATR connu à confirmation.

`hmm_causal` utilise les `train_size` premières barres comme train fixe,
incluant la chauffe des features (20 barres). La normalisation et le HMM
sont appris exclusivement sur les features disponibles dans ce préfixe.
Les sorties du train restent `INCONNU`, avec `is_warmup=True`. À partir de
la barre suivante, le filtre calcule `p(S_t | x_1,...,x_t)` par récurrence
avant en domaine logarithmique. Il n'utilise ni lissage arrière ni Viterbi.
Les états sont ordonnés par leur rendement moyen appris sur train ; leurs
identifiants ne sont pas universels entre fits.

Le préfixe doit rester identique pour comparer deux exécutions. Il n'y a
pas de réentraînement glissant implicite. Les barres invalides ou partielles
sont refusées, sans imputation ; les transitions suivent les observations
fournies, pas un temps physique continu. Les observations sans features
complètes restent inconnues. Les colonnes constantes utilisent une échelle
unitaire. Un train trop court ou des transitions dégénérées sont refusés.

`hmm_diagnostics` contient un JSON sur les lignes hors train : fin du train,
effectif, normalisation, moyennes des états dans les unités originales,
matrice de transition, historique de vraisemblance et convergence réelle
(le seul épuisement des itérations ne suffit pas). Ces diagnostics sont
également dans les attributs du résultat Python. La convergence ne garantit
pas une pertinence économique. Les méthodes historiques `price_action` et
`hmm_gaussian` conservent leur comportement rétrospectif et leur garde-fou.

```bash
regime-lib --url prix.parquet --asset TEST --source-timeframe H1 \
  --timestamp-convention close --timeframe H1 \
  --methods price_action_causal,hmm_causal \
  --method-params '{"hmm_causal":{"train_size":500}}' --output resultat
```

## Noms explicites des indicateurs simplifiés

| Nom public recommandé | Ancien identifiant conservé | Sens réel |
|---|---|---|
| `retrospective_pivot_breakout` | `mss_choch` | Tests rétrospectifs de cassure des pivots récents/anciens, sans séquence de confirmation |
| `quote_size_imbalance` | `ofi` | Moyenne glissante du déséquilibre des tailles affichées |
| `quote_absolute_imbalance_buckets` | `vpin` | Moyenne du déséquilibre absolu par buckets de tailles affichées |

Les nouveaux noms ont leurs propres sections de profil. Pour migrer un
profil personnalisé, recopier ses paramètres sous le nouveau nom. Les
anciens identifiants et labels restent disponibles pour lire/rejouer les
usages historiques. `indicator_name`, exporté dans le parquet et le manifeste,
donne le nom descriptif même pour ces identifiants historiques.

Pour `quote_size_imbalance`, on calcule par barre
`q = (ask_volume - bid_volume) / (ask_volume + bid_volume)`, puis une moyenne
glissante simple des `q` disponibles ; les totaux nuls sont omis. La chauffe
requiert `max(1, fenetre // 2)` observations disponibles. Ce n'est ni une
variation de carnet conditionnée aux prix ni un déséquilibre de transactions.
Les labels `TAILLE_ASK_DOMINANTE` et `TAILLE_BID_DOMINANTE` ne déduisent aucune
pression acheteuse ou vendeuse. La valeur s'appelle `quote_imbalance`.

Pour `quote_absolute_imbalance_buckets`, chaque barre apporte
`V = ask_volume + bid_volume` et `D = abs(ask_volume - bid_volume)`.
Le fractionnement répartit `D` proportionnellement au volume dans les buckets
de capacité `B`. Un bucket clos vaut `somme(D_fractionné)/B` ; la valeur publiée
est la moyenne des `n_buckets` derniers buckets clos. **Les signes opposés
ne s'annulent pas entre barres**, car la valeur absolue est prise par barre.
La valeur dépend donc du timeframe d'agrégation. Les labels deviennent
`DESEQUILIBRE_FAIBLE/MOYEN/ELEVE`, la valeur `quote_absolute_imbalance`.
Aucun volume exécuté, sens d'agression ou toxicité n'est estimé.

Pour `retrospective_pivot_breakout`, `CASSURE_HAUTE/BASSE_RECENTE` correspond
au test du pivot récent et `CASSURE_HAUTE/BASSE_ANCIENNE` au test du pivot
ancien, prioritaire si les deux tests sont vrais. Le test est répété à chaque
barre, pas seulement au franchissement. Ces labels ne représentent pas une
machine à états CHOCH puis MSS. L'autorisation de lookahead reste obligatoire.

## Convention ADX

Le calcul adopte l'amorçage de [TA-Lib ADX](https://github.com/TA-Lib/ta-lib/blob/main/src/ta_func/ta_ADX.c) :
sommes des mouvements/TR sur les transitions 1 à `n-1`, puis récurrence
`S_t = S_(t-1) - S_(t-1)/n + mouvement_t` à partir de `t=n`.
Le premier ADX est la moyenne des `n` DX de `n` à `2n-1`, puis lissage
Wilder avec coefficient `1/n`. Aucun arrondi intermédiaire. TR ou somme DI
nuls donnent un DX nul après chauffe. Une série plate devient donc RANGE.
Le premier ADX publié reste à `2n-1`. Les anciens résultats ADX doivent être
régénérés : la précédente EMA était amorcée par la première valeur.
Le test rationnel indépendant couvre DI, DX, ADX, chauffe et série plate.

## Scores et calibration explicite

Chaque méthode déclare `confidence_kind`, `confidence_description` et
`confidence_calibrated=False`. Ces champs sont exportés ; les scores natifs
ne deviennent pas comparables en probabilité par simple normalisation dans
[0,1]. Ils mesurent selon la méthode une marge, une constante par label,
un posterior transformé ou la probabilité filtrée d'un état latent.

`ConfidenceCalibrator` apprend une régression isotone sur un score et un
**événement binaire défini par l'utilisateur**. Le sens croissant/décroissant
est fixé avant évaluation. Utiliser un calibrateur par méthode/configuration,
avec le même événement et horizon si l'objectif est une comparaison commune.
Ne fournir que des scores valides (hors chauffe et inconnus) et des issues
observées. Les deux issues doivent être représentées dans le train.

```python
from regime_lib.study import ConfidenceCalibrator

cal = ConfidenceCalibrator("ret_fwd_5 > 0", increasing=True)
cal.fit(scores_train, issues_train, target_end=fins_cibles_train)
probabilites = cal.predict(scores_test)
diagnostic = cal.evaluate(scores_test, issues_test)
```

Toutes les dates test doivent suivre strictement la dernière fin de cible du
train : la purge est contrôlée. L'évaluation produit Brier, Brier de référence
(taux de base du train), Brier skill, ECE à dix bins fixes et table de fiabilité.
`cal.metadata` contient les bornes, effectifs et nœuds de la calibration pour
export JSON. Ni les issues test ni leur taux de base ne changent le modèle.
Cette API est explicite et indépendante de `Study.run` ; aucune cible de
succès ou calibration n'est imposée automatiquement. Les tests synthétiques
valident le mécanisme, pas la calibration empirique sur marché (C5).

## Excursions et horizons

Les horizons restent des nombres de barres de la grille temporelle déclarée.
Les trous, barres invalides ou partielles invalident toute la trajectoire.
Les `K` dernières lignes restent NaN.

`close_min_fwd_K` et `close_max_fwd_K` nomment explicitement les excursions
des clôtures relatives à `close[t]`, en incluant le point d'entrée. `dd` et
`ru` restent des alias compatibles. Ce ne sont pas des drawdowns de pic à creux.

L'entrée des nouvelles cibles est `close[t]`. Seuls les high/low des barres
`t+1 ... t+K` sont accessibles, avec le prix d'entrée comme extrême initial :

| Famille | Formule, avec H=max(close[t], high futurs), L=min(close[t], low futurs) |
|---|---|
| `mfe_long` | H / close[t] - 1 ≥ 0 |
| `mae_long` | L / close[t] - 1 ≤ 0 |
| `mfe_short` | 1 - L / close[t] ≥ 0 |
| `mae_short` | 1 - H / close[t] ≤ 0 |

Exemple : `mfe_long_fwd_20`. Les cibles `time_mfe_long_fwd_20`,
`time_mae_long_fwd_20`, `time_mfe_short_fwd_20`, `time_mae_short_fwd_20`
donnent le nombre de barres jusqu'au **premier** extrême, entre 0 et K.
Zéro signifie qu'aucune excursion n'améliore l'extrême initial. Elles ne
déterminent pas l'ordre intrabar des high/low et n'incluent pas de frais.
Les P&L short sont normalisés par le prix d'entrée, sans rendement inverse.
Pour les tests de modalités, « succès » signifie dépassement de la médiane
de référence ; une durée plus grande n'est pas présentée comme meilleure.
Les nouvelles familles fonctionnent dans Study et conservent sa purge par
fin d'horizon, même si un extrême a été atteint avant.

## Comparaison aux partitions PELT

`comparer_methodes` conserve la pureté pour compatibilité mais fournit aussi :

- `baseline_constante` : pureté 1 de la classification à label constant ;
- `baseline_majoritaire` : fréquence globale du label majoritaire de la méthode ;
- `gain_purete` : pureté moins cette fréquence ;
- `ami` et `ari` : information mutuelle et Rand ajustés au hasard ;
- `partition_degeneree` : absence de partition informative d'un côté.

Les comparaisons utilisent le même support valide commun. Une partition
constante donne AMI/ARI nuls, même si les deux côtés sont constants.
Les partitions à un label par barre ne gagnent pas artificiellement un
score d'accord parfait. Sans observations communes, les scores sont NaN.
Ces mesures restent descriptives et ne valident aucune stratégie.

## Validation

Les références et tests d'intégration sont dans
[les tests de complétion technique](../../tests/test_technical_completion.py).
Ils couvrent les dates de publication, troncatures et futurs modifiés,
le filtre HMM contre une récurrence probabiliste indépendante, les excursions
contre un exemple chiffré, les partitions triviales, la purge de calibration,
et le parcours CLI → Parquet → Study → exports.

Validation complète le 8 octobre 2026 : **608 tests réussis**, sans exclusion,
avec un avertissement joblib préexistant sur la détection des cœurs physiques.
