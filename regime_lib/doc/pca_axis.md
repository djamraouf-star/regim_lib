# PCA axis — axe principal de la trajectoire du prix

**Méthode** : `pca_axis`  
**Famille** : `methods/vector/`  
**Lookahead** : non (causal)

## Choix de conception

`pca_axis` applique l'ACP glissante aux deux coordonnées de la trajectoire
du prix :

```text
(rang temporel de la barre, log(close))
```

Ce choix vise à mesurer l'axe de variance maximale de l'évolution du prix
sur la fenêtre, et non celui de descripteurs de bougies. Une ACP de
`log_return`, de volatilité réalisée, de range normalisé et de position de
clôture décrit la co-variation de ces caractéristiques. Elle ne garantit
pas que son premier axe représente le drift directionnel : en particulier,
des caractéristiques presque constantes peuvent laisser une autre feature
dominer la covariance.

Le rang temporel est utilisé plutôt que la valeur brute du timestamp. Chaque
fenêtre est centrée et chaque coordonnée est standardisée avant le calcul de
la covariance. L'axe temporel et l'axe de prix ont ainsi la même échelle,
indépendamment des unités et du niveau absolu du prix. `log(close)` permet
d'interpréter les variations comme des variations relatives. Les observations
sont supposées également espacées dans le timeframe fourni ; les écarts
calendaires entre barres ne sont pas pondérés.

## Orientation et mesures

Le signe d'un vecteur propre n'étant pas déterminé par l'ACP, PC1 est orienté
de façon que sa composante temporelle soit positive. Le signe de sa
composante prix donne alors le sens haussier ou baissier.

Pour cette ACP à deux dimensions standardisées, la matrice de covariance a
une variance unitaire sur chacune des deux coordonnées. Les mesures sont
donc liées :

- `pca_direction` est la corrélation signée entre le rang temporel et
  `log(close)`, équivalente à l'écart signé entre les deux valeurs propres
  rapporté à leur somme. Elle varie de -1 à 1.
- `pca_eigen_ratio` est `λ1 / (λ1 + λ2)`.
- `pca_variance_ratio` est également `λ1 / (λ1 + λ2)`. Les deux ratios sont
  égaux pour cette définition à deux dimensions.

Un ratio élevé indique que les points temps-prix s'alignent davantage sur
un axe dominant. Ce ratio n'est pas l'Absorption Ratio classique calculé sur
un ensemble de facteurs ou de variables de marché : ici, il décrit la
concentration de variance entre le temps et le log-prix. La direction du
marché est portée par le signe de la relation temps-prix, et non par le
ratio seul.

## Régimes

| Régime | Condition |
|---|---|
| `HAUSSIER` | `pca_direction > seuil_direction` et `pca_eigen_ratio > seuil_eigen_ratio` |
| `BAISSIER` | `pca_direction < -seuil_direction` et `pca_eigen_ratio > seuil_eigen_ratio` |
| `NEUTRE` | mesures calculées mais conditions directionnelles non satisfaites |
| `INCONNU` | chauffe, fenêtre insuffisante, données non valides, barre partielle ou hors grille |

## Causalité

À l'indice `t`, l'ACP est calculée sur les `fenetre` observations allant de
`t-fenetre` à `t-1`. La barre `t` n'entre pas dans le calcul de ses propres
mesures. Aucun lookahead n'est nécessaire.

## Paramètres

| Paramètre | Défaut (H1) | Rôle |
|---|---:|---|
| `fenetre` | 100 | Nombre d'observations précédentes de la trajectoire |
| `seuil_direction` | 0.3 | Corrélation signée minimale pour classer une direction |
| `seuil_eigen_ratio` | 0.65 | Concentration minimale de variance sur PC1 |

Le profil `default.yaml` peut définir la fenêtre par timeframe et les deux
seuils globaux. Les valeurs données sont des seuils de classification, pas
des paramètres universels : elles doivent être évaluées sur l'actif et le
timeframe étudiés.
