# Calibration — préalable à la production

## Périmètre public

La calibration est un préalable à la mise en production de l'API, sous la
responsabilité de l'utilisateur. Elle ne constitue pas un module de fonctions
à livrer dans la version publique. Aucun moteur de calibration par actif,
par famille ou par indicateur ne fait partie du périmètre public prévu.
Chaque utilisateur choisit sa méthode, ses paramètres et ses critères de
validation en fonction de son usage.

Les valeurs de [default.yaml](../config/profiles/default.yaml) sont des
**paramètres arbitraires, propres à aucun actif**. Elles ne sont ni une
calibration EURUSD ni des paramètres validés pour la production. Les fenêtres
et seuils peuvent être remplacés par des profils utilisateur selon les
[règles de configuration](config.md).

## Exemples éventuels

Des calibrations spécifiques à **EURUSD et XAUUSD**, limitées à certains
indicateurs, peuvent être fournies à titre d'exemples. Elles n'imposent aucune
méthode et ne deviennent pas les paramètres par défaut de l'API.
Le profil EURUSD contient des exemples de surcharges ; le profil XAUUSD
hérite actuellement des défauts sans surcharge. Leur présence ne démontre
pas une validation empirique.

## Scores et validation

Un score `confidence` n'est pas nécessairement une probabilité de justesse du
régime. Les métadonnées `confidence_kind`, `confidence_description` et
`confidence_calibrated=False` précisent son interprétation. Une normalisation
dans [0,1] ne constitue pas une calibration probabiliste.

L'utilisateur distingue l'ajustement des paramètres de l'indicateur et,
s'il en a besoin, l'estimation d'une probabilité pour un événement explicite.
Il définit les données, les périodes et les critères avant l'évaluation,
vérifie la disponibilité temporelle des observations et réserve des données
indépendantes pour la validation. Les tests logiciels ne remplacent pas
cette validation ; la calibration seule ne démontre pas l'aptitude au trading.

## Historique et état du dépôt

Des prototypes de calibration et des études exploratoires ont été développés
avant cette décision de périmètre. Leur présence dans le dépôt ne constitue
pas un engagement de livraison ou une recommandation de méthode.
À la date de cette mise à jour documentaire, `ConfidenceCalibrator` reste
exporté par le paquet ; son retrait du code distribué reste à réaliser.
Les prototypes price développés localement ne sont pas publiés avec cette
mise à jour documentaire.
Cette page ne les présente plus comme un parcours d'utilisation de l'API publique.

La [note historique price](calibration_price.md) conserve le contexte du
chantier abandonné. Le statut historique C5 de l'audit reste inchangé ; sa
confirmation empirique relève de la préparation à la production, et non de
la livraison d'un module de calibration. Voir la [feuille de route](TODO.md).
