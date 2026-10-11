# TODO — regime_lib

Feuille de route mise à jour le 11 octobre 2026. Cette page résume les suites à donner ; le détail et
les preuves des constats sont tenus dans l'[audit synthétique](https://github.com/djamraouf-star/regim_lib/blob/main/AUDIT_SYNTHETIQUE_2026-10-04.md),
qui fait foi pour leurs statuts.

## État de l'audit

L'audit recense **34 constats corrigés, 1 partiel et 0 ouvert**. La suite
complète a été validée avec **608 tests réussis**, sans exclusion, le
8 octobre 2026. Ces résultats logiciels ne constituent pas une validation
empirique sur données de marché.

### Compléments techniques livrés

- Variantes causales des pivots et du HMM (**I9–I10**).
- Nom explicite de la variante de cassure de pivots et des proxys de cotations ;
  amorçage ADX corrigé et vérifié (**I11–I13**).
- Contrats des scores, excursions MFE/MAE
  long/short et temps aux extrêmes, baselines et scores ajustés PELT (**I8, I15, I20**).

Voir les [conventions, limites et exemples](technical_completion.md).

### Travail restant

- Régénérer les anciens résultats ADX ; utiliser les noms descriptifs des
  proxys pour les nouvelles analyses, avec migration explicite des profils.

Le projet reste un outil de recherche exploratoire, non validé pour le trading
en production. Aucun score global ni règle de trading n'est imposé.

## Préalable à la mise en production

La calibration est un préalable à la mise en production de l'API, sous la
responsabilité de son utilisateur. Elle ne fait pas partie des modules de
fonctions à livrer dans la version publique : aucun moteur ni protocole de
calibration par actif ou par famille n'est prévu dans cette feuille de route.
Chacun choisit sa méthode et les paramètres adaptés à son usage.

Les paramètres de [default.yaml](../config/profiles/default.yaml) sont
arbitraires et propres à aucun actif. Des calibrations spécifiques à EURUSD
et XAUUSD peuvent être fournies uniquement à titre d'exemples pour certains
indicateurs.

Cette décision de périmètre ne modifie pas le statut historique du constat
C5 ; sa confirmation empirique relève de la préparation à la production,
pas de la livraison d'un module de calibration.

## Principes actifs

- Le lookahead est interdit par défaut (`allow_lookahead=False`).
- Les sorties suivent un schéma uniforme ; les labels restent propres à chaque
  méthode et ne sont pas réputés comparables sémantiquement.
- Les paramètres effectifs, leur provenance et leurs empreintes sont exposés
  pour la reproductibilité.
- Les méthodes et leur inventaire sont décrits à partir du registre central.
- Les archives conservent l'historique et ne définissent pas la roadmap active.
