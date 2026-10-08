# TODO — regime_lib

État au 8 octobre 2026. Cette page résume les suites à donner ; le détail et
les preuves des constats sont tenus dans l'[audit synthétique](../../AUDIT_SYNTHETIQUE_2026-10-04.md),
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
- Contrats des scores, calibration explicite hors train, excursions MFE/MAE
  long/short et temps aux extrêmes, baselines et scores ajustés PELT (**I8, I15, I20**).

Voir les [conventions, limites et exemples](technical_completion.md).

### Travail restant

- **Confirmation empirique C5 (partiel)** : figer données, périodes et critères
  avant consultation d'une période indépendante ; vérifier calibration et
  stabilité. Les validations synthétiques ne constituent pas cette preuve.
- Régénérer les anciens résultats ADX ; utiliser les noms descriptifs des
  proxys pour les nouvelles analyses, avec migration explicite des profils.

Le projet reste un outil de recherche exploratoire, non validé pour le trading
en production. Aucun score global ni règle de trading n'est imposé.

## Principes actifs

- Le lookahead est interdit par défaut (`allow_lookahead=False`).
- Les sorties suivent un schéma uniforme ; les labels restent propres à chaque
  méthode et ne sont pas réputés comparables sémantiquement.
- Les paramètres effectifs, leur provenance et leurs empreintes sont exposés
  pour la reproductibilité.
- Les méthodes et leur inventaire sont décrits à partir du registre central.
- Les archives conservent l'historique et ne définissent pas la roadmap active.
