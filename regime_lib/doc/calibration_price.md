# Historique — calibration de la famille price

**Chantier retiré du périmètre public le 11 octobre 2026.** Cette note conserve
le contexte d'une exploration passée ; elle ne définit aucun travail à livrer.
La [politique de calibration](calibration.md) remplace l'ancien projet de
moteur différencié par famille.

L'exploration portait sur EURUSD M15, avec un horizon de cinq barres, et sur
les méthodes `price_action`, `price_action_causal` et
`retrospective_pivot_breakout` (alias `mss_choch`). Un inventaire des labels,
des scores et de leur disponibilité temporelle avait été préparé.

Deux objectifs avaient été distingués : la conformité du mouvement futur
au label et la persistance du label. Ils mesurent des propriétés différentes.
Les scores constants par label et les annotations rétrospectives nécessitaient
également une interprétation explicite. Ces pistes restent des notes de
recherche, sans méthode imposée aux utilisateurs ni validation générale.

Aucun moteur, protocole ou fonction de calibration propre à cette famille
n'est prévu dans la version publique. Les prototypes présents dans le dépôt
sont historiques ; cette mise à jour documentaire ne les retire pas du code.
