# Référence API

Ce document liste les modules documentés automatiquement via Sphinx.

La calibration est [hors du périmètre public à livrer](calibration.md).
Les références générées depuis le code peuvent encore montrer les prototypes
historiques : leur retrait du paquet reste à réaliser. Leur présence dans
l'autodocumentation ne constitue pas un engagement de livraison.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   regime_lib.core.base
   regime_lib.core.data_loader
   regime_lib.core.output
   regime_lib.core.registry
   regime_lib.methods.volatility.atr_volatility
   regime_lib.methods.trend.er_kaufman
   regime_lib.methods.trend.adx
   regime_lib.methods.price.price_action
   regime_lib.methods.statistical.hmm_gaussian
   regime_lib.study
   regime_lib.utils.validation
   regime_lib.utils.repro
   regime_lib.study.report
   regime_lib.cli
```
