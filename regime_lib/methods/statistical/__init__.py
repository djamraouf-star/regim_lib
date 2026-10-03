"""
Méthodes statistiques (HMM, Markov-switching, change-point).

⚠ Ces méthodes sont généralement OFFLINE : elles utilisent du lookahead
structurel (normalisation sur toute la série, fit du modèle sur toute la
série). Elles doivent définir `requires_lookahead = True`.
"""

from regime_lib.methods.statistical import hmm_gaussian  # noqa: F401
