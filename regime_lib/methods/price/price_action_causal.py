"""Événements de pivots publiés à confirmation, sans réécriture du passé."""
import numpy as np
import pandas as pd

from regime_lib.core.registry import register_method
from regime_lib.core.utils import unaligned_mask
from regime_lib.utils.validation import validated_detector, validity_mask
from .price_action import PriceActionDetector, _atr, _classify_from_pivots, _propagate_and_confidence


@register_method
class PriceActionCausalDetector(PriceActionDetector):
    name = "price_action_causal"
    availability = "bar_close"
    requires_lookahead = False
    regime_description = "Structure des pivots connus à confirmation ; publications historiques immuables."

    @validated_detector
    def fit_predict(self, df):
        n = len(df)
        radius = self.n_fractale
        high, low, close = (df[c].to_numpy(dtype=float) for c in ("high", "low", "close"))
        atr = _atr(high, low, close, self.n_atr_tol) if n else np.array([])
        valid = validity_mask(df).to_numpy() & ~unaligned_mask(df.index, self.timeframe)
        labels = np.full(n, "INCONNU", dtype=object)
        scores = np.zeros(n)
        observed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
        confirmed = observed.copy()
        kinds = np.full(n, "", dtype=object)
        revisions = np.zeros(n, dtype=bool)
        pivots = []
        label, score = "INCONNU", 0.0
        for t in range(n):
            i = t - radius
            if i >= radius and valid[i-radius:t+1].all():
                highs = high[i-radius:t+1]
                lows = low[i-radius:t+1]
                peak = high[i] > np.max(np.delete(highs, radius))
                trough = low[i] < np.min(np.delete(lows, radius))
                # Une barre extérieure doublement extrême est ambiguë : abstention.
                if peak != trough:
                    kind, value = ("S", high[i]) if peak else ("C", low[i])
                    replace = bool(pivots and pivots[-1][1] == kind)
                    accept = not replace or (value > pivots[-1][2] if peak else value < pivots[-1][2])
                    if accept:
                        if replace:
                            pivots.pop()
                        pivots.append((0, kind, value))
                        pivots = pivots[-4:]
                        observed.iloc[t], confirmed.iloc[t] = df.index[i], df.index[t]
                        kinds[t], revisions[t] = kind, replace
                        raw = _classify_from_pivots(pivots, np.array([atr[t]]), 1, self.k_atr)
                        ls, cs = _propagate_and_confidence(*raw, df.index[t:t+1])
                        label, score = ls[0], cs[0]
            if valid[t]:
                labels[t], scores[t] = label, score
        return pd.DataFrame({"regime": pd.Series(labels, index=df.index, dtype=object), "confidence": scores,
                             "pivot_observed_at": observed, "pivot_confirmed_at": confirmed,
                             "pivot_kind": pd.Series(kinds, index=df.index, dtype=object), "pivot_replaces_previous": revisions}, index=df.index)
