"""
Caractérisation des segments produits par la segmentation PELT.

Un segment est une **phase homogène** du prix. Pour l'utiliser comme
référence de projection, on le caractérise par des mesures objectives
calculées sur ses barres :

    - duree          : nombre de barres du segment
    - direction      : signe de (close_fin - close_debut) ∈ {-1, 0, +1}
    - pente          : close_fin - close_debut (variation absolue)
    - amplitude      : high_max - low_min
    - efficience     : |pente| / amplitude ∈ [0, 1]
                       1.0 = mouvement parfaitement directionnel
                       0.0 = oscillation pure (chop)
    - volatilite     : écart-type des log-returns intra-segment
    - pente_atr      : pente / ATR_moyen (normalisée par volatilité)

Ces features constituent les **régimes objectifs**. Chaque méthode sera
ensuite projetée dessus pour mesurer son accord avec la structure du
prix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from regime_lib.utils.validation import validate_alignment, validate_ohlcv, validity_mask
from regime_lib.core.temporal import resolve_timeframe, expected_grid


def caracteriser_segments(
    df: pd.DataFrame,
    segment_id: pd.Series,
    atr_fenetre: int = 14,
    *,
    timeframe: str | None = None,
    expected_index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """
    Produit un DataFrame de features par segment.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV indexé par DatetimeIndex.
    segment_id : pd.Series
        Identifiant de segment par barre (résultat de
        `segmenter_dataframe`).
    atr_fenetre : int
        Fenêtre pour l'ATR utilisé dans `pente_atr`.

    Returns
    -------
    pd.DataFrame
        Une ligne par segment, indexé par `segment_id`. Colonnes :
        debut, fin, n_barres, direction, pente, amplitude,
        efficience, volatilite, pente_atr.
    """
    if "close" not in df.columns:
        raise ValueError("La colonne 'close' est requise.")
    validate_ohlcv(df)
    validate_alignment(df.index, segment_id.index)
    if segment_id.isna().any():
        raise ValueError("Identifiants de segment manquants.")
    columns = ["segment_id", "debut", "fin", "n_barres", "direction", "pente",
               "amplitude", "efficience", "volatilite", "pente_atr", "is_valid"]
    if df.empty:
        return pd.DataFrame(columns=columns).set_index("segment_id")
    if atr_fenetre < 1:
        raise ValueError("atr_fenetre doit être positif.")
    # Un identifiant ne doit jamais recoller des épisodes disjoints.
    starts = segment_id.ne(segment_id.shift())
    if segment_id[starts].duplicated().any():
        raise ValueError("Identifiant de segment réutilisé sur des épisodes disjoints.")
    tf = resolve_timeframe(df, timeframe)
    grid = expected_grid(df.index, tf, expected_index)
    positions = grid.get_indexer(df.index)
    gap_before = pd.Series(np.r_[False, np.diff(positions) != 1], index=df.index)

    # Aligner segment_id sur l'index de df
    seg = pd.Series(
        segment_id.values, index=df.index, name="segment_id",
    )
    df_work = df.assign(segment_id=seg.values)

    # ATR pour la normalisation de pente
    high = df_work["high"].to_numpy(dtype=float)
    low = df_work["low"].to_numpy(dtype=float)
    close = df_work["close"].to_numpy(dtype=float)
    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    tr = np.fmax(
        np.fmax(high - low, np.abs(high - prev_close)),
        np.abs(low - prev_close),
    )
    atr = pd.Series(tr, index=df_work.index).rolling(
        atr_fenetre, min_periods=1
    ).mean()
    df_work["atr"] = atr.values

    # Log-return pour la volatilité
    log_return = np.log(
        df_work["close"] / df_work["close"].shift(1)
    )
    df_work["log_return"] = log_return.values

    # Agrégation par segment
    rows: list[dict] = []
    for sid, g in df_work.groupby("segment_id", sort=True):
        close_debut = float(g["close"].iloc[0])
        close_fin = float(g["close"].iloc[-1])
        high_max = float(g["high"].max())
        low_min = float(g["low"].min())
        amplitude = high_max - low_min
        pente = close_fin - close_debut
        efficience = (
            abs(pente) / amplitude if amplitude > 0 else 0.0
        )
        direction = int(np.sign(pente))
        atr_moy = float(g["atr"].mean())
        pente_atr = pente / atr_moy if atr_moy > 0 else 0.0
        volatilite = float(g["log_return"].std())

        rows.append({
            "segment_id": int(sid),
            "debut": g.index[0],
            "fin": g.index[-1],
            "n_barres": len(g),
            "direction": direction,
            "pente": pente,
            "amplitude": amplitude,
            "efficience": float(np.clip(efficience, 0.0, 1.0)),
            "volatilite": volatilite,
            "pente_atr": float(pente_atr),
            "is_valid": bool(validity_mask(g).all() and not gap_before.loc[g.index].iloc[1:].any()),
        })

    result = pd.DataFrame(rows).set_index("segment_id")
    metrics = ["direction", "pente", "amplitude", "efficience", "volatilite", "pente_atr"]
    result.loc[~result["is_valid"], metrics] = np.nan
    return result
