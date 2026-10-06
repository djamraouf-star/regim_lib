"""Contrats d'entrée communs aux études et rapports descriptifs."""

import pandas as pd

from regime_lib.core.data_loader import resample
from regime_lib.core.temporal import resolve_timeframe
from regime_lib.utils.validation import validate_ohlcv, validate_time_index, validity_mask

SERIES_FIELDS = ("asset", "timeframe", "source", "price_side", "adjustment", "calendar")


def series_identity(data: pd.DataFrame) -> dict[str, str]:
    """Exige une identité unique ; les champs optionnels sont contrôlés si présents."""
    if data.empty:
        raise ValueError("Aucune observation retenue.")
    for field in ("asset", "timeframe"):
        if field not in data:
            raise ValueError(f"Identité de série absente : {field}.")
    identity = {}
    for field in SERIES_FIELDS:
        if field in data:
            if data[field].isna().any() or data[field].nunique() != 1:
                raise ValueError(f"Identité ambiguë ({field}) : sélectionner une série unique.")
            identity[field] = data[field].iloc[0]
    return identity


def prepare_regimes(
    data: pd.DataFrame, configuration: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Sélectionne une configuration et contrôle une série méthode/timeframe."""
    data = data.copy()
    if configuration is not None:
        if "params_hash" not in data:
            raise ValueError("params_hash absent pour la sélection de configuration.")
        data = data[data.params_hash.eq(configuration)]
    if data.empty:
        raise ValueError("Méthode/configuration absente.")
    if "method" not in data or data.method.isna().any() or data.method.nunique() != 1:
        raise ValueError("Méthode ambiguë : sélectionner une méthode unique.")
    if "params_hash" in data and (data.params_hash.isna().any() or data.params_hash.nunique() != 1):
        raise ValueError("Configurations ambiguës : sélectionner params_hash.")
    identity = series_identity(data)
    if "timestamp" not in data:
        raise ValueError("Colonne timestamp absente.")
    data = data.set_index("timestamp").sort_index()
    validate_time_index(data.index)
    valid = validity_mask(data)
    coverage = {
        "n_observations": len(data), "n_valide": int(valid.sum()),
        "n_exclu": int((~valid).sum()),
        "n_partiel": int(data.is_partial.fillna(True).sum()) if "is_partial" in data else 0,
        "n_chauffe": int(data.is_warmup.fillna(True).sum()) if "is_warmup" in data else 0,
        "n_invalide": int((~data.is_valid.fillna(False).astype(bool)).sum()) if "is_valid" in data else 0,
        "n_inconnu": int(data.regime.eq("INCONNU").sum()) if "regime" in data else 0,
        "n_absent": int(data.regime.isna().sum()) if "regime" in data else 0,
    }
    data.attrs.update(series_identity=identity, exclusions=coverage)
    return data, valid


def prepare_prices(
    data: pd.DataFrame, identity: dict[str, str], *,
    asset: str | None = None, timeframe: str | None = None,
    required_index: pd.DatetimeIndex | None = None,
    expected_index: pd.DatetimeIndex | None = None,
) -> tuple[pd.DataFrame, str]:
    """Valide l'identité des prix, leur fréquence et la présence des timestamps."""
    data = data.copy()
    validate_ohlcv(data)
    if asset and data.attrs.get("asset") not in (None, asset):
        raise ValueError("Actif OHLCV explicite incompatible avec les métadonnées.")
    if identity and (asset or data.attrs.get("asset")) != identity["asset"]:
        raise ValueError("Actif OHLCV absent ou incompatible avec les régimes.")
    source_tf = resolve_timeframe(data, timeframe)
    target_tf = identity.get("timeframe", source_tf)
    for field in ("source", "price_side", "adjustment", "calendar"):
        if field in identity and identity[field] != data.attrs.get(field):
            raise ValueError(f"Identité OHLCV incompatible : {field}.")
    if source_tf != target_tf:
        if expected_index is not None:
            raise ValueError("Avec expected_index, fournir les OHLCV déjà au timeframe de l'étude.")
        data = resample(data, target_tf, source_timeframe=source_tf)
    if required_index is not None:
        validate_time_index(required_index)
        if not required_index.isin(data.index).all():
            raise ValueError("Des timestamps des régimes sont absents des prix : alignement incompatible.")
    return data, target_tf
