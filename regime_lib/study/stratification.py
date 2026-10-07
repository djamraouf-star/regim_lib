"""Comparaisons conditionnées par un contexte, sur la grille temporelle originale."""

import pandas as pd

from regime_lib.study.evaluation import par_modalite_tests, stats_conditionnelles
from regime_lib.study.inference import bootstrap_modalities
from regime_lib.study.targets import parser_cible
from regime_lib.study.validation import prepare_regimes


STRATIFICATION_COLUMNS = {
    "ctx_session", "ctx_event_type", "ctx_day_of_week",
    "ctx_is_rollover", "ctx_is_holiday", "ctx_in_event",
}
KEY_COLUMNS = ["fold", "feature", "cible", "contexte", "strate"]


def extract_strata(regimes, features, column, configurations):
    """Contrôle le contexte brut des seules séries/configurations retenues."""
    if column not in regimes:
        raise ValueError(f"Contexte de stratification absent : {column}.")
    identity = features.attrs["series_identity"]
    selected = regimes[
        regimes.asset.eq(identity["asset"]) & regimes.timeframe.eq(identity["timeframe"])
    ]
    pieces = []
    for method in sorted({name.split("__", 1)[0] for name in features}):
        rows, _ = prepare_regimes(
            selected[selected.method.eq(method)], (configurations or {}).get(method),
        )
        values = rows[column]
        present = values.dropna()
        if column in {"ctx_session", "ctx_event_type"}:
            valid = present.map(lambda value: isinstance(value, str) and bool(value.strip()))
        elif column == "ctx_day_of_week":
            valid = present.isin(range(7)) & present.map(lambda value: not isinstance(value, bool))
        else:
            from pandas.api.types import is_bool
            valid = present.map(is_bool)
        if not valid.all():
            raise ValueError(f"Valeurs invalides pour le contexte {column}.")
        pieces.append(values)
    combined = pd.concat(pieces)
    # Une absence déclarée et une valeur au même timestamp sont contradictoires.
    if combined.groupby(level=0).nunique(dropna=False).gt(1).any():
        raise ValueError(f"Contexte incohérent entre méthodes : {column}.")
    unique = combined[~combined.index.duplicated()]
    return unique.reindex(features.index).to_frame(column)


def evaluate_strata(feature, target, context, *, key, context_name, tests,
                    inference, timeframe, expected_index):
    """Masque les observations externes sans comprimer le temps ni recalculer les cibles."""
    missing = context.isna()
    outside = context.eq("HORS_SESSION").fillna(False) if context_name == "ctx_session" else missing & False
    usable_context = ~missing & ~outside
    base = dict(zip(KEY_COLUMNS[:3], key), contexte=context_name)
    coverage = [{**base, "strate": None, "scope": "fold",
                 "n_total": len(context), "n_contexte_absent": int(missing.sum()),
                 "n_hors_session": int(outside.sum()),
                 "n_contexte_valide": int(usable_context.sum()),
                 "n_commun": int((usable_context & feature.notna() & target.notna()).sum())}]
    tables = {}
    for value in sorted(context[usable_context].unique(), key=str):
        mask = usable_context & context.eq(value).fillna(False)
        f, y = feature.where(mask), target.where(mask)
        stratum_key = (*key, context_name, str(value))
        coverage.append({**base, "strate": str(value), "scope": "strate",
                         "n_total": int(mask.sum()), "n_contexte_absent": 0,
                         "n_hors_session": 0, "n_contexte_valide": int(mask.sum()),
                         "n_commun": int((f.notna() & y.notna()).sum())})
        if tests:
            table = par_modalite_tests(f, y, nom_cible=key[2], timeframe=timeframe,
                                       expected_index=expected_index)
            if inference is not None:
                table = bootstrap_modalities(
                    f, y, table, inference, key=stratum_key,
                    horizon=parser_cible(key[2])[1], timeframe=timeframe,
                    expected_index=expected_index,
                )
        else:
            table = stats_conditionnelles(f, y, timeframe=timeframe, expected_index=expected_index)
        tables[stratum_key] = table
    return tables, coverage


def flatten_strata(tables):
    records = []
    for key, table in tables.items():
        if table.empty:
            continue
        frame = table.rename_axis("modalite").reset_index()
        frame["modalite"] = frame["modalite"].astype(str)
        records.append(frame.assign(**dict(zip(KEY_COLUMNS, key))))
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=[*KEY_COLUMNS, "modalite"])
