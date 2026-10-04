"""
Orchestrateur d'étude : assemble features, cibles, split, évaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from regime_lib.core.data_loader import load_parquet, resample
from regime_lib.core.temporal import resolve_timeframe
from regime_lib.utils.validation import validate_time_index, validate_ohlcv
from regime_lib.study.evaluation import (
    hit_rate_directionnel,
    ic_spearman,
    stats_conditionnelles,
    par_modalite_tests,
)
from regime_lib.study.features import extraire_features
from regime_lib.study.split import (
    split_holdout,
    split_in_sample,
    split_walk_forward,
)
from regime_lib.study.targets import calculer_cible


SCHEMAS_SPLIT: dict[str, Callable] = {
    "in_sample": split_in_sample,
    "holdout": split_holdout,
    "walk_forward": split_walk_forward,
}


class Study:
    """
    Étude d'une ou plusieurs features par rapport à une ou plusieurs
    cibles.

    Attributes
    ----------
    df_features : pd.DataFrame
        Features extraites du parquet de régimes.
    df_cibles : pd.DataFrame
        Cibles calculées sur le OHLCV.
    resultats : pd.DataFrame | None
        Résultats de l'évaluation après `run()`.
    resultats_tests : dict | None
        Batterie de tests statistiques (Welch, Mann-Whitney,
        z de proportions) par paire (feature, cible), sur le fold 0.
        Clés : tuples `(nom_feature, nom_cible)`.
    """

    def __init__(
        self,
        regimes_path: str | Path,
        ohlcv_path: str | Path,
        features: list[str] | None = None,
        targets: list[str] | None = None,
        methodes: list[str] | None = None,
        methodes_causales_only: bool = True,
        split: str = "in_sample",
        tests_modalite: bool = True,
        ohlcv_data: pd.DataFrame | None = None,
        asset: str | None = None,
        timeframe: str | None = None,
        configurations: dict[str, str] | None = None,
        ohlcv_asset: str | None = None,
        ohlcv_timeframe: str | None = None,
        expected_index: pd.DatetimeIndex | None = None,
        **split_kwargs,
    ) -> None:
        if split not in SCHEMAS_SPLIT:
            raise ValueError(
                f"Schéma de split inconnu : {split!r}. "
                f"Valides : {sorted(SCHEMAS_SPLIT)}."
            )

        self.regimes_path = Path(regimes_path)
        self.ohlcv_path = Path(ohlcv_path)
        self.ohlcv_data = ohlcv_data
        self.asset = asset
        self.timeframe = timeframe
        self.configurations = configurations
        self.ohlcv_asset = ohlcv_asset
        self.ohlcv_timeframe = ohlcv_timeframe
        self.expected_index = expected_index
        self.couverture: pd.DataFrame | None = None
        self.support_commun: pd.DataFrame | None = None

        if features is None:
            features = ["regime"]
        if targets is None:
            targets = ["ret_fwd_5"]

        self.noms_features = features
        self.noms_cibles = targets
        self.methodes = methodes
        self.methodes_causales_only = methodes_causales_only
        self.schema_split = split
        self.tests_modalite = tests_modalite
        self.split_kwargs = split_kwargs

        self.df_features: pd.DataFrame | None = None
        self.df_cibles: pd.DataFrame | None = None
        self.resultats: pd.DataFrame | None = None
        self.resultats_tests: dict | None = None
        self._detail_conditionnel: dict | None = None

    # ------------------------------------------------------------------
    def run(self) -> "Study":
        """
        Exécute l'étude complète : chargement, calcul, évaluation.
        """
        self._charger_features()
        self._charger_cibles()
        self._evaluer()
        return self

    # ------------------------------------------------------------------
    def _charger_features(self) -> None:
        df_regimes = pd.read_parquet(self.regimes_path)
        df_features = extraire_features(
            df_regimes,
            methodes=self.methodes,
            features=self.noms_features,
            methodes_causales_only=self.methodes_causales_only,
            asset=self.asset, timeframe=self.timeframe,
            configurations=self.configurations,
        )
        self.df_features = df_features

    def _charger_cibles(self) -> None:
        if self.ohlcv_data is None:
            # Le pipeline détecte automatiquement le format (barres ou
            # ticks) via data_loader.
            df_ohlcv, _ = load_parquet(str(self.ohlcv_path), asset=self.ohlcv_asset,
                                       timeframe=self.ohlcv_timeframe)
        else:
            df_ohlcv = self.ohlcv_data.copy()
        validate_ohlcv(df_ohlcv)
        identity = self.df_features.attrs["series_identity"] if self.df_features is not None else {}
        source_asset = self.ohlcv_asset or df_ohlcv.attrs.get("asset")
        if self.ohlcv_asset and df_ohlcv.attrs.get("asset") not in (None, self.ohlcv_asset):
            raise ValueError("Actif OHLCV explicite incompatible avec les métadonnées.")
        if identity and source_asset != identity["asset"]:
            raise ValueError("Actif OHLCV absent ou incompatible avec les régimes.")
        source_tf = resolve_timeframe(df_ohlcv, self.ohlcv_timeframe)
        target_tf = identity.get("timeframe", source_tf)
        for field in ("source", "price_side", "adjustment", "calendar"):
            if field in identity and identity[field] != df_ohlcv.attrs.get(field):
                raise ValueError(f"Identité OHLCV incompatible : {field}.")
        if source_tf != target_tf:
            if self.expected_index is not None:
                raise ValueError("Avec expected_index, fournir les OHLCV déjà au timeframe de l'étude.")
            df_ohlcv = resample(df_ohlcv, target_tf, source_timeframe=source_tf)
        if self.df_features is not None and not self.df_features.index.isin(df_ohlcv.index).all():
            raise ValueError("Des timestamps des régimes sont absents des prix : alignement incompatible.")
        cibles = {}
        for nom in self.noms_cibles:
            cibles[nom] = calculer_cible(df_ohlcv, nom, timeframe=target_tf,
                                        expected_index=self.expected_index)
        self.df_cibles = pd.DataFrame(cibles)

    def _evaluer(self) -> None:
        if self.df_features is None or self.df_cibles is None:
            raise RuntimeError("Appeler run() d'abord.")

        validate_time_index(self.df_features.index)
        validate_time_index(self.df_cibles.index)
        common = self.df_features.index
        if not common.isin(self.df_cibles.index).all():
            raise ValueError("Timestamps de features absents des cibles : alignement incompatible.")

        features = self.df_features.loc[common]
        cibles = self.df_cibles.loc[common]
        # Support commun par cible, conservé sur la grille originale pour le split.
        feature_valid = features.notna().all(axis=1)
        self.support_commun = cibles.notna().mul(feature_valid, axis=0).astype(bool)
        coverage_rows = []
        for feat in features:
            for target in cibles:
                own = features[feat].notna() & cibles[target].notna()
                shared = self.support_commun[target]
                coverage_rows.append({
                    "feature": feat, "cible": target, "n_total": len(common),
                    "n_propre": int(own.sum()), "n_commun": int(shared.sum()),
                    "n_feature_invalide": int(features[feat].isna().sum()),
                    "n_cible_invalide": int(cibles[target].isna().sum()),
                    "couverture_propre": float(own.mean()) if len(own) else np.nan,
                    "couverture_commune": float(shared.mean()) if len(shared) else np.nan,
                    **self.df_features.attrs.get("exclusions", {}).get(feat.split("__", 1)[0], {}),
                })
        self.couverture = pd.DataFrame(coverage_rows)
        cibles = cibles.where(self.support_commun)

        # Découpage
        schema = SCHEMAS_SPLIT[self.schema_split]
        paires = schema(common, **self.split_kwargs)

        lignes = []
        detail = {}

        for fold_id, (idx_train, idx_test) in enumerate(paires):
            for col_feat in features.columns:
                for col_cible in cibles.columns:
                    f_train = features.loc[idx_train, col_feat]
                    c_train = cibles.loc[idx_train, col_cible]
                    f_test = features.loc[idx_test, col_feat]
                    c_test = cibles.loc[idx_test, col_cible]

                    ic_train = ic_spearman(f_train, c_train)
                    ic_test = ic_spearman(f_test, c_test)
                    hit_train = hit_rate_directionnel(f_train, c_train)
                    hit_test = hit_rate_directionnel(f_test, c_test)

                    lignes.append({
                        "fold": fold_id,
                        "feature": col_feat,
                        "cible": col_cible,
                        "n_train": ic_train["n"],
                        "n_test": ic_test["n"],
                        "ic_train": ic_train["ic"],
                        "ic_test": ic_test["ic"],
                        "pvalue_train": ic_train["pvalue"],
                        "pvalue_test": ic_test["pvalue"],
                        "hit_train": hit_train["hit_rate"],
                        "hit_test": hit_test["hit_rate"],
                    })

                    # Détail conditionnel (test uniquement)
                    cle = (fold_id, col_feat, col_cible)
                    detail[cle] = stats_conditionnelles(f_test, c_test)

        self.resultats = pd.DataFrame(lignes)
        self._detail_conditionnel = detail

        # Batterie de tests par modalité (fold 0 uniquement)
        if self.tests_modalite and len(paires) > 0:
            idx_test = paires[0][1]
            tests_dict: dict = {}
            for col_feat in features.columns:
                for col_cible in cibles.columns:
                    f_test = features.loc[idx_test, col_feat]
                    c_test = cibles.loc[idx_test, col_cible]
                    tests_dict[(col_feat, col_cible)] = par_modalite_tests(
                        f_test, c_test, nom_cible=col_cible,
                    )
            self.resultats_tests = tests_dict
        else:
            self.resultats_tests = None

    # ------------------------------------------------------------------
    def save(self, output_dir: str | Path) -> None:
        """Exporte les résultats en parquet et le rapport en markdown."""
        if self.resultats is None:
            raise RuntimeError("Appeler run() avant save().")

        from regime_lib.study.report import generer_rapport

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = output_dir / "study_results.parquet"
        self.resultats.to_parquet(parquet_path, index=False)
        if self.couverture is not None:
            self.couverture.to_parquet(output_dir / "study_coverage.parquet", index=False)
        if self.support_commun is not None:
            self.support_commun.to_parquet(output_dir / "study_common_support.parquet")

        # Export séparé des tests par modalité (un parquet long)
        if self.resultats_tests:
            tests_longs = []
            for (feat, cible), tab in self.resultats_tests.items():
                if tab.empty:
                    continue
                t = tab.reset_index().copy()
                t["feature"] = feat
                t["cible"] = cible
                t["modalite"] = t["modalite"].astype(str)
                tests_longs.append(t)
            if tests_longs:
                tests_path = output_dir / "study_tests_modalite.parquet"
                pd.concat(tests_longs, ignore_index=True).to_parquet(
                    tests_path, index=False
                )
                print(f"Tests     : {tests_path}")

        rapport_path = output_dir / "study_report.md"
        rapport = generer_rapport(
            self.resultats,
            detail_conditionnel=self._detail_conditionnel,
            tests_modalite=self.resultats_tests,
            schema_split=self.schema_split,
            noms_cibles=self.noms_cibles,
        )
        if self.couverture is not None:
            from regime_lib.study.report import _to_markdown
            rapport += "\n\n## Couverture et support commun\n\n"
            rapport += "Les estimations utilisent le même support valide par cible. "
            rapport += "Les motifs d'exclusion peuvent se chevaucher.\n\n"
            rapport += _to_markdown(self.couverture)
            if self.support_commun is not None and not self.support_commun.any().any():
                rapport += "\n\nAucune estimation : support valide commun vide.\n"
        rapport_path.write_text(rapport, encoding="utf-8")

        print(f"Résultats : {parquet_path}")
        print(f"Rapport   : {rapport_path}")

    def detail_conditionnel(
        self,
        feature: str,
        cible: str,
        fold: int = 0,
    ) -> pd.DataFrame:
        """Renvoie le tableau conditionnel pour une paire donnée."""
        if self._detail_conditionnel is None:
            raise RuntimeError("Appeler run() d'abord.")
        cle = (fold, feature, cible)
        if cle not in self._detail_conditionnel:
            raise KeyError(f"Aucun résultat pour {cle}.")
        return self._detail_conditionnel[cle]

    def tests_modalite_pour(
        self,
        feature: str,
        cible: str,
    ) -> pd.DataFrame:
        """Renvoie les tests statistiques par modalité pour une paire."""
        if self.resultats_tests is None:
            raise RuntimeError(
                "Tests de modalité non calculés. "
                "Vérifier tests_modalite=True et appeler run()."
            )
        cle = (feature, cible)
        if cle not in self.resultats_tests:
            raise KeyError(f"Aucun résultat pour {cle}.")
        return self.resultats_tests[cle]
