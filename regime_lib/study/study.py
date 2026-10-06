"""
Orchestrateur d'étude : assemble features, cibles, split, évaluation.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
from tempfile import TemporaryDirectory
from regime_lib.study.provenance import manifest, file_hash, frame_info

from regime_lib.study.inference import InferenceConfig, bootstrap_modalities, correct_family
from typing import Callable

import numpy as np
import pandas as pd

from regime_lib.core.data_loader import load_parquet
from regime_lib.core.temporal import expected_grid
from regime_lib.study.validation import prepare_prices
from regime_lib.utils.validation import validate_time_index
from regime_lib.study.evaluation import (
    hit_rate_directionnel,
    feature_type,
    FEATURE_TYPES,
    ic_spearman,
    stats_conditionnelles,
    par_modalite_tests,
    stabilite_modalites,
)
from regime_lib.study.features import extraire_features
from regime_lib.study.split import (
    split_holdout,
    split_in_sample,
    split_walk_forward,
)
from regime_lib.study.targets import calculer_cible, parser_cible


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
        z de proportions) par paire (feature, cible), sur chaque fold.
        Clés : tuples `(fold, nom_feature, nom_cible)`.
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
        feature_types: dict[str, str] | None = None,
        inference: InferenceConfig | None = None,
        **split_kwargs,
    ) -> None:
        if split not in SCHEMAS_SPLIT:
            raise ValueError(
                f"Schéma de split inconnu : {split!r}. "
                f"Valides : {sorted(SCHEMAS_SPLIT)}."
            )

        if inference is not None and not isinstance(inference, InferenceConfig):
            raise TypeError("inference doit être une InferenceConfig.")
        if inference is not None and not tests_modalite:
            raise ValueError("inference requiert tests_modalite=True.")
        self.inference = inference
        self.feature_types = dict(feature_types or {})
        if any(value not in FEATURE_TYPES for value in self.feature_types.values()):
            raise ValueError("Type de feature inconnu dans feature_types.")
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

        for target in targets:
            _, horizon = parser_cible(target)
            if inference is not None and inference.block_size < horizon:
                raise ValueError("block_size doit être >= à l’horizon de chaque cible.")

        self.noms_features = features
        self.noms_cibles = targets
        self.methodes = methodes
        self.methodes_causales_only = methodes_causales_only
        self.schema_split = split
        self.tests_modalite = tests_modalite
        self.split_kwargs = split_kwargs

        self.df_features: pd.DataFrame | None = None
        self.df_cibles: pd.DataFrame | None = None
        self.df_fin_cibles: pd.DataFrame | None = None
        self.resultats: pd.DataFrame | None = None
        self.resultats_tests: dict | None = None
        self.stabilite_modalites: pd.DataFrame | None = None
        self._detail_conditionnel: dict | None = None
        self.metadata: dict | None = None
        self._source_metadata: dict = {}

    # ------------------------------------------------------------------
    def run(self) -> "Study":
        """
        Exécute l'étude complète : chargement, calcul, évaluation.
        """
        self.resultats = None
        self.metadata = None
        self.resultats_tests = None
        self._detail_conditionnel = None
        self._source_metadata = {}
        self._charger_features()
        self._charger_cibles()
        self._evaluer()
        self.metadata = manifest(self, origin="run")
        return self

    # ------------------------------------------------------------------
    def _charger_features(self) -> None:
        source_hash = file_hash(self.regimes_path)
        df_regimes = pd.read_parquet(self.regimes_path)
        if file_hash(self.regimes_path) != source_hash:
            raise ValueError("Le parquet de régimes a changé pendant le chargement.")
        self._source_metadata["regimes"] = {"path": str(self.regimes_path), "sha256": source_hash}
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
        identity = self.df_features.attrs["series_identity"] if self.df_features is not None else {}
        df_ohlcv, target_tf = prepare_prices(
            df_ohlcv, identity, asset=self.ohlcv_asset, timeframe=self.ohlcv_timeframe,
            required_index=self.df_features.index if self.df_features is not None else None,
            expected_index=self.expected_index,
        )
        self._source_metadata["prices_prepared"] = frame_info(df_ohlcv)
        self._source_metadata["prices_origin"] = "ohlcv_data" if self.ohlcv_data is not None else str(self.ohlcv_path)
        grid = expected_grid(df_ohlcv.index, target_tf, self.expected_index)
        ends = {}
        cibles = {}
        for nom in self.noms_cibles:
            _, horizon = parser_cible(nom)
            ends[nom] = pd.Series(grid, index=grid).shift(-horizon).reindex(df_ohlcv.index)
            cibles[nom] = calculer_cible(df_ohlcv, nom, timeframe=target_tf,
                                        expected_index=self.expected_index)
        self.df_cibles = pd.DataFrame(cibles)
        self.df_fin_cibles = pd.DataFrame(ends)

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
        self._fold_metadata = [
            {"fold": fold, "train_start": train[0], "train_end_before_purge": train[-1],
             "test_start": test[0], "test_end": test[-1],
             "n_train_before_purge": len(train), "n_test_grid": len(test)}
            for fold, (train, test) in enumerate(paires)
        ]

        unknown = set(self.feature_types) - set(features.columns) - {
            name.split("__", 1)[-1] for name in features.columns
        }
        if unknown:
            raise ValueError(f"Features déclarées absentes : {sorted(unknown)}.")
        kinds = {
            name: feature_type(features[name], self.feature_types.get(
                name, self.feature_types.get(name.split("__", 1)[-1])))
            for name in features
        }
        lignes = []
        detail = {}
        tests_dict = {}
        timeframe = features.attrs.get("series_identity", {}).get("timeframe")

        for fold_id, (idx_train, idx_test) in enumerate(paires):
            for col_feat in features.columns:
                for col_cible in cibles.columns:
                    train_index = idx_train
                    if self.schema_split != "in_sample":
                        if self.df_fin_cibles is None:
                            raise ValueError("Dates de fin des cibles absentes : appeler run().")
                        ends = self.df_fin_cibles[col_cible].reindex(idx_train)
                        train_index = idx_train[ends.notna() & ends.lt(idx_test[0])]
                        if train_index.empty:
                            raise ValueError(f"Train vide après purge : fold {fold_id}, cible {col_cible}.")
                    f_train = features.loc[train_index, col_feat]
                    c_train = cibles.loc[train_index, col_cible]
                    f_test = features.loc[idx_test, col_feat]
                    c_test = cibles.loc[idx_test, col_cible]

                    ic_train = ic_spearman(f_train, c_train, type_feature=kinds[col_feat])
                    ic_test = ic_spearman(f_test, c_test, type_feature=kinds[col_feat])
                    hit_train = hit_rate_directionnel(f_train, c_train, type_feature=kinds[col_feat], nom_cible=col_cible)
                    hit_test = hit_rate_directionnel(f_test, c_test, type_feature=kinds[col_feat], nom_cible=col_cible)

                    lignes.append({
                        "fold": fold_id,
                        "feature": col_feat,
                        "cible": col_cible,
                        "feature_type": kinds[col_feat],
                        "n_purge_train": len(idx_train) - len(train_index),
                        "ic_status_train": ic_train["status"],
                        "ic_status_test": ic_test["status"],
                        "hit_status_train": hit_train["status"],
                        "hit_status_test": hit_test["status"],
                        "n_hit_train": hit_train["n"],
                        "n_hit_test": hit_test["n"],
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
                    detail[cle] = stats_conditionnelles(
                        f_test, c_test, timeframe=timeframe, expected_index=self.expected_index,
                    )
                    if self.tests_modalite:
                        tests_dict[cle] = par_modalite_tests(
                            f_test, c_test, nom_cible=col_cible,
                            timeframe=timeframe, expected_index=self.expected_index,
                        )

                        if self.inference is not None:
                            tests_dict[cle] = bootstrap_modalities(
                                f_test, c_test, tests_dict[cle], self.inference, key=cle,
                                horizon=parser_cible(col_cible)[1], timeframe=timeframe,
                                expected_index=self.expected_index,
                            )
        if self.inference is not None:
            correct_family(tests_dict, self.inference.confidence_level)

        self.resultats = pd.DataFrame(lignes)
        self._detail_conditionnel = detail

        self.resultats_tests = tests_dict if self.tests_modalite else None
        self.stabilite_modalites = stabilite_modalites(tests_dict, len(paires))

    # ------------------------------------------------------------------
    def save(self, output_dir: str | Path) -> None:
        """Prépare tous les fichiers avant publication ; manifeste écrit en dernier."""
        if self.resultats is None:
            raise RuntimeError("Appeler run() avant save().")
        output_dir = Path(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        current = manifest(self, origin="run" if self.metadata else "manual_evaluation")
        if self.metadata is not None and current != self.metadata:
            raise ValueError("L'étude a changé depuis run() : relancer run() avant save().")
        metadata = self.metadata or current
        managed = {
            "study_results.parquet", "study_target_ends.parquet", "study_coverage.parquet",
            "study_common_support.parquet", "study_tests_modalite.parquet",
            "study_conditionnel.parquet", "study_stability.parquet", "study_report.md",
            "study_inference.json", "study_metadata.json",
        }
        with TemporaryDirectory(dir=output_dir.parent, prefix=".study-export-") as temp:
            staging = Path(temp)
            self._write_export(staging)
            report = staging / "study_report.md"
            with report.open("a", encoding="utf-8") as stream:
                stream.write("\n\n## Traçabilité\n\n")
                stream.write(f"Mode : {metadata['analysis_mode']}.\n\n")
                stream.write(metadata['validation_claim'] + "\n\n")
                stream.write("Configuration, versions, périodes, exclusions et empreintes : `study_metadata.json`.\n")
            metadata = {**metadata, "artifacts": {p.name: file_hash(p) for p in sorted(staging.iterdir())}}
            (staging / "study_metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            output_dir.mkdir(parents=True, exist_ok=True)
            for name in managed:
                if (output_dir / name).resolve() in {self.regimes_path.resolve(), self.ohlcv_path.resolve()}:
                    raise ValueError("Un export écraserait un fichier source.")
            for name in sorted(managed - {"study_metadata.json"}):
                source = staging / name
                target = output_dir / name
                if source.exists():
                    os.replace(source, target)
                elif target.is_file():
                    target.unlink()
            os.replace(staging / "study_metadata.json", output_dir / "study_metadata.json")
        print(f"Étude exportée : {output_dir}")

    def _write_export(self, output_dir: str | Path) -> None:
        """Exporte les résultats en parquet et le rapport en markdown."""
        if self.resultats is None:
            raise RuntimeError("Appeler run() avant save().")

        from regime_lib.study.report import generer_rapport

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.inference is not None:
            (output_dir / "study_inference.json").write_text(
                json.dumps(self.inference.to_dict(), indent=2), encoding="utf-8")

        parquet_path = output_dir / "study_results.parquet"
        self.resultats.to_parquet(parquet_path, index=False)
        if self.df_fin_cibles is not None:
            self.df_fin_cibles.to_parquet(output_dir / "study_target_ends.parquet")
        if self.couverture is not None:
            self.couverture.to_parquet(output_dir / "study_coverage.parquet", index=False)
        if self.support_commun is not None:
            self.support_commun.to_parquet(output_dir / "study_common_support.parquet")

        # Export séparé des tests par modalité (un parquet long)
        if self.resultats_tests:
            tests_longs = []
            for key, tab in self.resultats_tests.items():
                fold, feat, cible = key if len(key) == 3 else (0, *key)
                if tab.empty:
                    continue
                t = tab.reset_index().copy()
                t["fold"] = fold
                t["feature"] = feat
                t["cible"] = cible
                t["modalite"] = t["modalite"].astype(str)
                tests_longs.append(t)
            if tests_longs:
                tests_path = output_dir / "study_tests_modalite.parquet"
                pd.concat(tests_longs, ignore_index=True).to_parquet(
                    tests_path, index=False
                )


        details = []
        for (fold, feat, cible), tab in (self._detail_conditionnel or {}).items():
            if tab.empty:
                continue
            table = tab.rename_axis("modalite").reset_index()
            table["modalite"] = table["modalite"].astype(str)
            table = table.assign(fold=fold, feature=feat, cible=cible)
            details.append(table)
        if details:
            pd.concat(details, ignore_index=True).to_parquet(
                output_dir / "study_conditionnel.parquet", index=False)
        if self.stabilite_modalites is not None:
            self.stabilite_modalites.to_parquet(output_dir / "study_stability.parquet", index=False)

        rapport_path = output_dir / "study_report.md"
        rapport = generer_rapport(
            self.resultats,
            detail_conditionnel=self._detail_conditionnel,
            tests_modalite=self.resultats_tests,
            schema_split=self.schema_split,
            noms_cibles=self.noms_cibles,
            stabilite=self.stabilite_modalites,
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
        fold: int = 0,
    ) -> pd.DataFrame:
        """Renvoie les tests statistiques par modalité pour une paire."""
        if self.resultats_tests is None:
            raise RuntimeError(
                "Tests de modalité non calculés. "
                "Vérifier tests_modalite=True et appeler run()."
            )
        cle = (fold, feature, cible)
        if cle not in self.resultats_tests:
            raise KeyError(f"Aucun résultat pour {cle}.")
        return self.resultats_tests[cle]
