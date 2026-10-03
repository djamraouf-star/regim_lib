"""
Orchestrateur d'étude : assemble features, cibles, split, évaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from regime_lib.core.data_loader import load_parquet
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
        **split_kwargs,
    ) -> None:
        if split not in SCHEMAS_SPLIT:
            raise ValueError(
                f"Schéma de split inconnu : {split!r}. "
                f"Valides : {sorted(SCHEMAS_SPLIT)}."
            )

        self.regimes_path = Path(regimes_path)
        self.ohlcv_path = Path(ohlcv_path)

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
        )
        self.df_features = df_features

    def _charger_cibles(self) -> None:
        # Charger le OHLCV. Le pipeline détecte automatiquement le format
        # (barres ou ticks) via data_loader.
        df_ohlcv, _ = load_parquet(str(self.ohlcv_path))
        cibles = {}
        for nom in self.noms_cibles:
            cibles[nom] = calculer_cible(df_ohlcv, nom)
        self.df_cibles = pd.DataFrame(cibles)

    def _evaluer(self) -> None:
        if self.df_features is None or self.df_cibles is None:
            raise RuntimeError("Appeler run() d'abord.")

        # Aligner les index : intersection stricte
        common = self.df_features.index.intersection(self.df_cibles.index)
        if len(common) == 0:
            raise ValueError(
                "Aucun timestamp commun entre features et cibles. "
                "Vérifier que les timeframes correspondent."
            )

        features = self.df_features.loc[common]
        cibles = self.df_cibles.loc[common]

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
