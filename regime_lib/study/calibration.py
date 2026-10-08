"""Calibration d'un score vers un événement binaire explicite, sur train seul.

La probabilité obtenue concerne l'événement choisi, jamais la « justesse »
universelle d'un régime. L'évaluation doit utiliser une période indépendante.
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from regime_lib.utils.validation import validate_alignment, validate_time_index


class ConfidenceCalibrator:
    def __init__(self, event: str, *, increasing: bool = True):
        if not isinstance(event, str) or not event.strip():
            raise ValueError("Décrire explicitement l'événement binaire à calibrer.")
        if type(increasing) is not bool:
            raise ValueError("Le sens de calibration doit être fixé avant le test.")
        self.event = event
        self.increasing = increasing
        self.model = None

    @staticmethod
    def _scores(scores):
        validate_time_index(scores.index)
        if scores.empty or not np.isfinite(scores.to_numpy(dtype=float)).all() or not scores.between(0, 1).all():
            raise ValueError("Scores non vides, finis et dans [0,1] requis.")

    @staticmethod
    def _outcomes(scores, outcomes):
        validate_alignment(scores.index, outcomes.index)
        if outcomes.isna().any() or not outcomes.isin([0, 1]).all():
            raise ValueError("L'événement doit être binaire et observé sur chaque ligne.")

    def fit(self, scores: pd.Series, outcomes: pd.Series, *, target_end: pd.Series):
        self._scores(scores)
        self._outcomes(scores, outcomes)
        validate_alignment(scores.index, target_end.index)
        if not isinstance(target_end.dtype, pd.DatetimeTZDtype) or str(target_end.dt.tz) != "UTC":
            raise ValueError("Les fins de cible doivent être des dates UTC.")
        if target_end.isna().any() or (target_end < scores.index).any():
            raise ValueError("Dates de fin de cible invalides.")
        if outcomes.nunique() != 2:
            raise ValueError("Le train doit observer les deux issues binaires.")
        model = IsotonicRegression(increasing=self.increasing, out_of_bounds="clip")
        model.fit(scores.to_numpy(), outcomes.to_numpy())
        self.model = model
        self.available_after = max(scores.index.max(), target_end.max())
        self.base_rate = float(outcomes.mean())
        self.metadata = {"event": self.event, "increasing": self.increasing,
                         "train_start": scores.index.min().isoformat(),
                         "available_after": self.available_after.isoformat(),
                         "n_train": len(scores), "base_rate": self.base_rate,
                         "score_knots": model.X_thresholds_.tolist(),
                         "probability_knots": model.y_thresholds_.tolist()}
        return self

    def predict(self, scores: pd.Series):
        if self.model is None:
            raise RuntimeError("Appeler fit sur la période de calibration.")
        self._scores(scores)
        if scores.index.min() <= self.available_after:
            raise ValueError("La prédiction doit suivre toutes les fins de cibles du train (purge).")
        result = pd.Series(self.model.predict(scores.to_numpy()), index=scores.index,
                           name="calibrated_probability")
        result.attrs["calibration"] = self.metadata.copy()
        return result

    def evaluate(self, scores: pd.Series, outcomes: pd.Series):
        self._outcomes(scores, outcomes)
        probability = self.predict(scores)
        bins = np.minimum((probability.to_numpy() * 10).astype(int), 9)
        table = pd.DataFrame({"bin": bins, "probability": probability.to_numpy(),
                              "outcome": outcomes.to_numpy()}).groupby("bin").agg(
                                  n=("outcome", "size"), probability=("probability", "mean"),
                                  frequency=("outcome", "mean"))
        brier = float(np.mean((probability - outcomes)**2))
        baseline = float(np.mean((self.base_rate - outcomes)**2))
        return {"event": self.event, "n": len(scores), "brier": brier,
                "baseline_brier": baseline, "brier_skill": 1 - brier / baseline if baseline else float("nan"),
                "ece": float(((table.probability-table.frequency).abs()*table.n).sum()/len(scores)),
                "reliability": table.reset_index().to_dict(orient="records")}
