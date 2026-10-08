"""HMM entraîné sur un préfixe fixe, puis filtré exclusivement vers l'avant."""
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from regime_lib.config.parameters import Parameter, ParameterSchema
from regime_lib.core.base import RegimeDetector
from regime_lib.core.registry import register_method
from regime_lib.core.utils import unaligned_mask
from regime_lib.utils.validation import validated_detector, validity_mask
from .hmm_gaussian import _construire_features


def filter_forward(model, x):
    """p(S_t | x_1,...,x_t), jamais de backward ni de Viterbi global."""
    means = model.means_
    covariance = model.covars_
    variances = np.diagonal(covariance, axis1=1, axis2=2) if covariance.ndim == 3 else covariance
    if not np.isfinite(variances).all() or (variances <= 0).any():
        raise ValueError("Variances HMM invalides.")
    with np.errstate(divide="ignore"):
        transition = np.log(model.transmat_)
        initial = np.log(model.startprob_)
    result = np.empty((len(x), len(means)))
    log_alpha = initial
    for i, observation in enumerate(x):
        emission = -0.5 * (np.log(2 * np.pi * variances) + (observation - means)**2 / variances).sum(axis=1)
        prior = initial if i == 0 else logsumexp(log_alpha[:, None] + transition, axis=0)
        joint = prior + emission
        normalizer = logsumexp(joint)
        if not np.isfinite(normalizer):
            raise ValueError("Filtrage HMM non fini.")
        log_alpha = joint - normalizer
        result[i] = np.exp(log_alpha)
    return result


@register_method
class HMMCausalDetector(RegimeDetector):
    name = "hmm_causal"
    availability = "bar_close"
    regime_dimension = "latent_state"
    regime_description = "États caractérisés sur train fixe ; posterior filtré à date hors train."
    confidence_kind = "filtered_state_probability"
    confidence_description = "Maximum p(état courant | observations passées) ; pas une probabilité de rendement."
    PARAM_SCHEMA = ParameterSchema(parameters={"n_states": Parameter(int, minimum=1),
        "n_iter": Parameter(int, minimum=1), "seed": Parameter(int),
        "train_size": Parameter(int, minimum=25)})

    def __init__(self, allow_lookahead=False, timeframe=None, profile=None,
                 n_states=None, n_iter=None, seed=None, train_size=None):
        super().__init__(allow_lookahead=allow_lookahead, timeframe=timeframe, profile=profile,
                         n_states=n_states, n_iter=n_iter, seed=seed, train_size=train_size)

    def _configure_regime_map(self):
        self.REGIME_MAP = {f"STATE_{i}": i for i in range(self.n_states)}
        self.REGIME_MAP["INCONNU"] = self.n_states

    @validated_detector
    def fit_predict(self, df):
        result = pd.DataFrame({"regime": "INCONNU", "confidence": 0.0,
                               "is_warmup": True, "hmm_diagnostics": ""}, index=df.index)
        if len(df) <= self.train_size:
            return result
        valid = validity_mask(df) & ~unaligned_mask(df.index, self.timeframe)
        if not valid.iloc[:self.train_size].all():
            raise ValueError("Le train HMM doit contenir uniquement des barres valides et closes.")
        features = _construire_features(df)
        train = features.loc[features.index < df.index[self.train_size]]
        if len(train) < max(2, self.n_states * 2):
            raise ValueError("Train HMM insuffisant après chauffe des features.")
        center = train.mean()
        scale = train.std().replace(0, 1.0)
        x = (features - center) / scale
        from hmmlearn.hmm import GaussianHMM
        model = GaussianHMM(n_components=self.n_states, covariance_type="diag",
                            n_iter=self.n_iter, random_state=self.seed)
        model.fit(x.loc[train.index].to_numpy())
        if not np.isfinite(model.transmat_).all() or not np.allclose(model.transmat_.sum(axis=1), 1):
            raise ValueError("Transitions HMM invalides ; ajuster le train ou n_states.")
        # Les numéros sont ordonnés par moyenne du rendement dans le train.
        order = np.argsort(model.means_[:, 0], kind="stable")
        posterior = filter_forward(model, x.to_numpy())[:, order]
        # Une observation invalide ne peut contaminer les publications suivantes.
        # Rejet explicite : pas d'imputation implicite de la séquence de Markov.
        if not valid.all():
            raise ValueError("Le filtrage HMM requiert des barres valides et closes.")
        future = features.index >= df.index[self.train_size]
        indices = features.index[future]
        result.loc[indices, "regime"] = [f"STATE_{i}" for i in posterior[future].argmax(axis=1)]
        result.loc[indices, "confidence"] = posterior[future].max(axis=1)
        result.loc[indices, "is_warmup"] = False
        history = list(model.monitor_.history)
        result.attrs["hmm_diagnostics"] = {
            "train_end": df.index[self.train_size-1].isoformat(),
            "train_rows": len(train), "iterations": model.monitor_.iter,
            "converged": bool(len(history) > 1 and 0 <= history[-1] - history[-2] < model.tol),
            "log_likelihood": history, "feature_names": list(train.columns),
            "normalization_mean": center.tolist(), "normalization_scale": scale.tolist(),
            "state_means": (model.means_[order] * scale.to_numpy() + center.to_numpy()).tolist(),
            "transition_matrix": model.transmat_[order][:, order].tolist(),
        }
        import json
        result.loc[indices, "hmm_diagnostics"] = json.dumps(result.attrs["hmm_diagnostics"], allow_nan=False)
        return result
