"""Small-sample fusion and acquisition; no API accepts unqueried doctor labels.

Ground-truth pool grades are known in ActiveHAI. Doctor responses are costly.
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import softmax
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def nll(probabilities, labels):
    return float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1)).mean())


def confusion_posterior(labels, human, mass=5.):
    """Dirichlet P(doctor response | true grade), pooled-accuracy shrinkage.

    Estimate the shared diagonal prior from these queried pairs only. This is
    empirical Bayes, not a claim of calibrated clinical reliability.
    """
    labels, human = np.asarray(labels, dtype=int), np.asarray(human, dtype=int)
    if labels.shape != human.shape or labels.ndim != 1 or mass <= 0:
        raise ValueError('Expected paired one-dimensional labels and positive mass')
    if not np.isin(labels, range(5)).all() or not np.isin(human, range(5)).all():
        raise ValueError('Labels must be in [0, 4]')
    accuracy = (np.sum(labels == human) + 1.) / (len(labels) + 2.)
    prior = np.full((5, 5), (1. - accuracy) / 4.)
    np.fill_diagonal(prior, accuracy)
    counts = mass * prior
    np.add.at(counts, (labels, human), 1.)
    return counts


def fuse(machine, human, likelihood, strength=1.):
    """P(y|x,h) proportional to P_machine(y|x) P(h|y)**strength."""
    log_machine = np.log(np.clip(machine, 1e-12, 1.))
    log_human = np.log(np.clip(likelihood[:, np.asarray(human, dtype=int)].T, 1e-12, 1.))
    return softmax(log_machine + strength * log_human, axis=-1)


@dataclass
class BudgetFusion:
    posterior: np.ndarray
    strength: float
    mass: float
    loo_nll: float

    @property
    def likelihood(self):
        return self.posterior / self.posterior.sum(1, keepdims=True)

    def predict(self, machine, human):
        return fuse(machine, human, self.likelihood, self.strength)

    def to_dict(self):
        return dict(posterior=self.posterior.tolist(), strength=self.strength,
                    mass=self.mass, loo_nll=self.loo_nll)

    @classmethod
    def from_dict(cls, value):
        return cls(np.asarray(value['posterior']), value['strength'], value['mass'], value['loo_nll'])


def fit_fusion(machine_oof, queried_labels, queried_human):
    """Select 2 scalars by LOO NLL, reusing the SAME queried labels.

    The held-out response is excluded from both counts and the shared prior.
    LOO is a development criterion on adaptively sampled data, not an unbiased
    performance estimate. Machine OOF predictions exclude that row's target.
    """
    y, h = np.asarray(queried_labels), np.asarray(queried_human)
    if len(y) < 2 or len(machine_oof) != len(y):
        raise ValueError('At least two queried pairs with aligned probabilities required')
    best = None
    for mass in (1., 5., 20.):
        held_out_likelihood = []
        for i in range(len(y)):
            keep = np.arange(len(y)) != i
            counts = confusion_posterior(y[keep], h[keep], mass)
            likelihood = counts / counts.sum(1, keepdims=True)
            held_out_likelihood.append(likelihood[:, h[i]])
        log_human = np.log(np.asarray(held_out_likelihood))
        for strength in (0., .5, 1., 1.5, 2.):
            p = softmax(np.log(np.clip(machine_oof, 1e-12, 1.)) + strength * log_human, axis=1)
            score = nll(p, y)
            # Deterministic tie-breaking prefers stronger shrinkage, then beta near 1.
            key = (score, -mass, abs(strength - 1.), strength)
            if best is None or key < best[0]:
                best = (key, mass, strength)
    _, mass, strength = best
    return BudgetFusion(confusion_posterior(y, h, mass), strength, mass, best[0][0])


@dataclass
class MachineAdapter:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    blend: float
    regularization_c: float

    def predict(self, features, frozen):
        probe = softmax(((features - self.mean) / self.scale) @ self.coef.T + self.intercept, axis=1)
        return (1. - self.blend) * frozen + self.blend * probe

    def save(self, path):
        np.savez_compressed(path, **vars(self))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(*(z[k].copy() for k in ('mean', 'scale', 'coef', 'intercept')),
                       float(z['blend']), float(z['regularization_c']))


def fit_machine(features, labels, frozen, pool_ids, calibration_ids, seed):
    """Known-y feature adapter; ZERO doctor responses.

    Select C and blend on the fixed known-y calibration split. Produce OOF
    predictions on the acquisition pool for subsequent fusion selection.
    Normalization is refitted inside every training split.
    """
    x, y = np.asarray(features), np.asarray(labels)
    train, cal = np.asarray(pool_ids), np.asarray(calibration_ids)
    if len(cal) == 0 or np.intersect1d(train, cal).size:
        raise ValueError('Need a nonempty disjoint known-label calibration split')
    scaler = StandardScaler().fit(x[train])
    xt = scaler.transform(x[train])
    xc = scaler.transform(x[cal])
    candidates = []
    with threadpool_limits(limits=1):
        for c in (.001, .01, .1, 1.):
            model = LogisticRegression(C=c, max_iter=2000, solver='lbfgs').fit(xt, y[train])
            probe = model.predict_proba(xc)
            for blend in (0., .25, .5, .75, 1.):
                score = nll((1-blend)*frozen[cal] + blend*probe, y[cal])
                candidates.append((score, blend, c, model))
        score, blend, c, model = min(candidates, key=lambda a: a[:3])
        adapter = MachineAdapter(scaler.mean_, scaler.scale_, model.coef_, model.intercept_, blend, c)
        oof = frozen.copy()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fit_ids, hold_ids in cv.split(x[train], y[train]):
            fit_ids, hold_ids = train[fit_ids], train[hold_ids]
            scale = StandardScaler().fit(x[fit_ids])
            m = LogisticRegression(C=c, max_iter=2000, solver='lbfgs').fit(scale.transform(x[fit_ids]), y[fit_ids])
            oof[hold_ids] = (1-blend)*frozen[hold_ids] + blend*m.predict_proba(scale.transform(x[hold_ids]))
    details = {'regularization_c': c, 'blend': blend, 'calibration_nll': score,
               'known_target_training_rows': len(train), 'known_target_calibration_rows': len(cal),
               'extra_human_queries': 0}
    return adapter, oof, details


def normalized_features(features, pool_ids):
    """Training-pool normalization only, for geometric diversity."""
    z = StandardScaler().fit(features[pool_ids]).transform(features)
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)


def expected_information(model, machine, labels, draws=64, seed=0):
    """Posterior fusion MI marginalized over every hypothetical doctor label.

    Uses estimated P(h|known y) instead of pretending the doctor always says y.
    No actual unqueried response is an input. MI is a model proxy, not clinical
    error probability. Fixed beta=1 for query scoring avoids beta=0 deadlock.
    """
    rng = np.random.default_rng(seed)
    samples = np.stack([rng.dirichlet(row, size=draws) for row in model.posterior], axis=1)
    log_p = np.log(np.clip(machine, 1e-12, 1.))
    utility = np.zeros(len(machine))
    weights = model.likelihood[np.asarray(labels)]
    for h in range(5):
        predictions = softmax(log_p[None] + np.log(np.clip(samples[:, None, :, h], 1e-12, 1)), axis=-1)
        mean = predictions.mean(0)
        entropy_mean = -(mean*np.log(np.clip(mean, 1e-12, 1))).sum(-1)
        mean_entropy = -(predictions*np.log(np.clip(predictions, 1e-12, 1))).sum(-1).mean(0)
        utility += weights[:, h] * np.maximum(entropy_mean - mean_entropy, 0.)
    return utility


def acquire_budget(model, features_normalized, labels, machine_oof, pool_ids,
                   selected, rng, mode='diverse', exploration=.25, seed=0):
    """One query/grade: expected information + diversity + random exploration."""
    if mode not in ('random', 'diverse') or not 0 <= exploration <= 1:
        raise ValueError('Invalid acquisition mode or exploration fraction')
    additions, diagnostics = [], []
    selected = np.asarray(selected, dtype=int)
    for c in range(5):
        ids = np.asarray([i for i in pool_ids if labels[i] == c and i not in selected], dtype=int)
        if not len(ids):
            raise ValueError(f'Pool exhausted for grade {c}')
        if mode == 'random' or rng.random() < exploration:
            chosen = int(rng.choice(ids))
            diagnostics.append({'grade': c, 'reason': 'random', 'id': chosen})
        else:
            info = expected_information(model, machine_oof[ids], labels[ids], seed=seed+c)
            same_class = selected[labels[selected] == c]
            if not len(same_class):
                raise ValueError('Initial queries must cover every grade')
            distance = np.maximum(1 - features_normalized[ids] @ features_normalized[same_class].T, 0).min(1)
            # Rank averaging has no units and does not privilege larger MI scales.
            utility = .5*rankdata(info, method='average') + .5*rankdata(distance, method='average')
            local = int(rng.choice(np.flatnonzero(np.isclose(utility, utility.max()))))
            chosen = int(ids[local])
            diagnostics.append({'grade': c, 'reason': 'information_and_diversity', 'id': chosen,
                                'expected_information': float(info[local]), 'distance': float(distance[local])})
        additions.append(chosen)
    return additions, diagnostics
