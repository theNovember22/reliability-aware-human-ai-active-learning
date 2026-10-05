"""ActiveHAI components. Pool scoring deliberately has no human-label argument."""
from contextlib import contextmanager
from dataclasses import dataclass
import random

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
import torch
from torch import nn
from torch.nn import functional as F


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@dataclass
class FoldData:
    probs: torch.Tensor
    labels: torch.Tensor
    human: torch.Tensor
    features: torch.Tensor

    def subset(self, ids):
        return FoldData(*(a[ids] for a in (self.probs, self.labels, self.human, self.features)))

    def to(self, device):
        return FoldData(*(a.to(device) for a in (self.probs, self.labels, self.human, self.features)))


def load_csv(path):
    df = pd.read_csv(path)
    required = ['Logits', 'True Label', 'Human Label', 'Feat']
    if list(df.columns) != required or df.empty:
        raise ValueError(f'{path}: expected nonempty columns {required}')
    # These files say Logits but contain normalized probabilities.
    p = np.stack([np.asarray([float(v) for v in s.split(',')]) for s in df.Logits])
    x = np.stack([np.asarray([float(v) for v in s.split(',')]) for s in df.Feat])
    y, h = df['True Label'].to_numpy(), df['Human Label'].to_numpy()
    if not np.isfinite(x).all() or not np.isfinite(p).all():
        raise ValueError('Nonfinite probabilities/features')
    if p.shape[1] != 5 or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError('Expected five normalized AI probabilities, not logits')
    for labels in (y, h):
        if not np.isin(labels, np.arange(5)).all():
            raise ValueError('Labels must be integers in [0, 4]')
    return FoldData(torch.tensor(p, dtype=torch.float32), torch.tensor(y, dtype=torch.long),
                    torch.tensor(h, dtype=torch.long), torch.tensor(x, dtype=torch.float32))


class Evaluator(nn.Module):
    def __init__(self, feature_dim=512, dropout=0.1):
        super().__init__()
        self.human_embedding = nn.Embedding(5, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, 5)

    def forward(self, human, features):
        return self.classifier(self.dropout(self.human_embedding(human) + features))


def fuse_log_probs(ai_probs, evaluator_logits):
    # log softmax(log p_AI + evaluator logits) == normalized product.
    return F.log_softmax(ai_probs.clamp_min(1e-12).log() + evaluator_logits, dim=-1)


def temperature_scale(probs, temperature):
    if temperature <= 0:
        raise ValueError('Temperature must be positive')
    return F.softmax(probs.clamp_min(1e-12).log() / temperature, dim=-1)


def fit_temperature(probs, labels):
    """Scalar NLL temperature; caller must supply calibration data, never test."""
    p = probs.detach().double().cpu()
    y = labels.detach().cpu()
    objective = lambda log_t: F.nll_loss(F.log_softmax(p.clamp_min(1e-12).log() / np.exp(log_t), -1), y).item()
    fit = minimize_scalar(objective, bounds=(-3., 3.), method='bounded')
    if not fit.success:
        raise RuntimeError('Temperature optimization failed')
    # Never accept worse calibration NLL than the identity on the fitting set.
    return float(np.exp(fit.x)) if objective(fit.x) < objective(0.) else 1.


@contextmanager
def dropout_inference(model):
    states = {m: m.training for m in model.modules()}
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.train()
    try:
        yield
    finally:
        for m, state in states.items():
            m.training = state


@torch.no_grad()
def predict_samples(model, features, human, ai_probs, passes=1, seed=0):
    if passes < 1:
        raise ValueError('passes must be >= 1')
    devices = [features.device.index or 0] if features.is_cuda else []
    # MC inference must not alter training RNG or future initialization.
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if passes > 1:
            with dropout_inference(model):
                return torch.stack([fuse_log_probs(ai_probs, model(human, features)).exp()
                                    for _ in range(passes)])
        states = {m: m.training for m in model.modules()}
        model.eval()
        try:
            return fuse_log_probs(ai_probs, model(human, features)).exp().unsqueeze(0)
        finally:
            for m, state in states.items():
                m.training = state


def entropy(p):
    return -(p * p.clamp_min(1e-12).log()).sum(-1)


def reliability_signals(samples, labels):
    """Dropout failure frequency, MI and limit-state margin statistics.

    For unqueried data, samples are conditional on pseudo h=y. Failure is
    therefore a COUNTERFACTUAL proxy, not an estimated real doctor error rate.
    """
    mean = samples.mean(0)
    mi = (entropy(mean) - entropy(samples).mean(0)).clamp_min(0)
    target = samples.gather(2, labels[None, :, None].expand(samples.shape[0], -1, 1)).squeeze(-1)
    others = samples.clone()
    others.scatter_(2, labels[None, :, None].expand(samples.shape[0], -1, 1), -1.)
    margins = target - others.max(-1).values
    return {'mi': mi, 'failure_frequency': (margins <= 0).float().mean(0),
            'margin_mean': margins.mean(0), 'margin_std': margins.std(0, unbiased=False)}


def train_evaluator(data, ids, seed, epochs=100, lr=3e-4, dropout=0.1, weight_decay=0.):
    seed_everything(seed)
    model = Evaluator(data.features.shape[1], dropout).to(data.features.device)
    selected = data.subset(ids)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for _ in range(epochs):
        model.train()
        log_p = fuse_log_probs(selected.probs, model(selected.human, selected.features))
        loss = F.nll_loss(log_p, selected.labels)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    model.eval()
    return model, float(loss.detach())


def initial_ids(labels, pool_ids, count, rng):
    y = labels.cpu().numpy()
    return [int(i) for c in range(5) for i in rng.choice(pool_ids[y[pool_ids] == c], count, replace=False)]


@torch.no_grad()
def acquire(model, features, labels, ai_probs, pool_ids, selected, rng,
            mode='baseline', candidates=100, window_start=55, window_length=5,
            passes=20, seed=0):
    """One sample/class. No unqueried human labels can enter this function."""
    additions, diagnostics = [], []
    y = labels.cpu().numpy()
    for c in range(5):
        available = np.array([i for i in pool_ids if y[i] == c and i not in selected], dtype=int)
        if not len(available):
            raise ValueError(f'Pool exhausted for class {c}')
        ids = rng.choice(available, min(candidates, len(available)), replace=False)
        if mode == 'random':
            additions.append(int(rng.choice(ids)))
            continue
        # Algorithm 1 in ActiveHAI uses pseudo expert h=y before querying.
        pseudo = labels[ids]
        model.eval()
        evaluator_p = model(pseudo, features[ids]).softmax(-1)
        score = (evaluator_p.gather(1, pseudo[:, None]).squeeze(1) - .5).abs().cpu().numpy()
        order = np.argsort(-score, kind='stable')
        start = min(window_start, max(0, len(ids) - window_length))
        window = order[start:start + window_length]
        if mode == 'reliability':
            samples = predict_samples(model, features[ids], pseudo, ai_probs[ids], passes, seed + c)
            signal = reliability_signals(samples, pseudo)
            # Novel heuristic: prefer epistemic uncertainty near a stochastic
            # decision boundary, but stay inside the original MWAC window.
            mi = signal['mi'].cpu().numpy()[window]
            pf = signal['failure_frequency'].cpu().numpy()[window]
            mi_rank = np.argsort(np.argsort(mi, kind='stable'), kind='stable') / max(1, len(mi)-1)
            utility = .5 * mi_rank + .5 * (4 * pf * (1-pf))
            best = np.flatnonzero(np.isclose(utility, utility.max()))
            local = window[int(rng.choice(best))]
            diagnostics.append({'class': c, 'counterfactual_failure_frequency': float(signal['failure_frequency'][local]),
                                'mi': float(signal['mi'][local]), 'candidate_count': len(ids)})
        else:
            local = int(rng.choice(window))
        additions.append(int(ids[local]))
    return additions, diagnostics


def metrics(probs, labels, bins=15):
    p = probs.detach().cpu().double()
    y = labels.detach().cpu()
    confidence, pred = p.max(1)
    correct = pred.eq(y)
    # bucket 0 includes confidence zero; bucket bins-1 includes exactly one.
    bucket = torch.clamp((confidence * bins).long(), max=bins-1)
    ece = sum(float((bucket == b).double().mean() *
                    (correct[bucket == b].double().mean() - confidence[bucket == b].mean()).abs())
              for b in range(bins) if (bucket == b).any())
    cm = confusion_matrix(y, pred, labels=np.arange(5))
    recall = np.divide(cm.diagonal(), cm.sum(1), out=np.zeros(5), where=cm.sum(1) != 0)
    return {'accuracy': accuracy_score(y, pred), 'macro_f1': f1_score(y, pred, average='macro', labels=list(range(5)), zero_division=0),
            'balanced_accuracy': float(recall.mean()), 'qwk': cohen_kappa_score(y, pred, weights='quadratic', labels=list(range(5))),
            'nll': F.nll_loss(p.clamp_min(1e-12).log(), y).item(), 'ece': ece,
            'brier': float(((p-F.one_hot(y, 5))**2).sum(1).mean()),
            'per_class_recall': recall.tolist(), 'confusion_matrix': cm.tolist()}
