"""Run from VS Code or a terminal: python run_dr5.py --help."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import torch

from src.dr5.core import (load_csv, seed_everything, temperature_scale, fit_temperature,
    train_evaluator, initial_ids, acquire, predict_samples, metrics)

ROOT = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=ROOT/'data/DR-5')
    p.add_argument('--output', type=Path, default=ROOT/'results/dr5')
    p.add_argument('--folds', type=int, nargs='+', default=list(range(5)))
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    p.add_argument('--mode', choices=['random', 'baseline', 'mc', 'reliability'], default='baseline')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--max-labels', type=int, default=8, help='Expert predictions per class; starts at 2')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--dropout', type=float, default=.1)
    p.add_argument('--weight-decay', type=float, default=0.)
    p.add_argument('--mc-passes', type=int, default=20)
    p.add_argument('--ai-calibration', choices=['none', 'temperature'], default='temperature')
    p.add_argument('--calibration-per-class', type=int, default=20,
                   help='Reserved AI calibration rows/class; no human queries unless --fusion-calibration')
    p.add_argument('--fusion-calibration', action='store_true', help='Costs extra human predictions: 5*calibration-per-class')
    p.add_argument('--validation-per-class', type=int, default=0,
                   help='Disjoint human-labeled validation rows/class, charged to query budget')
    p.add_argument('--stop-patience', type=int, default=0, help='0=fixed budget; otherwise stop on validation NLL plateau')
    p.add_argument('--stop-min-delta', type=float, default=.001)
    p.add_argument('--candidates', type=int, default=100)
    p.add_argument('--window-start', type=int, default=55)
    p.add_argument('--window-length', type=int, default=5)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--audit-only', action='store_true')
    return p


def split_ids(labels, calibration, validation, seed):
    rng = np.random.default_rng(seed)
    cal, val, pool = [], [], []
    for c in range(5):
        ids = rng.permutation(np.flatnonzero(labels.cpu().numpy() == c))
        if calibration + validation + 2 > len(ids):
            raise ValueError('Calibration/validation leave fewer than two pool rows/class')
        cal.extend(ids[:calibration]); val.extend(ids[calibration:calibration+validation])
        pool.extend(ids[calibration+validation:])
    return tuple(np.array(x, dtype=int) for x in (cal, val, pool))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(data_dir, folds):
    records = []
    for fold in folds:
        hashes = []
        for split in ['train', 'test']:
            path = data_dir / f'model_output_{fold}_{split}.csv'
            d = load_csv(path)
            # Feature equality detects exact duplicate feature rows only, not patient leakage.
            row_hashes = {hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in d.features}
            hashes.append(row_hashes)
            records.append({'fold': fold, 'split': split, 'rows': len(d.labels),
                'feature_dim': d.features.shape[1], 'class_counts': torch.bincount(d.labels, minlength=5).tolist(),
                'ai_accuracy': float(d.probs.argmax(1).eq(d.labels).float().mean()),
                'human_accuracy': float(d.human.eq(d.labels).float().mean()), 'sha256': sha256(path),
                'duplicate_feature_rows': len(d.labels)-len(row_hashes)})
        if hashes[0] & hashes[1]:
            raise ValueError(f'Exact train/test feature overlap in fold {fold}')
    return records


def run(args):
    if args.epochs < 1 or args.max_labels < 2 or args.mc_passes < 2:
        raise ValueError('epochs>=1, max-labels>=2, mc-passes>=2 required')
    if min(args.calibration_per_class, args.validation_per_class, args.stop_patience) < 0:
        raise ValueError('Split sizes and patience cannot be negative')
    if args.window_length < 1 or args.candidates < 1 or args.window_start < 0:
        raise ValueError('Invalid window/candidate settings')
    if args.stop_patience and not args.validation_per_class:
        raise ValueError('Stopping needs --validation-per-class; extra human cost is recorded')
    if (args.fusion_calibration or args.ai_calibration == 'temperature') and not args.calibration_per_class:
        raise ValueError('Calibration requires reserved calibration rows')
    if args.mode in ('mc', 'reliability') and args.dropout == 0:
        raise ValueError('MC dropout requires nonzero dropout during training')
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'manifest.json').exists():
        raise ValueError('Output already contains a run; choose a new --output to preserve it')
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        commit = 'archive (no git metadata)'
    manifest = {'config': config, 'base_commit': commit, 'python': platform.python_version(),
                'torch': torch.__version__, 'numpy': np.__version__,
                'source_hashes': {str(f.relative_to(ROOT)): sha256(f) for f in [ROOT/'run_dr5.py', ROOT/'src/dr5/core.py']},
                'data_hashes': {f.name: sha256(f) for fold in args.folds for f in
                    [args.data_dir/f'model_output_{fold}_train.csv', args.data_dir/f'model_output_{fold}_test.csv']},
                'test_usage': 'Final metrics only; no checkpoint, acquisition, temperature or stopping decisions',
                'limitation': 'No sample/patient IDs or backbone training provenance in supplied CSVs'}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    if args.audit_only:
        result = audit(args.data_dir, args.folds)
        (args.output/'audit.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2)); return
    rows = []
    passes = args.mc_passes if args.mode in ('mc', 'reliability') else 1
    for fold in args.folds:
        original = load_csv(args.data_dir/f'model_output_{fold}_train.csv').to(args.device)
        # Same partition across variants AND seeds; no test rows enter here.
        cal_ids, val_ids, pool_ids = split_ids(original.labels, args.calibration_per_class,
                                              args.validation_per_class, 10000 + fold)
        if min(int((original.labels[pool_ids] == c).sum()) for c in range(5)) < args.max_labels:
            raise ValueError('Requested expert budget exceeds available pool')
        ai_t = fit_temperature(original.probs[cal_ids], original.labels[cal_ids]) if args.ai_calibration == 'temperature' else 1.
        data = original.subset(np.arange(len(original.labels)))
        data.probs = temperature_scale(data.probs, ai_t)
        for seed in args.seeds:
            run_seed = fold * 1000 + seed
            rng = np.random.default_rng(run_seed)
            selected = initial_ids(data.labels, pool_ids, 2, rng)
            states, trace = [], []
            best_val, stale = float('inf'), 0
            for budget in range(2, args.max_labels+1):
                model, loss = train_evaluator(data, selected, run_seed, args.epochs,
                    args.lr, args.dropout, args.weight_decay)
                fusion_t = 1.
                if args.fusion_calibration:
                    cp = predict_samples(model, data.features[cal_ids], data.human[cal_ids], data.probs[cal_ids], passes, run_seed+100).mean(0)
                    fusion_t = fit_temperature(cp, data.labels[cal_ids])
                extra_queries = len(val_ids) + (len(cal_ids) if args.fusion_calibration else 0)
                record = {'fold': fold, 'seed': seed, 'mode': args.mode, 'labels_per_class': budget,
                          'active_human_queries': len(selected), 'extra_human_queries': extra_queries,
                          'total_human_queries': len(selected)+extra_queries, 'train_loss': loss,
                          'ai_temperature': ai_t, 'fusion_temperature': fusion_t}
                stop = False
                if len(val_ids):
                    vp = predict_samples(model, data.features[val_ids], data.human[val_ids], data.probs[val_ids], passes, run_seed+200).mean(0)
                    vm = metrics(temperature_scale(vp, fusion_t), data.labels[val_ids])
                    record.update({'validation_nll': vm['nll'], 'validation_accuracy': vm['accuracy'],
                                   'validation_failure_rate': 1-vm['accuracy']})
                    if vm['nll'] < best_val-args.stop_min_delta:
                        best_val, stale = vm['nll'], 0
                    else:
                        stale += 1
                    stop = bool(args.stop_patience and stale >= args.stop_patience)
                record['stopped'] = stop
                ckpt = {'state_dict': {k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'feature_dim': data.features.shape[1], 'dropout': args.dropout,
                        'selected_ids': list(selected), 'calibration_ids': cal_ids.tolist(),
                        'validation_ids': val_ids.tolist(), 'pool_ids': pool_ids.tolist(),
                        'record': record, 'config': config}
                torch.save(ckpt, args.output/f'fold{fold}_seed{seed}_budget{budget}.pt')
                states.append((model, dict(record), fusion_t))
                event = {'budget': budget, 'selected_ids': list(selected), 'record': dict(record)}
                if not stop and budget < args.max_labels:
                    new, signals = acquire(model, data.features, data.labels, data.probs, pool_ids,
                        selected, rng, args.mode, args.candidates, args.window_start, args.window_length,
                        args.mc_passes, run_seed+budget)
                    selected.extend(new)
                    event.update({'next_ids': new, 'acquisition_signals': signals})
                trace.append(event)
                if stop:
                    break
            # All active learning and stopping decisions are complete before test loading.
            test = load_csv(args.data_dir/f'model_output_{fold}_test.csv').to(args.device)
            test.probs = temperature_scale(test.probs, ai_t)
            machine = metrics(test.probs, test.labels)
            human_accuracy = float(test.human.eq(test.labels).float().mean())
            for model, record, fusion_t in states:
                tp = predict_samples(model, test.features, test.human, test.probs, passes, run_seed+300).mean(0)
                tp = temperature_scale(tp, fusion_t)
                record.update(metrics(tp, test.labels))
                record.update({'ai_accuracy': machine['accuracy'], 'human_accuracy': human_accuracy, 'ai_ece': machine['ece']})
                rows.append(record)
                np.savez_compressed(args.output/f"pred_fold{fold}_seed{seed}_budget{record['labels_per_class']}.npz",
                    probabilities=tp.cpu().numpy(), labels=test.labels.cpu().numpy())
            (args.output/f'trace_fold{fold}_seed{seed}.json').write_text(json.dumps(trace, indent=2))
            (args.output/'metrics.json').write_text(json.dumps(rows, indent=2))
            pd.DataFrame(rows).drop(columns=['confusion_matrix', 'per_class_recall']).to_csv(args.output/'metrics.csv', index=False)
            print(f"fold={fold} seed={seed} budget={rows[-1]['labels_per_class']} "
                  f"accuracy={rows[-1]['accuracy']:.3f} ECE={rows[-1]['ece']:.3f}", flush=True)
    frame = pd.DataFrame(rows)
    summary = frame.groupby('labels_per_class')[['accuracy','macro_f1','qwk','ece','nll','brier','total_human_queries']].agg(['mean','std'])
    summary.to_csv(args.output/'summary.csv')
    print(summary.to_string())


if __name__ == '__main__':
    run(parser().parse_args())
