"""Budget-preserving DR5 improvements: train everything before evaluating test sets."""
import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch

from run_dr5 import ROOT, sha256, split_ids
from src.dr5.core import load_csv, initial_ids, metrics
from src.dr5.budget import (BudgetFusion, MachineAdapter, acquire_budget,
                           fit_fusion, fit_machine, normalized_features)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=ROOT/'data/DR-5')
    p.add_argument('--output', type=Path, default=ROOT/'results/budget_v2')
    p.add_argument('--folds', type=int, nargs='+', default=list(range(5)))
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    p.add_argument('--machine', choices=['frozen', 'probe', 'both'], default='both')
    p.add_argument('--acquisition', choices=['random', 'diverse', 'both'], default='both')
    p.add_argument('--max-labels', type=int, default=8, help='Queries per grade: 8 means 40 total')
    p.add_argument('--exploration', type=float, default=.25)
    p.add_argument('--phase', choices=['train', 'evaluate', 'all'], default='all')
    return p


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def train(args):
    if args.max_labels < 2 or not 0 <= args.exploration <= 1:
        raise ValueError('Need max-labels >=2 and exploration in [0,1]')
    if not args.folds or len(set(args.folds)) != len(args.folds) or not set(args.folds) <= set(range(5)):
        raise ValueError('Choose distinct folds from 0 through 4')
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Choose distinct seeds')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output is not empty; choose a new path to preserve runs')
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    source_paths = ['run_budget_dr5.py', 'run_dr5.py', 'src/dr5/budget.py', 'src/dr5/core.py']
    manifest = {'config': config, 'python': platform.python_version(),
                'numpy': np.__version__, 'torch': torch.__version__, 'sklearn': sklearn.__version__,
                'source_hashes': {p: sha256(ROOT/p) for p in source_paths},
                'train_hashes': {str(f): sha256(args.data_dir/f'model_output_{f}_train.csv') for f in args.folds},
                'protocol': 'All folds, seeds and variants finish training before any test evaluation.',
                'query_scope': 'Development queries only. Every evaluated test case needs its doctor prediction.',
                'caveat': 'Known pool target labels are assumed free; these existing test folds are exploratory.',
                'complete': False}
    write_json(args.output/'manifest.json', manifest)
    machines = ['frozen', 'probe'] if args.machine == 'both' else [args.machine]
    acquisitions = ['random', 'diverse'] if args.acquisition == 'both' else [args.acquisition]
    run_files = []
    for fold in args.folds:
        data = load_csv(args.data_dir/f'model_output_{fold}_train.csv')
        x, y, ai = data.features.numpy(), data.labels.numpy(), data.probs.numpy().astype(float)
        cal, _, pool = split_ids(data.labels, 20, 0, 10000+fold)
        if min(np.sum(y[pool] == c) for c in range(5)) < args.max_labels:
            raise ValueError('Budget exceeds pool capacity')
        z = normalized_features(x, pool)
        oof_by_machine, details_by_machine = {'frozen': ai}, {'frozen': {'extra_human_queries': 0}}
        if 'probe' in machines:
            adapter, oof, details = fit_machine(x, y, ai, pool, cal, 10000+fold)
            adapter.save(args.output/f'machine_fold{fold}.npz')
            oof_by_machine['probe'], details_by_machine['probe'] = oof, details
        for machine in machines:
            for acquisition in acquisitions:
                for seed in args.seeds:
                    run_seed = fold*1000+seed
                    rng = np.random.default_rng(run_seed)
                    selected = initial_ids(data.labels, pool, 2, rng)
                    rounds = []
                    for budget in range(2, args.max_labels+1):
                        # The only training-time access to doctor responses is
                        # explicitly indexed by the unique query ledger.
                        queried_human = data.human[selected].numpy()
                        fusion = fit_fusion(oof_by_machine[machine][selected], y[selected], queried_human)
                        record = {'labels_per_class': budget, 'selected_ids': list(selected),
                                  'queried_human': queried_human.tolist(),
                                  'active_human_queries': len(selected), 'extra_human_queries': 0,
                                  'total_human_queries': len(selected), 'fusion': fusion.to_dict()}
                        if budget < args.max_labels:
                            new, diagnostics = acquire_budget(fusion, z, y, oof_by_machine[machine],
                                pool, selected, rng, acquisition, args.exploration, run_seed+budget)
                            record.update(next_ids=new, acquisition_signals=diagnostics)
                            selected.extend(new)
                        rounds.append(record)
                    name = f'{machine}_{acquisition}_fold{fold}_seed{seed}.json'
                    write_json(args.output/name, {'fold': fold, 'seed': seed, 'machine': machine,
                        'acquisition': acquisition, 'machine_details': details_by_machine[machine],
                        'calibration_ids': cal.tolist(), 'pool_ids': pool.tolist(), 'rounds': rounds})
                    run_files.append(name)
                    print(f'TRAINED {name}: {len(selected)} unique expert queries', flush=True)
    # Seal training artifacts before test data can be loaded, including adapters.
    artifacts = run_files + [p.name for p in sorted(args.output.glob('machine_fold*.npz'))]
    manifest.update(complete=True, run_files=run_files,
                    artifact_hashes={name: sha256(args.output/name) for name in artifacts})
    write_json(args.output/'manifest.json', manifest)


def evaluate(output):
    manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    if not manifest['complete']:
        raise ValueError('Training must finish for ALL planned variants before evaluation')
    if (output/'evaluation.json').exists():
        raise ValueError('Evaluation already exists; preserve it instead of overwriting')
    for name, digest in manifest['artifact_hashes'].items():
        if sha256(output/name) != digest:
            raise ValueError(f'Training artifact changed after sealing: {name}')
    for name, digest in manifest['source_hashes'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f'Source changed after training: {name}')
    data_dir = Path(manifest['config']['data_dir'])
    rows, tests, adapters, test_hashes = [], {}, {}, {}
    for name in manifest['run_files']:
        run = json.loads((output/name).read_text(encoding='utf-8'))
        fold, machine = run['fold'], run['machine']
        if fold not in tests:
            path = data_dir/f'model_output_{fold}_test.csv'
            tests[fold] = load_csv(path)
            test_hashes[str(fold)] = sha256(path)
        test = tests[fold]
        ai = test.probs.numpy().astype(float)
        p = ai
        if machine == 'probe':
            if fold not in adapters:
                adapters[fold] = MachineAdapter.load(output/f'machine_fold{fold}.npz')
            p = adapters[fold].predict(test.features.numpy(), ai)
        for state in run['rounds']:
            model = BudgetFusion.from_dict(state['fusion'])
            pred = model.predict(p, test.human.numpy())
            record = {k: run[k] for k in ('fold', 'seed', 'machine', 'acquisition')}
            record.update({k: state[k] for k in ('labels_per_class', 'active_human_queries',
                                               'extra_human_queries', 'total_human_queries')})
            record.update(metrics(torch.tensor(pred), test.labels))
            record.update(machine_accuracy=float(np.mean(p.argmax(1) == test.labels.numpy())),
                          frozen_ai_accuracy=float(np.mean(ai.argmax(1) == test.labels.numpy())),
                          human_accuracy=float(np.mean(test.human.numpy() == test.labels.numpy())),
                          evaluation_human_predictions=len(test.labels),
                          fusion_strength=model.strength, prior_mass=model.mass)
            rows.append(record)
            np.savez_compressed(output/f"pred_{Path(name).stem}_budget{state['labels_per_class']}.npz",
                                probabilities=pred, labels=test.labels.numpy())
    frame = pd.DataFrame(rows)
    frame.drop(columns=['confusion_matrix', 'per_class_recall']).to_csv(output/'metrics.csv', index=False)
    write_json(output/'metrics.json', rows)
    columns = ['accuracy', 'macro_f1', 'qwk', 'ece', 'nll', 'total_human_queries']
    summary = frame.groupby(['machine', 'acquisition', 'labels_per_class'])[columns].agg(['mean', 'std'])
    summary.to_csv(output/'summary.csv')
    endpoint = frame[frame.labels_per_class == manifest['config']['max_labels']]
    fold_summary = endpoint.groupby(['machine', 'acquisition', 'fold']).accuracy.mean().unstack('acquisition')
    fold_summary.to_csv(output/'paired_folds.csv')
    lines = ['# DR5 budget-preserving experiment', '',
             'All variants were trained and sealed before test evaluation. These previously observed folds remain exploratory.', '',
             '| Machine | Acquisition | Accuracy mean +/- SD (%) | Macro-F1 | ECE | Development queries |',
             '|---|---|---:|---:|---:|---:|']
    for (machine, acq), group in endpoint.groupby(['machine', 'acquisition']):
        lines.append(f'| {machine} | {acq} | {100*group.accuracy.mean():.2f} +/- {100*group.accuracy.std():.2f} | '
                     f'{group.macro_f1.mean():.4f} | {group.ece.mean():.4f} | {int(group.total_human_queries.iloc[0])} |')
    lines += ['', 'SD is across fold/seed runs, not a confidence interval. Seeds share test folds.',
              'Each test case also requires an expert response (200 per fold). The 40-query limit is for model development.',
              'The probe uses existing known training diagnoses, never additional expert predictions.',
              'No patient IDs or backbone provenance are available; patient/backbone leakage cannot be ruled out.', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(output/'evaluation.json', {'test_hashes': test_hashes, 'runs': len(manifest['run_files']),
               'training_manifest_sha256': sha256(output/'manifest.json'),
               'selection': 'No test-based parameter or checkpoint selection; report every predeclared variant.'})
    print('\n'.join(lines), flush=True)


def main():
    args = parser().parse_args()
    torch.set_num_threads(1)
    if args.phase in ('train', 'all'):
        train(args)
    if args.phase in ('evaluate', 'all'):
        evaluate(args.output)


if __name__ == '__main__':
    main()
