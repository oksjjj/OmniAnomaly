# -*- coding: utf-8 -*-
"""
Re-evaluate POT / best-F1 from saved score pickles (no re-training / re-scoring).

Example:
    python eval_from_scores.py --dataset SMAP
"""
import argparse
import json
import os
import pickle

import numpy as np

from main import print_metrics_summary
from omni_anomaly.eval_methods import pot_eval, bf_search
from omni_anomaly.train_logger import experiment_logging
from omni_anomaly.utils import default_pot_level, get_data


def parse_args():
    p = argparse.ArgumentParser(description='Re-evaluate from saved scores')
    p.add_argument('--dataset', type=str, required=True)
    p.add_argument('--result_dir', type=str, default='result')
    p.add_argument('--train_score', type=str, default='train_score.pkl')
    p.add_argument('--test_score', type=str, default='test_score.pkl')
    p.add_argument('--level', type=float, default=None,
                   help='POT low quantile (default: auto by dataset)')
    p.add_argument('--pot_q', type=float, default=1e-4,
                   help='POT risk q (paper: 1e-4)')
    p.add_argument('--bf_search_min', type=float, default=-400.)
    p.add_argument('--bf_search_max', type=float, default=400.)
    p.add_argument('--bf_search_step_size', type=float, default=1.)
    p.add_argument('--get_score_on_dim', action='store_true')
    p.add_argument('--log_dir', type=str, default='log')
    return p.parse_args()


def run_eval(args, log):
    level = args.level if args.level is not None else default_pot_level(args.dataset)
    train_path = os.path.join(args.result_dir, args.train_score)
    test_path = os.path.join(args.result_dir, args.test_score)

    with open(train_path, 'rb') as f:
        train_score = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_score = pickle.load(f)

    (_, _), (_, y_test) = get_data(args.dataset, do_preprocess=True)
    if y_test is None:
        raise RuntimeError(f'No test labels for dataset={args.dataset}')

    y_test = y_test[-len(test_score):]
    if args.get_score_on_dim:
        test_score = np.sum(test_score, axis=-1)
        train_score = np.sum(train_score, axis=-1)

    print(f'train_score: {train_score.shape}, test_score: {test_score.shape}')
    print(f'POT q={args.pot_q}, level={level}'
          f'{" (auto)" if args.level is None else " (manual)"}')
    log.info(
        'Re-eval dataset=%s q=%s level=%s train_score=%s test_score=%s',
        args.dataset, args.pot_q, level, train_path, test_path,
    )

    t, th = bf_search(
        test_score, y_test,
        start=args.bf_search_min,
        end=args.bf_search_max,
        step_num=int(abs(args.bf_search_max - args.bf_search_min) /
                     args.bf_search_step_size),
        display_freq=50,
    )
    pot_result = pot_eval(
        train_score, test_score, y_test,
        q=args.pot_q, level=level,
    )

    metrics = {
        'best-f1': t[0],
        'precision': t[1],
        'recall': t[2],
        'TP': t[3],
        'TN': t[4],
        'FP': t[5],
        'FN': t[6],
        'latency': t[-1],
        'threshold': th,
        'dataset': args.dataset,
        'pot_q': args.pot_q,
        'level': level,
        'level_source': 'manual' if args.level is not None else 'auto',
        'train_score': train_path,
        'test_score': test_path,
    }
    metrics.update(pot_result)

    out_path = os.path.join(args.result_dir, 'metrics_reeval.json')
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f'Metrics saved to {out_path}')
    print_metrics_summary(metrics)
    log.info('Re-eval finished. metrics=%s', out_path)
    return metrics


def main():
    args = parse_args()
    with experiment_logging(args.log_dir, args.dataset, mode='eval') as (log_path, log):
        print(f'Log file: {log_path}')
        log.info('Score re-evaluation started')
        run_eval(args, log)


if __name__ == '__main__':
    main()
