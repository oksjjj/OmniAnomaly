# -*- coding: utf-8 -*-
"""
OmniAnomaly entry point — PyTorch port of the official TensorFlow main.py
https://github.com/NetManAIOps/OmniAnomaly
"""
import argparse
import json
import logging
import os
import pickle
import time
import warnings
from pprint import pprint

import numpy as np
import torch

from omni_anomaly.checkpoint import load_checkpoint
from omni_anomaly.device import get_device
from omni_anomaly.eval_methods import pot_eval, bf_search
from omni_anomaly.model import OmniAnomaly
from omni_anomaly.prediction import Predictor
from omni_anomaly.train_logger import experiment_logging
from omni_anomaly.training import Trainer
from omni_anomaly.utils import get_data_dim, get_data, save_z


class ExpConfig:
    """Experiment configuration (defaults match the official ExpConfig)."""

    # dataset configuration
    dataset = "machine-1-1"
    x_dim = 38

    # model architecture configuration
    use_connected_z_q = True
    use_connected_z_p = True

    # model parameters
    z_dim = 3
    rnn_cell = 'GRU'  # 'GRU', 'LSTM' or 'Basic'
    rnn_num_hidden = 500
    window_length = 100
    dense_dim = 500
    posterior_flow_type = 'nf'  # 'nf' or None
    nf_layers = 20
    max_epoch = 10
    train_start = 0
    max_train_size = None
    batch_size = 50
    l2_reg = 0.0001  # kept for API parity; unused in the official graph
    initial_lr = 0.001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 40
    lr_anneal_step_freq = None
    std_epsilon = 1e-4

    # evaluation parameters
    test_n_z = 1
    test_batch_size = 50
    test_start = 0
    max_test_size = None

    bf_search_min = -400.
    bf_search_max = 400.
    bf_search_step_size = 1.

    valid_step_freq = 100
    gradient_clip_norm = 10.

    early_stop = True  # restore best valid weights at end (TrainLoop style)

    # pot parameters
    # recommend values for `level`:
    # SMAP: 0.07
    # MSL: 0.01
    # SMD group 1: 0.0050
    # SMD group 2: 0.0075
    # SMD group 3: 0.0001
    level = 0.01

    # outputs config
    save_z = False
    get_score_on_dim = False
    save_dir = 'model'
    restore_dir = None
    result_dir = 'result'
    train_score_filename = 'train_score.pkl'
    test_score_filename = 'test_score.pkl'

    # PyTorch-only
    log_dir = 'log'
    device = None  # auto: mps > cuda > cpu

    def to_dict(self):
        return {
            k: getattr(self, k)
            for k in dir(self)
            if not k.startswith('_') and not callable(getattr(self, k))
        }

    def update_from_args(self, args):
        for key, value in vars(args).items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)


def parse_args():
    parser = argparse.ArgumentParser(
        description='OmniAnomaly (PyTorch port of NetManAIOps/OmniAnomaly)',
    )
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--max_epoch', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--initial_lr', type=float, default=None)
    parser.add_argument('--z_dim', type=int, default=None)
    parser.add_argument('--window_length', type=int, default=None)
    parser.add_argument('--level', type=float, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--restore_dir', type=str, default=None)
    parser.add_argument('--result_dir', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default=None,
                        help='mps / cuda / cpu (default: auto)')
    parser.add_argument('--valid_step_freq', type=int, default=None)
    parser.add_argument('--no_early_stop', action='store_true',
                        help='Do not restore best validation weights at end')
    parser.add_argument('--posterior_flow_type', type=str, default=None,
                        help="'nf' or 'none'")
    return parser.parse_args()


def get_checkpoint_dir(config):
    base = config.save_dir or 'model'
    return os.path.join(base, config.dataset)


def run_experiment(config, device, log):
    os.makedirs(config.result_dir, exist_ok=True)
    if config.save_dir is not None:
        os.makedirs(get_checkpoint_dir(config), exist_ok=True)

    print('=' * 30 + ' Configurations ' + '=' * 30)
    print(json.dumps(config.to_dict(), indent=2, default=str))
    print(f'Using device: {device}')

    with open(os.path.join(config.result_dir, 'config.json'), 'w') as f:
        json.dump(config.to_dict(), f, indent=2, default=str)

    (x_train, _), (x_test, y_test) = get_data(
        config.dataset,
        config.max_train_size,
        config.max_test_size,
        train_start=config.train_start,
        test_start=config.test_start,
    )

    model = OmniAnomaly(config).to(device)
    checkpoint_dir = get_checkpoint_dir(config)
    log_dir = config.log_dir or 'log'

    trainer = Trainer(
        model=model,
        device=device,
        max_epoch=config.max_epoch,
        batch_size=config.batch_size,
        valid_batch_size=config.test_batch_size,
        initial_lr=config.initial_lr,
        lr_anneal_epochs=config.lr_anneal_epoch_freq,
        lr_anneal_factor=config.lr_anneal_factor,
        grad_clip_norm=config.gradient_clip_norm,
        valid_step_freq=config.valid_step_freq,
        early_stop=config.early_stop,
        log_dir=log_dir,
        dataset=config.dataset,
        checkpoint_dir=checkpoint_dir if config.save_dir is not None else None,
        config=config.to_dict(),
    )

    predictor = Predictor(
        model, device=device,
        batch_size=config.batch_size,
        n_z=config.test_n_z,
        last_point_only=True,
    )

    if config.restore_dir is not None:
        checkpoint, path = load_checkpoint(model, config.restore_dir, device)
        print(f'Model restored from {path}')
        log.info('Restored checkpoint: %s', path)

    if config.max_epoch > 0:
        train_start = time.time()
        best_valid_metrics = trainer.fit(x_train)
        train_time = (time.time() - train_start) / config.max_epoch
        best_valid_metrics['train_time'] = train_time
    else:
        best_valid_metrics = {}

    # train scores for POT
    train_score, train_z, train_pred_speed = predictor.get_score(x_train)
    if config.train_score_filename is not None:
        with open(os.path.join(config.result_dir, config.train_score_filename),
                  'wb') as file:
            pickle.dump(train_score, file)
    if config.save_z:
        save_z(train_z, 'train_z')

    if x_test is not None:
        test_start = time.time()
        test_score, test_z, pred_speed = predictor.get_score(x_test)
        test_time = time.time() - test_start
        if config.save_z:
            save_z(test_z, 'test_z')
        best_valid_metrics.update({
            'pred_time': pred_speed,
            'pred_total_time': test_time,
        })
        if config.test_score_filename is not None:
            with open(os.path.join(config.result_dir, config.test_score_filename),
                      'wb') as file:
                pickle.dump(test_score, file)

        if y_test is not None and len(y_test) >= len(test_score):
            if config.get_score_on_dim:
                test_score = np.sum(test_score, axis=-1)
                train_score = np.sum(train_score, axis=-1)

            t, th = bf_search(
                test_score, y_test[-len(test_score):],
                start=config.bf_search_min,
                end=config.bf_search_max,
                step_num=int(abs(config.bf_search_max - config.bf_search_min) /
                             config.bf_search_step_size),
                display_freq=50,
            )
            pot_result = pot_eval(
                train_score, test_score,
                y_test[-len(test_score):],
                level=config.level,
            )
            best_valid_metrics.update({
                'best-f1': t[0],
                'precision': t[1],
                'recall': t[2],
                'TP': t[3],
                'TN': t[4],
                'FP': t[5],
                'FN': t[6],
                'latency': t[-1],
                'threshold': th,
            })
            best_valid_metrics.update(pot_result)

    # final save (official always saves after training)
    if config.save_dir is not None:
        from omni_anomaly.checkpoint import save_checkpoint
        path = save_checkpoint(
            model, config.to_dict(), get_checkpoint_dir(config),
            filename='model.pt',
        )
        print(f'Model saved to {path}')

    metrics_path = os.path.join(config.result_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(best_valid_metrics, f, indent=2, default=str)

    print('=' * 30 + 'result' + '=' * 30)
    pprint(best_valid_metrics)
    log.info('Experiment finished')


def main():
    args = parse_args()
    config = ExpConfig()
    config.update_from_args(args)
    if args.no_early_stop:
        config.early_stop = False
    if args.posterior_flow_type is not None:
        pft = args.posterior_flow_type
        config.posterior_flow_type = None if pft.lower() in ('none', 'null') else pft
    config.x_dim = get_data_dim(config.dataset)

    if config.device:
        device = torch.device(config.device)
    else:
        device = get_device()

    log_dir = config.log_dir or 'log'
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    with experiment_logging(log_dir, config.dataset) as (log_path, log):
        print(f'Log file: {log_path}')
        log.info('Experiment started')
        log.info('Device: %s', device)
        run_experiment(config, device, log)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        main()
