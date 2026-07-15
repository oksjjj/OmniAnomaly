# -*- coding: utf-8 -*-
"""Trainer — PyTorch port of omni_anomaly.training.Trainer."""
import copy
import logging
import time

import numpy as np
import torch

from omni_anomaly.checkpoint import save_checkpoint
from omni_anomaly.train_logger import TrainHistory
from omni_anomaly.utils import BatchSlidingWindow

__all__ = ['Trainer']

logger = logging.getLogger('omni_anomaly.train')


def _clip_grad_norm_per_tensor(parameters, max_norm):
    """Match TF ``tf.clip_by_norm`` applied to each gradient tensor."""
    for p in parameters:
        if p.grad is not None:
            torch.nn.utils.clip_grad_norm_(p, max_norm)


class Trainer:
    """
    OmniAnomaly trainer.

    Early-stopping follows tfsnippet ``TrainLoop(early_stopping=True)``:
    track the best ``valid_loss`` and restore those weights when training
    finishes. Training is **not** aborted by patience (unlike some ports).
    """

    def __init__(
        self,
        model,
        device,
        n_z=None,
        max_epoch=256,
        max_step=None,
        batch_size=256,
        valid_batch_size=1024,
        valid_step_freq=100,
        initial_lr=0.001,
        lr_anneal_epochs=10,
        lr_anneal_factor=0.75,
        grad_clip_norm=10.0,
        early_stop=True,
        log_dir=None,
        dataset='default',
        checkpoint_dir=None,
        config=None,
    ):
        if max_epoch is None and max_step is None:
            raise ValueError('At least one of `max_epoch` and `max_step` '
                             'should be specified')

        self.model = model
        self.device = device
        self.n_z = n_z
        self.max_epoch = max_epoch
        self.max_step = max_step
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.valid_step_freq = valid_step_freq
        self.initial_lr = initial_lr
        self.lr_anneal_epochs = lr_anneal_epochs
        self.lr_anneal_factor = lr_anneal_factor
        self.grad_clip_norm = grad_clip_norm
        self.early_stop = early_stop
        self.dataset = dataset
        self.history = TrainHistory(log_dir, dataset) if log_dir else None
        self.checkpoint_dir = checkpoint_dir
        self.config = config or {}
        self.best_checkpoint_path = None

        # Official code never attaches kernel L2 regularizers; l2_reg in
        # ExpConfig is unused. Do not pass weight_decay.
        self.optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

    def _log(self, message):
        logger.info(message)
        print(message)

    def _record_step(self, **kwargs):
        if self.history is not None:
            self.history.add(**kwargs)

    def _eval_loss(self, data, batch_size):
        self.model.eval()
        total_loss = 0.0
        total_count = 0
        sw = BatchSlidingWindow(
            array_size=len(data),
            window_size=self.model.window_length,
            batch_size=batch_size,
        )
        with torch.no_grad():
            for (batch_x,) in sw.get_iterator([data]):
                x = torch.from_numpy(batch_x).to(self.device)
                loss = self.model.get_training_loss(x, n_z=self.n_z)
                total_loss += loss.item() * len(batch_x)
                total_count += len(batch_x)
        self.model.train()
        return total_loss / max(total_count, 1)

    def _save_history(self):
        if self.history is None:
            return {}
        json_path, csv_path = self.history.save()
        paths = {}
        if json_path:
            paths['train_history_json'] = json_path
        if csv_path:
            paths['train_history_csv'] = csv_path
        return paths

    def _save_best_checkpoint(self, best_valid_loss, epoch, step):
        if not self.checkpoint_dir:
            return None
        path = save_checkpoint(
            self.model,
            self.config,
            self.checkpoint_dir,
            extra={
                'best_valid_loss': float(best_valid_loss),
                'epoch': int(epoch),
                'step': int(step),
                'dataset': self.dataset,
            },
        )
        self.best_checkpoint_path = path
        self._log(f'Best model saved to {path} (valid_loss={best_valid_loss:.6f})')
        return path

    def fit(self, values, valid_portion=0.3):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError('`values` must be a 2-D array')

        n = int(len(values) * valid_portion)
        train_values, valid_values = values[:-n], values[-n:]

        train_sw = BatchSlidingWindow(
            array_size=len(train_values),
            window_size=self.model.window_length,
            batch_size=self.batch_size,
            shuffle=True,
            ignore_incomplete_batch=True,
        )

        lr = self.initial_lr
        best_valid_loss = None
        best_state = None
        train_batch_times = []
        valid_batch_times = []

        global_step = 0
        self._log(f'train_values: {train_values.shape}')

        epoch = 0
        stop = False
        while not stop:
            epoch += 1
            if self.max_epoch is not None and epoch > self.max_epoch:
                break

            self.model.train()
            epoch_start = time.time()

            for batch_x, in train_sw.get_iterator([train_values]):
                if self.max_step is not None and global_step >= self.max_step:
                    stop = True
                    break

                batch_start = time.time()
                x = torch.from_numpy(batch_x).to(self.device)

                self.optimizer.zero_grad()
                loss = self.model.get_training_loss(x, n_z=self.n_z)
                loss.backward()

                if self.grad_clip_norm:
                    _clip_grad_norm_per_tensor(
                        self.model.parameters(), self.grad_clip_norm,
                    )

                self.optimizer.step()
                train_batch_times.append(time.time() - batch_start)
                global_step += 1

                if global_step % self.valid_step_freq == 0:
                    valid_start = time.time()
                    valid_loss = self._eval_loss(
                        valid_values, self.valid_batch_size,
                    )
                    valid_batch_times.append(time.time() - valid_start)
                    train_duration = time.time() - epoch_start

                    is_best = (
                        best_valid_loss is None or valid_loss < best_valid_loss
                    )
                    if is_best:
                        best_valid_loss = valid_loss
                        if self.early_stop:
                            best_state = copy.deepcopy(self.model.state_dict())
                        self._save_best_checkpoint(
                            best_valid_loss, epoch, global_step,
                        )

                    self._log(
                        f'epoch={epoch} step={global_step} '
                        f'loss={loss.item():.6f} valid_loss={valid_loss:.6f} '
                        f'lr={lr:.6g} best_valid_loss={best_valid_loss:.6f} '
                        f'train_time={train_duration:.2f}s'
                    )
                    self._record_step(
                        epoch=epoch,
                        step=global_step,
                        loss=float(loss.item()),
                        valid_loss=float(valid_loss),
                        lr=float(lr),
                        best_valid_loss=float(best_valid_loss),
                        is_best=is_best,
                        train_time_sec=float(train_duration),
                        valid_time_sec=float(valid_batch_times[-1]),
                    )
                    epoch_start = time.time()

            if stop:
                break

            if self.lr_anneal_epochs and epoch % self.lr_anneal_epochs == 0:
                lr *= self.lr_anneal_factor
                for pg in self.optimizer.param_groups:
                    pg['lr'] = lr
                msg = f'Learning rate decreased to {lr}'
                self._log(msg)
                self._record_step(
                    event='lr_anneal', epoch=epoch, step=global_step, lr=float(lr),
                )

        # TrainLoop early_stopping: restore best variables at the end
        if self.early_stop and best_state is not None:
            self.model.load_state_dict(best_state)

        metrics = {
            'best_valid_loss': float(
                best_valid_loss if best_valid_loss is not None else float('nan')
            ),
            'train_time': float(np.mean(train_batch_times) if train_batch_times else 0),
            'valid_time': float(np.mean(valid_batch_times) if valid_batch_times else 0),
            'stopped_epoch': epoch,
            'stopped_step': global_step,
            'best_checkpoint': self.best_checkpoint_path,
        }
        metrics.update(self._save_history())
        return metrics
