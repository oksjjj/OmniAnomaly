# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import auc

from omni_anomaly.spot import SPOT


def _prepare_rank_inputs(score, label):
    y_true = np.asarray(label).reshape(-1).astype(bool)
    score = np.asarray(score, dtype=float)
    if score.ndim > 1:
        score = score.sum(axis=-1)
    score = score.reshape(-1)
    if len(y_true) != len(score):
        raise ValueError('score and label must have the same length')
    return score, y_true


def _calc_pa_curves(score, label, n_thresholds=1000):
    """Build ROC / PR curves with point adjustment at each threshold."""
    score, y_true = _prepare_rank_inputs(score, label)
    thresholds = np.linspace(score.min() - 1e-6, score.max() + 1e-6, n_thresholds)

    precisions = []
    recalls = []
    fprs = []
    tprs = []

    for th in thresholds:
        pred = (score < th).copy()
        pred_adj = adjust_predicts(score, label, pred=pred)

        tp = int(np.sum(pred_adj & y_true))
        fp = int(np.sum(pred_adj & ~y_true))
        fn = int(np.sum(~pred_adj & y_true))
        tn = int(np.sum(~pred_adj & ~y_true))

        precisions.append(tp / (tp + fp + 1e-8))
        recalls.append(tp / (tp + fn + 1e-8))
        fprs.append(fp / (fp + tn + 1e-8))
        tprs.append(tp / (tp + fn + 1e-8))

    order_pr = np.argsort(recalls)
    order_roc = np.argsort(fprs)
    auprc = auc(np.array(recalls)[order_pr], np.array(precisions)[order_pr])
    auroc = auc(np.array(fprs)[order_roc], np.array(tprs)[order_roc])
    return auroc, auprc


def calc_rank_metrics(score, label, n_pa_thresholds=1000):
    """
    AUROC / AUPRC with point adjustment at each threshold.

    Matches F1 / POT evaluation: ``adjust_predicts`` is applied whenever
    converting scores to binary predictions.
    """
    score, y_true = _prepare_rank_inputs(score, label)
    if len(np.unique(y_true)) < 2:
        nan = float('nan')
        return {'auroc': nan, 'auprc': nan, 'point_adjustment': True}

    auroc, auprc = _calc_pa_curves(score, label, n_thresholds=n_pa_thresholds)
    return {
        'auroc': float(auroc),
        'auprc': float(auprc),
        'point_adjustment': True,
    }


def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.

    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN


def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.

    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):

    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score < threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def calc_seq(score, label, threshold, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label)


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).


    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target = calc_seq(score, label, threshold, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)
    print(m, m_t)
    return m, m_t


def pot_eval(init_score, score, label, q=1e-4, level=0.02):
    """
    Run POT method on given score.

    Args:
        init_score (np.ndarray): Anomaly scores of the train set (for init).
        score (np.ndarray): Anomaly scores of the test set.
        label: Ground-truth labels for the test set.
        q (float): Detection level / risk. Paper uses ``1e-4``.
        level (float): Low quantile for the initial threshold ``t``
            (SMAP 0.07, MSL 0.01, SMD subset-specific).

    Notes:
        ``SPOT.run(dynamic=False)`` overwrites ``extreme_quantile`` with
        ``init_threshold`` after the first exceedance.  We therefore take the
        GPD-fitted extreme quantile from ``initialize()`` as ``pot_th``,
        matching the intended POT procedure (paper / standard SPOT).
    """
    s = SPOT(q)
    s.fit(init_score, score)
    s.initialize(level=level, min_extrema=True)

    # GPD extreme quantile computed in initialize (before run can overwrite it)
    gpd_extreme_quantile = float(s.extreme_quantile)
    pot_th = -gpd_extreme_quantile

    ret = s.run(dynamic=False)
    print(len(ret['alarms']))
    print(len(ret['thresholds']))
    print('POT threshold (GPD extreme quantile):', pot_th,
          '(init_threshold on negated scores:', float(s.init_threshold), ')')

    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    p_t = calc_point2point(pred, label)
    print('POT result: ', p_t, pot_th, p_latency)
    return {
        'pot-f1': p_t[0],
        'pot-precision': p_t[1],
        'pot-recall': p_t[2],
        'pot-TP': p_t[3],
        'pot-TN': p_t[4],
        'pot-FP': p_t[5],
        'pot-FN': p_t[6],
        'pot-threshold': pot_th,
        'pot-latency': p_latency,
        'pot-q': q,
        'pot-level': level,
        'point_adjustment': True,
    }
