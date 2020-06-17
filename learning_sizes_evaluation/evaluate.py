import logging
from collections import namedtuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.scale import SymmetricalLogTransform

logger = logging.getLogger(__name__)


def coverage_accuracy_relational(golds, preds, notes=None):
    """Compute coverage and selectivity for a comparison task."""
    total_n = len(golds)
    answered_count = 0
    correct_count = 0
    for i, gold in enumerate(golds):
        pred = preds[i]
        if pred is not None:
            answered_count += 1
            if pred == gold:
                correct_count += 1
                if notes is not None:
                    logger.debug(f'Correct: {notes[i]}')
            else:
                if notes is not None:
                    logger.debug(f'Incorrect: {notes[i]}')
    coverage = answered_count / total_n
    selectivity = correct_count / answered_count
    return coverage, selectivity


def precision_recall(input, point_predictions):
    """Compute coverage and selectivity."""
    res = check_preds(input, point_predictions)
    nan_count = sum([x is None for x in res])
    logger.info(f'Number of nans: {nan_count}')
    coverage = 1 - (nan_count / len(res))
    logger.info(f'coverage: {coverage}')
    res_clean = [x for x in res if x is not None]
    selectivity = np.mean(res_clean)
    logger.info(f'selectivity: {selectivity}')
    return selectivity, coverage


def check_preds(input, point_predictions):
    """Check whether predictions are correct."""
    res = []
    for _, row in input.iterrows():
        min = row['min']
        max = row['max']
        object = row['object']
        pred_size = point_predictions[object]
        if pred_size is not None:
            correct = max > pred_size > min
        else:
            correct = None
        res.append(correct)
    return res


def range_distance(input, point_predictions):
    """Compute statistics for error distances"""
    distances = get_distances(input, point_predictions)

    distances_squared = [d ** 2 for d in distances]

    distances_nonzero = [d for d in distances if d > 0]
    distances_nonzero_squared = [d ** 2 for d in distances_nonzero]

    dist_mean = np.mean(distances)
    dist_median = np.median(distances_nonzero)
    dist_mean_squared = np.mean(distances_squared)
    logger.info(f'Mean distance: {dist_mean}')
    logger.info(f'Median distance FOR INCORRECT: {dist_median}')
    logger.info(f'Mean distance squared: {dist_mean_squared}')
    return dist_mean, dist_mean_squared, dist_median


def get_distances(input, point_predictions):
    """Compute error distances (i.e. distance to correct range)"""
    distances = []
    for _, row in input.iterrows():
        minimum = row['min']
        maximum = row['max']
        object = row['object']
        pred_size = point_predictions[object]
        if pred_size is not None:
            correct = maximum > pred_size > minimum
            if correct:
                distance = 0.
            else:
                distance_to_min = abs(pred_size - minimum)
                distance_to_max = abs(pred_size - maximum)
                distance = min(distance_to_min, distance_to_max)

            distances.append(distance)
    return distances


Result = namedtuple('Result',
                    ['system', 'selectivity', 'coverage', 'mean_distance', 'mean_distance_squared', 'median_distance'])
RelationalResult = namedtuple('RelationalResult', ['system', 'selectivity', 'coverage'])


def distances_hist(distances_results, keys, save=False):
    """Plot a histogram of error distances."""
    maximums = []
    comp = []
    for k in keys:
        distances = distances_results[k]
        comp.append(distances)
        maximums.append(max(distances))
    tr = SymmetricalLogTransform(base=10, linthresh=1, linscale=1)
    ss = tr.transform([0., max(maximums) + 1])
    bins = tr.inverted().transform(np.linspace(*ss, num=30))
    plt.style.use('seaborn-deep')

    fig, ax = plt.subplots()
    plt.hist(comp, bins, label=keys)
    plt.legend(loc='upper right')
    ax.set_xscale('symlog')
    plt.xlabel('Distance [m]')
    if save:
        plt.savefig('distances_hist.png')
    plt.show()
