import logging
import os
import pickle

import pandas as pd
import yaml
from box import Box
from logging_setup_dla.logging import set_up_root_logger
from pandas import DataFrame

from learning_sizes_evaluation.evaluate import precision_recall, range_distance, check_preds, Result, get_distances, \
    distances_hist
from learning_sizes_evaluation.monte_carlo_permutation_test import permutation_test

set_up_root_logger(f'DEV', os.path.join(os.getcwd(), 'logs'))

logger = logging.getLogger(__name__)


def main():
    """Show an evaluation of results on the development set."""
    with open("config.yml", "r") as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile))

    paths = cfg[cfg.evaluate]
    input: DataFrame = pd.read_csv(paths.input)
    input = input.astype({'object': str})
    input_large = input[input['min'] > 1]
    logger.info(f'Number of large objects in input: {len(input_large.index)}')
    predictions_correct = dict()
    results = list()
    distances_results = dict()
    for k, v in paths.results.items():
        with open(v, 'rb') as f:
            predictions = pickle.load(f)
            # predictions['nan'] = predictions[float('nan')]
        logger.info(f'\nResults for {k}')
        logger.info(f'Loaded results: {v}')
        predictions_correct[k] = [bool(p) for p in check_preds(input, predictions)]
        selectivity, coverage = precision_recall(input, predictions)
        mean, mean_squared, median = range_distance(input, predictions)
        results.append(Result(k, selectivity, coverage, mean, mean_squared, median))

        distances_results[k] = get_distances(input, predictions)

    results_df = pd.DataFrame(results)
    results_df.to_csv('results_dev.csv')

    distances_hist(distances_results, ['regex', 'bootstrap_no_visual_no_coref'])
    distances_hist(distances_results, ['bootstrap_no_visual_coref', 'bootstrap_no_visual_no_coref'])

    for k1 in paths.results.keys():
        for k2 in paths.results.keys():
            pcorrect1 = predictions_correct[k1]
            pcorrect2 = predictions_correct[k2]
            predictions_correct1 = [p is True for p in pcorrect1]
            predictions_correct2 = [p is True for p in pcorrect2]
            p = permutation_test(predictions_correct1, predictions_correct2)
            logger.info(f'p-value {p} for {k1} {k2}')


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise
