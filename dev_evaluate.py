import logging
import os
import pickle

import yaml
from box import Box
from logging_setup_dla.logging import set_up_root_logger
from pandas import DataFrame
import pandas as pd

from learning_sizes_evaluation.evaluate import precision_recall, range_distance, check_preds, Result
from learning_sizes_evaluation.monte_carlo_permutation_test import permutation_test

set_up_root_logger(f'DEV', os.path.join(os.getcwd(), 'logs'))

logger = logging.getLogger(__name__)


def main():
    with open("config.yml", "r") as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile))

    paths = cfg[cfg.evaluate]
    input: DataFrame = pd.read_csv(paths.input)
    input = input.astype({'object': str})
    input_large = input[input['min'] > 1]
    logger.info(f'Number of large objects in input: {len(input_large.index)}')
    predictions_correct = dict()
    results = list()
    for k, v in paths.results.items():
        with open(v, 'rb') as f:
            predictions = pickle.load(f)
            # predictions['nan'] = predictions[float('nan')]
        logger.info(f'\nResults for {k}')
        logger.info(f'Loaded results: {v}')
        predictions_correct[k] = check_preds(input, predictions)
        selectivity, coverage = precision_recall(input, predictions)
        mean, mean_squared, median = range_distance(input, predictions)
        results.append(Result(k, selectivity, coverage, mean, mean_squared, median))

            # logger.info('Results on large objects')
            # precision_recall(input_large, predictions)
    results_df = pd.DataFrame(results)
    results_df.to_csv('results_dev.csv')

    for k1 in paths.results.keys():
        for k2 in paths.results.keys():
            predictions_correct1 = [p is True for p in predictions_correct[k1]]
            predictions_correct2 = [p is True for p in predictions_correct[k2]]
            p = permutation_test(predictions_correct1, predictions_correct2)
            logger.info(f'p-value {p} for {k1} {k2}')




if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise
