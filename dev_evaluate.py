import json
import logging
import os
import pickle

import yaml
from box import Box
from logging_setup_dla.logging import set_up_root_logger
from pandas import DataFrame
import pandas as pd

from learning_sizes_evaluation.evaluate import precision_recall, range_distance

set_up_root_logger(f'DEV', os.path.join(os.getcwd(), 'logs'))

logger = logging.getLogger(__name__)


def main():
    with open("config.yml", "r") as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile))

    paths = cfg[cfg.evaluate]
    input: DataFrame = pd.read_csv(paths.input)
    input = input.astype({'object': str})
    for k, v in paths.results.items():
        with open(v, 'rb') as f:
            predictions = pickle.load(f)
            # predictions['nan'] = predictions[float('nan')]
        logger.info(f'\nResults for {k}')
        precision_recall(input, predictions)
        range_distance(input, predictions)




if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception")
        raise
