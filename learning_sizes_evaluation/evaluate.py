import logging

import numpy as np

logger = logging.getLogger(__name__)


def precision_recall(input, point_predictions):
    res = []
    for _, row in input.iterrows():
        min = row['min']
        max = row['max']
        object = row['object']
        pred_size = point_predictions[object]
        logger.debug(f'pred: {pred_size} (type {type(pred_size)}) min: {min} max: {max}')
        if pred_size is not None:
            correct = max > pred_size > min
        else:
            correct = None
        res.append(correct)
    nan_count = sum([x is None for x in res])
    logger.info(f'Number of nans: {nan_count}')
    logger.info(f'Recall: {1 - (nan_count / len(res))}')
    res_clean = [x for x in res if x is not None]
    precision = np.mean(res_clean)
    logger.info(f'Precision: {precision}')


def range_distance(input, point_predictions):
    distances = []
    for _, row in input.iterrows():
        minimum = row['min']
        maximum = row['max']
        object = row['object']
        pred_size = point_predictions[object]
        logger.debug(f'pred: {pred_size} (type {type(pred_size)}) min: {minimum} max: {maximum}')
        if pred_size is not None:
            correct = maximum > pred_size > minimum
            if correct:
                distance = 0.
            else:
                distance_to_min = abs(pred_size - minimum)
                distance_to_max = abs(pred_size - maximum)
                distance = min(distance_to_min, distance_to_max)

            distances.append(distance)

    distances_squared = [d ** 2 for d in distances]

    distances_nonzero = [d for d in distances if d > 0]
    distances_nonzero_squared = [d ** 2 for d in distances_nonzero]


    logger.info(f'Mean distance: {np.mean(distances)}')
    logger.info(f'Median distance FOR INCORRECT: {np.median(distances_nonzero)}')
    logger.info(f'Mean distance squared: {np.mean(distances_squared)}')
    logger.info(f'Median distance squared FOR INCORRECT: {np.median(distances_nonzero_squared)}')