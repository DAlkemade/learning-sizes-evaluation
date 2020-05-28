from learning_sizes_evaluation.evaluate import coverage_accuracy


def test_cov_acc():
    gold = [True, True, False]
    preds = [True,False, None]
    cov, acc = coverage_accuracy(gold, preds)
    assert cov == 2/3
    assert acc == .5