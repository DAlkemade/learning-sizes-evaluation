from learning_sizes_evaluation.evaluate import coverage_accuracy_relational


def test_cov_acc():
    gold = [True, True, False]
    preds = [True,False, None]
    cov, acc = coverage_accuracy_relational(gold, preds)
    assert cov == 2/3
    assert acc == .5