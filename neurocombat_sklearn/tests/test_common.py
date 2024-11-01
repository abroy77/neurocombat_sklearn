from sklearn.utils.estimator_checks import check_estimator

from neurocombat_sklearn import CombatModel

def test_all_transformers():
    return check_estimator(CombatModel())
