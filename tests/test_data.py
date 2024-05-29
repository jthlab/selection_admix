import numpy as np

from bmws.data import Dataset

def test_from_records():
    records = [
        {'t': 10, 'theta': [0.3, 0.7], 'obs': [10, 5]},
        {'t': 9, 'theta': [0.2, 0.8], 'obs': [2, 1]},
    ]
    dataset = Dataset.from_records(records)
    np.testing.assert_allclose(dataset.theta[0], [0.3, 0.7])
    np.testing.assert_allclose(dataset.obs, np.array([[10, 5], [0, 0], [2, 1]]))
    assert dataset.t[0] == 10
    assert dataset.t[1] == 9
    assert dataset.t[2] == 9
