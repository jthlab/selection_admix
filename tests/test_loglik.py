import jax
import numpy as np

from bmws.data import Dataset
from bmws.betamix import BetaMixture, loglik

def test_loglik_same_admix_noadmix():
    'test that the loglikelihood is the same in the admixed and no admix model'
    data = [[5, 1]]  #  , [0, 0], [1, 0], [0, 0], [3, 2]]  # (n, d)
    records = [
        {'t': t, 'theta': np.ones(1), 'obs': (n, d)}
        for t, (n, d) in enumerate(data)
    ]
    ds = Dataset.from_records(records)
    beta0 = jax.vmap(lambda _: BetaMixture.uniform(100))(np.arange(1))
    s = np.zeros([len(data), 1])
    Ne = np.full_like(s, 1e4)
    print(loglik(s, Ne, ds, beta0))
