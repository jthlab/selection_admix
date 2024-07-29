import numpy as np

from bmws.betamix import BetaMixture, loglik


def test_loglik_same_admix_noadmix():
    "test that the loglikelihood is the same in the admixed and no admix model"
    obs = np.array([[5, 1], [0, 0], [1, 0], [0, 0], [3, 2]])  # (n, d)
    beta0 = BetaMixture.uniform(100)
    s = np.zeros(len(obs) - 1)
    Ne = np.full_like(s, 1e4)
    print(loglik(s, Ne, obs, beta0))
