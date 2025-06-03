import random
from typing import NamedTuple, TypedDict

import jax
import numpy as np
import statsmodels.api as sm
from jax import numpy as jnp

from .util import tree_stack


class Dataset(NamedTuple):
    r"""
    A dataset of alleles sampled over time.

    Params:
        thetas: array of shape [T, N, K] giving the admixture loadings for each sample at each time point.
        obs: array of shape [T, N, 2]; obs[t, i, 0] is the number of alleles sampled from indiv. i at time t, while
            obs[t, i, 1] is the number of derived alleles that were observed.
    """

    t: int
    theta: jnp.ndarray
    obs: tuple[int, int]

    @property
    def K(self):
        return self.theta.shape[-1]

    @property
    def T(self):
        "The length of the time series"
        return self.t[0] + 1

    @classmethod
    def from_records(cls, records: list[dict]):
        """Contruct dataset from a list of records.

        Args:
            records: list of dictionaries. each dictionary should have keys 't' (time before present when then individuals were sampled),
                'theta' (admixture loadings), 'n' (number of alleles observed), and 'd' (number of derived alleles observed.).
        """
        # expand records so that there is just one binomial observation per record
        new_records = [
            dict(t=r["t"], theta=r["theta"], obs=[1, 1])
            for r in records
            for _ in range(r["obs"][1])
        ]
        new_records.extend(
            [
                dict(t=r["t"], theta=r["theta"], obs=[1, 0])
                for r in records
                for _ in range(r["obs"][0] - r["obs"][1])
            ]
        )
        random.shuffle(new_records)
        T_MIN = min(r["t"] for r in new_records)
        T_MAX = max(r["t"] for r in new_records)
        K = len(new_records[0]["theta"])
        assert all(
            len(r["theta"]) == K for r in new_records
        ), "All records must have the same number of admixture loadings."
        pi = jnp.ones(K) / K
        dss = []
        for t in range(T_MAX, T_MIN - 1, -1):
            dss.extend(
                [
                    Dataset(
                        t=r["t"], theta=jnp.array(r["theta"]), obs=jnp.array(r["obs"])
                    )
                    for r in new_records
                    if r["t"] == t
                ]
            )
            # transition "dummy record" to move the ts one step towards present
            if t > T_MIN:
                dss.append(Dataset(t=t - 1, theta=pi, obs=jnp.array([0, 0])))
        ret = tree_stack(dss)
        assert ret.t.max() == T_MAX
        assert ret.t.min() == T_MIN
        # assert len(ret.obs) == len(records) + T_MAX, (len(ret.obs), len(records), T_MAX)
        return ret


def mean_paths(data: Dataset):
    K = data.K
    mean_paths = []
    Xt = sm.add_constant(np.arange(data.T).reshape(-1, 1))  # add constant for intercept
    for i in range(data.K):
        x = np.array([int(y) for x, y in zip(data.obs, data.t) if x[0] > 0])
        y = np.array(
            [
                theta[i] * x[1]
                for x, y, theta in zip(data.obs, data.t, data.theta)
                if x[0] > 0
            ]
        )
        # logistic regression using statsmodels
        X = sm.add_constant(np.array(x).reshape(-1, 1))  # add constant for intercept
        model = sm.Logit(y, X)
        result = model.fit(disp=0)  # disp=0 to suppress output
        # predict at time x=T
        pred = result.predict(Xt)
        mean_paths.append(pred)
    return np.transpose(mean_paths)
