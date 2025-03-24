from typing import NamedTuple, TypedDict

import jax
import numpy as np
from jax import numpy as jnp


def tree_stack(trees):
    """Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = jax.tree.flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [np.stack(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


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
        T_MIN = min(r["t"] for r in records)
        T_MAX = max(r["t"] for r in records)
        K = len(records[0]["theta"])
        assert all(
            len(r["theta"]) == K for r in records
        ), "All records must have the same number of admixture loadings."
        pi = jnp.ones(K) / K
        dss = []
        for t in range(T_MAX, T_MIN - 1, -1):
            dss.extend(
                [
                    Dataset(
                        t=r["t"], theta=jnp.array(r["theta"]), obs=jnp.array(r["obs"])
                    )
                    for r in records
                    if r["t"] == t
                ]
            )
            # transition "dummy record" to move the ts one step towards present
            if t > T_MIN:
                dss.append(Dataset(t=t - 1, theta=pi, obs=jnp.array([0, 0])))
        return tree_stack(dss)
