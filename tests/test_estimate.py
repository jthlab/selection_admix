import numpy as np
import jax
import jax.numpy as jnp

from bmws.betamix import BetaMixture
from bmws.data import Dataset
from bmws.estimate import sample_paths, _sample_path

def test_sample_posterior():
    ds = Dataset.from_records([
        dict(t=20, theta=np.array([.1, .3, .6]), obs=(1, 1)),
        dict(t=18, theta=np.array([1., 0., 0.]), obs=(1, 0)),
        dict(t=10, theta=np.array([.1, .4, .5]), obs=(2, 1)),
        dict(t=10, theta=np.array([.1, 0., .9]), obs=(5, 2)),
        dict(t=10, theta=np.array([.5, 0., .5]), obs=(1, 1)),
        dict(t=10, theta=np.array([0, 1., 0]), obs=(100, 1)),
        dict(t=2, theta=np.array([.2, .5, .3]), obs=(10, 5)),
        dict(t=2, theta=np.array([.5, .2, .3]), obs=(100, 0)),
        dict(t=1, theta=np.array([.5, .2, .3]), obs=(1000, 0)),
        dict(t=0, theta=np.array([.2, .2, .6]), obs=(35, 0)),
        ]
    )
    s = np.zeros([21, 3])
    Ne = np.full_like(s, 1e4)
    data = ds
    pi = jax.vmap(lambda _: BetaMixture.uniform(20))(jnp.arange(3))
    key = jax.random.PRNGKey(1)
    paths = sample_paths(s, Ne, data, pi, k=3, seed=1)
    breakpoint()
