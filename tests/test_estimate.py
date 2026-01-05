import numpy as np
import jax
import jax.numpy as jnp

from bmws.betamix import sample_path, Selection, prior_from_beta, em

from bmws.data import Dataset

def test_sample_posterior():
    data = Dataset.from_records([
        dict(t=20, theta=np.array([.1, .3, .6]), obs=(1, 1)),
        dict(t=18, theta=np.array([1., 0., 0.]), obs=(1, 0)),
        dict(t=10, theta=np.array([.1, .4, .5]), obs=(2, 1)),
        dict(t=10, theta=np.array([.1, 0., .9]), obs=(5, 2)),
        dict(t=10, theta=np.array([.5, 0., .5]), obs=(1, 1)),
        dict(t=10, theta=np.array([.01, .98, .01]), obs=(100, 1)),
        dict(t=2, theta=np.array([.2, .5, .3]), obs=(10, 5)),
        dict(t=2, theta=np.array([.5, .2, .3]), obs=(100, 0)),
        dict(t=1, theta=np.array([.5, .2, .3]), obs=(1000, 0)),
        dict(t=0, theta=np.array([.2, .2, .6]), obs=(35, 0)),
        ]
    )
    sln = Selection(T=21, s=np.zeros([4, 3]))
    key = jax.random.PRNGKey(1)
    key, subkey = jax.random.split(key)
    a = jnp.ones(3)
    b = 100. * a
    prior = prior_from_beta(a, b, 10_000, 1000, subkey)
    key, subkey = jax.random.split(key)
    path = sample_path(sln, data, prior, subkey)
    print(path)


def test_em():
    data = Dataset.from_records([
        dict(t=20, theta=np.array([.1, .3, .6]), obs=(1, 1)),
        dict(t=18, theta=np.array([1., 0., 0.]), obs=(1, 0)),
        dict(t=10, theta=np.array([.1, .4, .5]), obs=(2, 1)),
        dict(t=10, theta=np.array([.1, 0., .9]), obs=(5, 2)),
        dict(t=10, theta=np.array([.5, 0., .5]), obs=(1, 1)),
        dict(t=10, theta=np.array([.01, .98, .01]), obs=(100, 1)),
        dict(t=2, theta=np.array([.2, .5, .3]), obs=(10, 5)),
        dict(t=2, theta=np.array([.5, .2, .3]), obs=(100, 0)),
        dict(t=1, theta=np.array([.5, .2, .3]), obs=(1000, 0)),
        dict(t=0, theta=np.array([.2, .2, .6]), obs=(35, 0)),
        ]
    )
    sln = Selection(T=21, s=np.zeros([4, 3]))
    key = jax.random.PRNGKey(1)
    key, subkey = jax.random.split(key)
    a = jnp.ones(3)
    b = 100. * a
    prior = prior_from_beta(a, b, 10_000, subkey)
    key, subkey = jax.random.split(key)
    path = em(sln, data, prior, subkey)
    breakpoint()
