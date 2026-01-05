from bmws.data import Dataset
import jax.numpy as jnp
import bmws.sim
import numpy as np

def test_sim():
    ds = Dataset.from_records([
        dict(t=20, theta=np.array([.1, .3, .6]), obs=(100, 1)),
        dict(t=18, theta=np.array([1., 0., 0.]), obs=(100, 0)),
        dict(t=10, theta=np.array([.1, .4, .5]), obs=(200, 1)),
        dict(t=10, theta=np.array([.1, 0., .9]), obs=(500, 2)),
        dict(t=10, theta=np.array([.5, 0., .5]), obs=(100, 1)),
        dict(t=2, theta=np.array([.2, .5, .3]), obs=(1000, 5)),
        dict(t=2, theta=np.array([.5, .2, .3]), obs=(10000, 0)),
        dict(t=1, theta=np.array([.5, .2, .3]), obs=(100000, 0)),
        dict(t=0, theta=np.array([0., 1., 0.]), obs=(3005, 0)),
        dict(t=0, theta=np.array([.2, .2, .6]), obs=(3005, 0)),
        ]
    )
    # ds = ds._replace(theta = jnp.array([[.33, .33, .34] for _ in ds.theta]))
    s = np.zeros([ds.T, 3])
    s[int(ds.T / 2):] = [-.1, 0, .1]
    Ne = np.full_like(s, 1e4)
    mdl = dict(s=s, Ne=Ne, f0=np.array([.5, .5, .5]))
    data = ds
    res = bmws.sim.sim_admix(mdl, data, 1, estimate_kwargs={'alpha': 100., 'beta': 0., 'gamma': 0.})
