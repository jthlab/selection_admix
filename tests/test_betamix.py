from functools import partial
import numpy as np
import jax
from jax import jit, vmap
import jax.numpy as jnp
from pytest import fixture

from bmws.data import Dataset
from bmws.betamix import _transition, _binom_sampling, _binom_sampling_admix, forward, SpikedBeta, BetaMixture

@fixture(params=[1, 5])
def K(request):
    return request.param

@fixture
def beta(K):
    betas = [SpikedBeta(log_p=np.array([-1., -2.]), f_x=BetaMixture.uniform(32)) for _ in range(K)]
    return jax.tree.map(lambda *v: jnp.stack(v), *betas)

@fixture
def dataset(K, rng):
    records = [
        {'t': 10, 'theta': rng.dirichlet(np.ones(K)), 'obs': [10, 5]},
        {'t': 9, 'theta': rng.dirichlet(np.ones(K)), 'obs': [2, 1]},
        {'t': 2, 'theta': rng.dirichlet(np.ones(K)), 'obs': [1, 0]},
    ]
    return Dataset.from_records(records)

@fixture
def s(K):
    return np.zeros([10, K])

@fixture
def Ne(K):
    return np.full([10, K], 100.0)


def test_interp():
    f = partial(jax.scipy.stats.beta.pdf, a=1, b=1)
    b = BetaMixture.interpolate(f, 32, (0.3, 0.45))

def test_sampling_no_obs(beta):
    beta = jax.tree_map(lambda a: a[0], beta)
    beta1, _ = _binom_sampling(0, 0, beta)
    np.testing.assert_allclose(beta1.log_p, beta.log_p)
    np.testing.assert_allclose(beta1.f_x.a, beta.f_x.a)
    np.testing.assert_allclose(beta1.f_x.b, beta.f_x.b)
    np.testing.assert_allclose(beta1.f_x.c, beta.f_x.c)

def test_admix_sampling_no_obs(beta, dataset, K):
    datum = jax.tree.map(lambda a: a[1], dataset)
    assert datum.obs[0] == datum.obs[1] == 0
    beta1, _ = _binom_sampling_admix(beta, datum)
    np.testing.assert_allclose(beta1.log_p, beta.log_p)
    # these identities don't hold exactly because of the resampling scheme used to maintain constant M
    # np.testing.assert_allclose(beta1.f_x.a, beta.f_x.a)
    # np.testing.assert_allclose(beta1.f_x.b, beta.f_x.b)
    # np.testing.assert_allclose(beta1.f_x.c, beta.f_x.c)
    # but we can instead check that the densities are about equal
    x = np.linspace(0, 1, 1000)
    y = vmap(lambda x: abs(beta1.f_x(x) - beta.f_x(x)))(x)
    tv = np.trapz(y, x) / 2
    assert tv < .1


def test_forward(dataset, s, Ne, K):
    beta0 = vmap(lambda _: BetaMixture.uniform(32))(jnp.arange(K))
    beta, ll = forward(s, Ne, dataset, beta0)

def test_no_sampling_admix(beta, K):
    '''Test that the binomial sampling with and without admixture are the same when there is only one population'''
    datum = Dataset(**{'t': 10, 'theta': np.eye(K)[0], 'obs': [0, 0]})
    beta1, _ = _binom_sampling_admix(beta, datum)
    np.testing.assert_allclose(beta1.log_p, beta.log_p)

def test_equal_binom_sampling_admix_noadmix(beta, K, rng):
    '''Test that the binomial sampling with and without admixture are the same when there is only one population'''
    datum = Dataset(**{'t': 10, 'theta': np.eye(K)[0], 'obs': [0, 0]})
    beta0 = jax.tree_map(lambda a: a[0], beta)
    beta1, ll1 = _binom_sampling(0, 0, beta0)
    beta2, ll2 = _binom_sampling_admix(beta, datum)
    np.testing.assert_allclose(ll1, ll2)
    for x in rng.uniform(0, 1, 10):
        y1 = beta1.f_x(x)
        y2 = vmap(BetaMixture.__call__, (0, None))(beta2.f_x, x)
        np.testing.assert_allclose(y1, y2)


        

def test_dataset_forward():
    ds = Dataset.from_records([
        dict(t=20, theta=np.array([.1, .3, .6]), obs=(1, 1)),
        dict(t=18, theta=np.array([1., 0., 0.]), obs=(1, 0)),
        dict(t=10, theta=np.array([.5, 0., .5]), obs=(1, 1)),
        dict(t=2, theta=np.array([.5, .2, .3]), obs=(10, 5)),
        dict(t=0, theta=np.array([.2, .2, .6]), obs=(3, 0)),
        ]
    )
    s = np.zeros([21, 3])
    Ne = np.full_like(s, 1e4)
    data = ds
    pi = jax.vmap(lambda _: BetaMixture.uniform(20))(jnp.arange(3))
    res = forward(s, Ne, data, pi)
    breakpoint()
    print(res)

