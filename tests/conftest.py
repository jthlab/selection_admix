import numpy as np
from pytest import fixture

@fixture(params=range(3))
def seed(request):
    return request.param

@fixture
def rng(seed):
    return np.random.default_rng(seed)

@fixture
def times():
    return tuple(range(0, 200, 10))[::-1]


@fixture
def T(times):
    return len(times)


@fixture
def Ne(T):
    return np.array([100.0] * T)


@fixture
def s(times):
    return np.zeros(len(times[:-1]))


@fixture
def obs(times):
    T = len(times)
    n = np.random.randint(2, 20, size=T)
    derived = np.random.randint(0, n, size=T)
    obs = np.transpose([n, derived])
    obs[-1][:] = 10
    return obs
