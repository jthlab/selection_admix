from jax.scipy.stats import beta
from jax import grad

print(grad(beta.pdf, argnums=2)(0., 1., 2.))
