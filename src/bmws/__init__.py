import logging

import jax
import platformdirs

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", platformdirs.user_cache_dir("bmws"))

logging.getLogger("absl").setLevel(logging.ERROR)

from bmws.betamix import BetaMixture
from bmws.common import Observation
from bmws.estimate import estimate, estimate_em, sample_paths
from bmws.sim import sim_and_fit, sim_wf

__all__ = [
    "estimate",
    "estimate_em",
    "sim_and_fit",
    "sim_wf",
    "Observation",
    "BetaMixture",
]
