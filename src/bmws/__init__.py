import jax

jax.config.update("jax_enable_x64", True)
import logging

logging.getLogger("absl").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)

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
