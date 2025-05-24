import jax
import platformdirs

jax.config.update("jax_compilation_cache_dir", platformdirs.user_cache_dir("bmws"))
