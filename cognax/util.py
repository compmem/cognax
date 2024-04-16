from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=["fn", "n_times"])
def vmap_n(fn, n_times, *args, **kwargs):
    """vmap a function n times"""
    for i in range(n_times):
        fn = jax.vmap(fn)

    return fn(*args, **kwargs)


def tree_stack(trees):
    return jax.tree_util.tree_map(lambda *v: jnp.stack(v), *trees)
