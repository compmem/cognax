from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=["fn", "n_times"])
def vmap_n(fn, n_times, *args, **kwargs):
    """vmap a function n times"""
    for i in range(n_times):
        fn = jax.vmap(fn)

    return fn(*args, **kwargs)


# from https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75?permalink_comment_id=4634557#gistcomment-4634557
def tree_stack(trees):
    return jax.tree_util.tree_map(lambda *v: jnp.stack(v), *trees)