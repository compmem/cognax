from collections import namedtuple

import jax
import jax.random as random
import jax.numpy as jnp

from numpyro.infer.util import initialize_model
from numpyro.infer.initialization import init_to_uniform

EvoState = namedtuple("EvoState", ["state", "rng_key"])


class EvoMAP:
    """
    This should allow you to use evolutionary strategies to get the MAP of a numpyro model.
    Discrete variables must be enumerated out.

    Depends on `evosax`.

    **Example**

    ```python
        import jax
        import jax.numpy as jnp
        import jax.random as random
        import numpyro
        import numpyro.distributions as dist
        from evosax import DE

        def model(obs):
            a = numpyro.sample('mu', dist.Normal(0, 1))
            with numpyro.plate('n_obs', 100):
                numpyro.sample('obs', dist.Normal(a, 1), obs=obs)

        obs = dist.Normal(loc=jnp.array(3.), scale=1).sample(jax.random.PRNGKey(0), (100,))

        evo = EvoMAP(model, DE, popsize=100)
        state = evo.init(random.PRNGKey(0), obs)

        j_update = jax.jit(evo.update)

        for i in tqdm(range(2000)):
            state, loss = j_update(state)

        print(evo.get_params(state))
    ```
    """

    def __init__(self, model, strategy, popsize):
        """
        Args:
            model: Python callable with Pyro primitives for the model.
            strategy: evosax.Strategy
            popsize (int): evosax population size
        """
        self.model = model

        self.strategy = strategy
        self.popsize = popsize

    def init(self, rng_key, *args, **kwargs):
        init_key, rng_key = random.split(rng_key)

        # only if no guide was specified
        params_info, potential_fn_gen, postprocess_fn, _ = initialize_model(
            rng_key,
            self.model,
            dynamic_args=True,
            init_strategy=init_to_uniform,
            model_args=args,
            model_kwargs=kwargs,
            validate_grad=False,
        )
        self.batch_potential_fn = jax.vmap(potential_fn_gen(*args, **kwargs))
        self.constrain_fn = postprocess_fn(*args, **kwargs)

        self.strategy = self.strategy(
            popsize=self.popsize, pholder_params=params_info.z
        )

        init_state = self.strategy.initialize(init_key)

        if jax.config.values.get("jax_enable_x64"):
            init_state = init_state.replace(best_fitness=-jnp.finfo(jnp.float64).max)

        return EvoState(init_state, rng_key)

    def update(self, evo_state):
        ask_key, rng_key = random.split(evo_state.rng_key)

        candidates, state = self.strategy.ask(ask_key, evo_state.state)
        fitness = self.batch_potential_fn(candidates)
        state = self.strategy.tell(candidates, fitness, state)

        return EvoState(state, rng_key), fitness

    def get_params(self, evo_state):
        return self.constrain_fn(
            self.strategy.param_reshaper.reshape_single(evo_state.state.best_member)
        )
