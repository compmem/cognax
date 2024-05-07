# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0
# modified from https://github.com/pyro-ppl/numpyro/blob/master/test/test_constraints.py

from collections import namedtuple

import pytest

from jax import jit, vmap
import jax.numpy as jnp
from jax.tree_util import tree_map

from numpyro.distributions import constraints
from cognax.joint_modeling.collection import constraint_collection

_a = jnp.asarray


class T(namedtuple("TestCase", ["constraint_cls", "params", "kwargs"])):
    pass


PARAMETRIZED_CONSTRAINTS = {
    "constraint_collection": T(
        constraint_collection,
        ([constraints.greater_than(_a(1.0)), constraints.greater_than(_a(0.0))],),
        dict(slices=[(0, 1), (1, 2)]),
    ),
}


@pytest.mark.parametrize(
    "cls, cst_args, cst_kwargs",
    PARAMETRIZED_CONSTRAINTS.values(),
    ids=PARAMETRIZED_CONSTRAINTS.keys(),
)
def test_parametrized_constraint_pytree(cls, cst_args, cst_kwargs):
    constraint = cls(*cst_args, **cst_kwargs)

    # test that singleton constraints objects can be used as pytrees
    def in_cst(constraint, x):
        return x**2

    def out_cst(constraint, x):
        return constraint

    jitted_in_cst = jit(in_cst)
    jitted_out_cst = jit(out_cst)

    assert jitted_in_cst(constraint, 1.0) == 1.0
    assert jitted_out_cst(constraint, 1.0) == constraint

    assert jnp.allclose(
        vmap(in_cst, in_axes=(None, 0), out_axes=0)(constraint, jnp.ones(3)),
        jnp.ones(3),
    )

    assert (
        vmap(out_cst, in_axes=(None, 0), out_axes=None)(constraint, jnp.ones(3))
        == constraint
    )

    if len(cst_args) > 0:
        # test creating and manipulating vmapped constraints
        vmapped_cst_args = tree_map(lambda x: x[None], cst_args)

        vmapped_csts = jit(vmap(lambda args: cls(*args, **cst_kwargs), in_axes=(0,)))(
            vmapped_cst_args
        )
        assert vmap(lambda x: x == constraint, in_axes=0)(vmapped_csts).all()

        twice_vmapped_cst_args = tree_map(lambda x: x[None], vmapped_cst_args)

        vmapped_csts = jit(
            vmap(
                vmap(lambda args: cls(*args, **cst_kwargs), in_axes=(0,)),
                in_axes=(0,),
            ),
        )(twice_vmapped_cst_args)
        assert vmap(vmap(lambda x: x == constraint, in_axes=0), in_axes=0)(
            vmapped_csts
        ).all()


@pytest.mark.parametrize(
    "cls, cst_args, cst_kwargs",
    PARAMETRIZED_CONSTRAINTS.values(),
    ids=PARAMETRIZED_CONSTRAINTS.keys(),
)
def test_parametrized_constraint_eq(cls, cst_args, cst_kwargs):
    constraint = cls(*cst_args, **cst_kwargs)
    constraint2 = cls(*cst_args, **cst_kwargs)
    assert constraint == constraint2
    assert constraint != 1

    # check that equality checks are robust to constraints parametrized
    # by abstract values
    @jit
    def check_constraints(c1, c2):
        return c1 == c2

    assert check_constraints(constraint, constraint2)
