"""Tests for the PNO + GNN alpha policy (network.model.GraphAlphaModel).

These encode the point of the redesign (PNO_GNN_integration_spec.md §10):
order-independence, P-independence, the DGSEM-graph coupling, and the
jit/grad training contract. The spec's physical-sanity check (smooth -> alpha
~ 0, shock -> alpha elevated) needs a TRAINED network and lives in the
training/evaluation pipeline (compare.py), not here.

Run from the repo root:
    .venv_spectral/bin/python -m pytest nn/tests/test_graph_model.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from jax_dgsem import GLLBasis, Mesh1D
from jax_dgsem.physics import GAMMA
from network.model import GraphAlphaModel
from network.policy import (alpha_features, apply_alpha, build_alpha_model,
                            graph_features, model_meta)

PERIODIC = ("periodic", None)


def make_mesh(P, n_elem):
    return Mesh1D(GLLBasis(P), n_elem, 0.0, 1.0, PERIODIC, PERIODIC)


def make_model(seed=0, stable_init=False, **kw):
    """Small graph model; stable_init=False so outputs actually vary."""
    return GraphAlphaModel(pno_channels=4, pno_hidden=8, gnn_hidden=8,
                           fusion_hidden=8, depth=1,
                           key=jax.random.PRNGKey(seed),
                           stable_init=stable_init, **kw)


def euler_state(mesh, kind):
    """Euler state on the mesh nodes: smooth density sine or a sharp jump."""
    x = np.asarray(mesh.node_positions(0.0))
    if kind == "smooth":
        rho = 1.0 + 0.2 * np.sin(2.0 * np.pi * x)
    else:
        rho = np.where(x < 0.5, 1.0, 0.125)
    u = np.zeros_like(x)
    p = np.ones_like(x)
    E = p / (GAMMA - 1.0) + 0.5 * rho * u ** 2
    return jnp.asarray(np.stack([rho, rho * u, E]))


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------

def test_shape_and_range_contract():
    mesh = make_mesh(4, 8)
    alpha, alpha_boundary = apply_alpha(make_model(), euler_state(mesh, "shock"),
                                        mesh, "graph")
    assert alpha.shape == (8, 4)                 # (n_elem, P), solver contract
    assert bool(jnp.all((alpha > 0.0) & (alpha < 1.0)))
    assert alpha_boundary.shape == (8,)          # one per element-to-element face
    assert bool(jnp.all((alpha_boundary > 0.0) & (alpha_boundary < 1.0)))


def test_stable_init_outputs_alpha_init():
    """Zero readout + logit bias -> the untrained policy is constant alpha_init
    (the stable Stage-2 cold start, same contract as NodalAlphaModel), for
    BOTH the subcell and element-boundary readouts (they share proj_w/proj_b)."""
    mesh = make_mesh(4, 8)
    model = make_model(stable_init=True, alpha_init=0.05)
    alpha, alpha_boundary = apply_alpha(model, euler_state(mesh, "shock"), mesh,
                                        "graph")
    np.testing.assert_allclose(np.asarray(alpha), 0.05, atol=1e-12)
    np.testing.assert_allclose(np.asarray(alpha_boundary), 0.05, atol=1e-12)


def test_features_have_no_onehot():
    """The graph input is exactly [residual, modal spectrum] -- no position
    block; alpha_features routes 'graph' to it."""
    mesh = make_mesh(4, 6)
    U = euler_state(mesh, "shock")
    feats = alpha_features(U, mesh, "graph")
    assert feats.shape == (2, 6, 5)
    np.testing.assert_array_equal(np.asarray(feats),
                                  np.asarray(graph_features(U, mesh)))


# ---------------------------------------------------------------------------
# Invariances (the whole point of the redesign)
# ---------------------------------------------------------------------------

def test_point_permutation_equivariance():
    """Shuffle the interior nodes of every element together with the matching
    rows/cols of the operators (quads, D, Phi) and the residual rows: the
    per-point latents must permute identically -- position enters ONLY through
    the physical operators, never through array slots. (Face nodes stay in
    their slots: 'which nodes sit on the face' is itself physical structure,
    like D.)"""
    mesh = make_mesh(4, 6)
    U = euler_state(mesh, "shock")
    model = make_model()
    feats = graph_features(U, mesh)
    z = model.point_latents(feats, mesh)

    perm = np.array([0, 3, 2, 1, 4])             # interior shuffle, faces fixed
    mesh_p = eqx.tree_at(
        lambda m: (m.quads, m.D, m.Phi), mesh,
        (mesh.quads[perm], mesh.D[perm][:, perm], mesh.Phi[perm]))
    feats_p = feats.at[0].set(feats[0][:, perm])  # residual rows follow points;
                                                  # the MODE axis is untouched
    z_p = model.point_latents(feats_p, mesh_p)
    np.testing.assert_allclose(np.asarray(z_p), np.asarray(z)[:, perm, :],
                               rtol=0.0, atol=1e-12)


def test_element_roll_equivariance():
    """Reorder the elements (periodic roll = the adjacency-preserving element
    permutation): per-element outputs must be identical, for both the subcell
    alpha and the element-boundary alpha."""
    mesh = make_mesh(3, 8)
    model = make_model()
    feats = graph_features(euler_state(mesh, "shock"), mesh)
    alpha, alpha_boundary = model(feats, mesh)
    s = 3
    alpha_roll, alpha_boundary_roll = model(jnp.roll(feats, s, axis=1), mesh)
    np.testing.assert_allclose(np.asarray(alpha_roll),
                               np.asarray(jnp.roll(alpha, s, axis=0)),
                               rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(alpha_boundary_roll),
                               np.asarray(jnp.roll(alpha_boundary, s, axis=0)),
                               rtol=0.0, atol=1e-12)


def test_p_generalisation_same_params():
    """One instantiated model runs at P=3 and P=8 with no shape error and the
    same global-feature width -- the property the CNN + one-hot could not
    have (its weights were tied to Nn). alpha_boundary is per-ELEMENT, so its
    shape must stay (n_elem,) regardless of P."""
    model = make_model()
    for P in (3, 8):
        mesh = make_mesh(P, 6)
        U = euler_state(mesh, "shock")
        alpha, alpha_boundary = apply_alpha(model, U, mesh, "graph")
        assert alpha.shape == (6, P)
        assert alpha_boundary.shape == (6,)
        g, nodal = model.pno(graph_features(U, mesh)[1], mesh.Phi)
        assert g.shape == (6, 4)                 # channels independent of P
        assert nodal.shape == (6, P + 1, 4)


def test_n_elem_generalisation_same_params():
    model = make_model()
    for n_elem in (4, 32):
        mesh = make_mesh(4, n_elem)
        alpha, alpha_boundary = apply_alpha(model, euler_state(mesh, "shock"),
                                            mesh, "graph")
        assert alpha.shape == (n_elem, 4)
        assert alpha_boundary.shape == (n_elem,)


# ---------------------------------------------------------------------------
# The DGSEM-graph coupling
# ---------------------------------------------------------------------------

def test_surface_coupling_and_locality():
    """depth=1 (one volume + one surface round): alpha in element e must react
    to features of the face-sharing neighbours e+-1 (the coupling the old
    per-element CNN could not represent) and must NOT react to elements two
    faces away (message passing is local, not array-global)."""
    mesh = make_mesh(3, 8)
    model = make_model()
    feats = graph_features(euler_state(mesh, "shock"), mesh)
    jac = np.asarray(jax.jacobian(lambda f: model(f, mesh)[0])(feats))
    # jac: (n_elem, P, channel, n_elem, Nn)
    e = 4
    assert np.any(jac[e, :, :, e + 1, :] != 0.0)     # surface round works
    assert np.any(jac[e, :, :, e - 1, :] != 0.0)
    assert np.all(jac[e, :, :, e + 2, :] == 0.0)     # depth-1 locality
    assert np.all(jac[e, :, :, e - 2, :] == 0.0)


def test_boundary_alpha_locality():
    """alpha_boundary[e] (element e's right face, shared with e+1's left face)
    must react to exactly the two elements it couples -- e and e+1 -- and
    nothing else: z[e,-1] only ever picks up a surface message from e+1 (and
    z[e+1,0] only from e), so this is even tighter than the subcell interfaces
    tested above, which can also reach one element past their own (e-1 or
    e+2) through the WITHIN-element node they pair with."""
    mesh = make_mesh(3, 8)
    model = make_model()
    feats = graph_features(euler_state(mesh, "shock"), mesh)
    jac = np.asarray(jax.jacobian(lambda f: model(f, mesh)[1])(feats))
    # jac: (n_elem,) alpha_boundary, (channel, n_elem, Nn) features
    e = 4
    assert np.any(jac[e, :, e, :] != 0.0)            # reacts to its own element
    assert np.any(jac[e, :, e + 1, :] != 0.0)         # and its right neighbour
    assert np.all(jac[e, :, e - 1, :] == 0.0)         # nothing else
    assert np.all(jac[e, :, e + 2, :] == 0.0)


def test_nonperiodic_boundary_gets_no_wraparound():
    """On a reflective mesh the first/last elements must not exchange face
    messages (the wrap rows are zeroed): alpha[0] is independent of the last
    element's features. alpha_boundary still has shape (n_elem,) (its last
    entry, the non-existent wrap face, is simply never consumed by the
    non-periodic solver path -- see jax_dgsem.solver._apply_surface_and_mass
    _inverse)."""
    basis = GLLBasis(3)
    mesh = Mesh1D(basis, 6, 0.0, 1.0)                # reflective both sides
    model = make_model()
    feats = graph_features(euler_state(mesh, "shock"), mesh)
    alpha, alpha_boundary = model(feats, mesh)
    assert alpha_boundary.shape == (6,)
    jac = np.asarray(jax.jacobian(lambda f: model(f, mesh)[0])(feats))
    assert np.all(jac[0, :, :, -1, :] == 0.0)
    assert np.all(jac[-1, :, :, 0, :] == 0.0)


# ---------------------------------------------------------------------------
# Training contract
# ---------------------------------------------------------------------------

def test_jit_grad_vmap_smoke():
    """The training path: filter_jit + vmap over states + filter_grad w.r.t.
    the params -- finite, non-trivial gradients, no shape errors. Loss uses
    BOTH outputs so alpha_boundary is exercised through the same jit/vmap/grad
    stack as alpha."""
    mesh = make_mesh(4, 8)
    model = make_model()
    Us = jnp.stack([euler_state(mesh, "shock"), euler_state(mesh, "smooth")])

    @eqx.filter_jit
    def loss(m, Ub):
        a, ab = jax.vmap(lambda u: apply_alpha(m, u, mesh, "graph"))(Ub)
        return jnp.mean(a ** 2) + jnp.mean(ab ** 2)

    val, grads = eqx.filter_value_and_grad(loss)(model, Us)
    assert bool(jnp.isfinite(val))
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert all(bool(jnp.all(jnp.isfinite(l))) for l in leaves)
    assert any(bool(jnp.any(l != 0.0)) for l in leaves)


def test_meta_roundtrip_rebuilds_same_tree():
    """model_meta -> build_from_meta reproduces the trained architecture
    (checkpoint loading contract)."""
    from training.config import TrainConfig
    from network.policy import build_from_meta

    cfg = TrainConfig()                              # model_type "graph"
    assert cfg.model_type == "graph"
    tmpl = build_from_meta(model_meta(cfg), jax.random.PRNGKey(0))
    model = build_alpha_model(cfg.model_type, cfg.P, cfg.width,
                              cfg.kernel_size, cfg.depth,
                              jax.random.PRNGKey(1),
                              pno_channels=cfg.pno_channels,
                              pno_hidden=cfg.pno_hidden,
                              fusion_hidden=cfg.fusion_hidden)
    a = jax.tree_util.tree_leaves(eqx.filter(tmpl, eqx.is_array))
    b = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    assert len(a) == len(b)
    assert all(x.shape == y.shape for x, y in zip(a, b))
