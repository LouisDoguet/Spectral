"""Contract tests for the element-GNN -> OPNO alpha policy (model_type "opno").

What is pinned down here (and why it matters after the GNN_PNO post-mortem):
  - the NodalAlphaModel output contract (shape/range/stable init) is honoured;
  - P- and n_elem-independence: the SAME weights run at any order and any
    element count with no shape error;
  - the physical symmetries: node-permutation invariance of the pooled element
    encoder, element-roll equivariance on a periodic mesh, and GNN locality
    (an element's alpha can only see `depth` neighbours per side);
  - the bounded-backward energy transform: exactly zero gradient below the
    clip floor (the GNN_PNO log10(e + 1e-12) had a ~4e11 backward factor
    there, the root of the exploding-BPTT failure);
  - jit/grad/vmap and the meta -> rebuild -> load round-trip used by every
    evaluation script.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from jax_dgsem import GLLBasis, Mesh1D
from network.model import (ENERGY_FLOOR, RES_SCALES, OPNOAlphaModel,
                           log_energy, load_model, save_model)
from network.policy import (apply_alpha, build_alpha_model, build_from_meta,
                            model_meta, opno_features)

PERIODIC = ("periodic", None)


def make_mesh(P=6, n_elem=12):
    return Mesh1D(GLLBasis(P), n_elem, 0.0, 1.0, PERIODIC, PERIODIC)


def smooth_state(mesh):
    """A smooth positive Euler state on the mesh's nodes."""
    x = mesh.node_positions(0.0)
    rho = 1.0 + 0.3 * jnp.sin(2.0 * jnp.pi * x)
    u = 0.2 * jnp.cos(2.0 * jnp.pi * x)
    p = 1.0 + 0.2 * jnp.cos(4.0 * jnp.pi * x)
    E = p / 0.4 + 0.5 * rho * u * u
    return jnp.stack([rho, rho * u, E])


def make_model(depth=2, key=0, **kw):
    return OPNOAlphaModel(len(RES_SCALES), width=16, opno_hidden=12,
                          opno_channels=6, fusion_hidden=12, depth=depth,
                          key=jax.random.PRNGKey(key), stable_init=False, **kw)


def test_shape_and_range_contract():
    mesh = make_mesh(P=6, n_elem=12)
    model = make_model()
    alpha = apply_alpha(model, smooth_state(mesh), mesh, "opno")
    assert alpha.shape == (12, 6)                  # (n_elem, P)
    assert bool(jnp.all((alpha > 0.0) & (alpha < 1.0)))
    assert bool(jnp.all(jnp.isfinite(alpha)))


def test_stable_init_outputs_alpha_init():
    mesh = make_mesh()
    a0 = 1e-3
    model = OPNOAlphaModel(len(RES_SCALES), depth=2,
                           key=jax.random.PRNGKey(0), alpha_init=a0,
                           stable_init=True)
    alpha = apply_alpha(model, smooth_state(mesh), mesh, "opno")
    assert np.allclose(np.asarray(alpha), a0, atol=1e-9)


def test_encoder_node_permutation_invariance():
    """Permuting the node order (residual tokens + their xi coordinates
    together) must leave the pooled element latents bit-unchanged; the mode
    row is NOT permuted (mode order = frequency is meaningful)."""
    mesh = make_mesh(P=6, n_elem=8)
    model = make_model()
    feats = opno_features(smooth_state(mesh), mesh)
    n_res = len(RES_SCALES)

    perm = np.array([3, 0, 6, 1, 5, 2, 4])         # Nn = 7
    feats_p = feats.at[:n_res].set(feats[:n_res][:, :, perm])
    mesh_p = eqx.tree_at(lambda m: m.quads, mesh, mesh.quads[perm])

    h = model.element_latents(feats, mesh)
    h_p = model.element_latents(feats_p, mesh_p)
    assert np.allclose(np.asarray(h), np.asarray(h_p), atol=1e-12)


def test_element_roll_equivariance():
    """Rolling the elements of a periodic mesh rolls the alpha field."""
    mesh = make_mesh(P=5, n_elem=10)
    model = make_model()
    feats = opno_features(smooth_state(mesh), mesh)
    a = model(feats, mesh)
    a_roll = model(jnp.roll(feats, 3, axis=1), mesh)
    assert np.allclose(np.asarray(a_roll), np.asarray(jnp.roll(a, 3, axis=0)),
                       atol=1e-10)


def test_p_generalisation_same_params():
    """The same weights run at P=4 and P=8 (no weight shape depends on Nn)."""
    model = make_model()
    for P in (4, 8):
        mesh = make_mesh(P=P, n_elem=6)
        alpha = apply_alpha(model, smooth_state(mesh), mesh, "opno")
        assert alpha.shape == (6, P)
        assert bool(jnp.all(jnp.isfinite(alpha)))


def test_n_elem_generalisation_same_params():
    model = make_model()
    for n in (4, 24):
        mesh = make_mesh(P=6, n_elem=n)
        alpha = apply_alpha(model, smooth_state(mesh), mesh, "opno")
        assert alpha.shape == (n, 6)


def test_gnn_locality():
    """Perturbing one element's features changes alpha only within `depth`
    elements per side (the GNN is the ONLY inter-element pathway)."""
    depth, j0, n_elem = 2, 8, 16
    mesh = make_mesh(P=6, n_elem=n_elem)
    model = make_model(depth=depth)
    feats = opno_features(smooth_state(mesh), mesh)
    feats_pert = feats.at[0, j0, :].add(0.7)       # residual channel of elem j0

    d = np.abs(np.asarray(model(feats_pert, mesh)) - np.asarray(model(feats, mesh)))
    per_elem = d.max(axis=1)
    dist = np.minimum(np.abs(np.arange(n_elem) - j0),
                      n_elem - np.abs(np.arange(n_elem) - j0))
    assert per_elem[j0] > 1e-7                     # the perturbed element reacts
    assert np.all(per_elem[dist > depth] == 0.0)   # nothing beyond depth hops


def test_energy_floor_zero_gradient():
    """log_energy must be flat (zero gradient) below the clip floor and
    responsive above it -- the bounded-backward fix for the GNN_PNO
    log10(e + 1e-12) explosion."""
    g = jax.grad(lambda e: jnp.sum(log_energy(e)))
    below = g(jnp.array([0.0, 1e-14, ENERGY_FLOOR * 0.5]))
    above = g(jnp.array([ENERGY_FLOOR * 10.0, 1e-3, 0.5]))
    assert np.all(np.asarray(below) == 0.0)
    assert np.all(np.asarray(above) > 0.0)
    # backward factor stays modest right above the floor
    assert float(g(jnp.array([2e-6]))[0]) < 1e5


def test_jit_grad_vmap_smoke():
    mesh = make_mesh(P=6, n_elem=8)
    model = make_model()
    feats = opno_features(smooth_state(mesh), mesh)

    alpha_jit = eqx.filter_jit(lambda m, f: m(f, mesh))(model, feats)
    assert bool(jnp.all(jnp.isfinite(alpha_jit)))

    grads = eqx.filter_grad(lambda m: jnp.sum(m(feats, mesh) ** 2))(model)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert all(bool(jnp.isfinite(g).all()) for g in leaves)
    assert any(float(jnp.max(jnp.abs(g))) > 0 for g in leaves)

    batch = jnp.stack([feats, feats * 1.1])
    ab = jax.vmap(lambda f: model(f, mesh))(batch)
    assert ab.shape == (2, 8, 6)


def test_meta_roundtrip_rebuilds_same_tree(tmp_path):
    """model_meta -> build_from_meta must rebuild a template that loads a
    saved opno checkpoint and reproduces its outputs exactly."""
    from training.config import TrainConfig
    cfg = TrainConfig(model_type="opno", width=16, depth=2, opno_hidden=12,
                      opno_channels=6, fusion_hidden=12)
    key = jax.random.split(jax.random.PRNGKey(cfg.seed))[1]
    model = build_alpha_model(cfg.model_type, cfg.P, cfg.width,
                              cfg.kernel_size, cfg.depth, key,
                              opno_hidden=cfg.opno_hidden,
                              opno_channels=cfg.opno_channels,
                              fusion_hidden=cfg.fusion_hidden)
    meta = model_meta(cfg)
    assert meta["model_type"] == "opno"
    assert meta["data_channels"] == ["residual_asinh", "menergy"]

    path = str(tmp_path / "model.eqx")
    save_model(path, model)
    rebuilt = load_model(path, build_from_meta(meta, jax.random.PRNGKey(99)))

    mesh = make_mesh(P=cfg.P, n_elem=8)
    feats = opno_features(smooth_state(mesh), mesh)
    assert np.allclose(np.asarray(model(feats, mesh)),
                       np.asarray(rebuilt(feats, mesh)), atol=0.0)
