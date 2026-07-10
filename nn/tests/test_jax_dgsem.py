"""Isolated tests for the differentiable hybrid DGSEM stack.

Run from the repo root:
    .venv_spectral/bin/python nn/tests/test_jax_dgsem.py

The C++ comparison (the make-or-break check) runs only if build/spectral
exists; it reruns Sod for 200 steps and compares against the JAX solver at
the VTU's ASCII precision.
"""

import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from jax_dgsem import GLLBasis, Mesh1D, time_loop
from jax_dgsem.solver import rk4_step, hybrid_residual, cfl_time_step
from jax_dgsem.physics import (physical_flux, chandrashekar_ec,
                               entropy_stable_lf, logmean, pressure)
from jax_dgsem.indicator import persson_peraire_alpha, modal_energy, \
    postprocess_alpha

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_basis():
    for P in (3, 5, 6):
        b = GLLBasis(P)
        q, w, D = np.asarray(b.quads), np.asarray(b.weights), np.asarray(b.D)
        assert abs(w.sum() - 2.0) < 1e-14
        assert np.allclose(D.sum(1), 0.0, atol=1e-13)
        for deg in range(P + 1):
            du = deg * q ** (deg - 1) if deg > 0 else np.zeros_like(q)
            assert np.allclose(D @ q ** deg, du, atol=1e-10), (P, deg)
    print("test_basis OK")


def test_physics():
    a = jnp.array([1.0, 2.0, 3.0])
    assert np.allclose(logmean(a, a), a)
    assert np.allclose(logmean(jnp.array(1.0), jnp.array(2.0)),
                       1.0 / np.log(2.0))
    U = jnp.array([1.2, 0.3, 2.5])
    assert np.allclose(chandrashekar_ec(U, U), physical_flux(U), atol=1e-12)
    assert np.allclose(entropy_stable_lf(U, U), physical_flux(U), atol=1e-12)
    # logmean gradient is finite at a == b (the jnp.where safe-branch trick)
    g = jax.grad(lambda x: logmean(x, jnp.array(2.0)))(jnp.array(2.0))
    assert np.isfinite(float(g))
    print("test_physics OK")


def test_freestream():
    P, n_elem = 5, 20
    b = GLLBasis(P)
    Uc = jnp.array([1.0, 0.5, 3.0])
    mesh = Mesh1D(b, n_elem, 0.0, 1.0, ("wall", Uc), ("wall", Uc))
    U0 = jnp.broadcast_to(Uc[:, None, None], (3, n_elem, P + 1))
    for alpha in (jnp.zeros(n_elem), jnp.ones(n_elem), jnp.full(n_elem, 0.3)):
        r = hybrid_residual(U0, alpha, mesh)
        assert float(jnp.max(jnp.abs(r))) < 1e-12
    print("test_freestream OK")


def _read_vtu(path):
    txt = open(path).read()

    def arr(name):
        m = re.search(rf'Name="{name}"[^>]*>\s*([^<]+)<', txt)
        return np.fromstring(m.group(1), sep=" ")

    return {k: arr(k) for k in ("rho", "velocity", "pressure", "alpha")}


def test_vs_cpp():
    binary = os.path.join(REPO, "build", "spectral")
    if not os.path.exists(binary):
        print("test_vs_cpp SKIPPED (build/spectral not found)")
        return
    P, N, L, dt, nsteps = 5, 50, 2.0, 5e-5, 200
    subprocess.run(
        [binary, "--case", "sod", "--solver", "hybrid_dgsem", "--P", str(P),
         "--N", str(N), "--dt", str(dt), "--T", str(nsteps * dt),
         "--output", "jaxval", "--verbose", "0"],
        cwd=REPO, check=True, capture_output=True)
    f = _read_vtu(os.path.join(REPO, "results", f"jaxval_{nsteps:06d}.vtu"))

    b = GLLBasis(P)
    UL = np.array([1.0, 0.0, 2.5])
    UR = np.array([0.125, 0.0, 0.25])
    mesh = Mesh1D(b, N, 0.0, L, ("wall", UL), ("wall", UR))
    x = np.asarray(mesh.node_positions(0.0))
    s = 0.5 * (1.0 - np.tanh((x - 0.5 * L) / (2.0 * L / N)))
    rho = 0.125 + (1.0 - 0.125) * s
    p = 0.1 + (1.0 - 0.1) * s
    U0 = jnp.stack([jnp.asarray(rho), jnp.zeros_like(jnp.asarray(rho)),
                    jnp.asarray(p / 0.4)])

    Uf, _ = time_loop(
        U0, lambda U: persson_peraire_alpha(U, mesh, alpha_max=0.5),
        nsteps, dt, mesh, return_trajectory=False)

    for name, mine in (("rho", Uf[0]), ("velocity", Uf[1] / Uf[0]),
                       ("pressure", pressure(Uf))):
        err = np.abs(np.asarray(mine).ravel() - f[name]).max()
        assert err < 5e-6, (name, err)  # VTU is written with 6 sig digits
    a = np.asarray(persson_peraire_alpha(Uf, mesh, alpha_max=0.5))
    assert np.abs(a - f["alpha"].reshape(N, P + 1)[:, 0]).max() < 5e-6
    print("test_vs_cpp OK (matches C++ hybrid DGSEM at VTU precision)")


def test_gradient_flow():
    sys.path.insert(0, os.path.join(REPO, "nn"))
    from network.model import AlphaModel

    P, N = 5, 16
    b = GLLBasis(P)
    Uc = jnp.array([1.0, 0.0, 2.5])
    mesh = Mesh1D(b, N, 0.0, 1.0, ("wall", Uc), ("wall", Uc))
    x = mesh.node_positions(0.0)
    rho = 1.0 + 0.2 * jnp.sin(2 * jnp.pi * x)
    U0 = jnp.stack([rho, 0.1 * rho, 2.5 * jnp.ones_like(rho)])
    dt = cfl_time_step(U0, mesh, 0.3)
    model = AlphaModel(P + 1, key=jax.random.PRNGKey(1))
    # move the head off its zero init so gradients reach every layer
    model = eqx.tree_at(lambda m: m.head.weight, model,
                        0.01 * jnp.ones_like(model.head.weight))

    def loss(model):
        def body(U, _):
            alpha = postprocess_alpha(model(modal_energy(U, mesh).T),
                                      hard_clip=False)
            return rk4_step(U, alpha, dt, mesh), jnp.sum(alpha ** 2)

        Uf, cs = jax.lax.scan(body, U0, None, length=10)
        return jnp.sum((Uf - U0) ** 2) + jnp.sum(cs)

    grads = eqx.filter_grad(loss)(model)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert all(bool(jnp.isfinite(g).all()) for g in leaves)
    nonzero = sum(int(jnp.sum(g != 0)) for g in leaves)
    total = sum(g.size for g in leaves)
    assert nonzero > total // 2, f"only {nonzero}/{total} grads nonzero"
    print(f"test_gradient_flow OK ({nonzero}/{total} nonzero, all finite)")


if __name__ == "__main__":
    test_basis()
    test_physics()
    test_freestream()
    test_gradient_flow()
    test_vs_cpp()
