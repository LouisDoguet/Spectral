# Integration spec: replace the CNN+one-hot shock sensor with a GNN + PNO network (JAX)

> **For Claude Code (project mode).** This is a work order, not code. You have the real
> repository; the author of this spec did not — everything below is *design intent*,
> *contracts*, and *shapes*. **You** write the actual JAX/Flax code, adapting names and
> conventions to what already exists in the repo. Framework is **JAX** (Flax `nn.Module`
> for the learnable parts, `jraph` for the graph/message passing, `jax.numpy` throughout).
> **Start with the DISCOVERY step and confirm real conventions before writing anything.**

---

## 0. Discovery (do this first, before writing any code)

Locate and report back on:

1. **The current model file** — the file defining the existing sensor pipeline
   (Preconditioning → OneHot → CNN → ResNet×2 → Pooling). This is where the new
   `PolynomialNeuralOperator`, `DGSEMGraphNet`, and top-level classes will be **added**
   (§2), so identifying it correctly is the whole basis of the integration. Grep for
   `Conv`, `ResNet`, `one_hot`/`onehot`, `alpha`, `nn.Module`. Report the file path, the
   existing Flax module class name(s), the file's conventions (base class, import style,
   docstring/annotation style, dataclass vs NamedTuple for containers), and how the module
   is initialised/applied (`.init` / `.apply`, where params live).
2. **Tensor layout of the two inputs**:
   - density residual — confirm exact shape. Spec assumes `(N_points, 1)`.
   - modal energy — confirm exact shape. Spec assumes per-element `(N_elem, P+1)`.
     If it is currently per-point `(N_points, P+1)`, note it and reconcile (§5).
3. **Element ↔ point mapping** — is there an existing array giving, per integration point,
   its element id, and per element its order `P` and its local point ids? If not, we build
   one (§2, `ElementIndex`). Report what exists.
4. **Dimensionality** — 1D, 2D tensor-product, or 3D? This drives edge construction (§3).
   Spec describes 1D; for 2D/3D the intra-element graph and face coupling generalise by
   tensor product — **flag it and ask the user before implementing multi-D edges.**
5. **Order regime** — is `P` globally fixed, or p-adaptive (varies per element / over
   time)? This decides whether the PNO runs a batched fixed-`P` path or a ragged
   per-element path (§4). In JAX this matters extra: ragged/variable-length axes fight
   `jit`, so if `P` varies you likely need per-order batching or padding-with-masking —
   call this out explicitly (§4, §5).
6. **Reusable transform** — the modal↔nodal (Legendre/Vandermonde) basis used to *compute*
   the modal energy already exists in the solver. Find it. The PNO reuses it; do not
   reimplement Legendre evaluation.
7. **Output contract** — where `alpha` is consumed by the solver, its expected shape and
   range, and whether it must be per-point or per-element (§9).
8. **Autodiff context** — is training solver-in-the-loop (gradients flow through the
   solver into the sensor)? If so, note which graph quantities are constants (no grad) vs
   differentiable (§8, item 4).

Report findings, then proceed.

---

## 1. Design summary (what we are building and why)

The existing sensor is **order-dependent** in two illegitimate ways:
- the **CNN** assumes integration points lie on a fixed array with index-based adjacency;
- the **one-hot** hard-codes each point's array slot.

Neither is physical. Replace them with two size-/permutation-invariant branches:

- **PNO (Polynomial Neural Operator)** — a neural *operator* over the modal energy. Maps a
  modal signal of *any length* `P+1` to (a) a fixed-size global element feature and (b) a
  per-point nodal reconstruction, using **shared weights independent of `P`**.
- **GNN** — message passing on a graph whose edges are the *actual* DGSEM coupling:
  intra-element edges weighted by the differentiation matrix `D`, inter-element edges via
  the face/mortar map. Permutation-equivariant, independent of point count.

They fuse into a **pointwise MLP** emitting `alpha` per integration point.

```
modal_energy ─► PNO ─► pno_global      (N_elem,   C)      ─┐
                └────► nodal_from_pno   (N_points, C)  ─┐   │
residual, xi ──────────────────────────────────────►  GNN ─┤
                                                           ▼
                                        fuse ► pointwise MLP ► alpha (N_points, 1)
```

**Invariances that MUST hold:**
- permute point order within an element → identical output (bit-identical at eval);
- permute element order / change element count → identical per-element output;
- change `P` (per element or globally) → runs with no shape error and no retraining,
  because both branches use shared weights + pooling and the inverse transform is a
  formula, not a learned matrix.

Mode order is **kept** — mode index = frequency, physically meaningful. Do **not** make
the network invariant to permuting the modal vector.

---

## 2. Where the new code goes — classes in the EXISTING model file

**Do not create a new package.** Add the new networks as **classes inside the repo's
existing model file** — the same file that currently defines the sensor (the one found in
discovery §0.1, e.g. `model.py` / `models.py` / whatever the repo calls it). They must sit
alongside the current classes and follow the file's existing conventions exactly: same base
class (Flax `nn.Module` or whatever the existing modules subclass), same import style,
same naming/docstring/type-annotation conventions, same parameter-init pattern, same place
in the file's ordering (define leaf modules before the top-level module that composes them).

**New classes to add to the model file:**

| Class (adapt name to repo style) | Role | Defined in §  |
|----------------------------------|------|---------------|
| `PolynomialNeuralOperator`       | modal energy → global + nodal features (P-independent) | §4 |
| `DGSEMGraphNet`                   | jraph message passing on the DGSEM graph               | §6 |
| `ShockSensorNet` (or reuse/rename the existing sensor class) | top-level: PNO + GNN + fusion + output MLP | §7 |

**Pure-function helpers (not `nn.Module`s)** — the graph/edge construction (§3), the
`ElementIndex` container, and the Legendre-basis wrapper (§4) are *not* learnable and should
**not** be Flax modules. Put them where the repo keeps such utilities: if the model file
already holds free functions / dataclasses, add them there too; otherwise add them to the
existing mesh/operator/utils module that already owns the `D` matrix and the Vandermonde
basis. Prefer colocating them with the operators they reuse rather than duplicating.

**Reuse before creating.** The modal↔nodal (Vandermonde/Legendre) basis already exists in
the solver — the same one used to produce the modal energy. Reuse it directly; only add a
thin wrapper (a helper function, not a class) if shapes don't match §4's needs. Do not
reimplement Legendre evaluation.

**`ElementIndex`** is a small pure-data container (a dataclass / NamedTuple in the repo's
preferred style) holding, for the current mesh: element id per point (`point2elem`, shape
`(N_points,)`), order per element (`elem_order`, shape `(N_elem,)`), and point ids per
element (`elem_points`). Build it from whatever mesh structure already exists
(discovery §0.3). It is a mesh constant — build once, not per timestep.

---

## 3. Graph/edge helpers — edges from the real DGSEM operators

*(Pure functions + `ElementIndex`, added per §2 — not Flax modules.)*

Two edge sets, both derived from operators the solver already has. Describe, don't invent.

**Intra-element edges (volume coupling).** Within each element, connect every pair of
points; the edge weight from point `a` to point `b` is the differentiation-matrix entry
`D[a,b]` for that element's order. This is the exact operator DGSEM uses to differentiate
inside an element, so the graph carries the true internal coupling rather than array
adjacency. Output: an `edge_index` of shape `(2, E_intra)` and a matching `edge_attr` of
shape `(E_intra, 1)` holding the `D` entries. **Reuse the solver's `D`; do not recompute.**

**Inter-element edges (surface coupling).** Across each element face, connect the touching
boundary points (both directions). For a conforming 1D mesh these are (last point of
element `e`) ↔ (first point of element `e+1`), weight `1`. For non-conforming / mortar
interfaces, the weight is the mortar interpolation coefficient, and the pairing comes from
the solver's face/mortar connectivity — **get it from there, never from array order.**
Output: `edge_index` `(2, E_inter)`, `edge_attr` `(E_inter, 1)`.

**JAX note.** Build these as static `jnp` arrays once per mesh/`P` change and close over
them (or pass as non-traced args) so `jit` sees fixed shapes. If `E_intra`/`E_inter` change
when `P` or the mesh changes, that triggers recompilation — acceptable if it happens rarely
(mesh/order change), not per step. `jraph` can also pad graphs to fixed size
(`jraph.pad_with_graphs`) with a mask if you need a single compiled shape across meshes —
recommend this if the mesh/order changes frequently during training.

---

## 4. `PolynomialNeuralOperator` class — Polynomial Neural Operator (P-independent)

*(New Flax module class in the existing model file, per §2.)*

**Purpose.** Turn the raw modal energy vector (a hand-computed per-mode statistic) into
learned features, in a way that accepts any number of modes `P+1`.

**The P-independence mechanism (this is the core idea — implement exactly this):**
- Attach a **normalised frequency coordinate** to each mode: `coord_k = k / P` for
  `k = 0..P`. This is the modal-space analogue of the point coordinate `xi` that replaced
  the one-hot in physical space.
- Form one **token per mode**: `(coord_k, energy_k)` → a length-2 vector.
- Apply a **shared per-mode MLP** (same learned weights to every mode of every element),
  mapping `2 → hidden → channels`. Because the same weights hit each mode independently,
  the parameter count is fixed no matter how many modes exist.
- **Pool over the mode axis** (mean, or attention if you want to get fancy later) to get
  the fixed-size **global element feature**.

**Two outputs.** After the per-mode MLP produces `mode_feat` of shape `(P+1, channels)`
for an element:
1. **Global feature** — pool `mode_feat` over modes → `(channels,)` per element. Stacked
   over the mesh: `pno_global` of shape `(N_elem, channels)`. Meaning: "how energetic /
   rough is this whole element," fixed size regardless of `P`.
2. **Nodal reconstruction** — multiply the element's Legendre basis matrix
   `Phi` (shape `(n_points, P+1)`, `Phi[i,k] = L_k(xi_i)`, from the reused Legendre basis) by
   `mode_feat` → `(n_points, channels)` per element. Stacked over the mesh:
   `nodal_from_pno` of shape `(N_points, channels)`. Meaning: the mode content projected
   back onto the physical integration points, one feature vector per point.

**Why it's an operator, not a layer.** The whole map — variable-length modal signal in,
variable-length nodal field out, same learned weights either way — is a function between
function spaces. The learned part (per-mode MLP) is P-independent by construction; the
inverse transform (`Phi @ mode_feat`) is a *formula* evaluated at whatever `xi` and however
many modes the element has, so it adapts to any `P` with no learned weights involved. Only
the per-mode MLP is learnable; `Phi` is fixed.

**Shape summary (per element, then stacked):**

| quantity          | shape                | notes                                  |
|-------------------|----------------------|----------------------------------------|
| `energy`          | `(P+1,)`             | mode-ordered, kept ordered             |
| `coord`           | `(P+1,)`             | `k/P`, the P-independence trick        |
| tokens            | `(P+1, 2)`           | `(coord, energy)` per mode             |
| `mode_feat`       | `(P+1, channels)`    | shared MLP output                      |
| global (pooled)   | `(channels,)`        | → stack → `pno_global (N_elem, C)`     |
| `Phi`             | `(n_points, P+1)`    | reuse solver Vandermonde               |
| nodal recon       | `(n_points, C)`      | → concat → `nodal_from_pno (N_points,C)`|

**Fixed-P shortcut.** If discovery §0.5 says `P` is globally fixed, skip the ragged
per-element path: stack energy to `(N_elem, P+1)`, build one `coord`, run the shared MLP on
the flattened `(N_elem·(P+1), 2)` — identical math, `jit`-friendly, faster. Keep the ragged
/ padded-with-mask path only if `P` genuinely varies (and see §5 for the padding approach
that keeps it compilable in JAX).

---

## 5. Reconciling the modal-energy shape (and JAX raggedness)

Spec assumes modal energy is **per element** `(N_elem, P+1)`. If the repo carries it
**per point** `(N_points, P+1)` (duplicated across an element's points), collapse to
per-element first (take the element's first row / mean; assert rows within an element match,
to catch silent bugs). Report which layout the repo uses.

**If `P` varies per element (p-adaptivity):** `jax.jit` dislikes ragged axes. Prefer one of:
- **Pad to `P_max` + mask** — store energy as `(N_elem, P_max+1)` with a boolean mask;
  the per-mode MLP runs on all slots, and pooling ignores masked modes. Keeps a single
  compiled shape. `coord_k = k / P_elem` still uses each element's *true* `P`.
- **Per-order batching** — group elements by `P`, run each group through the same shared
  weights, concatenate results. No padding, but multiple compiled shapes (one per distinct
  `P`). Fine if the set of orders is small and stable.
Recommend padding+mask as the default; note the choice in the code and to the user.

---

## 6. `DGSEMGraphNet` class — message passing on the DGSEM graph

*(New Flax module class in the existing model file, per §2.)*

**Purpose.** Let each point exchange information with the points it's actually coupled to —
inside its element (volume) and across faces to neighbouring elements (surface) — instead
of with array-adjacent slots.

**Node input features.** Per point, concatenate: density residual (`1`), reference
coordinate `xi` (`1`, replacing the one-hot), and the PNO nodal reconstruction
(`channels`). Total node feature width `= 2 + channels`. Stacked: `(N_points, 2+channels)`.

**Two message-passing rounds, mirroring the DG operator:**
1. **Volume round** — messages only along **intra-element** edges, weighted by `D`. This is
   the graph analogue of `D` acting within the element.
2. **Surface round** — messages only along **inter-element** edges. This is the graph
   analogue of the numerical-flux exchange across faces, and it is the piece the old
   per-element CNN could not represent at all.

Each round: a message function combining the neighbour's features with the edge weight,
sum-aggregation at the receiver, then an update function. Make the block **residual** (add
the round's input back to its output) to match the original ResNet×2 depth and stabilise
training. Implement with `jraph` (e.g. a `GraphNetwork` / `GraphConvolution`-style update),
using `edge_attr` to carry the `D` / mortar weight into the message.

**Output.** Updated per-point embedding, shape `(N_points, gnn_hidden)`.

**JAX note.** `jraph` expects a `GraphsTuple` (`nodes`, `edges`, `senders`, `receivers`,
plus `n_node`/`n_edge`). Assemble it from the §3 graph-helper outputs. Segment-sum aggregation
(`jax.ops.segment_sum` / jraph's built-ins) is permutation-invariant, which is exactly the
property we want — do not introduce any order-dependent op (no reshape-to-grid, no conv).

---

## 7. `ShockSensorNet` class — top-level wiring

*(New Flax module class in the existing model file — or rename/replace the current sensor
class in place, keeping its public apply signature, per §2.)*

**Purpose.** Drop-in replacement for the CNN+one-hot sensor, same external contract.

**Forward flow:**
1. Apply the **existing preconditioning** module (reuse it — do not drop it) to `residual`
   (and to `energy` before the PNO if the current code preconditions the modal input).
2. **PNO branch** → `pno_global (N_elem, C)` and `nodal_from_pno (N_points, C)`.
3. Build **node features** = concat(`residual`, `xi`, `nodal_from_pno`) → `(N_points, 2+C)`.
4. **GNN branch** on those node features + the two edge sets → `(N_points, gnn_hidden)`.
5. **Fusion** — broadcast `pno_global` to points via `point2elem` (gather:
   `pno_global[point2elem]` → `(N_points, C)`), concat with the GNN embedding →
   `(N_points, gnn_hidden + C)`.
6. **Output MLP** — pointwise (shared weights per point), `(gnn_hidden+C) → 32 → 1`,
   sigmoid → `alpha (N_points, 1)` in `[0,1]`.

**Contract to preserve.** Keep the same public apply signature the solver already calls so
downstream code is untouched — same method name, same returned `alpha` shape/range. The
learnable submodules (preconditioning, PNO MLP, GNN, output MLP) all live under one parent
Flax module so a single `params` pytree covers the whole sensor.

---

## 8. Impacted files — integration checklist

Work through these against the real repo (confirm names in discovery):

1. **[edit — primary] the existing model file** — add `PolynomialNeuralOperator`,
   `DGSEMGraphNet`, and the top-level `ShockSensorNet` as new classes here, alongside the
   current sensor classes and matching the file's conventions (§2). If the current sensor
   class name is the natural home for the top-level module, rename/replace it in place
   rather than adding a parallel class, keeping its public apply signature. **No new
   package/files for the networks.**
2. **[edit] operator/mesh/utils module** — add the graph/edge helper functions and
   `ElementIndex` (§3) here (or in the model file if that's where free functions already
   live), colocated with the existing `D` matrix and Vandermonde basis they reuse. Add the
   Legendre-basis wrapper only if no reusable one exists.
3. **[edit] model construction / factory** — wherever the old sensor is built, swap in the
   new top-level class. Keep the public apply name so the solver call site is unchanged.
4. **[edit] input assembly** — the code building the CNN input and the one-hot. Replace
   with: assemble `residual`, `xi` (from the solver's LGL node array — reuse it),
   per-element `energy`, the basis matrices, and build the graph once via the §3 helpers.
   **Delete the one-hot construction.**
5. **[edit] graph caching + autodiff** — build `ElementIndex` + edges once per mesh/`P`
   change and close over them / pass as static args; keep them on-device. Edge indices are
   constants (never need grad). Decide whether `D` edge weights are frozen or differentiable
   in solver-in-the-loop training (discovery §0.8) and mark them `stop_gradient` if frozen.
6. **[edit] training script** — output shape is unchanged `(N_points, 1)`, so the loss
   should slot in. Confirm the label pipeline still matches. If the old pooling produced
   per-element `alpha`, decide per-point (recommended, drop pooling) vs a final element pool
   (§9). Ensure the new `params` pytree replaces the old one in the optimiser state.
7. **[edit] config** — add `pno_channels`, `gnn_hidden`, `residual`, and the `P`-handling
   mode (fixed / padded+mask / per-order). Remove CNN/one-hot hyperparameters (kernel size,
   one-hot length, pooling type).
8. **[remove]** the CNN, one-hot, and pooling code paths from the model file once tests
   pass. Optionally keep behind a config flag during transition for an A/B baseline.
9. **[add] dependency** — `jraph` (and confirm `flax`, `optax` already present) in the
   environment file.

---

## 9. Output contract decision

The original ends with **Pooling** → possibly one `alpha` per element. The new design
naturally produces **per-point** `alpha`. Prefer per-point and **remove pooling** — it is
the exact step that would discard the neighbour information the GNN just built. Only add a
final element-wise pool (mean over each element's points) if the solver strictly requires
one `alpha` per element; if so, document it.

---

## 10. Tests to add (these encode the whole point of the redesign)

1. **Point-permutation invariance** — shuffle point order within an element (and the
   matching rows of residual / xi / edges); assert output `alpha` is bit-identical
   (`jnp.allclose`, tight tol). *This is the real test that order-dependence is gone.*
2. **Element-permutation / count invariance** — reorder elements; assert per-element
   outputs unchanged.
3. **P-generalisation** — run one element at `P=3` and one at `P=8` through the same
   instantiated params with no shape error; assert `pno_global` width is identical
   (`channels`) for both.
4. **Shape contract** — `alpha.shape == (N_points, 1)`, values in `[0,1]`.
5. **Physical sanity** — smooth field → `alpha ≈ 0`; injected discontinuity → `alpha`
   elevated in the troubled element AND slightly elevated at the shared face of the
   neighbour (evidence the surface round works).
6. **jit/grad smoke test** — `jax.jit` the apply fn and take a `jax.grad` w.r.t. params on
   a dummy batch; assert no shape/recompilation errors and finite gradients.

---

## 11. Open questions to resolve with the user before/while implementing

- **1D vs 2D/3D** — drives §3 edge construction; do not guess for multi-D.
- **`P` fixed vs p-adaptive** — batched PNO path vs padded+mask / per-order (§4, §5).
- **`alpha` per-point vs per-element** — output contract (§9).
- **Solver-in-the-loop?** — whether `D` edge weights carry gradients (§8.4).
- **Which pooling for the PNO global feature** — mean to start; attention later if needed.
