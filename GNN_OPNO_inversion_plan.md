# Implementation plan: invert the sensor — element GNN first, OPNO decoder second

Branch `GNN->OPNO`, cut from the CNN-era code (7c8d8f6). The previous
PNO→GNN point-level sensor lives on `GNN_PNO` and is used here as a reference
implementation only. This plan is the inversion requested: **first** a GNN that
learns how *elements* relate to each other through their features (density
residual + modal spectrum), **then** an OPNO (orthogonal-polynomial neural
operator) that enriches the modal spectrum with the neighbour-aware element
latents and decodes a smooth alpha field.

## 0. Why the inversion can work where PNO→GNN did not

The GNN_PNO diagnostic (see the training analysis on that branch) found the
architecture was never the blocker — training was. Three quantitative failures,
all baked into the new design as constraints:

1. **Backward-exploding input transform.** `log10(e + 1e-12)` had a bounded
   forward value but a ~4e11 backward factor for floor-level mode energies;
   ‖dα/dU‖ reached 6.9e3 (vs 9.9 for the working CNN) and the 512-step BPTT
   gradient hit 1e28–1e33 (finite → not skipped → unit-norm noise updates).
   → Here every log uses `log10(clip(e, E_FLOOR, 1))`: identical forward above
   the floor, exactly zero gradient below it.
2. **Position features drowned state features.** The residual channel was
   normalized by its *global max*, crushing smooth-element signal to ~1e-2 of
   the O(1) positional channels (raw xi + Legendre patterns) — under noisy
   gradients the policy collapsed to a position-keyed alpha comb.
   → Here the residual is encoded with `asinh(res / s)` at *fixed* physical
   scales (no state-dependent normalization at all: bounded gradient, no
   global coupling, magnitude information preserved), and no raw positional
   channel reaches the decoder/readout — position enters only through the
   physical basis Phi and inside the pooled element encoder.
3. **Chaotic-horizon BPTT + unguarded finite explosions.** The CNN tolerates
   the full 512-step gradient (norm 77); a stiffer policy does not.
   → Truncated BPTT (config `bptt_window`, stop-gradient between windows) and
   a train-step guard that *skips* updates whose global gradient norm exceeds
   `grad_skip_norm` (`apply_if_finite` only catches NaN, not finite 1e30).

Also carried over from the diagnostic: **no `alpha_boundary` lever on this
branch.** The solver keeps the validated entropy-stable-LF element interfaces
unconditionally (this branch's solver never had the EC-blend machinery; porting
it back from GNN_PNO stays a possible later fine-tune, after subcell training
is healthy).

## 1. Architecture (new class `OPNOAlphaModel`, model_type `"opno"`)

```
features (3, n_elem, Nn) = [asinh(res/s1), asinh(res/s2), modal energy spectrum]
        (rows 0-1: node axis)                (row 2: MODE axis, order kept)

STAGE A - element encoder (P-independent by token pooling):
  residual tokens  (xi_i, r1_i, r2_i)  -> shared MLP -> mean-pool over nodes ─┐
  spectrum tokens  (k/P, logE_k)       -> shared MLP -> mean-pool over modes ─┴─ concat -> Linear -> h0_e (width,)

STAGE B - element GNN ("how elements relate"):
  depth rounds over the element line graph (jnp.roll neighbours, periodic-aware):
      m_l = relu(MsgL(h)), m_r = relu(MsgR(h))          # direction-aware messages
      h   = relu(h + Upd([h, m_l from e-1, m_r from e+1]))
  Non-periodic ends: the missing neighbour's message is zeroed.

STAGE C - OPNO decoder ("enrich the spectrum with the neighbour view"):
  mode tokens t_{e,k} = [k/P, logE_k, h_e]  -> shared per-mode MLP -> mode feats (n_elem, Nn, C)
  nodal reconstruction  ztilde_{e,i,:} = sum_k Phi[i,k] * modefeat_{e,k,:}   (formula, any P)
  pointwise fuse        z_{e,i} = relu(W [ztilde_{e,i,:}, r1_{e,i}, r2_{e,i}, h_e])

READOUT (same contract as NodalAlphaModel):
  interface latents zi = 0.5 (z_{e,j} + z_{e,j+1})  ->  sigmoid(zi . w + b)  ->  alpha (n_elem, P)
```

Properties:
- **P-independent**: no weight shape depends on Nn (token MLPs shared over
  modes/nodes, pooling, Phi is a mesh constant). Same weights run at any P and
  any n_elem.
- **Smooth alpha by construction**: within an element alpha comes from a
  Legendre expansion of the (nonlinearly fused) mode features; across elements
  the GNN latents vary smoothly with the element neighbourhood — this is the
  "smoothly transmitted alpha from a global view".
- **Subcell localization retained**: the per-node residual channels shortcut
  into the pointwise fuse, so within-element localization does not have to
  survive the element-level pooling.
- Receptive field = `depth` elements each side (PP itself is 0-hop; depth=2
  gives the ±2-element view).

## 2. Files

| file | change |
|---|---|
| `nn/network/model.py` | + `TokenMLP`, `ElementEncoder`, `ElementGraphNet`, `ModalOPNO`, `OPNOAlphaModel` (stable_init identical in spirit to NodalAlphaModel: zero readout weight + logit(alpha_init) bias) |
| `nn/network/policy.py` | + `opno_features` (asinh residual channels with the edge-node artifact fix ported from GNN_PNO, + modal_energy row); + `call_model` / `apply_alpha` routing (opno models take the mesh); build/meta/rebuild for `"opno"` |
| `nn/training/config.py` | model fields (`opno_hidden`, `opno_channels`, `fusion_hidden`; `width`/`depth` reused for the GNN), guards (`bptt_window=64`, `grad_skip_norm=1e3`), `pretrain_epochs=300` + `pretrain_target_mse=1e-2`, default `model_type="opno"` |
| `nn/training/train.py` | route through `apply_alpha`; TBPTT windows in `rollout_cost`; grad-norm skip + per-batch gnorm logging in `train_step`/`History`; comb detector (per-interface-index alpha spread) in diagnostics + epoch log; CLI `--model-type opno`, `--bptt-window` |
| `nn/training/pretrain.py` | route through `call_model`; early stop at `pretrain_target_mse` |
| `nn/training/viz_snapshot.py`, `nn/main.py`, `nn/evaluation/compare.py` | one-line switch to the routing helper |
| `nn/tests/test_opno_model.py` | contracts: shape/range, stable init, node-permutation invariance of the encoder pooling, element-roll equivariance, P- and n_elem-generalisation with the same params, GNN locality (depth-hop influence), clip-floor zero-gradient, jit/grad/vmap smoke, meta round-trip |

Out of scope (deliberate): C++/.nnx export of the opno model (as for the graph
model, deferred until the policy is worth exporting); alpha_boundary.

## 3. Acceptance checks (run after coding)

1. `pytest nn/tests` green.
2. Stiffness: ‖dα/dU‖ (power iteration) at smooth and PP-shocked states for the
   untrained + briefly-pretrained opno model — must sit near the CNN's O(10),
   nowhere near the graph model's O(1e3-1e5).
3. Gradient growth vs rollout length L ∈ {64, 128, 256, 512} with
   `bptt_window=64` — bounded/linear-ish, no super-exponential detonation.
4. Smoke train: `--light --model-type opno` a few epochs — finite losses,
   sane gnorms, snapshot renders, no NaN-batch storm.
