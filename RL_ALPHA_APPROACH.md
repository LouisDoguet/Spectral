# Reinforcement Learning Approach for Neural Network Alpha Prediction

## Problem Statement

The hybrid DGSEM solver needs to adaptively choose the blending factor `alpha` at each element and timestep to balance:
- **High-order accuracy** in smooth regions (alpha ≈ 0)
- **Stability and shock capture** in discontinuous regions (alpha ≈ 1)

Currently, hybrid DGSEM uses the **Persson-Peraire modal energy indicator** to compute alpha, which is:
- Heuristic-based (not learned from data)
- Expensive to compute (requires Legendre coefficients)
- Not necessarily optimal for all flow types

**Goal:** Train a neural network to predict the optimal alpha directly from the solution state (ρ, ρu, E) without computing modal coefficients.

---

## Reinforcement Learning Framework

### 1. Core Concept

Instead of having a fixed "ground truth" alpha, we treat this as a **policy learning problem**:

- **State:** Local solution (density, momentum, energy) at all nodes
- **Action:** Predicted alpha value(s) for elements
- **Reward:** A metric that measures solution quality

The neural network learns a policy (mapping state → alpha) that maximizes cumulative reward over time.

### 2. Training Data Generation

#### Phase 1: Generate Simulation Trajectories

For each test case (Sod shock tube, etc.):

1. **Run N independent hybrid DGSEM simulations** with different alpha configurations:
   - One with PP indicator (baseline)
   - Several with fixed alpha values: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
   - (Optional) Random/stratified alpha values per element

2. **Record at each timestep:**
   - State vector: [ρ, ρu, E] at all nodes
   - Applied alpha (for that simulation)
   - Computed reward metrics

#### Phase 2: Define Reward Signals

For each timestep `t` and element `e`, compute a reward based on:

**Primary rewards:**
- `R_entropy`: Entropy stability measure
  - Entropy conservative → R = 0
  - Entropy increasing (stable) → R > 0
  - Entropy decreasing (unstable) → R < 0

- `R_gradient`: Smoothness of solution
  - Smooth regions should have low alpha (reward high alpha less)
  - High-gradient regions need blending

- `R_accuracy`: If reference solution available
  - Compare against reference (e.g., exact solution)
  - Error-based reward: R = -|u_computed - u_ref|

**Combined reward:**
```
R(state, alpha_applied) = w_entropy * R_entropy 
                        + w_gradient * R_gradient 
                        + w_accuracy * R_accuracy
```

Where weights (w_entropy, w_gradient, w_accuracy) are tuned.

#### Phase 3: Create Training Pairs

For each recorded trajectory:

```
Input:  [ρ, ρu, E] at nodes (same as before)
Output: alpha_values that were applied
Label:  Reward obtained from that alpha choice
```

### 3. Network Architecture & Training Strategy

#### Network Design
- **Input:** Solution state (density, momentum, energy) at all nodes
- **Hidden layers:** Learn patterns that correlate with optimal alpha
- **Output:** Predicted alpha ∈ [0, 1] for each element

#### Supervised Learning from Trajectories

**Step A: Offline Training**
1. Collect trajectories from simulations with various alpha choices
2. Label each (state, alpha) pair with the reward it achieved
3. Train network to predict: `high_reward_alpha = network(state)`

This is **imitation learning from optimal trajectories**.

**Step B: Refinement (Optional)**
- Use the trained network to run new simulations
- Measure actual performance
- Fine-tune rewards based on real outcomes

### 4. Detailed Training Loop

```
┌─────────────────────────────────────────────────────────┐
│ OFFLINE PHASE: Generate Training Data                   │
├─────────────────────────────────────────────────────────┤
│ For each test case (e.g., Sod shock tube):             │
│   For each alpha_config in [PP, 0.0, 0.2, ..., 1.0]:  │
│     • Run hybrid DGSEM to T_final                       │
│     • At each timestep:                                 │
│       - Record state (ρ, ρu, E)                         │
│       - Compute reward (entropy, gradient, accuracy)    │
│       - Store (state, alpha_config, reward) tuple       │
│ ─────────────────────────────────────────────────────────│
│ Output: Training dataset D = {(state_i, α_i, R_i)}     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ SUPERVISED LEARNING PHASE: Train Network                │
├─────────────────────────────────────────────────────────┤
│ Network learns: α_pred = f_NN(state)                   │
│                                                          │
│ Loss function (one of):                                 │
│   1. MSE: ||α_pred - α_optimal||²                      │
│      where α_optimal = argmax_α R(state, α)            │
│                                                          │
│   2. Regression on reward:                              │
│      Predict both α and expected R(α)                   │
│      Train to maximize predicted reward                 │
│                                                          │
│ Training: Standard SGD/Adam on collected trajectories   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ VALIDATION PHASE: Test Trained Network                  │
├─────────────────────────────────────────────────────────┤
│ Run hybrid DGSEM using NN-predicted alpha:              │
│   For test case not in training set:                    │
│     • At each timestep: α = network(state)              │
│     • Compute residual with NN-chosen alpha             │
│     • Compare metrics to:                               │
│       - Persson-Peraire baseline                        │
│       - Fixed alpha=0.5                                 │
│     • Measure: entropy, error, convergence              │
└─────────────────────────────────────────────────────────┘
```

### 5. Key Components

#### A. Reward Definition

**Entropy Stability Reward:**
```
dS/dt = entropy production rate
R_entropy = max(0, dS/dt) / dS_max   (clip to [0,1])
  - Higher entropy production (stable) → higher reward
  - Entropy decrease (bad) → penalty
```

**Gradient-based Reward:**
```
grad = ||∇ρ|| + ||∇u|| + ||∇p||
R_gradient = {
    +1.0  if (alpha is low AND grad is low)     [smooth region]
    +1.0  if (alpha is high AND grad is high)   [shock region]
    -0.5  otherwise                              [mismatch]
}
```

**Error-based Reward (if reference available):**
```
L2_error = ||u_computed - u_ref||_L2
R_accuracy = exp(-λ * L2_error)  where λ is tuning param
```

#### B. Training Dataset Structure

```
For each sample i:
  state_i:     shape (n_nodes,) or (n_elements,) 
               = [ρ_1, ..., ρ_N, u_1, ..., u_N, E_1, ..., E_N]
  
  alpha_i:     shape (n_elements,)
               = actual alpha values used in simulation i
  
  reward_i:    shape (n_elements,) or scalar
               = computed reward for that timestep
```

#### C. Loss Function

**Option 1: Regression to optimal alpha**
```
Given state s, find which α ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
yielded highest reward, call it α_best(s).

Loss = MSE(network_pred(s), α_best(s))
     = mean[(f_NN(s) - α_best(s))²]
```

**Option 2: Direct reward maximization**
```
Train network to predict expected reward:
r_pred = network(state)
Loss = -mean[r_pred]  (maximize reward)

Or combined:
Loss = MSE(α_pred, α_best) + λ * (-R_predicted)
```

---

## Implementation Roadmap

### Stage 1: Data Collection
1. Modify main.cpp to allow running multiple simulations with different alpha configs
2. Output state + reward pairs at each timestep
3. Accumulate into training dataset file

### Stage 2: Network Training
1. Load dataset
2. Normalize inputs (state vectors)
3. Train network using collected (state, optimal_alpha) pairs
4. Save trained model

### Stage 3: Integration with HybridDGSEM
1. Modify HybridDGSEM to accept optional neural network
2. Replace `computeAlpha()` with neural network prediction when available
3. Validate on test cases

### Stage 4: Validation
1. Compare NN-predicted alpha vs. Persson-Peraire on held-out test cases
2. Measure: entropy stability, error, computational cost
3. Iterate on reward definition if needed

---

## Advantages of This Approach

✅ **Learned from physics:** Rewards based on entropy stability and accuracy  
✅ **Generalizable:** Can train on one problem, test on variations  
✅ **Fast inference:** Neural network much faster than computing Legendre modes  
✅ **Flexible:** Can adjust rewards to emphasize different goals  
✅ **Interpretable:** Alpha values are still in [0,1], same as Persson-Peraire  

---

## Challenges & Considerations

⚠️ **Expensive data generation:** Need to run many hybrid DGSEM simulations  
⚠️ **Reward definition:** Need to carefully tune reward weights  
⚠️ **Overfitting:** Network may overfit to training scenarios  
⚠️ **Generalization:** Performance on unseen flow types needs testing  
⚠️ **Cost vs. benefit:** Training overhead vs. inference speedup  

---

## Alternative: Simplified Version (Faster to implement)

If full RL is too heavy, start with:

1. **Collect one trajectory** (single hybrid DGSEM run with Persson-Peraire)
2. **For each element and timestep,** compute reward:
   - Entropy production rate
   - Solution gradient magnitude
3. **Train network to predict** (state → expected_reward_and_alpha)
4. **Use network predictions** in new hybrid DGSEM runs

This skips the "multiple alpha trials" phase and instead learns from a single reference trajectory, making it faster but potentially less optimal.

---

## Summary

| Step | Action | Input | Output |
|------|--------|-------|--------|
| 1 | Data Collection | Test case + alpha configs | Trajectories with rewards |
| 2 | Training | Trajectories | Trained network |
| 3 | Integration | Network + HybridDGSEM | Hybrid solver with NN alpha |
| 4 | Validation | Test cases | Performance metrics |

