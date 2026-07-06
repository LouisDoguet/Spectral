/**
 * @file train_hybrid_alpha.cpp
 * @brief Train a policy network to choose the hybrid-DGSEM blending factor alpha
 *        by one-step-lookahead reward maximization (RL / policy iteration).
 *
 * This is the data-generation + training pipeline described in
 * RL_ALPHA_APPROACH.md, specialized to what the hybrid-DGSEM residual actually
 * exposes. Key structural fact exploited here: in computeHybridResidual() the
 * inter-element interface fluxes are ALWAYS the entropy-stable flux, so the
 * residual of element e depends only on alpha[e]. Therefore a single residual
 * evaluation at a candidate alpha value gives, for every element simultaneously,
 * the one-step update it would experience at that alpha.
 *
 * Pipeline:
 *   1. Run a baseline hybrid-DGSEM trajectory (Persson-Peraire alpha) on the
 *      Sod shock tube.
 *   2. At sampled timesteps, for each candidate alpha in {0.0, 0.1, ..., 1.0}:
 *        - evaluate the hybrid residual,
 *        - form the forward-Euler update U' = U - dt*R for every element,
 *        - score each element with a reward:
 *            R = -(new-extrema overshoot) - lambda * alpha   (with a hard
 *                penalty for states that lose positivity of rho or p).
 *      The argmax-over-alpha is the optimal blending label for that element.
 *   3. Train a per-element network (standardized density profile -> alpha).
 *   4. Save the model (.nn).
 *
 * Rationale for the reward:
 *   - Smooth regions: every candidate keeps the solution monotone (overshoot=0),
 *     so the lambda*alpha cost selects alpha=0 -> full high-order accuracy.
 *   - Shock regions: low alpha creates new extrema (Gibbs), large overshoot;
 *     the smallest alpha that suppresses the oscillation wins -> just enough
 *     first-order dissipation. This is exactly the desired blending policy.
 */

#include "../lib/test_cases.h"
#include "../lib/base/base.h"
#include "../lib/time/hybrid_alpha.h"
#include "../lib/space/mesh.h"
#include "../lib/time/hybrid_solver.h"

#include "activation.h"
#include "container.h"
#include "layer.h"
#include "loss_function.h"
#include "network.h"
#include "tensor.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

static constexpr double GAM = 1.4;

int main(int argc, char *argv[]) {
  // ---------------- Configuration ----------------
  const int P = 6;
  const int N = 50;
  const double L = 2.0;
  const double T_final = 0.4;
  const double dt = 1e-4;

  const int EPOCHS = 40000;
  const double LEARNING_RATE = 1e-3;
  const int SAMPLE_STRIDE = 25;   // capture a training snapshot every N steps
  double LAMBDA = 0.02;     // dissipation cost (prefer smaller alpha)
  if (argc > 2)
      LAMBDA = std::stod(argv[2]);

  std::string model_name = "alpha_hybrid.nn";
  if (argc > 1)
    model_name = argv[1];

  // Candidate alpha grid (the discrete action space).
  std::vector<double> cand;
  for (int k = 0; k <= 10; ++k)
    cand.push_back(0.1 * k);

  std::cout << "\n======================================================\n";
  std::cout << "HYBRID-DGSEM ALPHA POLICY TRAINING (one-step reward RL)\n";
  std::cout << "======================================================\n\n";
  std::cout << "  P=" << P << "  N=" << N << "  T=" << T_final << "  dt=" << dt
            << "\n  sample stride=" << SAMPLE_STRIDE << "  lambda=" << LAMBDA
            << "\n  candidates: " << cand.size() << " values in [0,1]\n"
            << "  output model: " << model_name << "\n\n";

  // ---------------- Build Sod shock tube + baseline solver ----------------
  base::Lagrange basis(P);
  // Sod: left (1,0,1), right (0.125,0,0.1), interface at L/2.
  mesh::Mesh *M = S1D::generateMesh(&basis, N, L, 1.0, 0.0, 1.0, 0.125, 0.0, 0.1,
                                    0.5 * L, -1.0);

  solver::HybridDGSEM S(M, P + 1);
  S.setIndicatorParams(/*alpha_max=*/0.5, 
                       /*alpha_min=*/0.001, 
                       /*diffuse=*/true);
  S.setVerbosity(0);

  const int n = P + 1;       // nodes per element
  const int n_elem = N;
  const int n_steps = static_cast<int>(std::ceil(T_final / dt));

  std::vector<double> X; // flattened features (n per sample)
  std::vector<double> Y; // labels (1 per sample)
  std::vector<double> alpha_all(n_elem, 0.0);
  std::vector<double> feat;

  // ---------------- Trajectory + reward sweep ----------------
  std::cout << "STEP 1: Generating trajectory and optimal-alpha labels...\n";
  int n_snapshots = 0;

  for (int step = 0; step <= n_steps; ++step) {
    if (step > 0 && step % SAMPLE_STRIDE == 0) {
      // `reward[e][c]` for current state (U unchanged by residual evaluation).
      std::vector<std::vector<double>> reward(
          n_elem, std::vector<double>(cand.size(), -1e18));

      for (size_t c = 0; c < cand.size(); ++c) {
        std::fill(alpha_all.begin(), alpha_all.end(), cand[c]);
        M->computeHybridResidual(alpha_all.data());

        for (int e = 0; e < n_elem; ++e) {
          const elm::Element *E = M->getElem(e);
          const double *rho = E->getU1();
          const double *rhou = E->getU2();
          const double *en = E->getU3();
          const double *dr = E->getDivF1();
          const double *dm = E->getDivF2();
          const double *de = E->getDivF3();

          // Admissible density band from this element + neighbour boundary
          // nodes (the local stencil that should bound the update).
          double mmin = 1e18, mmax = -1e18;
          for (int j = 0; j < n; ++j) {
            mmin = std::min(mmin, rho[j]);
            mmax = std::max(mmax, rho[j]);
          }
          if (e > 0) {
            double v = M->getElem(e - 1)->getRho(n - 1);
            mmin = std::min(mmin, v);
            mmax = std::max(mmax, v);
          }
          if (e < n_elem - 1) {
            double v = M->getElem(e + 1)->getRho(0);
            mmin = std::min(mmin, v);
            mmax = std::max(mmax, v);
          }

          bool bad = false;
          double overshoot = 0.0;
          for (int j = 0; j < n; ++j) {
            double rp = rho[j] - dt * dr[j];
            double mp = rhou[j] - dt * dm[j];
            double ep = en[j] - dt * de[j];
            double pp = (GAM - 1.0) * (ep - 0.5 * mp * mp / rp);
            if (!std::isfinite(rp) || !std::isfinite(pp) || rp <= 1e-10 ||
                pp <= 1e-10) {
              bad = true;
              break;
            }
            if (rp > mmax)
              overshoot += rp - mmax;
            if (rp < mmin)
              overshoot += mmin - rp;
          }

          double scale = (mmax - mmin > 1e-12) ? (mmax - mmin) : 1.0;
          reward[e][c] = bad ? -1e9
                             : -(overshoot / scale) - LAMBDA * cand[c];
        }
      }

      // argmax over candidates -> optimal alpha label; features from density.
      for (int e = 0; e < n_elem; ++e) {
        int best = 0;
        for (size_t c = 1; c < cand.size(); ++c)
          if (reward[e][c] > reward[e][best])
            best = static_cast<int>(c);

        solver::modal_energy_features(M->getElem(e), n, feat);
        X.insert(X.end(), feat.begin(), feat.end());
        Y.push_back(cand[best]);
      }
      ++n_snapshots;
    }

    if (step < n_steps)
      S.step(dt);
  }

  const int n_samples = static_cast<int>(Y.size());
  std::cout << "  Captured " << n_snapshots << " snapshots -> " << n_samples
            << " training samples (" << n << " features each)\n\n";

  // Quick label distribution sanity check.
  {
    std::vector<int> hist(cand.size(), 0);
    for (double y : Y) {
      int idx = static_cast<int>(std::round(y * 10.0));
      idx = std::max(0, std::min((int)cand.size() - 1, idx));
      hist[idx]++;
    }
    std::cout << "  Label histogram (alpha -> count):\n";
    for (size_t c = 0; c < cand.size(); ++c)
      std::cout << "    " << cand[c] << " : " << hist[c] << "\n";
    std::cout << "\n";
  }

  // ---------------- Train per-element policy network ----------------
  std::cout << "STEP 2: Training policy network...\n";
  TENSOR::Tensor input(n_samples, n);
  TENSOR::Tensor target(n_samples, 1);
  input.setData(X);
  target.setData(Y);

  auto arch = std::make_shared<CONT::Sequential>();
  arch->add(std::make_shared<LAYER::ReLU>(n, 32));
  arch->add(std::make_shared<LAYER::ReLU>(32, 32));
  arch->add(std::make_shared<LAYER::Sigmoid>(32, 1));

  auto loss = std::make_shared<LFUN::MSE>();
  Network net(arch, loss, LEARNING_RATE);
  std::cout << "  Architecture: ReLU(" << n << "->32) -> ReLU(32->32) -> "
            << "Sigmoid(32->1)\n\n";

  net.train(input, target, EPOCHS);

  // ---------------- Save ----------------
  net.save(model_name);
  std::cout << "\n Model saved to: " << model_name << "\n";
  std::cout << "======================================================\n";

  delete M;
  return 0;
}
