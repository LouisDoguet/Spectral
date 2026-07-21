#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "../phy/entropy_flux.h"
#include "../phy/physics.h"
#include "mesh.h"

namespace mesh {


/// @brief Physical 1D Euler flux `F(U)` at a single node.
///
/// Used for the subcell-flux endpoints and the entropy-stable surface lifts.
inline void physicalFlux(double rho, double rhou, double en, double &f1,
                         double &f2, double &f3) {
  double p;
  phy::getP(&p, &rho, &rhou, &en, 1);
  const double u = rhou / rho;
  f1 = rhou;             // rho * u
  f2 = rhou * u + p;     // rho * u^2 + p
  f3 = u * (en + p);     // u * (E + p)
}

Mesh::Mesh(const int n, base::_Basis *basis, double xL, double xR) : n(n) {
  /// Basis + space definition
  primary_basis = basis;
  double dx_mesh = xR - xL;
  double dx = (double)dx_mesh / (n);

  /// Definition of global arrays
  int nquads = basis->getOrder() + 1;
  global_rho = new double[n * nquads];
  global_rhou = new double[n * nquads];
  global_e = new double[n * nquads];
  global_AV = new double[n * nquads];

  /// Element creation
  this->elem = new elm::Element *[n];

  /// Mesh generation
  /// Arrays attributed to the corresponding elements
  double x_iter = xL;
  for (int e = 0; e < n; ++e) {
    elem[e] =
        new elm::Element(e, basis, x_iter, x_iter + dx,
                          &global_rho[e * nquads], &global_rhou[e * nquads],
                          &global_e[e * nquads], &global_AV[e * nquads]);
    elem[e]->setFlux();
    x_iter += dx;
  }
}

Mesh::Mesh(const int n, base::_Basis *basis, double xL, double xR,
           double *init_u1, double *init_u2, double *init_u3, double u1_L,
           double u2_L, double u3_L, double u1_R, double u2_R, double u3_R)
    : n(n) {
  primary_basis = basis;

  // Default boundaries: fixed-state (Dirichlet) walls holding the prescribed
  // ghost states. Cases needing another type (e.g. reflecting walls) override
  // these via setLeftBC / setRightBC after construction.
  bc_left = new bc::Wall(u1_L, u2_L, u3_L);
  bc_right = new bc::Wall(u1_R, u2_R, u3_R);

  double dx_mesh = xR - xL;
  double dx = (double)dx_mesh / (n);

  /// Global buffer
  int nquads = basis->getOrder() + 1;
  global_rho = new double[n * nquads];
  global_rhou = new double[n * nquads];
  global_e = new double[n * nquads];
  global_AV = new double[n * nquads];

  /// Pointers to the elements
  this->elem = new elm::Element *[n];

  double x_iter = xL;
  for (int e = 0; e < n; ++e) {
    double xL_elem = xL + e * dx;
    double xR_elem = xL + (e + 1) * dx;
    /// Assigns initial conditions to the global buffer
    /// Automatic setup of the viscosity at 0.
    for (int q = 0; q < nquads; q++) {
      global_rho[e * nquads + q] = init_u1[e * nquads + q];
      global_rhou[e * nquads + q] = init_u2[e * nquads + q];
      global_e[e * nquads + q] = init_u3[e * nquads + q];
      global_AV[e * nquads + q] = 0.;
    }

    /// Construct the element e with for values it's position in the global
    /// buffer
    elem[e] =
        new elm::Element(e, basis, xL_elem, xR_elem, &global_rho[e * nquads],
                          &global_rhou[e * nquads], &global_e[e * nquads],
                          &global_AV[e * nquads]);

    /// Computes F from U
    elem[e]->setFlux();
    x_iter += dx;
  }
}

void Mesh::computeElements() {
  for (int e = 0; e < n; ++e) {
    elem[e]->computeDivFlux();
  }
}

void Mesh::computeInterfaces() {
  const int P = elem[0]->getBasis()->getOrder();

  for (int e = 0; e < n - 1; ++e) {
    /// Select iteracting elements
    elm::Element *LeftElem = elem[e];
    elm::Element *RightElem = elem[e + 1];

    /// Set up of the useful values
    double u1L = *(LeftElem->getU1(P));
    double u1R = *(RightElem->getU1(0));
    double u2L = *(LeftElem->getU2(P));
    double u2R = *(RightElem->getU2(0));
    double u3L = *(LeftElem->getU3(P));
    double u3R = *(RightElem->getU3(0));

    double f1L = *(LeftElem->getF1(P));
    double f1R = *(RightElem->getF1(0));
    double f2L = *(LeftElem->getF2(P));
    double f2R = *(RightElem->getF2(0));
    double f3L = *(LeftElem->getF3(P));
    double f3R = *(RightElem->getF3(0));

    /// Compute pressure
    double pL, pR;
    phy::getP(&pL, &u1L, &u2L, &u3L, 1);
    phy::getP(&pR, &u1R, &u2R, &u3R, 1);

    /// Compute max wave speed
    double lambdaL = reimann::computeMaxWaveSpeed(u1L, u2L / u1L, pL);
    double lambdaR = reimann::computeMaxWaveSpeed(u1R, u2R / u1R, pR);
    double lambda = std::max(lambdaL, lambdaR);

    /// Reimann problem
    double f1star = reimann::Rusanov(f1L, f1R, u1L, u1R, lambda);
    double f2star = reimann::Rusanov(f2L, f2R, u2L, u2R, lambda);
    double f3star = reimann::Rusanov(f3L, f3R, u3L, u3R, lambda);

    /// Boundary lift L_i = (F* - F^int) at the boundary node; the 1/(J*w)
    /// scaling is applied later by applyMassInverse.
    LeftElem->correctDivF1(P, f1star - f1L);
    LeftElem->correctDivF2(P, f2star - f2L);
    LeftElem->correctDivF3(P, f3star - f3L);

    RightElem->correctDivF1(0, f1R - f1star);
    RightElem->correctDivF2(0, f2R - f2star);
    RightElem->correctDivF3(0, f3R - f3star);
  }
}

void Mesh::applyDirichlet() {
  const int P = elem[0]->getBasis()->getOrder();

  // --- LEFT BOUNDARY ---
  if (bc_left) {
    double u1_int_L = *(elem[0]->getU1(0));
    double u2_int_L = *(elem[0]->getU2(0));
    double u3_int_L = *(elem[0]->getU3(0));
    double f1_int_L = *(elem[0]->getF1(0));
    double f2_int_L = *(elem[0]->getF2(0));
    double f3_int_L = *(elem[0]->getF3(0));

    // Ghost (exterior) state supplied by the boundary condition.
    double u1_L, u2_L, u3_L;
    bc_left->ghostState(u1_int_L, u2_int_L, u3_int_L, u1_L, u2_L, u3_L);

    // Dynamic lambda with safety factor for stability
    double p_int_L;
    double p_ext_L;
    phy::getP(&p_int_L, &u1_int_L, &u2_int_L, &u3_int_L, 1);
    phy::getP(&p_ext_L, &u1_L, &u2_L, &u3_L, 1);
    double f1_ext_L = u2_L;                           // rho * u
    double f2_ext_L = (u2_L * u2_L / u1_L) + p_ext_L; // rho * u^2 + p
    double f3_ext_L = (u2_L / u1_L) * (u3_L + p_ext_L);
    double lam_int_L =
        reimann::computeMaxWaveSpeed(u1_int_L, u2_int_L / u1_int_L, p_int_L);
    double lam_ext_L = reimann::computeMaxWaveSpeed(u1_L, u2_L / u1_L, p_ext_L);
    double lambda_L = std::max(lam_ext_L, lam_int_L); // DIFF COEFF

    /// Imposes same flux at the domain limit
    double f1s_L =
        reimann::Rusanov(f1_ext_L, f1_int_L, u1_L, u1_int_L, lambda_L);
    double f2s_L =
        reimann::Rusanov(f2_ext_L, f2_int_L, u2_L, u2_int_L, lambda_L);
    double f3s_L =
        reimann::Rusanov(f3_ext_L, f3_int_L, u3_L, u3_int_L, lambda_L);

    elem[0]->correctDivF1(0, f1_int_L - f1s_L);
    elem[0]->correctDivF2(0, f2_int_L - f2s_L);
    elem[0]->correctDivF3(0, f3_int_L - f3s_L);
  }

  // --- RIGHT BOUNDARY ---
  if (bc_right) {
    int last = n - 1;
    double u1_int_R = *(elem[last]->getU1(P));
    double u2_int_R = *(elem[last]->getU2(P));
    double u3_int_R = *(elem[last]->getU3(P));
    double f1_int_R = *(elem[last]->getF1(P));
    double f2_int_R = *(elem[last]->getF2(P));
    double f3_int_R = *(elem[last]->getF3(P));

    // Ghost (exterior) state supplied by the boundary condition.
    double u1_R, u2_R, u3_R;
    bc_right->ghostState(u1_int_R, u2_int_R, u3_int_R, u1_R, u2_R, u3_R);

    double p_int_R;
    double p_ext_R;
    phy::getP(&p_int_R, &u1_int_R, &u2_int_R, &u3_int_R, 1);
    phy::getP(&p_ext_R, &u1_R, &u2_R, &u3_R, 1);
    double f1_ext_R = u2_R;
    double f2_ext_R = (u2_R * u2_R / u1_R) + p_ext_R;
    double f3_ext_R = (u2_R / u1_R) * (u3_R + p_ext_R);

    double lam_int_R =
        reimann::computeMaxWaveSpeed(u1_int_R, u2_int_R / u1_int_R, p_int_R);
    double lam_ext_R = reimann::computeMaxWaveSpeed(u1_R, u2_R / u1_R, p_ext_R);
    double lambda_R = std::max(lam_ext_R, lam_int_R); // DIFF COEFF

    double f1s_R =
        reimann::Rusanov(f1_int_R, f1_ext_R, u1_int_R, u1_R, lambda_R);
    double f2s_R =
        reimann::Rusanov(f2_int_R, f2_ext_R, u2_int_R, u2_R, lambda_R);
    double f3s_R =
        reimann::Rusanov(f3_int_R, f3_ext_R, u3_int_R, u3_R, lambda_R);

    elem[last]->correctDivF1(P, f1s_R - f1_int_R);
    elem[last]->correctDivF2(P, f2s_R - f2_int_R);
    elem[last]->correctDivF3(P, f3s_R - f3_int_R);
  }
}

void Mesh::computeResidual() {
  // --- Standard strong-form DG residual (Rusanov, non entropy-stable) --------
  // U -> F(U)
  for (int e = 0; e < n; ++e) {
    elem[e]->setFlux();
  }
  // DG volume term: divF <- divergence of the nodal flux
  this->computeElements();
  // DG surface term: Rusanov numerical flux at interior faces
  this->computeInterfaces();
  // Boundary term: Dirichlet Rusanov lift
  this->applyDirichlet();
  // Closure: divF <- (1/J) * Minv * divF on each element
  for (int e = 0; e < n; ++e) {
    elem[e]->applyMassInverse();
  }
}

/// @note NOT USED - FIRST DEV OF HYBRID POLY/RBF BASE SWITCH
void Mesh::adaptBasis(sens::PerssonPeraire &sensor, int truncation,
                      double s_shock, double s_smooth) {
  if (alt_basis == nullptr || primary_basis == nullptr)
    return;

  for (int e = 0; e < n; ++e) {
    elm::Element *E = elem[e];
    // Modal coefficients depend on the current basis -> refresh first.
    E->computeLegendreCoefficients();
    const double Se = sensor.SmoothnessIndicator(*E, truncation);
    if (Se < 1e-30)
      continue; // perfectly smooth -> nothing to do
    const double logSe = std::log10(Se);

    base::_Basis *current = const_cast<base::_Basis *>(E->getBasis());
    if (current == primary_basis && logSe > s_shock) {
      E->setBasis(alt_basis);
    } else if (current == alt_basis && logSe < s_smooth) {
      E->setBasis(primary_basis);
    }
  }
}

// ---------------------------------------------------------------------------
// Hybrid-scheme building blocks
// ---------------------------------------------------------------------------
// Both the DG and the FV residual are assembled from Nn+1 "subcell interface
// fluxes" B_0 .. B_Nn per element: the residual at node j is the flux jump
// divF_j = B_{j+1} - B_j (before the mass-inverse). The two schemes differ only
// in how the interior fluxes B_1 .. B_{Nn-1} are built; the endpoints B_0, B_Nn
// carry the physical flux and are overwritten by the shared ES surface lift.

void Mesh::buildSubcellFluxDG(int e, double *B1, double *B2, double *B3) const {
  const int Nn = elem[0]->getBasis()->getOrder() + 1;
  const double *w = elem[0]->getBasis()->getWeights();
  const double *D = elem[0]->getBasis()->getD();
  const double *rho = elem[e]->getU1();
  const double *rhou = elem[e]->getU2();
  const double *en = elem[e]->getU3();

  // Entropy-conservative (Chandrashekar) two-point flux for every node pair.
  std::vector<double> ec1(Nn * Nn), ec2(Nn * Nn), ec3(Nn * Nn);
  for (int j = 0; j < Nn; ++j)
    for (int k = 0; k < Nn; ++k)
      entropy::chandrashekar_ec(rho[j], rhou[j], en[j], rho[k], rhou[k], en[k],
                                ec1[j * Nn + k], ec2[j * Nn + k],
                                ec3[j * Nn + k]);

  // Left endpoint: physical flux.
  physicalFlux(rho[0], rhou[0], en[0], B1[0], B2[0], B3[0]);

  // SBP recurrence (Hennemann Eq. 7): telescoping of the split-form EC volume
  //   B_{j+1} = B_j + 2 w_j * sum_k D_{jk} f*_EC(u_j, u_k).
  for (int j = 0; j < Nn - 1; ++j) {
    double d1 = 0.0, d2 = 0.0, d3 = 0.0;
    for (int k = 0; k < Nn; ++k) {
      const double Djk = D[j * Nn + k];
      d1 += Djk * ec1[j * Nn + k];
      d2 += Djk * ec2[j * Nn + k];
      d3 += Djk * ec3[j * Nn + k];
    }
    const double w2j = 2.0 * w[j];
    B1[j + 1] = B1[j] + w2j * d1;
    B2[j + 1] = B2[j] + w2j * d2;
    B3[j + 1] = B3[j] + w2j * d3;
  }

  // Right endpoint: physical flux.
  physicalFlux(rho[Nn - 1], rhou[Nn - 1], en[Nn - 1], B1[Nn], B2[Nn], B3[Nn]);
}

void Mesh::buildSubcellFluxFV(int e, double *B1, double *B2, double *B3) const {
  const int Nn = elem[0]->getBasis()->getOrder() + 1;
  const double *rho = elem[e]->getU1();
  const double *rhou = elem[e]->getU2();
  const double *en = elem[e]->getU3();

  // Left endpoint: physical flux.
  physicalFlux(rho[0], rhou[0], en[0], B1[0], B2[0], B3[0]);

  // Interior: first-order entropy-stable flux across each subcell interface.
  for (int j = 0; j < Nn - 1; ++j)
    entropy::entropy_stable_lf(rho[j], rhou[j], en[j], rho[j + 1], rhou[j + 1],
                               en[j + 1], B1[j + 1], B2[j + 1], B3[j + 1]);

  // Right endpoint: physical flux.
  physicalFlux(rho[Nn - 1], rhou[Nn - 1], en[Nn - 1], B1[Nn], B2[Nn], B3[Nn]);
}

void Mesh::storeSubcellDivergence(int e, const double *B1, const double *B2,
                                  const double *B3) {
  const int Nn = elem[0]->getBasis()->getOrder() + 1;
  // divF_j = B_{j+1} - B_j; applyMassInverse() later scales by 1/(J w_j).
  for (int j = 0; j < Nn; ++j) {
    elem[e]->setDivF1(j, B1[j + 1] - B1[j]);
    elem[e]->setDivF2(j, B2[j + 1] - B2[j]);
    elem[e]->setDivF3(j, B3[j + 1] - B3[j]);
  }
}

void Mesh::applyEntropyStableInterfaces() {
  const int Nn = elem[0]->getBasis()->getOrder() + 1;
  const int NL = Nn - 1;
  // Mirrors computeInterfaces() but with the entropy-stable flux instead of
  // Rusanov: L_i = (F* - F^int) at each interface node.
  for (int e = 0; e < n - 1; ++e) {
    const double uL1 = elem[e]->getRho(NL);
    const double uL2 = *(elem[e]->getU2(NL));
    const double uL3 = *(elem[e]->getU3(NL));
    const double uR1 = elem[e + 1]->getRho(0);
    const double uR2 = *(elem[e + 1]->getU2(0));
    const double uR3 = *(elem[e + 1]->getU3(0));

    double f1s, f2s, f3s;
    entropy::entropy_stable_lf(uL1, uL2, uL3, uR1, uR2, uR3, f1s, f2s, f3s);

    double fL1, fL2, fL3, fR1, fR2, fR3;
    physicalFlux(uL1, uL2, uL3, fL1, fL2, fL3);
    physicalFlux(uR1, uR2, uR3, fR1, fR2, fR3);

    elem[e]->correctDivF1(NL, f1s - fL1);
    elem[e]->correctDivF2(NL, f2s - fL2);
    elem[e]->correctDivF3(NL, f3s - fL3);

    elem[e + 1]->correctDivF1(0, fR1 - f1s);
    elem[e + 1]->correctDivF2(0, fR2 - f2s);
    elem[e + 1]->correctDivF3(0, fR3 - f3s);
  }
}

void Mesh::applyEntropyStableBoundaries() {
  const int Nn = elem[0]->getBasis()->getOrder() + 1;
  const int NL = Nn - 1;

  // Left domain boundary (element 0, node 0): lift (F^int - F*).
  if (bc_left) {
    const double ui1 = elem[0]->getRho(0);
    const double ui2 = *(elem[0]->getU2(0));
    const double ui3 = *(elem[0]->getU3(0));
    // Ghost (exterior) state supplied by the boundary condition.
    double ug1, ug2, ug3;
    bc_left->ghostState(ui1, ui2, ui3, ug1, ug2, ug3);
    double fi1, fi2, fi3;
    physicalFlux(ui1, ui2, ui3, fi1, fi2, fi3);
    double fs1, fs2, fs3;
    entropy::entropy_stable_lf(ug1, ug2, ug3, ui1, ui2, ui3, fs1, fs2, fs3);
    elem[0]->correctDivF1(0, fi1 - fs1);
    elem[0]->correctDivF2(0, fi2 - fs2);
    elem[0]->correctDivF3(0, fi3 - fs3);
  }

  // Right domain boundary (last element, node Nn-1): lift (F* - F^int).
  if (bc_right) {
    const int last = n - 1;
    const double ui1 = elem[last]->getRho(NL);
    const double ui2 = *(elem[last]->getU2(NL));
    const double ui3 = *(elem[last]->getU3(NL));
    // Ghost (exterior) state supplied by the boundary condition.
    double ug1, ug2, ug3;
    bc_right->ghostState(ui1, ui2, ui3, ug1, ug2, ug3);
    double fi1, fi2, fi3;
    physicalFlux(ui1, ui2, ui3, fi1, fi2, fi3);
    double fs1, fs2, fs3;
    entropy::entropy_stable_lf(ui1, ui2, ui3, ug1, ug2, ug3, fs1, fs2, fs3);
    elem[last]->correctDivF1(NL, fs1 - fi1);
    elem[last]->correctDivF2(NL, fs2 - fi2);
    elem[last]->correctDivF3(NL, fs3 - fi3);
  }
}

// ---------------------------------------------------------------------------
// Entropy-stable residuals (DG / FV / hybrid)
// ---------------------------------------------------------------------------
void Mesh::computeDGResidual() {
  const int Nn = elem[0]->getBasis()->getOrder() + 1;
  std::vector<double> B1(Nn + 1); 
  std::vector<double> B2(Nn + 1); 
  std::vector<double> B3(Nn + 1);
  // Volume term: high-order entropy-conservative subcell fluxes.
  for (int e = 0; e < n; ++e) {
    buildSubcellFluxDG(e, B1.data(), B2.data(), B3.data());
    storeSubcellDivergence(e, B1.data(), B2.data(), B3.data());
  }
  // Surface term + closure.
  applyEntropyStableInterfaces();
  applyEntropyStableBoundaries();
  for (int e = 0; e < n; ++e)
    elem[e]->applyMassInverse();
}

void Mesh::computeFVResidual() {
  const int Nn = elem[0]->getBasis()->getOrder() + 1;
  std::vector<double> B1(Nn + 1); 
  std::vector<double> B2(Nn + 1); 
  std::vector<double> B3(Nn + 1);
  // Volume term: first-order entropy-stable subcell fluxes.
  for (int e = 0; e < n; ++e) {
    buildSubcellFluxFV(e, B1.data(), B2.data(), B3.data());
    storeSubcellDivergence(e, B1.data(), B2.data(), B3.data());
  }
  // Surface term + closure (identical to the DG residual).
  applyEntropyStableInterfaces();
  applyEntropyStableBoundaries();
  for (int e = 0; e < n; ++e)
    elem[e]->applyMassInverse();
}

void Mesh::computeHybridResidual(const double *alpha) {
  const int Nn = elem[0]->getBasis()->getOrder() + 1;
  const int P = Nn - 1;   // number of interior subcell interfaces per element

  // Blend the SUBCELL FLUXES per interface, then difference. Matches
  // nn/jax_dgsem/solver.py hybrid_residual + _alpha_on_interfaces.
  std::vector<double> Bd1(Nn + 1), Bd2(Nn + 1), Bd3(Nn + 1);
  std::vector<double> Bf1(Nn + 1), Bf2(Nn + 1), Bf3(Nn + 1);
  std::vector<double> Bh1(Nn + 1), Bh2(Nn + 1), Bh3(Nn + 1);
  for (int e = 0; e < n; ++e) {
    buildSubcellFluxDG(e, Bd1.data(), Bd2.data(), Bd3.data());
    buildSubcellFluxFV(e, Bf1.data(), Bf2.data(), Bf3.data());
    for (int k = 0; k <= Nn; ++k) {
      // interfaces 0 and Nn are the element boundaries (physical flux, a=0);
      // interior interfaces 1..Nn-1 take alpha[e*P + (k-1)].
      const double a = (k == 0 || k == Nn) ? 0.0 : alpha[e * P + (k - 1)];
      Bh1[k] = (1.0 - a) * Bd1[k] + a * Bf1[k];
      Bh2[k] = (1.0 - a) * Bd2[k] + a * Bf2[k];
      Bh3[k] = (1.0 - a) * Bd3[k] + a * Bf3[k];
    }
    storeSubcellDivergence(e, Bh1.data(), Bh2.data(), Bh3.data());
  }
  // Shared entropy-stable surface term + closure (identical for DG/FV/hybrid).
  applyEntropyStableInterfaces();
  applyEntropyStableBoundaries();
  for (int e = 0; e < n; ++e)
    elem[e]->applyMassInverse();
}

void Mesh::densityResidualDifference(std::vector<double> &out) {
  const int Nn = elem[0]->getBasis()->getOrder() + 1;
  out.assign(n * Nn, 0.0);
  // DG residual density component, stash.
  computeDGResidual();
  std::vector<double> dg(n * Nn);
  for (int e = 0; e < n; ++e)
    for (int j = 0; j < Nn; ++j) dg[e * Nn + j] = *(elem[e]->getDivF1(j));
  // FV residual density component, difference.
  computeFVResidual();
  for (int e = 0; e < n; ++e)
    for (int j = 0; j < Nn; ++j)
      out[e * Nn + j] = dg[e * Nn + j] - *(elem[e]->getDivF1(j));
}

void Mesh::perssonPeraireIndicator(std::vector<double> &eind) {
  const int P = elem[0]->getBasis()->getOrder();
  eind.assign(n, 0.0);
  for (int e = 0; e < n; ++e) {
    elem[e]->computeLegendreCoefficients();
    const double *m_coef = elem[e]->getModes();
    double total = 0.0;
    for (int j = 0; j <= P; ++j) total += m_coef[j] * m_coef[j];
    if (total < 1e-300) { eind[e] = 0.0; continue; }
    double total_m1 = total - m_coef[P] * m_coef[P];
    double E_N = m_coef[P] * m_coef[P] / total;
    double E_Nm = (P > 0 && total_m1 > 1e-300)
                      ? m_coef[P - 1] * m_coef[P - 1] / total_m1 : 0.0;
    eind[e] = std::max(E_N, E_Nm);
  }
}

void Mesh::setLeftBC(bc::_BoundaryConditions *b) {
  delete bc_left;
  bc_left = b;
}

void Mesh::setRightBC(bc::_BoundaryConditions *b) {
  delete bc_right;
  bc_right = b;
}

double Mesh::computeCFLTimeStep(double cfl) const {
  const int Nn = elem[0]->getBasis()->getOrder() + 1;
  const int P = Nn - 1;

  double lambda_max = 0.0;
  double dx_min = std::numeric_limits<double>::max();

  for (int e = 0; e < n; ++e) {
    // GLL nodes include the element endpoints, so getX(P) - getX(0) is the
    // element width even after a basis switch; keep the smallest.
    const double dx_e = elem[e]->getX(P) - elem[e]->getX(0);
    dx_min = std::min(dx_min, dx_e);

    for (int j = 0; j < Nn; ++j) {
      const double rho = elem[e]->getRho(j);
      const double u = elem[e]->getU(j);
      const double p = elem[e]->getP(j);
      lambda_max =
          std::max(lambda_max, reimann::computeMaxWaveSpeed(rho, u, p));
    }
  }

  // Degenerate (quiescent, p=0) field: no wave speed -> no CFL restriction.
  if (lambda_max <= 0.0)
    return 0.0;

  return cfl * dx_min / ((2.0 * P + 1.0) * lambda_max);
}

Mesh::~Mesh() {
  for (int e = 0; e < n; e++)
    delete elem[e];
  delete[] elem;
  delete[] global_rho;
  delete[] global_rhou;
  delete[] global_e;
  delete[] global_AV;
  delete bc_left;
  delete bc_right;
}

std::ostream &operator<<(std::ostream &os, const Mesh &m) {
  os << "----- MESH -----" << std::endl
     << "N. ELEM      : " << m.n << std::endl
     << "TOTAL POINTS : " << m.getTotalPoints() << std::endl;

  os << "PRIMARY BASIS:" << std::endl;
  if (m.primary_basis)
    os << *(m.primary_basis);
  else
    os << "  (none)" << std::endl;

  if (m.alt_basis) {
    os << "ALT BASIS:" << std::endl << *(m.alt_basis);
  }
  return os;
};

} // namespace mesh
