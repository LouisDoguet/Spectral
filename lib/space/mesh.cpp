#include <cmath>
#include <iomanip>
#include <iostream>

#include "../phy/physics.h"
#include "mesh.h"

namespace mesh {

Mesh::Mesh(const int n, base::_Basis *basis, double xL, double xR) : n(n) {
  primary_basis = basis;
  double dx_mesh = xR - xL;
  double dx = (double)dx_mesh / (n);

  int nquads = basis->getOrder() + 1;
  global_rho = new double[n * nquads];
  global_rhou = new double[n * nquads];
  global_e = new double[n * nquads];
  global_AV = new double[n * nquads];

  this->elem = new elem::Element *[n];

  double x_iter = xL;
  for (int e = 0; e < n; ++e) {
    elem[e] = new elem::Element(
        e, basis, x_iter, x_iter + dx, 
        &global_rho[e * nquads], &global_rhou[e * nquads], &global_e[e * nquads], 
        &global_AV[e * nquads]);
    elem[e]->setFlux();
    x_iter += dx;
  }
}

Mesh::Mesh(const int n, base::_Basis *basis, double xL, double xR,
           double *init_u1, double *init_u2, double *init_u3, double u1_L,
           double u2_L, double u3_L, double u1_R, double u2_R, double u3_R)
    : n(n), u1_L(u1_L), u2_L(u2_L), u3_L(u3_L), u1_R(u1_R), u2_R(u2_R),
      u3_R(u3_R) {
  primary_basis = basis;

  double dx_mesh = xR - xL;
  double dx = (double) dx_mesh / (n);

  /// Global buffer
  int nquads = basis->getOrder() + 1;
  global_rho = new double[n * nquads];
  global_rhou = new double[n * nquads];
  global_e = new double[n * nquads];
  global_AV = new double[n * nquads];

  /// Pointers to the elements
  this->elem = new elem::Element *[n];

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
        new elem::Element(e, basis, xL_elem, xR_elem, 
                          &global_rho[e * nquads], &global_rhou[e * nquads], &global_e[e * nquads], 
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
    elem::Element *LeftElem = elem[e];
    elem::Element *RightElem = elem[e + 1];

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
  double u1_int_L = *(elem[0]->getU1(0));
  double u2_int_L = *(elem[0]->getU2(0));
  double u3_int_L = *(elem[0]->getU3(0));
  double f1_int_L = *(elem[0]->getF1(0));
  double f2_int_L = *(elem[0]->getF2(0));
  double f3_int_L = *(elem[0]->getF3(0));

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
  double f1s_L = reimann::Rusanov(f1_ext_L, f1_int_L, u1_L, u1_int_L, lambda_L);
  double f2s_L = reimann::Rusanov(f2_ext_L, f2_int_L, u2_L, u2_int_L, lambda_L);
  double f3s_L = reimann::Rusanov(f3_ext_L, f3_int_L, u3_L, u3_int_L, lambda_L);

  elem[0]->correctDivF1(0, f1_int_L - f1s_L);
  elem[0]->correctDivF2(0, f2_int_L - f2s_L);
  elem[0]->correctDivF3(0, f3_int_L - f3s_L);

  // --- RIGHT BOUNDARY ---
  int last = n - 1;
  double u1_int_R = *(elem[last]->getU1(P));
  double u2_int_R = *(elem[last]->getU2(P));
  double u3_int_R = *(elem[last]->getU3(P));
  double f1_int_R = *(elem[last]->getF1(P));
  double f2_int_R = *(elem[last]->getF2(P));
  double f3_int_R = *(elem[last]->getF3(P));

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

  double f1s_R = reimann::Rusanov(f1_int_R, f1_ext_R, u1_int_R, u1_R, lambda_R);
  double f2s_R = reimann::Rusanov(f2_int_R, f2_ext_R, u2_int_R, u2_R, lambda_R);
  double f3s_R = reimann::Rusanov(f3_int_R, f3_ext_R, u3_int_R, u3_R, lambda_R);

  elem[last]->correctDivF1(P, f1s_R - f1_int_R);
  elem[last]->correctDivF2(P, f2s_R - f2_int_R);
  elem[last]->correctDivF3(P, f3s_R - f3_int_R);
}

void Mesh::computeResidual() {
  for (int e = 0; e < n; ++e) {
    elem[e]->setFlux();
  }
  this->computeElements();
  this->computeInterfaces();
  this->applyDirichlet();
  // Final stage: divFk <- (1/J) * Minv * divFk on each element.
  for (int e = 0; e < n; ++e) {
    elem[e]->applyMassInverse();
  }
}

void Mesh::adaptBasis(sens::PerssonPeraire& sensor, int truncation,
                      double s_shock, double s_smooth) {
  if (alt_basis == nullptr || primary_basis == nullptr) return;

  for (int e = 0; e < n; ++e) {
    elem::Element* E = elem[e];
    // Modal coefficients depend on the current basis -> refresh first.
    E->computeLegendreCoefficients();
    const double Se = sensor.SmoothnessIndicator(*E, truncation);
    if (Se < 1e-30) continue;                  // perfectly smooth -> nothing to do
    const double logSe = std::log10(Se);

    base::_Basis* current = const_cast<base::_Basis*>(E->getBasis());
    if (current == primary_basis && logSe > s_shock) {
      E->setBasis(alt_basis);
    } else if (current == alt_basis && logSe < s_smooth) {
      E->setBasis(primary_basis);
    }
  }
}

Mesh::~Mesh() {
  for (int e = 0; e < n; e++)
    delete elem[e];
  delete[] elem;
  delete[] global_rho;
  delete[] global_rhou;
  delete[] global_e;
  delete[] global_AV;
}

std::ostream &operator<<(std::ostream &os, const Mesh &m) {
  os << "----- MESH -----" << std::endl << "N. ELEM  : " << m.n << std::endl;
  return os;
};

} // namespace mesh
