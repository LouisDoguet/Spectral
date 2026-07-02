#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../phy/entropy_flux.h"
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
    elem[e] =
        new elem::Element(e, basis, x_iter, x_iter + dx,
                          &global_rho[e * nquads], &global_rhou[e * nquads],
                          &global_e[e * nquads], &global_AV[e * nquads]);
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
  double dx = (double)dx_mesh / (n);

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
        new elem::Element(e, basis, xL_elem, xR_elem, &global_rho[e * nquads],
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

void Mesh::adaptBasis(sens::PerssonPeraire &sensor, int truncation,
                      double s_shock, double s_smooth) {
  if (alt_basis == nullptr || primary_basis == nullptr)
    return;

  for (int e = 0; e < n; ++e) {
    elem::Element *E = elem[e];
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

void Mesh::computeHybridResidual(const double *alpha) {
  const int Nn = elem[0]->getBasis()->getOrder() + 1;
  const double *w = elem[0]->getBasis()->getWeights();
  const double *D = elem[0]->getBasis()->getD();

  const int stride = Nn + 1;
  std::vector<double> fb1(n * stride, 0.0);
  std::vector<double> fb2(n * stride, 0.0);
  std::vector<double> fb3(n * stride, 0.0);

  std::vector<double> ec1(Nn * Nn), ec2(Nn * Nn), ec3(Nn * Nn);

  // ---- Phase 1: element-local blended subcell fluxes
  // -------------------------
  for (int e = 0; e < n; ++e) {
    const double *rho = elem[e]->getU1();
    const double *rhou = elem[e]->getU2();
    const double *en = elem[e]->getU3();
    const double alp = alpha[e];
    double *B1 = fb1.data() + e * stride;
    double *B2 = fb2.data() + e * stride;
    double *B3 = fb3.data() + e * stride;

    for (int j = 0; j < Nn; ++j)
      for (int k = 0; k < Nn; ++k)
        entropy::chandrashekar_ec(rho[j], rhou[j], en[j], rho[k], rhou[k],
                                  en[k], ec1[j * Nn + k], ec2[j * Nn + k],
                                  ec3[j * Nn + k]);

    {
      double p0;
      phy::getP(&p0, const_cast<double *>(rho), const_cast<double *>(rhou),
                const_cast<double *>(en), 1);
      B1[0] = rhou[0];
      B2[0] = rhou[0] * rhou[0] / rho[0] + p0;
      B3[0] = rhou[0] / rho[0] * (en[0] + p0);
    }

    for (int j = 0; j < Nn - 1; ++j) {
      double d1 = 0.0, d2 = 0.0, d3 = 0.0;
      for (int k = 0; k < Nn; ++k) {
        const double Djk = D[j * Nn + k];
        d1 += Djk * ec1[j * Nn + k];
        d2 += Djk * ec2[j * Nn + k];
        d3 += Djk * ec3[j * Nn + k];
      }
      const double w2j = 2.0 * w[j];
      const double fbar1_hi = B1[j] + w2j * d1;
      const double fbar2_hi = B2[j] + w2j * d2;
      const double fbar3_hi = B3[j] + w2j * d3;

      double fv1, fv2, fv3;
      entropy::entropy_stable_lf(rho[j], rhou[j], en[j], rho[j + 1],
                                 rhou[j + 1], en[j + 1], fv1, fv2, fv3);

      B1[j + 1] = alp * fv1 + (1.0 - alp) * fbar1_hi;
      B2[j + 1] = alp * fv2 + (1.0 - alp) * fbar2_hi;
      B3[j + 1] = alp * fv3 + (1.0 - alp) * fbar3_hi;
    }
    {
                                const_cast<double*>(rhou + NL),
                                const_cast<double*>(en   + NL), 1);
                                B1[Nn] = rhou[NL];
                                B2[Nn] = rhou[NL] * rhou[NL] / rho[NL] + pN;
                                B3[Nn] = rhou[NL] / rho[NL] * (en[NL] + pN);
    }

    for (int j = 0; j < Nn; ++j) {
      elem[e]->setDivF1(j, B1[j + 1] - B1[j]);
      elem[e]->setDivF2(j, B2[j + 1] - B2[j]);
      elem[e]->setDivF3(j, B3[j + 1] - B3[j]);
    }
  }

  // ---- Phase 2: inter-element ES interface flux surface corrections
  // ----------
  for (int e = 0; e < n - 1; ++e) {
    const int NL = Nn - 1;
    const double uL1 = elem[e]->getRho(NL);
    const double uL2 = *(elem[e]->getU2(NL));
    const double uL3 = *(elem[e]->getU3(NL));
    const double uR1 = elem[e + 1]->getRho(0);
    const double uR2 = *(elem[e + 1]->getU2(0));
    const double uR3 = *(elem[e + 1]->getU3(0));

    double f1s, f2s, f3s;
    entropy::entropy_stable_lf(uL1, uL2, uL3, uR1, uR2, uR3, f1s, f2s, f3s);

    double pL, pR;
    phy::getP(&pL, const_cast<double *>(&uL1), const_cast<double *>(&uL2),
              const_cast<double *>(&uL3), 1);
    phy::getP(&pR, const_cast<double *>(&uR1), const_cast<double *>(&uR2),
              const_cast<double *>(&uR3), 1);
    const double fL1 = uL2, fL2 = uL2 * uL2 / uL1 + pL,
                 fL3 = uL2 / uL1 * (uL3 + pL);
    const double fR1 = uR2, fR2 = uR2 * uR2 / uR1 + pR,
                 fR3 = uR2 / uR1 * (uR3 + pR);

    elem[e]->correctDivF1(NL, f1s - fL1);
    elem[e]->correctDivF2(NL, f2s - fL2);
    elem[e]->correctDivF3(NL, f3s - fL3);

    elem[e + 1]->correctDivF1(0, fR1 - f1s);
    elem[e + 1]->correctDivF2(0, fR2 - f2s);
    elem[e + 1]->correctDivF3(0, fR3 - f3s);
  }

  // ---- Phase 3: domain boundary ES flux corrections -------------------------
  {
    const double ui1 = elem[0]->getRho(0);
    const double ui2 = *(elem[0]->getU2(0));
    const double ui3 = *(elem[0]->getU3(0));
    double pi, pe;
    phy::getP(&pi, const_cast<double *>(&ui1), const_cast<double *>(&ui2),
              const_cast<double *>(&ui3), 1);
    phy::getP(&pe, const_cast<double *>(&u1_L), const_cast<double *>(&u2_L),
              const_cast<double *>(&u3_L), 1);
    const double fe1 = u2_L, fe2 = u2_L * u2_L / u1_L + pe,
                 fe3 = u2_L / u1_L * (u3_L + pe);
    const double fi1 = ui2, fi2 = ui2 * ui2 / ui1 + pi,
                 fi3 = ui2 / ui1 * (ui3 + pi);
    double fs1, fs2, fs3;
    entropy::entropy_stable_lf(u1_L, u2_L, u3_L, ui1, ui2, ui3, fs1, fs2, fs3);
    elem[0]->correctDivF1(0, fi1 - fs1);
    elem[0]->correctDivF2(0, fi2 - fs2);
    elem[0]->correctDivF3(0, fi3 - fs3);
    (void)fe1;
    (void)fe2;
    (void)fe3;

    const int last = n - 1;
    const int NL = Nn - 1;
    const double ui1r = elem[last]->getRho(NL);
    const double ui2r = *(elem[last]->getU2(NL));
    const double ui3r = *(elem[last]->getU3(NL));
    double pir, per;
    phy::getP(&pir, const_cast<double *>(&ui1r), const_cast<double *>(&ui2r),
              const_cast<double *>(&ui3r), 1);
    phy::getP(&per, const_cast<double *>(&u1_R), const_cast<double *>(&u2_R),
              const_cast<double *>(&u3_R), 1);
    const double fir1 = ui2r, fir2 = ui2r * ui2r / ui1r + pir,
                 fir3 = ui2r / ui1r * (ui3r + pir);
    double fsr1, fsr2, fsr3;
    entropy::entropy_stable_lf(ui1r, ui2r, ui3r, u1_R, u2_R, u3_R, fsr1, fsr2,
                               fsr3);
    elem[last]->correctDivF1(NL, fsr1 - fir1);
    elem[last]->correctDivF2(NL, fsr2 - fir2);
    elem[last]->correctDivF3(NL, fsr3 - fir3);
    (void)per;
  }

  // ---- Phase 4: mass-matrix inverse + Jacobian scaling ----------------------
  for (int e = 0; e < n; ++e)
    elem[e]->applyMassInverse();
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
