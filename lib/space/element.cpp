#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>

#include "../math/math.h"
#include "../phy/physics.h"
#include "element.h"

namespace elm {

void Element::setBasis(base::_Basis* newBasis) {
  if (newBasis == basis) return;
  const int N = basis->getOrder() + 1;

  double* new_rho  = new double[N];
  double* new_rhou = new double[N];
  double* new_e    = new double[N];
  double* new_AV   = new double[N];

  const double* new_quads = newBasis->getQuads();
  basis->interpolate(rho,  new_quads, N, new_rho);
  basis->interpolate(rhou, new_quads, N, new_rhou);
  basis->interpolate(e,    new_quads, N, new_e);
  basis->interpolate(AV,   new_quads, N, new_AV);

  std::copy(new_rho,  new_rho  + N, rho);
  std::copy(new_rhou, new_rhou + N, rhou);
  std::copy(new_e,    new_e    + N, e);
  std::copy(new_AV,   new_AV   + N, AV);

  delete[] new_rho;
  delete[] new_rhou;
  delete[] new_e;
  delete[] new_AV;

  basis = newBasis;
  setFlux();
}

/**
 * @brief Sets the Jacobian from the element position
 */
void Element::setJ(double xL, double xR) {
  this->xL = xL;
  this->xR = xR;
  double dx = xR - xL;
  this->J = dx / 2.;
  this->invJ = 1 / J;
}

/// Core constructor: all public constructors delegate here
Element::Element(int id, base::_Basis *basis, double xL, double xR, double *rho,
                 double *rhou, double *e, double* AV, bool ownsMemory)
    : id(id), basis(basis), rho(rho), rhou(rhou), e(e), AV(AV), ownsMemory(ownsMemory) {
  setJ(xL, xR);
  int n = basis->getOrder() + 1;
  F1 = new double[n];
  F2 = new double[n];
  F3 = new double[n];
  divF1 = new double[n];
  divF2 = new double[n];
  divF3 = new double[n];
  legendreCoefficients = new double[n];
}

/// No-values: allocates zeroed U arrays, delegates to core
Element::Element(const int id, base::_Basis *sharedBasis, double xL, double xR)
    : Element(id, sharedBasis, xL, xR,
              new double[sharedBasis->getOrder() + 1](),
              new double[sharedBasis->getOrder() + 1](),
              new double[sharedBasis->getOrder() + 1](), 
              new double[sharedBasis->getOrder() + 1](), true) {}

/// Scalar init: delegates to no-values, then fills U arrays
Element::Element(const int id, base::_Basis *sharedBasis, double xL, double xR,
                 double rho_init, double rhou_init, double e_init)
    : Element(id, sharedBasis, xL, xR) {
  int n = sharedBasis->getOrder() + 1;
  std::fill(rho, rho + n, rho_init);
  std::fill(rhou, rhou + n, rhou_init);
  std::fill(e, e + n, e_init);
  std::fill(AV, AV + n, 0.);
}

/// External buffers: delegates to core with ownsMemory=false
Element::Element(const int id, base::_Basis *sharedBasis, double xL, double xR,
                 double *external_rho, double *external_rhou,
                 double *external_e, double* AV)
    : Element(id, sharedBasis, xL, xR, external_rho, external_rhou, external_e, AV, false) {}

/**
 * @brief Sets the flux from the Euler system solved
 */
void Element::setFlux() {
  int n = basis->getOrder() + 1;
  double *p = new double[n];
  phy::getP(p, rho, rhou, e, n);
  phy::computeFlux(F1, F2, F3, rho, rhou, e, p, n);
  delete[] p;
}

void Element::computeLegendreCoefficients() {
  mat::computeLegendreCoeffs(legendreCoefficients, rho, basis->getQuads(),
                             basis->getWeights(), basis->getOrder());
}

void Element::computeDivFlux() {
  const int N = basis->getOrder() + 1;
  const double* D = basis->getD();
  const double* w = basis->getWeights();

  // Reference derivative dF/dxi = D * F  (J factor lives in applyMassInverse)
  cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1.0, D, N, F1, 1, 0.0, divF1, 1);
  cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1.0, D, N, F2, 1, 0.0, divF2, 1);
  cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1.0, D, N, F3, 1, 0.0, divF3, 1);

  // Weight by w_i so divFk currently holds the volume residual V_i = w_i * (DF)_i
  for (int i = 0; i < N; ++i) {
    divF1[i] *= w[i];
    divF2[i] *= w[i];
    divF3[i] *= w[i];
  }
}

void Element::applyMassInverse() {
  const int N = basis->getOrder() + 1;
  const double* Minv = basis->getMinv();

  double* tmp = new double[N];

  cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, invJ, Minv, N, divF1, 1, 0.0, tmp, 1);
  std::copy(tmp, tmp + N, divF1);
  cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, invJ, Minv, N, divF2, 1, 0.0, tmp, 1);
  std::copy(tmp, tmp + N, divF2);
  cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, invJ, Minv, N, divF3, 1, 0.0, tmp, 1);
  std::copy(tmp, tmp + N, divF3);

  delete[] tmp;
}

double* Element::computePressure() {
  const int N = this->getBasis()->getOrder()+1;
  double* pressure = new double[N];
  for (int i=0; i<N; ++i ){
    pressure[i] = (this->e[i] - 0.5 * this->rhou[i] * (this->rhou[i] / this->rho[i]) ) / (0.4);
  }
  return pressure;
}

const double* Element::computePressureLaplacian() {
  double* pressure = this->computePressure();
  double* lap_pressure = new double[this->basis->getOrder()+1];
  mat::computeLaplacian(this->basis, this->invJ, pressure, lap_pressure);
  delete[] pressure;
  return lap_pressure;
}

const double* Element::computeVelocityDivergence() {
  int N = this->getBasis()->getOrder()+1;
  double* v = new double[N];
  double* dvdx = new double[N];

  for (int q=0; q<N; ++q) v[q]=this->getU(q);

  cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1., this->getBasis()->getD(), N, v, 1, 0., dvdx, 1);
  for (int i=0; i<N; ++i) { 
      dvdx[i]*= this->getInvJ()[N*i + i];
  }
  return dvdx;
}

Element::~Element() {
  if (ownsMemory) {
    delete[] rho;
    delete[] rhou;
    delete[] e;
  }
  delete[] F1;
  delete[] F2;
  delete[] F3;
  delete[] divF1;
  delete[] divF2;
  delete[] divF3;
  delete[] legendreCoefficients;
}

std::ostream &operator<<(std::ostream &os, const Element &e) {
  os << "----- ELEM -----" << std::endl
     << "ID  : " << e.id << std::endl
     << "xL  : " << e.xL << std::endl
     << "xR  : " << e.xR << std::endl;
  return os;
}
} // namespace elem
