#ifndef ELEMENT_H
#define ELEMENT_H

#include "../base/base.h"
#include <cstring>

static const double GAMMA = 1.4;

namespace elm {
/**
 * @brief Class storing an element
 */
class Element {
public:
  /**
   * @brief Constructor basic (no values) DEV/DEBUG
   */
  Element(const int id, base::_Basis *sharedBasis, double xL, double xR);
  /**
   * @brief Constructor with init values DEV/DEBUG
   */
  Element(const int id, base::_Basis *sharedBasis, double xL, double xR,
          double rho_init, double rhou_init, double e_init);

  /**
   * @brief Constructor of element object
   * @param id ID of the element
   * @param shareBasis Basis of the domain
   * @param xL Position of the left limit of the element
   * @param xR Position of the right limit of the element
   * @param external_rho Pointer to the general density
   * @param external_rhou Pointer to the general momentum
   * @param external_e Pointer to the general energy
   * @param AV1 Pointer to the artificial viscosity of general density
   * @param AV2 Pointer to the artificial viscosity of general momentum
   * @param AV3 Pointer to the artificial viscosity of general energy
   */
  Element(const int id, base::_Basis *sharedBasis, double xL, double xR,
          double *external_rho, double *external_rhou, double *external_e, double* artVisc);

  /// SETTERS
  /// @brief Migrate this element from its current basis to `newBasis`.
  /// Re-interpolates rho, rhou, e and AV from the old basis's nodes onto the
  /// new basis's nodes (both must share the same order P), updates the basis
  /// pointer and recomputes the flux. No-op if `newBasis == current basis`.
  void setBasis(base::_Basis *newBasis);
  void setJ(double xL, double xR);
  void setID(int ID) { this->id = ID; }
  void setU1(double *rho) { this->rho = rho; }
  void setU2(double *rhou) { this->rhou = rhou; }
  void setU3(double *e) { this->e = e; }
  
  /// @brief Sets the Artificial Viscosity of the element
  /// @param arr Array containing AV values
  /// @note Requires `memcpy` because `diffuse` deletes `eps` array to avoid memory leaks
  void setAV(double *arr) {
    int n = basis->getOrder() + 1;
    memcpy(this->AV, arr, n * sizeof(double));
  }
  
  void setRho(int pos, double val) { rho[pos] = val; }
  void setRhoU(int pos, double val) { rhou[pos] = val; }
  void setE(int pos, double val) { e[pos] = val; }

  void setDivF1(int pos, double val) { divF1[pos] = val; }
  void setDivF2(int pos, double val) { divF2[pos] = val; }
  void setDivF3(int pos, double val) { divF3[pos] = val; }

  void setAV(int pos, double val) { AV[pos] = val;}
    /// From U -> F
  void setFlux();

  /// GETTERS
  const int *getID() const { return &id; }
  const base::_Basis *getBasis() const { return basis; };
  const double *getInvJ() const { return &invJ; }
  const double *getU1() const { return rho; }
  const double *getU2() const { return rhou; }
  const double *getU3() const { return e; }
  const double *getF1() const { return F1; }
  const double *getF2() const { return F2; }
  const double *getF3() const { return F3; }
  const double *getDivF1() const { return divF1; }
  const double *getDivF2() const { return divF2; }
  const double *getDivF3() const { return divF3; }
  double getX(int q) const { return xL + (basis->getQuads()[q] + 1.0) * J; }
  // Variation to get a single quad value
  double *getU1(int q) const { return rho + q; }
  double *getU2(int q) const { return rhou + q; }
  double *getU3(int q) const { return e + q; }
  double *getF1(int q) const { return F1 + q; }
  double *getF2(int q) const { return F2 + q; }
  double *getF3(int q) const { return F3 + q; }
  double getU(int q) const { return *(rhou + q) / *(rho + q); }
  double getRho(int q) const { return *(rho + q); }
  double getP(int q) const { return (GAMMA - 1.0) * (*(e + q) - 0.5 * *(rhou + q) * *(rhou + q) / *(rho + q)); }
  double *getModes() const { return legendreCoefficients; }

  /// Modify the flux (USED FOR REIMANN CORRECTION)
  double *getDivF1(int q) const { return divF1 + q; }
  double *getDivF2(int q) const { return divF2 + q; }
  double *getDivF3(int q) const { return divF3 + q; }

  double *getAV() const {return AV;}
  double *getAV(int q) const {return AV + q;}


  void correctDivF1(int pos, double val) { divF1[pos] += val; }
  void correctDivF2(int pos, double val) { divF2[pos] += val; }
  void correctDivF3(int pos, double val) { divF3[pos] += val; }


  /// @brief Accumulates the DG volume residual into divFk: `divFk_i = w_i * (D F_k)_i`. \n
  /// Computes the residual of the element
  void computeDivFlux();

  /// @brief Final stage of the DG residual: divFk <- (1/J) * Minv * divFk.
  /// Routes the volume term and the boundary lifts through the basis's mass
  /// matrix. For Lagrange (Minv = diag(1/w)) this collapses back to the
  /// classical (1/J) * D * F + (1/(J w_b)) * [F*-F]_b form; for RBF the
  /// boundary correction is spread to interior nodes via Minv.
  void applyMassInverse();
  /// Computes Legendre coefficients from Lagrange polynomials
  void computeLegendreCoefficients();
  /// Computes element pressure
  double* computePressure();
  /// Computes the pressure Laplacian
  const double* computePressureLaplacian();
  /// Computes the divergence of velocity
  const double* computeVelocityDivergence();

  ~Element();

private:
  double xL;
  double xR;

  int id;
  base::_Basis *basis;
  double J;
  double invJ;

  bool ownsMemory; // Boolean for destructor
  double *rho;
  double *rhou;
  double *e;

  double *F1;
  double *F2;
  double *F3;

  double *divF1;
  double *divF2;
  double *divF3;

  double *AV;

  double *legendreCoefficients;

  friend std::ostream &operator<<(std::ostream &, const Element &);

private:
  /// Core constructor: all public constructors delegate to this
  Element(int id, base::_Basis *basis, double xL, double xR, double *rho,
          double *rhou, double *e, double* AV, bool ownsMemory);
};
} // namespace elem

#endif
