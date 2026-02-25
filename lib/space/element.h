#ifndef ELEMENT_H
#define ELEMENT_H

#include "../base/gll.h"

namespace elem {
/**
 * @brief Class storing an element
 */
class Element {
public:
  /**
   * @brief Constructor basic (no values) DEV/DEBUG
   */
  Element(const int id, gll::Basis *sharedBasis, double xL, double xR);
  /**
   * @brief Constructor with init values DEV/DEBUG
   */
  Element(const int id, gll::Basis *sharedBasis, double xL, double xR,
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
   */
  Element(const int id, gll::Basis *sharedBasis, double xL, double xR,
          double *external_rho, double *external_rhou, double *external_e);

  /// SETTERS
  void setBasis(gll::Basis *sharedBasis) { this->basis = sharedBasis; }
  void setJ(double xL, double xR);
  void setID(int ID) { this->id = ID; }
  void setU1(double *rho) { this->rho = rho; }
  void setU2(double *rhou) { this->rhou = rhou; }
  void setU3(double *e) { this->e = e; }

  /// GETTERS
  const int *getID() const { return &id; }
  const gll::Basis *getBasis() const { return basis; };
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
  double *getDivF1(int q) const { return divF1 + q; }
  double *getDivF2(int q) const { return divF2 + q; }
  double *getDivF3(int q) const { return divF3 + q; }

  /// Modify the flux (USED FOR REIMANN CORRECTION)
  void correctDivF1(int pos, double val) { divF1[pos] += val; }
  void correctDivF2(int pos, double val) { divF2[pos] += val; }
  void correctDivF3(int pos, double val) { divF3[pos] += val; }
  /// From U -> F
  void setFlux();
  /// df/dx using the base's derivative matrix
  void computeDivFlux();

  ~Element();

private:
  double xL;
  double xR;

  int id;
  gll::Basis *basis;
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

  friend std::ostream &operator<<(std::ostream &, const Element &);
};
} // namespace elem

#endif
