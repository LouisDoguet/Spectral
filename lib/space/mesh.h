#ifndef MESH_H
#define MESH_H

#include "../base/base.h"
#include "../sensor/sensor.h"
#include "element.h"

namespace mesh {
/**
 * @brief Class storing complete mesh
 */
class Mesh {
public:
  /**
   * @brief Construct a new mesh
   * @param n Mesh number of points
   * @param basis Basis of the elements
   * @param xL Beginning of the mesh
   * @param xR End of the mesh
   */
  Mesh(const int n, base::_Basis *basis, double xL, double xR);
  /**
   * @brief Construct a new mesh, with initial parameters
   * @param n Mesh number of points
   * @param basis Basis of the elements
   * @param xL Beginning of the mesh
   * @param xR End of the mesh
   * @param init_u1 U1 initial values (assigned to the entire element)
   * @param init_u2 U2 initial values (assigned to the entire element)
   * @param init_u3 U3 initial values (assigned to the entire element)
   * @param u1_L Left BC
   * @param u2_L
   * @param u3_L
   * @param u1_R Right BC
   * @param u2_R
   * @param u3_R
   */
  Mesh(const int n, base::_Basis *basis, double xL, double xR, double *init_u1,
       double *init_u2, double *init_u3, double u1_L, double u2_L, double u3_L,
       double u1_R, double u2_R, double u3_R);

  /// GETTERS
  const elem::Element *getElem(int i) const { return elem[i]; }
  elem::Element *getElem(int i) { return elem[i]; }
  int getNumElements() const { return n; }
  // Getters for global contiguous buffers
  double *getGlobalU1() { return global_rho; }
  double *getGlobalU2() { return global_rhou; }
  double *getGlobalU3() { return global_e; }
  const base::_Basis *getBasis() { return elem[0]->getBasis(); }
  int getTotalPoints() const {
    return n * (elem[0]->getBasis()->getOrder() + 1);
  }

  /// Flux solving routines
  void computeElements();   // Computes df/dx
  void computeInterfaces(); // Computes the Reimann problem at interface
  /**
   * @brief Find R(U) for an element (dU/dt = -dF/dx)
   * From the initialized U, for all elements:
   * - Computes F
   * - Applies the divergence (dFdx)
   * - Applies the Reimann correction at the elements boundaries
   * For the first and last elements:
   * - Applies Boundary Conditions
   */
  void computeResidual();

  /// Boundary conditions
  void applyDirichlet();

  /// @brief Register the alternative basis (e.g. IMQ) used to resolve
  ///   discontinuities. Must share the same order P as the primary basis.
  void setAltBasis(base::_Basis* alt) { alt_basis = alt; }
  base::_Basis* getPrimaryBasis() const { return primary_basis; }
  base::_Basis* getAltBasis()    const { return alt_basis; }

  /// @brief Per-element basis switching driven by the Persson-Peraire shock
  ///   indicator log10(Se):
  ///     - currently on primary, logSe > s_shock  -> migrate to alt
  ///     - currently on alt,     logSe < s_smooth -> migrate back to primary
  ///   Each migration triggers Element::setBasis (re-interpolation + setFlux).
  ///   Caller is responsible for calling this between time steps only.
  void adaptBasis(sens::PerssonPeraire& sensor, int truncation,
                  double s_shock, double s_smooth);

  ~Mesh();

private:
  const int n;
  elem::Element **elem;
  base::_Basis* primary_basis = nullptr;  // smooth-region basis (Lagrange)
  base::_Basis* alt_basis     = nullptr;  // shock-resolving basis (RBF)

  /// Unified variables
  double *global_rho;
  double *global_rhou;
  double *global_e;
  /// Unified Artificial Viscosities
  double *global_AV;


  /// Boudary conditions
  double u1_L;
  double u2_L;
  double u3_L;
  double u1_R;
  double u2_R;
  double u3_R;

  friend std::ostream &operator<<(std::ostream &, const Mesh &);
};
} // namespace mesh

#endif
