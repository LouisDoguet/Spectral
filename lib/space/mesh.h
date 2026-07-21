#ifndef MESH_H
#define MESH_H

#include "../base/base.h"
#include "../boundary_conditions/boundary_conditions.h"
#include "../sensor/sensor.h"
#include "element.h"
#include <vector>

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
  
  /// @brief Getter for the element `i`
  /// @param i Index of the element
  /// @return const `element` pointer
  const elm::Element *getElem(int i) const { return elem[i]; }

  /// @brief Getter for the element `i`
  /// @param i Index of the element
  /// @return `element` pointer
  elm::Element *getElem(int i) { return elem[i]; }

  int getNumElements() const { return n; }

  // Getters for global contiguous buffers
  double *getGlobalU1() { return global_rho; }
  double *getGlobalU2() { return global_rhou; }
  double *getGlobalU3() { return global_e; }

  /// @brief Getter for the mesh basis
  /// @return `_Basis` pointer
  const base::_Basis *getBasis() { return elem[0]->getBasis(); }
  int getTotalPoints() const {
    return n * (elem[0]->getBasis()->getOrder() + 1);
  }

  /// Flux solving routines
  void computeElements();   // Computes `df/dx`
  void computeInterfaces(); // Computes the Reimann problem at interface

  /**
   * @brief Standard nodal DG residual `(dU/dt = -dF/dx)`.
   *
   * The classical strong-form DGSEM used by the plain RK4 solver: 
   * 
   * - volume term :  divergence of the nodal flux `F(U)`          (`computeElements`)
   * 
   * - surface term: Rusanov numerical flux at element faces       (`computeInterfaces`)
   * 
   * - boundaries:   Dirichlet Rusanov lift                        (`applyDirichlet`)
   * 
   * - closure:      `(1/J) * Minv * divF`                         (`applyMassInverse`)
   *
   * @note This is NOT entropy stable (Rusanov + strong form). The
   *       entropy-stable high-order operator is computeDGResidual().
   */
  void computeResidual();

  /**
   * @brief Entropy-stable high-order DG residual (Hennemann et al. 2021).
   *
   * The alpha = 0 limit of the hybrid scheme:
   * 
   * - volume term:  entropy-conservative (Chandrashekar) split-form flux,
   *                 assembled as subcell fluxes via the SBP recurrence;
   * 
   * - surface term: entropy-stable (EC + LLF) flux at element faces / boundaries.
   *
   * Result is left in each element's divF array (post mass-inverse), ready to be
   * gathered by the solver exactly like computeResidual().
   */
  void computeDGResidual();

  /**
   * @brief First-order entropy-stable finite-volume residual.
   *
   * Each LGL node is treated as a subcell. 
   * 
   * The interior subcell fluxes are the entropy-stable two-point flux `f*_ES(u_j, u_{j+1})` and the element faces use the same ES flux as the DG
   * residual, so the two residuals share an identical surface term.
   */
  void computeFVResidual();

  /**
   * @brief Hybrid DGSEM residual: blend of the DG and FV SUBCELL FLUXES
   *        (Hennemann et al. 2021, Eq. 18), matching nn/jax_dgsem/solver.py.
   *
   *   B_hyb = (1 - alpha) * B_DG + alpha * B_FV   (per subcell interface)
   *   divF  = B_hyb[j+1] - B_hyb[j]               (then surface + mass-inverse)
   *
   * Blending the fluxes (not the residuals) keeps the scheme conservative and
   * entropy-stable for a per-interface alpha. For a per-element CONSTANT alpha
   * it is algebraically identical to the old residual blend.
   *
   * @param alpha  Per interior subcell interface blending factors, size
   *               n_elem * P (P = Nn-1 interior interfaces per element), in
   *               element-major order: alpha[e*P + i], i = 0..P-1. The two
   *               element-boundary interfaces (0 and Nn) are never blended
   *               (physical flux, alpha = 0), matching jax _alpha_on_interfaces.
   *               `alpha=0` everywhere: pure EC DGSEM. `alpha=1`: pure ES FV.
   */
  void computeHybridResidual(const double* alpha);

  /// @brief Per-node density residual difference (DG - FV), size n_elem*Nn,
  /// element-major. Matches nn/network/policy.py channel_residual (before its
  /// max-abs normalization). NOTE: overwrites the elements' divF (recomputed by
  /// the next residual call), so use it only inside computeAlpha().
  void densityResidualDifference(std::vector<double>& out);

  /// @brief Persson-Peraire modal-energy indicator E_ind per element
  /// (n_elem), = max(E_N, E_{N-1}). Matches nn/jax_dgsem/indicator.py.
  void perssonPeraireIndicator(std::vector<double>& eind);

  /// Boundary conditions
  void applyDirichlet();

  /// @brief Replace the left / right boundary condition. The mesh takes
  ///   ownership of the passed object and deletes any previously held one.
  ///   Passing nullptr disables the corresponding boundary lift.
  void setLeftBC(bc::_BoundaryConditions *b);
  void setRightBC(bc::_BoundaryConditions *b);
  const bc::_BoundaryConditions *getLeftBC() const { return bc_left; }
  const bc::_BoundaryConditions *getRightBC() const { return bc_right; }

  /**
   * @brief Largest explicit time step allowed by the CFL condition for the
   *        current field.
   *
   * For a nodal DG / DGSEM discretisation of polynomial degree P the linear
   * stability limit of an explicit Runge-Kutta integrator is
   *
   *     dt <= CFL * dx / ((2*P + 1) * max|lambda|)
   *
   * The (2*P + 1) factor is the RKDG eigenvalue/CFL scaling of Cockburn & Shu,
   * "Runge-Kutta Discontinuous Galerkin Methods for Convection-Dominated
   * Problems", J. Sci. Comput. 16 (2001) 173-261: a degree-P DG operator is
   * stable up to a Courant number of 1/(2*P + 1). Here `dx` is the (smallest)
   * element width and `lambda = |u| + c`, c = sqrt(gamma*p/rho), is the Euler
   * spectral radius, maximised over every solution node.
   *
   * @param cfl Safety factor (Courant number), typically 0 < cfl <= 1.
   * @return Maximum stable dt; the caller may pass it straight to run().
   */
  double computeCFLTimeStep(double cfl) const;

  /// @brief Register the alternative basis (e.g. IMQ) used to resolve
  ///   discontinuities. Must share the same order P as the primary basis.
  void setAltBasis(base::_Basis* alt) { alt_basis = alt; }
  base::_Basis* getPrimaryBasis() const { return primary_basis; }
  base::_Basis* getAltBasis()    const { return alt_basis; }

  /// @brief Per-element basis switching driven by the Persson-Peraire shock indicator `log10(Se)`:
  ///
  ///     - currently on primary, `logSe > s_shock`  -> migrate to alt
  ///
  ///     - currently on alt,     `logSe < s_smooth` -> migrate back to primary
  ///
  ///   Each migration triggers `Element::setBasis` (re-interpolation + setFlux).
  ///   Caller is responsible for calling this between time steps only.
  /// @note NOT USED ANYMORE - Base switch not efficient
  void adaptBasis(sens::PerssonPeraire& sensor, int truncation,
                  double s_shock, double s_smooth);

  ~Mesh();

private:
  /// @name Hybrid-scheme building blocks (shared by the DG / FV / hybrid residuals)
  /// @{
  /// Build the Nn+1 high-order (entropy-conservative) subcell interface fluxes
  /// of element e via the SBP recurrence. B* must have length Nn+1.
  void buildSubcellFluxDG(int e, double *B1, double *B2, double *B3) const;
  /// Build the Nn+1 first-order (entropy-stable) subcell interface fluxes of
  /// element e. B* must have length Nn+1.
  void buildSubcellFluxFV(int e, double *B1, double *B2, double *B3) const;
  /// Store the subcell-flux divergence divF_j = B_{j+1} - B_j into element e.
  void storeSubcellDivergence(int e, const double *B1, const double *B2,
                              const double *B3);
  /// Entropy-stable surface lift at interior element faces (shared DG/FV term).
  void applyEntropyStableInterfaces();
  /// Entropy-stable surface lift at the two domain boundaries.
  void applyEntropyStableBoundaries();
  /// @}

  const int n;
  elm::Element **elem;
  base::_Basis* primary_basis = nullptr;  // smooth-region basis (Lagrange)
  base::_Basis* alt_basis     = nullptr;  // shock-resolving basis (RBF)

  /// Unified variables
  double *global_rho;
  double *global_rhou;
  double *global_e;
  /// Unified Artificial Viscosities
  double *global_AV;


  /// Boundary conditions (owned). Default to fixed-state bc::Wall built from the
  /// ghost states passed to the constructor; swap with setLeft/RightBC.
  bc::_BoundaryConditions *bc_left = nullptr;
  bc::_BoundaryConditions *bc_right = nullptr;

  friend std::ostream &operator<<(std::ostream &, const Mesh &);
};
} // namespace mesh

#endif
