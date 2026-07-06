#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

namespace bc {

/**
 * @brief Abstract boundary condition for a 1D Euler domain edge.
 *
 * A boundary condition is expressed purely as the *ghost (exterior) state* it
 * presents just outside the domain. The mesh then evaluates exactly the same
 * numerical flux it uses at interior faces (Rusanov for the strong-form DG
 * residual, the entropy-stable LLF flux for the DGSEM/FV residuals), feeding it
 * the interior boundary-node state and this ghost state. Keeping the flux
 * machinery in one place means a new boundary type only has to answer the
 * question "what lies just outside the boundary?".
 */
class _BoundaryConditions {
public:
    virtual ~_BoundaryConditions();

    /**
     * @brief Fill the exterior (ghost) conservative state from the interior one.
     * @param rho_in,rhou_in,e_in     Interior boundary-node state (rho, rho*u, E).
     * @param rho_out,rhou_out,e_out  Ghost state consumed by the boundary flux.
     */
    virtual void ghostState(double rho_in, double rhou_in, double e_in,
                            double &rho_out, double &rhou_out,
                            double &e_out) const = 0;

    /// Human-readable name (diagnostics / logging).
    virtual const char *name() const = 0;
};

/**
 * @brief Fixed-state (Dirichlet) wall.
 *
 * The ghost state is a prescribed constant, independent of the interior field.
 * This reproduces the classic far-field inflow/outflow used by the shock-tube
 * cases (Sod, Lax, Shu-Osher) and is the default boundary type of the mesh.
 */
class Wall : public _BoundaryConditions {
public:
    Wall(double rho, double rhou, double e) : rho_(rho), rhou_(rhou), e_(e) {}

    void ghostState(double /*rho_in*/, double /*rhou_in*/, double /*e_in*/,
                    double &rho_out, double &rhou_out,
                    double &e_out) const override {
        rho_out = rho_;
        rhou_out = rhou_;
        e_out = e_;
    }

    const char *name() const override { return "Wall"; }

private:
    double rho_, rhou_, e_; ///< prescribed exterior conservative state
};

/**
 * @brief Reflecting (solid slip) wall.
 *
 * The ghost mirrors the interior state with the momentum reversed:
 *   rho_out = rho_in,  (rho u)_out = -(rho u)_in,  E_out = E_in.
 * Density, pressure and speed are preserved while the sign of the velocity is
 * flipped, so the numerical flux sees zero normal velocity at the interface —
 * an exact reflecting wall (only the pressure term survives in the momentum
 * flux). This is the physically correct boundary for the Woodward-Colella
 * two-blast-wave problem, whose tube is closed at both ends.
 */
class Reflective : public _BoundaryConditions {
public:
    void ghostState(double rho_in, double rhou_in, double e_in,
                    double &rho_out, double &rhou_out,
                    double &e_out) const override {
        rho_out = rho_in;
        rhou_out = -rhou_in; // reversed momentum -> zero velocity at the wall
        e_out = e_in;        // same rho and |u| -> same pressure and energy
    }

    const char *name() const override { return "Reflective"; }
};

} // namespace bc

#endif
