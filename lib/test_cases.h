#ifndef TEST_CASES_H
#define TEST_CASES_H

#include "base/base.h"
#include "diffusion/diffusion.h"
#include "sensor/sensor.h"
#include "space/mesh.h"
#include "time/solver.h"

namespace S1D
{
    /**
     * @brief Generic two-state Riemann initial condition on the domain [0, L].
     *
     * Left state  (x < x0): (rhoL, uL, pL)
     * Right state (x > x0): (rhoR, uR, pR)
     *
     * @param delta tanh smoothing half-width of the initial discontinuity.
     *              delta < 0 -> 2*dx (default), delta == 0 -> sharp jump.
     *
     * Building block reused by the Sod and Lax shock tubes.
     */
    mesh::Mesh* generateMesh(
        base::_Basis* basis,
        const int N_elem, const double L,
        const double rhoL, const double uL, const double pL,
        const double rhoR, const double uR, const double pR,
        double x0, double delta);

    // --- Standard 1D Euler benchmark cases --------------------------------
    // Each builds its own canonical domain and initial state on N_elem
    // elements of the supplied basis. `delta` controls the initial
    // discontinuity smoothing (see generateMesh); < 0 selects 2*dx.

    /// Lax shock tube. Domain [0,1], discontinuity at x0=0.5. Recommended T=0.16.
    mesh::Mesh* generateLax(base::_Basis* basis, int N_elem, double delta = -1.0);

    /// Shu-Osher shock/entropy-wave interaction. Domain [-5,5], shock at
    /// x=-4 impinging on a sinusoidal density field. Recommended T=1.8.
    mesh::Mesh* generateShuOsher(base::_Basis* basis, int N_elem,
                                 double delta = -1.0);

    /// Woodward-Colella two interacting blast waves. Domain [0,1] with
    /// reflecting walls at both ends. Recommended T=0.038.
    mesh::Mesh* generateWoodwardColella(base::_Basis* basis, int N_elem,
                                        double delta = -1.0);

    void RunShockTube(
        solver::_Solver* solver,
        double T_final, double dt, std::string case_name,
        diff::_Diffusion* diffusion = nullptr, sens::_Sensor* sensor = nullptr);
} // namespace S1D

#endif
