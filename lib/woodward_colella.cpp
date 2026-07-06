#include "test_cases.h"
#include "base/base.h"
#include "boundary_conditions/boundary_conditions.h"
#include "space/mesh.h"
#include <cmath>

namespace S1D {

// Woodward-Colella two interacting blast waves (Woodward & Colella 1984).
// Two strong shocks are launched from high-pressure pockets at each end of a
// closed tube; they collide and interact, producing a demanding pattern of
// shocks and contact discontinuities.
//
//   rho = 1, u = 0 everywhere, gamma = 1.4, on domain [0,1] with three
//   pressure regions:
//     x < 0.1        : p = 1000
//     0.1 <= x <= 0.9: p = 0.01
//     x > 0.9        : p = 100
//   Recommended final time T = 0.038.
//
// The tube is closed: both ends use true reflecting walls (bc::Reflective),
// which is essential here since the blast waves reach the boundaries and
// reflect back into the domain before the final time.
mesh::Mesh* generateWoodwardColella(base::_Basis* basis, int N_elem,
                                    double delta) {
    const double gamma = 1.4;
    const int    P       = basis->getOrder();
    const double xL      = 0.0;
    const double xR      = 1.0;
    const double L       = xR - xL;

    const double x1 = 0.1; // left  high-pressure / mid interface
    const double x2 = 0.9; // mid   / right high-pressure interface

    const double pLeft = 1000.0, pMid = 0.01, pRight = 100.0;
    const double rho0  = 1.0;    // uniform density, fluid at rest

    const double dx      = L / N_elem;
    const int    N_nodes = N_elem * (P + 1);

    if (delta < 0.0) delta = 2.0 * dx;

    // Pressure profile: piecewise-constant, optionally tanh-smoothed at the
    // two interfaces to avoid an under-resolved sharp jump at high P.
    auto pressure = [&](double x) -> double {
        if (delta == 0.0)
            return (x < x1) ? pLeft : (x > x2 ? pRight : pMid);
        double sL = 0.5 * (1.0 - std::tanh((x - x1) / delta)); // 1 left of x1
        double sR = 0.5 * (1.0 + std::tanh((x - x2) / delta)); // 1 right of x2
        return pMid + (pLeft - pMid) * sL + (pRight - pMid) * sR;
    };

    double *rho_i  = new double[N_nodes];
    double *rhou_i = new double[N_nodes];
    double *e_i    = new double[N_nodes];

    for (int i = 0; i < N_nodes; ++i) {
        int    elem = i / (P + 1);
        int    q    = i % (P + 1);
        double x    = xL + (elem * dx) + (basis->getQuads()[q] + 1.0) * dx / 2.0;

        double p = pressure(x);
        rho_i[i]  = rho0;
        rhou_i[i] = 0.0;
        e_i[i]    = p / (gamma - 1.0); // u = 0 -> no kinetic energy
    }

    // The ghost states below are placeholders; they are immediately replaced
    // by reflecting walls, which derive the ghost state from the interior.
    double bc_rhoL = rho0, bc_rhouL = 0.0, bc_eL = pLeft  / (gamma - 1.0);
    double bc_rhoR = rho0, bc_rhouR = 0.0, bc_eR = pRight / (gamma - 1.0);

    mesh::Mesh *mesh = new mesh::Mesh(N_elem, basis, xL, xR,
                                      rho_i, rhou_i, e_i,
                                      bc_rhoL, bc_rhouL, bc_eL,
                                      bc_rhoR, bc_rhouR, bc_eR);

    // Closed tube: reflecting walls at both ends.
    mesh->setLeftBC(new bc::Reflective());
    mesh->setRightBC(new bc::Reflective());
    return mesh;
}

} // namespace S1D
