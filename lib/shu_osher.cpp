#include "test_cases.h"
#include "base/base.h"
#include "space/mesh.h"
#include <cmath>

namespace S1D {

// Shu-Osher problem (Shu & Osher 1989): a Mach-3 shock advecting into a
// sinusoidal density field. It couples a strong discontinuity with smooth
// high-frequency structures, so it probes both the shock-capturing and the
// high-order accuracy of the scheme at once.
//
//   Left  state (x < -4): rho=3.857143, u=2.629369, p=10.33333   (post-shock)
//   Right state (x > -4): rho=1 + 0.2*sin(5x), u=0, p=1          (entropy wave)
//   Domain [-5,5], gamma=1.4, recommended final time T=1.8.
mesh::Mesh* generateShuOsher(base::_Basis* basis, int N_elem, double delta) {
    const double gamma = 1.4;
    const int    P       = basis->getOrder();
    const double xL      = -5.0;
    const double xR      =  5.0;
    const double L       = xR - xL;
    const double x_shock = -4.0;

    const double dx      = L / N_elem;
    const int    N_nodes = N_elem * (P + 1);

    // Post-shock (left) constant state.
    const double rhoL = 3.857143, uL = 2.629369, pL = 10.33333;

    // Sinusoidal density field on the right, at rest, unit pressure.
    auto rho_right = [](double x) { return 1.0 + 0.2 * std::sin(5.0 * x); };

    if (delta < 0.0) delta = 2.0 * dx;

    double *rho_i  = new double[N_nodes];
    double *rhou_i = new double[N_nodes];
    double *e_i    = new double[N_nodes];

    for (int i = 0; i < N_nodes; ++i) {
        int    elem = i / (P + 1);
        int    q    = i % (P + 1);
        double x    = xL + (elem * dx) + (basis->getQuads()[q] + 1.0) * dx / 2.0;

        double rhoR = rho_right(x);
        double rho, u, p;
        if (delta == 0.0) {
            // Sharp shock front.
            bool left = (x < x_shock);
            rho = left ? rhoL : rhoR;
            u   = left ? uL   : 0.0;
            p   = left ? pL   : 1.0;
        } else {
            // Tanh blend between the post-shock state and the entropy wave.
            double s = 0.5 * (1.0 - std::tanh((x - x_shock) / delta));
            rho = rhoR + (rhoL - rhoR) * s;
            u   = 0.0  + (uL   - 0.0 ) * s;
            p   = 1.0  + (pL   - 1.0 ) * s;
        }

        rho_i[i]  = rho;
        rhou_i[i] = rho * u;
        e_i[i]    = p / (gamma - 1.0) + 0.5 * rho * u * u;
    }

    // Left: fixed inflow at the post-shock state. Right: local (unperturbed)
    // sinusoidal state, which the shock does not reach within the run time.
    double bc_rhoL = rhoL, bc_rhouL = rhoL * uL,
           bc_eL   = pL / (gamma - 1.0) + 0.5 * rhoL * uL * uL;
    double rhoR_bc = rho_right(xR);
    double bc_rhoR = rhoR_bc, bc_rhouR = 0.0,
           bc_eR   = 1.0 / (gamma - 1.0);

    mesh::Mesh *mesh = new mesh::Mesh(N_elem, basis, xL, xR,
                                      rho_i, rhou_i, e_i,
                                      bc_rhoL, bc_rhouL, bc_eL,
                                      bc_rhoR, bc_rhouR, bc_eR);
    return mesh;
}

} // namespace S1D
