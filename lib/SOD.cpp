#include "base/gll.h"
#include "diffusion/diffusion.h"
#include "math/math.h"
#include "space/mesh.h"
#include "time/rk4.h"
#include "S1D.h"
#include <boost/program_options.hpp>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
namespace po = boost::program_options;

namespace S1D {
void RunShockTube(
        const int N_elem, const int P, const int Q,
        const double L, const double T_final,
        const double dt, const double eps,
        const double rhoL, const double uL, const double pL, 
        const double rhoR, const double uR, const double pR, 
        int x0, double delta) {

    double gamma = 1.4;
    
    const double dx   = L / N_elem;
    const int    N_nodes  = N_elem * (P + 1);
    gll::Basis *basis  = new gll::Basis(P);
    double *rho_i  = new double[N_nodes];
    double *rhou_i = new double[N_nodes];
    double *e_i    = new double[N_nodes];
    double bc_rhoL, bc_rhouL, bc_eL;
    double bc_rhoR, bc_rhouR, bc_eR;

    if (delta < 0.0) delta = 2.0 * dx;

    for (int i = 0; i < N_nodes; ++i) {
        int    elem = i / (P + 1);
        int    q    = i % (P + 1);
        double x    = (elem * dx) + (basis->getQuads()[q] + 1.0) * dx / 2.0;

        double rho, u, p;
        if (delta == 0.0) {
        // Sharp discontinuity (may be unstable for high P)
        rho = (x < x0) ? rhoL : rhoR;
        u   = 0.0;
        p   = (x < x0) ? pL   : pR;
        } else {
        // Tanh-smoothed discontinuity
        double s = 0.5 * (1.0 - std::tanh((x - x0) / delta));
        rho = rhoR + (rhoL - rhoR) * s;
        u   = 0.0;
        p   = pR   + (pL   - pR)   * s;
        }

        rho_i[i]  = rho;
        rhou_i[i] = rho * u;
        e_i[i]    = p / (gamma - 1.0) + 0.5 * rho * u * u;
    }

    bc_rhoL  = rhoL; bc_rhouL = 0.0; bc_eL = pL / (gamma - 1.0);
    bc_rhoR  = rhoR; bc_rhouR = 0.0; bc_eR = pR / (gamma - 1.0);

    std::string case_name = "results/sod_shock_tube";

    //-- MESH --
    mesh::Mesh *mesh = new mesh::Mesh(N_elem, basis, 0.0, L,
                                      rho_i, rhou_i, e_i,
                                      bc_rhoL, bc_rhouL, bc_eL,
                                      bc_rhoR, bc_rhouR, bc_eR);

    //-- SOLVER --
    solver::RK4 solver(mesh, Q);

    //-- DIFFUSION MODE --
    DIFF::Constant constant_diff(eps);
    if (eps>0) solver.setDiffusion(&constant_diff);

    //-- RUN --
    solver.run(T_final, dt, 10, case_name);

    delete mesh;
    delete basis;
    delete[] rho_i;
    delete[] rhou_i;
    delete[] e_i;
}   
}
