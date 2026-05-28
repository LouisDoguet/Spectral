#include "base/base.h"
#include "diffusion/diffusion.h"
#include "math/math.h"
#include "space/mesh.h"
#include "time/solver.h"
#include "sensor/sensor.h"
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

mesh::Mesh* generateMesh(
        const int N_elem, const int P, const double L,
        const double rhoL, const double uL, const double pL, 
        const double rhoR, const double uR, const double pR, 
        double x0, double delta) {
        
    double gamma = 1.4;
    
    const double dx   = L / N_elem;
    const int    N_nodes  = N_elem * (P + 1);
    base::InverseMultiQuadratic *basis  = new base::InverseMultiQuadratic(P,2.);
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

    return mesh;    
};

void RunShockTube(
    solver::_Solver* solver, diff::_Diffusion* diffusion, sens::_Sensor* sensor, 
    double T_final, double dt, std::string case_name) {

    //-- DIFFUSION MODE --
    if (diffusion) solver->setDiffusion(diffusion);
    if (sensor) solver->setSensor(sensor);

    //-- RUN --
    solver->run(T_final, dt, 10, case_name);
}   
}
