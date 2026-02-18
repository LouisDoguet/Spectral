#include <iostream>
#include <cstdlib>
#include "lib/base/gll.h"
#include "lib/space/mesh.h"
#include "lib/time/rk4.h"
#include <cmath>

int main() {
    //-- SIMULATION --
    const int P = 10;           
    const int N_elem = 100;    
    const double L = 1.0;      
    const double T_final = 0.5; 
    const double dt = 5e-5;    
    const int save_freq = 100; 

    //-- SETUP BASIS -- 
    gll::Basis* basis = new gll::Basis(P);
    double* rho_i = new double[N_elem]; 
    double* rhou_i = new double[N_elem]; 
    double* e_i = new double[N_elem];

    //-- INIT --
    for (int e = 0; e < N_elem; ++e) {
        double x_mid = ((double)e + 0.5) * (L / N_elem);
        double rho = 1.0 + 0.5 * std::exp(-100.0 * std::pow(x_mid - 0.3, 2));
        double u = 1.0; double p = 1.0;
        rho_i[e] = rho; rhou_i[e] = rho * u; 
        e_i[e] = p / (1.4 - 1.0) + 0.5 * rho * u * u; 
    }

    //-- MESH INIT --
    mesh::Mesh* mesh = new mesh::Mesh(N_elem, basis, 0.0, L, rho_i, rhou_i, e_i);
    //-- SOLVER INIT --
    solver::RK4 solver(mesh);
    
    //-- SIMULATION -- 
    solver.run(T_final, dt, save_freq, "results/smooth_advection");


    // Cleanup
    delete mesh; delete basis; delete[] rho_i; delete[] rhou_i; delete[] e_i;
    return 0;
}
