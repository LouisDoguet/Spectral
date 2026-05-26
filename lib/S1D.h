#ifndef S1D_H
#define S1D_H

namespace S1D
{
    mesh::Mesh* generateMesh(
        const int N_elem, const int P, const double L,
        const double rhoL, const double uL, const double pL, 
        const double rhoR, const double uR, const double pR, 
        double x0, double delta);

    void RunShockTube(
        solver::_Solver* solver, diff::_Diffusion* diffusion, sens::_Sensor* sensor, 
        double T_final, double dt, std::string case_name);
} // namespace S1D


#endif