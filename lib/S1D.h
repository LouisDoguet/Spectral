#ifndef S1D_H
#define S1D_H

#include "base/base.h"
#include "diffusion/diffusion.h"
#include "sensor/sensor.h"
#include "space/mesh.h"
#include "time/solver.h"

namespace S1D
{
    mesh::Mesh* generateMesh(
        base::_Basis* basis,
        const int N_elem, const double L,
        const double rhoL, const double uL, const double pL, 
        const double rhoR, const double uR, const double pR, 
        double x0, double delta);

    void RunShockTube(
        solver::_Solver* solver, 
        double T_final, double dt, std::string case_name,
        diff::_Diffusion* diffusion = nullptr, sens::_Sensor* sensor = nullptr);
} // namespace S1D


#endif