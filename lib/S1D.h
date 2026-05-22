#ifndef S1D_H
#define S1D_H

namespace S1D
{
    void RunShockTube(
        const int N_elem, const int P, const int Q,
        const double L, const double T_final,
        const double dt, const double eps,
        const double rhoL, const double uL, const double pL, 
        const double rhoR, const double uR, const double pR, 
        int x0, double delta);
} // namespace S1D


#endif