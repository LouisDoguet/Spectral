#ifndef PHYSICS_H
#define PHYSICS_H

/// Check page 60 Laizet notes

namespace phy {
    /**
     * @brief Computes pressure thanks to perfect gas law
     * @param p Pressure to be filled
     * @param rho
     * @param u
     * @param e
     * @param n Size
     * @return void
     */
    void getP(double* p, double* rho, double* rhou, double* e, int n);
    
    /**
     * @brief Computes F (Flux) from U (flow field) values
     * @param f1 rho*u
     * @param f2 rho*u^2 + p
     * @param f3 u*(e+p)
     * @param u1 rho
     * @param u2 u
     * @param u3 E
     * @param n Size
     * @return void
     */
    void computeFlux(double* f1, double* f2, double* f3, double* u1, double* u2, double* u3, double* p, int n);

}

namespace reimann {
    /**
     */
    double computeMaxWaveSpeed(double rho, double u, double p);    
    double Rusanov(double FL, double FR, double UL, double UR, double lambda);

}



#endif
