#include "entropy_flux.h"
#include <cmath>
#include <algorithm>

static constexpr double GAMMA  = 1.4;
static constexpr double GM1    = GAMMA - 1.0;   // 0.4

namespace entropy {

double logmean(double a, double b) {
    // Ismail-Roe (2009) stable evaluation via Taylor series when a ≈ b.
    double xi = a / b;
    double f  = (xi - 1.0) / (xi + 1.0);
    double f2 = f * f;
    if (f2 < 1e-4) {
        // Taylor: 2 * (1 + f2/3 + f4/5 + f6/7)
        double u = 1.0 + f2 * (1.0/3.0 + f2 * (1.0/5.0 + f2 / 7.0));
        return (a + b) / (2.0 * u);
    }
    return (b - a) / (std::log(b) - std::log(a));
}

void chandrashekar_ec(double rho1, double rhou1, double e1,
                      double rho2, double rhou2, double e2,
                      double& f1, double& f2, double& f3) {
    const double u1 = rhou1 / rho1;
    const double u2 = rhou2 / rho2;
    const double p1 = GM1 * (e1 - 0.5 * rhou1 * u1);
    const double p2 = GM1 * (e2 - 0.5 * rhou2 * u2);

    const double beta1 = 0.5 * rho1 / p1;
    const double beta2 = 0.5 * rho2 / p2;

    const double rho_ln  = logmean(rho1, rho2);
    const double beta_ln = logmean(beta1, beta2);

    const double u_avg   = 0.5 * (u1 + u2);
    const double rho_avg = 0.5 * (rho1 + rho2);
    const double beta_avg = 0.5 * (beta1 + beta2);

    // p̂ = {ρ} / (2 {β})
    const double p_hat = 0.5 * rho_avg / beta_avg;

    // ĥ = 1/(2(γ-1)β^ln) + p̂/ρ^ln + {u}²/2
    const double h_hat = 1.0 / (2.0 * GM1 * beta_ln) + p_hat / rho_ln + 0.5 * u_avg * u_avg;

    f1 = rho_ln * u_avg;
    f2 = f1 * u_avg + p_hat;
    f3 = f1 * h_hat;
}

void entropy_stable_lf(double rho1, double rhou1, double e1,
                       double rho2, double rhou2, double e2,
                       double& f1, double& f2, double& f3) {
    chandrashekar_ec(rho1, rhou1, e1, rho2, rhou2, e2, f1, f2, f3);

    const double u1 = rhou1 / rho1;
    const double u2 = rhou2 / rho2;
    const double p1 = GM1 * (e1 - 0.5 * rhou1 * u1);
    const double p2 = GM1 * (e2 - 0.5 * rhou2 * u2);
    const double c1 = std::sqrt(GAMMA * p1 / rho1);
    const double c2 = std::sqrt(GAMMA * p2 / rho2);
    const double lambda = std::max(std::abs(u1) + c1, std::abs(u2) + c2);

    f1 -= 0.5 * lambda * (rho2  - rho1);
    f2 -= 0.5 * lambda * (rhou2 - rhou1);
    f3 -= 0.5 * lambda * (e2    - e1);
}

} // namespace entropy
