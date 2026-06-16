#ifndef ENTROPY_FLUX_H
#define ENTROPY_FLUX_H

namespace entropy {

/**
 * @brief Numerically stable logarithmic mean.
 *   log_mean(a, b) = (b - a) / (ln b - ln a)
 *   Uses a Taylor expansion near a == b to avoid cancellation.
 */
double logmean(double a, double b);

/**
 * @brief Chandrashekar (2013) entropy-conservative two-point flux for 1D Euler.
 *
 * The flux satisfies  (v_k - v_j)^T * f*_EC(j,k) == psi_k - psi_j
 * (zero entropy production), where v = entropy variables and psi = entropy
 * potential.
 *
 * @param rho1 rhou1 e1   Left state  (rho, rho*u, total energy)
 * @param rho2 rhou2 e2   Right state
 * @param f1 f2 f3        Output flux components
 */
void chandrashekar_ec(double rho1, double rhou1, double e1, double rho2,
                      double rhou2, double e2, double &f1, double &f2,
                      double &f3);

/**
 * @brief Entropy-stable (ES) two-point flux: Chandrashekar EC + LLF
 * dissipation.
 *
 *   f*_ES = f*_EC - (lambda_max / 2) * (U_k - U_j)
 *
 * This is entropy stable because (v_k - v_j)^T * (U_k - U_j) >= 0
 * (by convexity of the entropy).
 */
void entropy_stable_lf(double rho1, double rhou1, double e1, double rho2,
                       double rhou2, double e2, double &f1, double &f2,
                       double &f3);

} // namespace entropy

#endif
