#ifndef HYBRID_SOLVER_H
#define HYBRID_SOLVER_H

#include "solver.h"
#include "hybrid_alpha.h"
#include <vector>

namespace solver {

/**
 * @brief Entropy-stable hybrid DGSEM solver (Hennemann et al., JCP 2021).
 *
 * Each time step:
 *  1. Compute per-element blending factor alpha[e] from the Persson-Peraire
 *     modal energy indicator applied to density (Eqs. 40-48 of the paper).
 *  2. Advance U^n -> U^{n+1} with classical RK4, using the hybrid residual
 *     Mesh::computeHybridResidual(alpha) = (1-alpha) R_DG + alpha R_FV, a
 *     direct convex combination of the entropy-stable DG and FV residuals
 *     (Mesh::computeDGResidual / Mesh::computeFVResidual).
 *
 * The blending factor controls the amount of first-order FV dissipation:
 *   alpha = 0 -> pure high-order entropy-conservative DGSEM (R_DG).
 *   alpha = 1 -> pure first-order entropy-stable FV (R_FV).
 *
 * Parameters (set via setIndicatorParams):
 *   alpha_max  : cap on alpha (default 0.5, recommended for N=4).
 *   alpha_min  : values below this are clipped to 0 (default 0.001).
 *   diffuse    : whether to spread alpha to neighbours (default true).
 */
class HybridDGSEM : public _Solver {
public:
    HybridDGSEM(mesh::Mesh* mesh, int n_plot = 0);

    /**
     * @brief Configure the blending indicator thresholds.
     * @param alpha_max  Maximum blending factor (0..1, default 0.5).
     * @param alpha_min  Zero-clipping threshold (default 0.001).
     * @param diffuse    Spread alpha to face-adjacent elements (default true).
     */
    void setIndicatorParams(double alpha_max = 0.5,
                            double alpha_min = 0.001,
                            bool   diffuse   = true) {
        alpha_max_  = alpha_max;
        alpha_min_  = alpha_min;
        diffuse_    = diffuse;
    }

    void step(double dt) override;
    void run(double T_final, double dt, int save_freq,
             std::string prefix) override;

    /** Read-only access to the last computed alpha field. */
    const std::vector<double>& getAlpha() const { return alpha_; }

    /**
     * @brief Use a trained policy network to predict alpha instead of the
     *        Persson-Peraire modal indicator. Pass nullptr to revert to PP.
     * The alpha_max cap, alpha_min clipping and neighbour diffusion are still
     * applied to the network output for safety.
     */
    void setAlphaNet(HybridAlphaNet* net) { alpha_net_ = net; }

private:
    std::vector<double> alpha_;
    HybridAlphaNet* alpha_net_ = nullptr;

    double alpha_max_ = 0.5;
    double alpha_min_ = 0.001;
    bool   diffuse_   = true;

    /** Compute alpha[e] from the Persson-Peraire indicator (Eqs. 40-48). */
    void computeAlpha();

    void set_stage_state(double dt, double coeff);
    void accumulate_stage(double coeff);
    void finalize_step(double dt);
};

} // namespace solver

#endif
