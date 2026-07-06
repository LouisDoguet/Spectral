#include "hybrid_solver.h"
#include "../diffusion/hybrid_alpha_net.h"
#include "../math/math.h"
#include <cblas.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace solver {

HybridDGSEM::HybridDGSEM(mesh::Mesh* mesh, int n_plot)
    : _Solver("HybridDGSEM", mesh, n_plot),
      alpha_(mesh->getNumElements(), 0.0) {
    // Export the per-element blending factor alpha as a piecewise-constant VTU
    // field, so the hybrid indicator can be inspected in ParaView alongside the
    // solution. alpha_ is indexed by element position; the exporter only hands
    // the field lambda an Element reference, so we recover the index by matching
    // the element pointer (cheap: export is infrequent).
    exporter->addElemField("alpha", [this](elm::Element& el) -> double {
        const int n_elem = m->getNumElements();
        for (int e = 0; e < n_elem; ++e)
            if (m->getElem(e) == &el) return alpha_[e];
        return 0.0;
    });
}

// ---------------------------------------------------------------------------
// Alpha computation: Persson-Peraire modal energy indicator (Eqs. 40-48)
// ---------------------------------------------------------------------------
void HybridDGSEM::computeAlpha() {
    const int n_elem = m->getNumElements();

    // ---- Policy-network branch (Beck-style learned indicator) --------------
    // The network predicts a raw alpha in [0,1] per element from the local
    // density profile; we then apply the same clipping/capping/diffusion tail
    // used by the Persson-Peraire branch for robustness.
    if (alpha_net_) {
        alpha_net_->fillAlpha(m, alpha_);
        for (int e = 0; e < n_elem; ++e) {
            double a = alpha_[e];
            if (a < alpha_min_)              a = 0.0;
            else if (a > 1.0 - alpha_min_)   a = 1.0;
            alpha_[e] = std::min(a, alpha_max_);
        }
        if (diffuse_ && n_elem > 1) {
            std::vector<double> tmp(alpha_);
            for (int e = 0; e < n_elem; ++e) {
                double nb = 0.0;
                if (e > 0)        nb = std::max(nb, tmp[e-1]);
                if (e < n_elem-1) nb = std::max(nb, tmp[e+1]);
                alpha_[e] = std::max(tmp[e], 0.5 * nb);
            }
        }
        return;
    }

    // ---- Persson-Peraire modal energy indicator (Eqs. 40-48) ---------------
    const int P      = m->getElem(0)->getBasis()->getOrder();
    const double N1  = static_cast<double>(P + 1);

    // T(N) = 0.5 * 10^(-1.8 * (N+1)^{1/4})   (Eq. 42)
    const double T_thresh = 0.5 * std::pow(10.0, -1.8 * std::pow(N1, 0.25));

    // s chosen so that alpha(E=0) = 0.0001  (Eq. 44-45)
    const double s = 9.21024;

    for (int e = 0; e < n_elem; ++e) {
        elm::Element* E = m->getElem(e);
        E->computeLegendreCoefficients();
        const double* m_coef = E->getModes();   // Legendre modal coefficients

        // Modal energy ||m||^2 (Eq. 39)
        double total = 0.0;
        for (int j = 0; j <= P; ++j) total += m_coef[j] * m_coef[j];
        if (total < 1e-300) { alpha_[e] = 0.0; continue; }

        // Truncated energies for modes N and N-1 (Eq. 40)
        double total_m1 = total - m_coef[P] * m_coef[P];   // sum_{j=0}^{N-1}
        double E_N  = m_coef[P]   * m_coef[P]   / total;
        double E_Nm = (P > 0 && total_m1 > 1e-300)
                       ? m_coef[P-1] * m_coef[P-1] / total_m1
                       : 0.0;
        double Eind = std::max(E_N, E_Nm);

        // Smooth sigmoid blending factor  (Eq. 43)
        double raw = 1.0 / (1.0 + std::exp(-s / T_thresh * (Eind - T_thresh)));

        // Clip  (Eq. 46)
        if (raw < alpha_min_)        raw = 0.0;
        else if (raw > 1.0 - alpha_min_) raw = 1.0;

        // Cap at alpha_max  (Eq. 47)
        alpha_[e] = std::min(raw, alpha_max_);
    }

    // Single-sweep neighbour diffusion  (Eq. 48): alpha_e = max(alpha_e, 0.5*alpha_neighbour)
    if (diffuse_ && n_elem > 1) {
        std::vector<double> tmp(alpha_);
        for (int e = 0; e < n_elem; ++e) {
            double nb = 0.0;
            if (e > 0)        nb = std::max(nb, tmp[e-1]);
            if (e < n_elem-1) nb = std::max(nb, tmp[e+1]);
            alpha_[e] = std::max(tmp[e], 0.5 * nb);
        }
    }
}

// ---------------------------------------------------------------------------
// RK4 stage helpers (identical logic to RK4, but call computeHybridResidual)
// ---------------------------------------------------------------------------
void HybridDGSEM::set_stage_state(double dt, double coeff) {
    const double a = -dt * coeff;
    cblas_dcopy(total_points, rho_n,  1, m->getGlobalU1(), 1);
    cblas_dcopy(total_points, rhou_n, 1, m->getGlobalU2(), 1);
    cblas_dcopy(total_points, e_n,    1, m->getGlobalU3(), 1);
    collect_residuals();
    cblas_daxpy(total_points, a, global_df1, 1, m->getGlobalU1(), 1);
    cblas_daxpy(total_points, a, global_df2, 1, m->getGlobalU2(), 1);
    cblas_daxpy(total_points, a, global_df3, 1, m->getGlobalU3(), 1);
}

void HybridDGSEM::accumulate_stage(double coeff) {
    collect_residuals();
    cblas_daxpy(total_points, -coeff, global_df1, 1, rho_acc,  1);
    cblas_daxpy(total_points, -coeff, global_df2, 1, rhou_acc, 1);
    cblas_daxpy(total_points, -coeff, global_df3, 1, e_acc,    1);
}

void HybridDGSEM::finalize_step(double dt) {
    const double a = dt / 6.0;
    cblas_dcopy(total_points, rho_n,  1, m->getGlobalU1(), 1);
    cblas_dcopy(total_points, rhou_n, 1, m->getGlobalU2(), 1);
    cblas_dcopy(total_points, e_n,    1, m->getGlobalU3(), 1);
    cblas_daxpy(total_points, a, rho_acc,  1, m->getGlobalU1(), 1);
    cblas_daxpy(total_points, a, rhou_acc, 1, m->getGlobalU2(), 1);
    cblas_daxpy(total_points, a, e_acc,    1, m->getGlobalU3(), 1);
}

// ---------------------------------------------------------------------------
// step(): compute alpha then advance one RK4 step using hybrid residual
// ---------------------------------------------------------------------------
void HybridDGSEM::step(double dt) {
    // Compute per-element blending factors from current state.
    computeAlpha();

    save_state();

    // Stage 1
    m->computeHybridResidual(alpha_.data());
    accumulate_stage(1.0);

    // Stage 2
    set_stage_state(dt, 0.5);
    m->computeHybridResidual(alpha_.data());
    accumulate_stage(2.0);

    // Stage 3
    set_stage_state(dt, 0.5);
    m->computeHybridResidual(alpha_.data());
    accumulate_stage(2.0);

    // Stage 4
    set_stage_state(dt, 1.0);
    m->computeHybridResidual(alpha_.data());
    accumulate_stage(1.0);

    finalize_step(dt);
}

// ---------------------------------------------------------------------------
// run(): time-stepping loop (mirrors RK4::run)
// ---------------------------------------------------------------------------
void HybridDGSEM::run(double T_final, double dt, int save_freq,
                      std::string prefix) {
    int n_steps = static_cast<int>(std::ceil(T_final / dt));
    print_start(n_steps, dt);
    for (int step = 0; step <= n_steps; ++step) {
        if (step % save_freq == 0) {
            // Refresh alpha_ from the current state so the exported field is in
            // sync with the solution being written (step() recomputes it anyway).
            computeAlpha();
            print_progress(step, n_steps, step * dt);
            exporter->write(step, step * dt, prefix);
            if (!snapshot_dir.empty())
                export_snapshot(step, step * dt, snapshot_dir);
        }
        this->step(dt);
    }
    exporter->writePVD(prefix);
    print_end(n_steps);
}

} // namespace solver
