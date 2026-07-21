#include "hybrid_solver.h"
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
      alpha_(mesh->getNumElements() *
             mesh->getElem(0)->getBasis()->getOrder(), 0.0) {
    // alpha_ holds one value per interior subcell interface: n_elem * P, in
    // element-major order alpha_[e*P + i]. Export a per-element representative
    // (the max over the element's interfaces = where the scheme is most
    // dissipative) as a piecewise-constant VTU field for ParaView.
    exporter->addElemField("alpha", [this](elm::Element& el) -> double {
        const int n_elem = m->getNumElements();
        const int P = m->getElem(0)->getBasis()->getOrder();
        for (int e = 0; e < n_elem; ++e)
            if (m->getElem(e) == &el) {
                double a = 0.0;
                for (int i = 0; i < P; ++i) a = std::max(a, alpha_[e * P + i]);
                return a;
            }
        return 0.0;
    });
}

// ---------------------------------------------------------------------------
// Alpha computation: Persson-Peraire modal energy indicator (Eqs. 40-48)
// ---------------------------------------------------------------------------
void HybridDGSEM::computeAlpha() {
    const int n_elem = m->getNumElements();
    const int P      = m->getElem(0)->getBasis()->getOrder();  // interfaces/elem

    // ---- Policy-network branch: fills PER-INTERFACE alpha (size n_elem*P) ---
    // The equinox net writes one alpha per interior subcell interface directly.
    // We then apply the same clip/cap tail as jax deployment (no neighbour
    // diffusion for the nodal policy).
    if (alpha_net_) {
        alpha_net_->fillAlpha(m, alpha_);
        for (double& a : alpha_) {
            if (a < alpha_min_)            a = 0.0;
            else if (a > 1.0 - alpha_min_) a = 1.0;
            a = std::min(a, alpha_max_);
        }
        return;
    }

    // ---- Persson-Peraire modal energy indicator (Eqs. 40-48), PER ELEMENT --
    const double N1  = static_cast<double>(P + 1);
    const double T_thresh = 0.5 * std::pow(10.0, -1.8 * std::pow(N1, 0.25)); // Eq.42
    const double s = 9.21024;                                               // Eq.44-45

    std::vector<double> pe(n_elem, 0.0);   // per-element blending factor
    for (int e = 0; e < n_elem; ++e) {
        elm::Element* E = m->getElem(e);
        E->computeLegendreCoefficients();
        const double* m_coef = E->getModes();

        double total = 0.0;
        for (int j = 0; j <= P; ++j) total += m_coef[j] * m_coef[j];
        if (total < 1e-300) { pe[e] = 0.0; continue; }

        double total_m1 = total - m_coef[P] * m_coef[P];
        double E_N  = m_coef[P] * m_coef[P] / total;
        double E_Nm = (P > 0 && total_m1 > 1e-300)
                       ? m_coef[P-1] * m_coef[P-1] / total_m1 : 0.0;
        double Eind = std::max(E_N, E_Nm);

        double raw = 1.0 / (1.0 + std::exp(-s / T_thresh * (Eind - T_thresh))); // Eq.43
        if (raw < alpha_min_)            raw = 0.0;                             // Eq.46
        else if (raw > 1.0 - alpha_min_) raw = 1.0;
        pe[e] = std::min(raw, alpha_max_);                                     // Eq.47
    }

    // Single-sweep neighbour diffusion (Eq. 48), per element.
    if (diffuse_ && n_elem > 1) {
        std::vector<double> tmp(pe);
        for (int e = 0; e < n_elem; ++e) {
            double nb = 0.0;
            if (e > 0)        nb = std::max(nb, tmp[e-1]);
            if (e < n_elem-1) nb = std::max(nb, tmp[e+1]);
            pe[e] = std::max(tmp[e], 0.5 * nb);
        }
    }

    // Broadcast the per-element alpha onto the element's P interior interfaces.
    for (int e = 0; e < n_elem; ++e)
        for (int i = 0; i < P; ++i)
            alpha_[e * P + i] = pe[e];
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
