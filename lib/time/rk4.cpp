#include "solver.h"
#include "../base/base.h"
#include "../diffusion/diffusion.h"
#include <cblas.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace solver {

void RK4::set_stage_state(double dt, double coeff) {
  double alpha = -dt * coeff;
  cblas_dcopy(total_points, rho_n, 1, m->getGlobalU1(), 1);
  cblas_dcopy(total_points, rhou_n, 1, m->getGlobalU2(), 1);
  cblas_dcopy(total_points, e_n, 1, m->getGlobalU3(), 1);
  collect_residuals();
  cblas_daxpy(total_points, alpha, global_df1, 1, m->getGlobalU1(), 1);
  cblas_daxpy(total_points, alpha, global_df2, 1, m->getGlobalU2(), 1);
  cblas_daxpy(total_points, alpha, global_df3, 1, m->getGlobalU3(), 1);
}

void RK4::accumulate_stage(double coeff) {
  collect_residuals();
  cblas_daxpy(total_points, -coeff, global_df1, 1, rho_acc, 1);
  cblas_daxpy(total_points, -coeff, global_df2, 1, rhou_acc, 1);
  cblas_daxpy(total_points, -coeff, global_df3, 1, e_acc, 1);
}

void RK4::finalize_step(double dt) {
  double alpha = dt / 6.0;
  cblas_dcopy(total_points, rho_n, 1, m->getGlobalU1(), 1);
  cblas_dcopy(total_points, rhou_n, 1, m->getGlobalU2(), 1);
  cblas_dcopy(total_points, e_n, 1, m->getGlobalU3(), 1);
  cblas_daxpy(total_points, alpha, rho_acc, 1, m->getGlobalU1(), 1);
  cblas_daxpy(total_points, alpha, rhou_acc, 1, m->getGlobalU2(), 1);
  cblas_daxpy(total_points, alpha, e_acc, 1, m->getGlobalU3(), 1);
}

void RK4::step(double dt) {
  // Basis adaptation must happen BEFORE save_state so rho_n/rhou_n/e_n and the
  // global U buffers stay consistent throughout the four stages. adaptBasis
  // relies on the Persson-Peraire smoothness indicator, so the registered
  // sensor must be (or derive from) PerssonPeraire.
  if (adapt_alt && adapt_pp) {
    if (auto* pp = dynamic_cast<sens::PerssonPeraire*>(adapt_pp))
      m->adaptBasis(*pp, adapt_trunc, adapt_s_shock, adapt_s_smooth);
  }
  save_state();
  m->computeResidual();
  if (diffusion)
    diffusion->apply(m);
  accumulate_stage(1.0);
  set_stage_state(dt, 0.5);
  m->computeResidual();
  if (diffusion)
    diffusion->apply(m);
  accumulate_stage(2.0);
  set_stage_state(dt, 0.5);
  m->computeResidual();
  if (diffusion)
    diffusion->apply(m);
  accumulate_stage(2.0);
  set_stage_state(dt, 1.0);
  m->computeResidual();
  if (diffusion)
    diffusion->apply(m);
  accumulate_stage(1.0);
  finalize_step(dt);
}

void RK4::run(double T_final, double dt, int save_freq, std::string prefix) {
  int n_steps = static_cast<int>(std::ceil(T_final / dt));
  print_start(n_steps, dt);
  for (int step = 0; step <= n_steps; ++step) {
    if (step % save_freq == 0) {
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
