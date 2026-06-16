#include "solver.h"
#include "../base/base.h"
#include "../diffusion/diffusion.h"
#include <cblas.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace solver {

_Solver::_Solver(std::string name, mesh::Mesh *mesh, int n_plot) : m(mesh), name(name) {
  total_points = m->getTotalPoints();
  int P = m->getElem(0)->getBasis()->getOrder();
  this->n_plot = (n_plot > 0) ? n_plot : (P + 1);

  exporter = new post::VTUExporter(mesh, this->n_plot);

  rho_n = new double[total_points];
  rhou_n = new double[total_points];
  e_n = new double[total_points];

  rho_acc = new double[total_points];
  rhou_acc = new double[total_points];
  e_acc = new double[total_points];

  global_df1 = new double[total_points];
  global_df2 = new double[total_points];
  global_df3 = new double[total_points];
}

void _Solver::save_state() {
  cblas_dcopy(total_points, m->getGlobalU1(), 1, rho_n, 1);
  cblas_dcopy(total_points, m->getGlobalU2(), 1, rhou_n, 1);
  cblas_dcopy(total_points, m->getGlobalU3(), 1, e_n, 1);
  std::memset(rho_acc, 0, total_points * sizeof(double));
  std::memset(rhou_acc, 0, total_points * sizeof(double));
  std::memset(e_acc, 0, total_points * sizeof(double));
}

void _Solver::collect_residuals() {
  int n_elem = m->getNumElements();
  int n_quads = total_points / n_elem;
  for (int e = 0; e < n_elem; ++e) {
    const elem::Element *el = m->getElem(e);
    std::memcpy(&global_df1[e * n_quads], el->getDivF1(),
                n_quads * sizeof(double));
    std::memcpy(&global_df2[e * n_quads], el->getDivF2(),
                n_quads * sizeof(double));
    std::memcpy(&global_df3[e * n_quads], el->getDivF3(),
                n_quads * sizeof(double));
  }
}

void _Solver::export_snapshot(int step, double time, std::string dir) {
  std::stringstream ss;
  ss << dir << "/snap_" << std::setfill('0') << std::setw(6) << step << ".bin";
  std::ofstream f(ss.str(), std::ios::binary);

  int n_elem = m->getNumElements();
  int P      = m->getElem(0)->getBasis()->getOrder();

  f.write(reinterpret_cast<const char *>(&n_elem), sizeof(int32_t));
  f.write(reinterpret_cast<const char *>(&P),      sizeof(int32_t));
  f.write(reinterpret_cast<const char *>(&time),   sizeof(double));
  f.write(reinterpret_cast<const char *>(m->getGlobalU1()), total_points * sizeof(double));
  f.write(reinterpret_cast<const char *>(m->getGlobalU2()), total_points * sizeof(double));
  f.write(reinterpret_cast<const char *>(m->getGlobalU3()), total_points * sizeof(double));
}

std::ostream &operator<<(std::ostream &os, const _Solver &s) {
  os << "===== SOLVER =====" << std::endl
     << "NAME         : " << s.name << std::endl
     << "TOTAL POINTS : " << s.total_points << std::endl
     << "PLOT POINTS  : " << s.n_plot << std::endl
     << "DIFFUSION    : " << (s.diffusion ? "enabled" : "disabled") << std::endl
     << "SENSOR       : " << (s.sensor ? "enabled" : "disabled") << std::endl
     << "BASIS ADAPT. : " << (s.adapt_alt ? "enabled" : "disabled") << std::endl;
  if (s.m)
    os << *(s.m);
  return os;
}

void _Solver::print_start(int n_steps, double dt) const {
  if (verbose_ < 1) return;
  int nelem = m->getNumElements();
  int P     = m->getElem(0)->getBasis()->getOrder();
  std::printf("[%s] Starting — %d elements, P=%d, %d steps, dt=%.2e\n",
              name.c_str(), nelem, P, n_steps, dt);
}

void _Solver::print_progress(int step, int n_steps, double t) const {
  if (verbose_ < 2) return;
  std::printf("[%s] step %5d/%d  t=%.6f\n", name.c_str(), step, n_steps, t);
}

void _Solver::print_end(int n_steps) const {
  if (verbose_ < 1) return;
  int nelem = m->getNumElements();
  int nquad = m->getElem(0)->getBasis()->getOrder() + 1;
  std::printf("[%s] Finished — %d steps, %d nodes, %lld total ops\n",
              name.c_str(), n_steps, nelem * nquad,
              static_cast<long long>(n_steps) * nelem * nquad);
}

_Solver::~_Solver() {
  delete exporter;
  delete[] rho_n;
  delete[] rhou_n;
  delete[] e_n;
  delete[] rho_acc;
  delete[] rhou_acc;
  delete[] e_acc;
  delete[] global_df1;
  delete[] global_df2;
  delete[] global_df3;
}

} // namespace solver
