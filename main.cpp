#include "lib/base/gll.h"
#include "lib/diffusion/diffusion.h"
#include "lib/math/math.h"
#include "lib/space/mesh.h"
#include "lib/time/rk4.h"
#include <boost/program_options.hpp>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#ifdef WITH_ONNX
#include <memory>
#endif
namespace po = boost::program_options;

int main(int argc, char *argv[]) {
  po::options_description opts("Available options.");
  opts.add_options()
    ("P",     po::value<int>()->default_value(5),      "Polynomial order")
    ("N",     po::value<int>()->default_value(50),     "Number of elements")
    ("Q",     po::value<int>()->default_value(0),      "Output points per element (0 = P+1)")
    ("L",     po::value<double>()->default_value(1.),  "Domain size")
    ("eps",   po::value<double>()->default_value(0.0), "Constant artificial viscosity (0 = disabled)")
    ("snap",  po::value<std::string>()->default_value(""), "Directory for ML training snapshots (empty = disabled)")
    ("model", po::value<std::string>()->default_value(""), "Path to ONNX model for neural-network diffusion")
    ("delta", po::value<double>()->default_value(-1.0),   "Tanh smoothing half-width for initial discontinuity (default: 2*dx, 0=sharp)")
    ("help",  "Print help message.");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, opts), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << opts << std::endl;
    return 0;
  }

  const int    P        = vm["P"].as<int>();
  const int    N_elem   = vm["N"].as<int>();
  const int    Q        = vm["Q"].as<int>();
  const double L        = vm["L"].as<double>();
  const double eps      = vm["eps"].as<double>();
  const std::string snap_dir = vm["snap"].as<std::string>();
  const std::string model    = vm["model"].as<std::string>();
  double delta               = vm["delta"].as<double>();

  //-- SIMULATION PARAMETERS --
  const int    N_nodes  = N_elem * (P + 1);
  const double gamma    = 1.4;
  const double dt       = 1e-4;
  const double T_final  = 0.2;
  const int    save_freq = 10;

  //-- SETUP BASIS & BUFFERS --
  gll::Basis *basis  = new gll::Basis(P);
  double *rho_i  = new double[N_nodes];
  double *rhou_i = new double[N_nodes];
  double *e_i    = new double[N_nodes];
  double bc_rhoL, bc_rhouL, bc_eL;
  double bc_rhoR, bc_rhouR, bc_eR;

  // ==========================================================
  // SOD SHOCK TUBE
  // ==========================================================
  {
    const double dx   = L / N_elem;
    const double x0   = 0.5 * L;
    const double rhoL = 1.0,   uL = 0.0, pL = 1.0;
    const double rhoR = 0.125, uR = 0.0, pR = 0.1;

    // Default smoothing: 2 element widths.  --delta 0 gives a sharp IC.
    if (delta < 0.0) delta = 2.0 * dx;

    for (int i = 0; i < N_nodes; ++i) {
      int    elem = i / (P + 1);
      int    q    = i % (P + 1);
      double x    = (elem * dx) + (basis->getQuads()[q] + 1.0) * dx / 2.0;

      double rho, u, p;
      if (delta == 0.0) {
        // Sharp discontinuity (may be unstable for high P)
        rho = (x < x0) ? rhoL : rhoR;
        u   = 0.0;
        p   = (x < x0) ? pL   : pR;
      } else {
        // Tanh-smoothed discontinuity
        double s = 0.5 * (1.0 - std::tanh((x - x0) / delta));
        rho = rhoR + (rhoL - rhoR) * s;
        u   = 0.0;
        p   = pR   + (pL   - pR)   * s;
      }

      rho_i[i]  = rho;
      rhou_i[i] = rho * u;
      e_i[i]    = p / (gamma - 1.0) + 0.5 * rho * u * u;
    }

    bc_rhoL  = rhoL; bc_rhouL = 0.0; bc_eL = pL / (gamma - 1.0);
    bc_rhoR  = rhoR; bc_rhouR = 0.0; bc_eR = pR / (gamma - 1.0);
  }
  std::string case_name = "results/sod_shock_tube";

  // ==========================================================
  // SMOOTH ADVECTION (Gaussian Pulse) — swap in if needed
  // ==========================================================
  /*
  for (int i = 0; i < N_nodes; ++i) {
    int    elem = i / (P + 1);
    int    q    = i % (P + 1);
    double x    = (elem * (L / N_elem))
                + (basis->getQuads()[q] + 1.0) * (L / N_elem) / 2.0;
    double rho  = 1.0;
    double u    = 1.0;
    double p    = 1.0 + 0.5 * std::exp(-200.0 * std::pow(x - L / 3.0, 2));
    rho_i[i]  = rho;
    rhou_i[i] = rho * u;
    e_i[i]    = p / (gamma - 1.0) + 0.5 * rho * u * u;
  }
  bc_rhoL = rho_i[0];  bc_rhouL = rhou_i[0];  bc_eL = e_i[0];
  bc_rhoR = rho_i[N_nodes-1]; bc_rhouR = rhou_i[N_nodes-1]; bc_eR = e_i[N_nodes-1];
  std::string case_name = "results/smooth_advection";
  */

  //-- MESH --
  mesh::Mesh *mesh = new mesh::Mesh(N_elem, basis, 0.0, L,
                                    rho_i, rhou_i, e_i,
                                    bc_rhoL, bc_rhouL, bc_eL,
                                    bc_rhoR, bc_rhouR, bc_eR);

  //-- SOLVER --
  solver::RK4 solver(mesh, Q);

  //-- DIFFUSION MODE --
  DIFF::Constant constant_diff(eps);
#ifdef WITH_ONNX
  std::unique_ptr<DIFF::ONNX> onnx_diff;
  if (!model.empty())
    onnx_diff = std::make_unique<DIFF::ONNX>(model, N_nodes);
#endif

  std::cout << "--- Diffusion : ";
  if (!model.empty()) {
#ifdef WITH_ONNX
    solver.setDiffusion(onnx_diff.get());
    std::cout << "ONNX (" << model << ")";
#else
    std::cout << "NONE (compiled without WITH_ONNX — ignoring --model)";
#endif
  } else if (eps > 0.0) {
    solver.setDiffusion(&constant_diff);
    std::cout << "Constant (eps=" << eps << ")";
  } else {
    std::cout << "None";
  }
  std::cout << " ---" << std::endl;

  //-- SNAPSHOT EXPORT --
  if (!snap_dir.empty()) {
    std::filesystem::create_directories(snap_dir);
    solver.setSnapshotDir(snap_dir);
    std::cout << "--- Snapshots : " << snap_dir << " ---" << std::endl;
  }

  //-- RUN --
  solver.run(T_final, dt, save_freq, case_name);
  const double* lap_press = mesh->getElem(20)->computePressureLaplacian();
  mat::print(lap_press, 6);

  //-- CLEANUP --
  delete mesh;
  delete basis;
  delete[] rho_i;
  delete[] rhou_i;
  delete[] e_i;
  return 0;
}
