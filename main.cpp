#include "lib/base/gll.h"
#include "lib/diffusion/diffusion.h"
#include "lib/math/math.h"
#include "lib/space/mesh.h"
#include "lib/time/rk4.h"
#include "lib/S1D.h"
#include "lib/sensor/sensor.h"
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
    ("P",     po::value<int>()->default_value(5),       "Polynomial order")
    ("N",     po::value<int>()->default_value(50),      "Number of elements")
    ("Q",     po::value<int>()->default_value(0),       "Output points per element (0 = P+1)")
    ("L",     po::value<double>()->default_value(1.),   "Domain size")
    ("T",     po::value<double>()->default_value(0.1),  "Final time")
    ("dt",    po::value<double>()->default_value(5e-5),  "Timestep")
    ("eps",   po::value<double>()->default_value(0.0),  "Constant artificial viscosity (0 = disabled)")
    ("snap",  po::value<std::string>()->default_value(""), "Directory for ML training snapshots (empty = disabled)")
    ("model", po::value<std::string>()->default_value(""), "Path to ONNX model for neural-network diffusion")
    ("delta", po::value<double>()->default_value(-1.0),   "Tanh smoothing half-width for initial discontinuity (default: 2*dx, 0=sharp)")
    ("sensor",                                             "Enable Persson-Peraire sensor diffusion")
    ("trunc", po::value<int>()->default_value(1),          "PP sensor: number of high modes to monitor")
    ("s0",    po::value<double>()->default_value(-3.0),    "PP sensor: shock threshold in log10(Se/S) scale")
    ("kappa", po::value<double>()->default_value(1.0),     "PP sensor: transition half-width (log10 units)")
    ("eps0",  po::value<double>()->default_value(0.01),    "PP sensor: maximum viscosity")
    ("help",  "Print help message.");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, opts), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << opts << std::endl;
    return 0;
  }

  const double gamma = 1.4;

  const int    P        = vm["P"].as<int>();
  const int    N_elem   = vm["N"].as<int>();
  const int    Q        = vm["Q"].as<int>();
  const double L        = vm["L"].as<double>();
  const double eps      = vm["eps"].as<double>();
  const double T_final  = vm["T"].as<double>();
  const double dt       = vm["dt"].as<double>();
  const std::string snap_dir = vm["snap"].as<std::string>();
  const std::string model    = vm["model"].as<std::string>();
  double delta               = vm["delta"].as<double>();
  const int    trunc         = vm["trunc"].as<int>();
  const double s0            = vm["s0"].as<double>();
  const double kappa         = vm["kappa"].as<double>();
  const double eps0          = vm["eps0"].as<double>();

  const double dx   = L / N_elem;
  const double x0   = 0.5 * L;
  const double rhoL = 1.0,   uL = 0.0, pL = 1.0;
  const double rhoR = 0.125, uR = 0.0, pR = 0.1;

  const int N_nodes  = N_elem * (P + 1);
  gll::Basis *basis  = new gll::Basis(P);
  double *rho_i  = new double[N_nodes];
  double *rhou_i = new double[N_nodes];
  double *e_i    = new double[N_nodes];
  double bc_rhoL, bc_rhouL, bc_eL;
  double bc_rhoR, bc_rhouR, bc_eR;

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

  //-- MESH --
  mesh::Mesh *MESH = new mesh::Mesh(N_elem, basis, 0.0, L,
                                    rho_i, rhou_i, e_i,
                                    bc_rhoL, bc_rhouL, bc_eL,
                                    bc_rhoR, bc_rhouR, bc_eR);

  std::filesystem::create_directories("results");

  solver::RK4 solver(MESH, Q);

  if (vm.count("sensor")) {
    DIFF::PerssonPeraire pp_diff(trunc, s0, kappa, eps0);
    solver.setDiffusion(&pp_diff);
    std::cout << "Diffusion : Persson-Peraire sensor\n"
              << "  trunc=" << trunc << "  s0=" << s0
              << "  kappa=" << kappa << "  eps0=" << eps0 << "\n";
    solver.run(T_final, dt, 10, "results/sod_pp");
  } else if (eps > 0.0) {
    DIFF::Constant cd(eps);
    solver.setDiffusion(&cd);
    std::cout << "Diffusion : constant eps=" << eps << "\n";
    solver.run(T_final, dt, 10, "results/sod_const");
  } else {
    std::cout << "Diffusion : none\n";
    solver.run(T_final, dt, 10, "results/sod_nodiff");
  }
}