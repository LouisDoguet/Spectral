#include "lib/base/gll.h"
#include "lib/math/math.h"
#include "lib/space/mesh.h"
#include "lib/time/rk4.h"
#include <boost/program_options.hpp>
#include <boost/program_options/value_semantic.hpp>
#include <cmath>
#include <iostream>
namespace po = boost::program_options;

int main(int argc, char *argv[]) {
  po::options_description opts("Available options.");
  opts.add_options()("P", po::value<int>()->default_value(5),
                     "Polynomial order")(
      "N", po::value<int>()->default_value(50), "Number of elements")(
      "L", po::value<double>()->default_value(1.), "Domain size")(
      "Q", po::value<int>()->default_value(10),
      "Number of quadrature points per elements")("help",
                                                  "Print help message.");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, opts), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << opts << std::endl;
    return 0;
  }
  const int P = vm["P"].as<int>();
  const int Q = vm["Q"].as<int>();
  const int N_elem = vm["N"].as<int>();
  const double L = vm["L"].as<double>();

  //-- SIMULATION PARAMETERS --
  const int N_nodes = N_elem * Q;
  const double gamma = 1.4;

  // Use a smaller dt for P=5 to satisfy the strict CFL condition of Spectral
  // Elements
  const double dt = 1e-4;
  const double T_final = 0.4; // Adjusted for Sod Tube timing
  const int save_freq = 10;

  //-- SETUP BASIS & BUFFERS --
  gll::Basis *basis = new gll::Basis(P, Q);
  std::cout << basis;
  double *rho_i = new double[N_nodes];
  double *rhou_i = new double[N_nodes];
  double *e_i = new double[N_nodes];

  // Boundary Condition Holders
  double bc_rhoL, bc_rhouL, bc_eL;
  double bc_rhoR, bc_rhouR, bc_eR;

  // ==========================================================
  // CASE 1: SOD SHOCK TUBE (Uncomment to use)
  // ==========================================================
  /*
  for (int i = 0; i < N_nodes; ++i) {
      int e = i / (P + 1);
      int q = i % (P + 1);
      // Correct physical x-coordinate mapping
      double x = (e * (L / N_elem)) + (basis->getQuads()[q] + 1.0) * (L /
  N_elem) / 2.0;

      double rho, u, p;
      if (x < 0.5) {
          rho = 1.0; u = 0.0; p = 1.0;   // Left State
      } else {
          rho = 0.125; u = 0.0; p = 0.1; // Right State
      }
      rho_i[i] = rho;
      rhou_i[i] = rho * u;
      e_i[i] = p / (gamma - 1.0) + 0.5 * rho * u * u;
  }
  bc_rhoL = 1.0;   bc_rhouL = 0.0; bc_eL = 2.5;
  bc_rhoR = 0.125; bc_rhouR = 0.0; bc_eR = 0.25;
  std::string case_name = "results/sod_shock_tube";
  */

  // ==========================================================
  // CASE 2: SMOOTH ADVECTION (Gaussian Pulse)
  // ==========================================================

  for (int i = 0; i < N_nodes; ++i) {
    int e = i / Q; // Element iteration
    int q = i % Q; // Quadrature iteration
    double x =
        (e * (L / N_elem)) + ((double)basis->getQ()) * (L / N_elem) / 2.0;

    double rho = 1.0;
    double u = 1.0; // Advection speed
    // Gaussian pulse in pressure
    double p = 1.0 + 0.5 * std::exp(-200.0 * std::pow(x - 1. * L / 3., 2));

    rho_i[i] = rho;
    rhou_i[i] = rho * u;
    e_i[i] = p / (gamma - 1.0) + 0.5 * rho * u * u;
  }
  // For Advection, match BCs to the "ambient" flow state
  bc_rhoL = rho_i[0];
  bc_rhouL = rhou_i[0];
  bc_eL = e_i[0];
  bc_rhoR = rho_i[N_nodes - 1];
  bc_rhouR = rhou_i[N_nodes - 1];
  bc_eR = e_i[N_nodes - 1];
  std::string case_name = "results/smooth_advection";

  //-- MESH INIT --
  mesh::Mesh *mesh =
      new mesh::Mesh(N_elem, basis, 0.0, L, rho_i, rhou_i, e_i, bc_rhoL,
                     bc_rhouL, bc_eL, bc_rhoR, bc_rhouR, bc_eR);

  std::cout << *(mesh->getElem(0));
  std::cout << std::endl;
  // std::cout << *(mesh->getElem(0)->getBasis());
  std::cout << std::endl;
  mat::print(mesh->getElem(0)->getBasis()->getQuads(), Q);

  //-- SOLVER INIT --
  solver::RK4 solver(mesh);

  //-- RUN --
  solver.run(T_final, dt, save_freq, case_name);

  //-- CLEANUP --
  delete mesh;
  delete basis;
  delete[] rho_i;
  delete[] rhou_i;
  delete[] e_i;
  return 0;
}
