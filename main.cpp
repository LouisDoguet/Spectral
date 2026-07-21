#include "lib/test_cases.h"
#include "lib/base/base.h"
#include "lib/diffusion/diffusion.h"
#include "lib/math/math.h"
#include "lib/sensor/sensor.h"
#include "lib/space/mesh.h"
#include "lib/time/hybrid_solver.h"
#include "lib/time/solver.h"
#include <boost/program_options.hpp>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
namespace po = boost::program_options;

int main(int argc, char *argv[]) {
  po::options_description opts("Available options.");
  opts.add_options()("P", po::value<int>()->default_value(5),
                     "Polynomial order")(
      "N", po::value<int>()->default_value(50),
      "Number of elements")("Q", po::value<int>()->default_value(0),
                            "Output points per element on a uniform grid; "
                            "0 = export the raw P+1 computed (GLL) nodes")(
      "L", po::value<double>()->default_value(2.), "Domain size")(
      "T", po::value<double>()->default_value(0.2),
      "Final time")("dt", po::value<double>()->default_value(5e-5),
                    "Timestep; set to -1 for an automatic CFL-based dt")(
      "cfl", po::value<double>()->default_value(0.5),
      "Courant number used when dt<0 (auto CFL time step)")(
      "eps", po::value<double>()->default_value(1.),
      "RBF Epsilon value (inactive if no RBF elements)")(
      "output", po::value<std::string>()->default_value("results/"),
      "Path to generated output ParaView files")(
      "delta", po::value<double>()->default_value(-1.0),
      "Tanh smoothing half-width for initial discontinuity (default: 2*dx, "
      "0=sharp)")(
      "case", po::value<std::string>()->default_value("sod"),
      "1D benchmark case: sod | lax | shu-osher | woodward-colella")(
      "solver", po::value<std::string>()->default_value("rk4"),
                  "Solver type: rk4 | hybrid_dgsem")(
      "alpha_max", po::value<double>()->default_value(0.5),
      "hybrid_dgsem: maximum FV blending factor")(
      "sensor", po::value<std::string>()->default_value(""),
      "Sensor for the solver")(
      "base0", po::value<std::string>()->default_value("Lagrange"),
      "Original solving base")(
      "base1", po::value<std::string>()->default_value(""),
      "Replacement base (when sensor recognize discontinuity)")(
      "trunc", po::value<int>()->default_value(1),
      "PP sensor: number of high modes to monitor")(
      "s0", po::value<double>()->default_value(-3.0),
      "PP sensor: shock threshold in log10(Se/S) scale")(
      "kappa", po::value<double>()->default_value(1.0),
      "PP sensor: transition half-width (log10 units)")(
      "eps0", po::value<double>()->default_value(0.01),
      "PP sensor: maximum viscosity")(
      "nn_model", po::value<std::string>()->default_value(""),
      "Path to neural network model file (.nn)")(
      "verbose", po::value<int>()->default_value(1),
      "Verbosity level: 0=silent, 1=start/end, 2=all saved iterations")(
      "help", "Print help message.");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, opts), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << opts << std::endl;
    return 0;
  }

  const int P = vm["P"].as<int>();
  const int N_elem = vm["N"].as<int>();
  const int Q = vm["Q"].as<int>();
  const double L = vm["L"].as<double>();
  const double eps = vm["eps"].as<double>();
  double T_final = vm["T"].as<double>();
  double dt = vm["dt"].as<double>();
  const double cfl = vm["cfl"].as<double>();
  std::string output = vm["output"].as<std::string>();
  const std::string solver_type = vm["solver"].as<std::string>();
  const double alpha_max = vm["alpha_max"].as<double>();
  const std::string sensor = vm["sensor"].as<std::string>();
  const std::string base0 = vm["base0"].as<std::string>();
  const std::string base1 = vm["base1"].as<std::string>();
  double delta = vm["delta"].as<double>();
  const int trunc = vm["trunc"].as<int>();
  const double s0 = vm["s0"].as<double>();
  const double kappa = vm["kappa"].as<double>();
  const double eps0 = vm["eps0"].as<double>();
  const std::string nn_model = vm["nn_model"].as<std::string>();
  const int verbose = vm["verbose"].as<int>();
  const std::string test_case = vm["case"].as<std::string>();

  // Per-case default final time, applied only when the user left --T at its
  // default so an explicit --T always wins.
  if (vm["T"].defaulted()) {
    if (test_case == "lax")
      T_final = 0.16;
    else if (test_case == "shu-osher")
      T_final = 1.8;
    else if (test_case == "woodward-colella")
      T_final = 0.038;
  }

  if (output != "results/")
    output = "results/" + output;
  else
    output = "results/spectral1D";

  const double dx = L / N_elem;
  const double x0 = 0.5 * L;
  const double rhoL = 1.0, uL = 0.0, pL = 1.0;
  const double rhoR = 0.125, uR = 0.0, pR = 0.1;

  int base0_id = 0;
  if (base0 == "RBF::IMQ")
    base0_id = 1;
  else if (base0 == "RBF::GAUSSIAN")
    base0_id = 2;

  int base1_id = 0;
  if (base1 == "RBF::IMQ")
    base1_id = 1;
  else if (base1 == "RBF::GAUSSIAN")
    base1_id = 2;

  int sensor_id = 0;
  if (sensor == "PerssonPeraire")
    sensor_id = 1;

  base::_Basis *primary_base = nullptr;
  switch (base0_id) {
  case 1:
    primary_base = new base::InverseMultiQuadratic(P, eps);
    break;

  case 2:
    primary_base = new base::Gaussian(P, eps);
    break;

  default:
    primary_base = new base::Lagrange(P);
    break;
  }

  base::RBF *secondary_base = nullptr;
  switch (base1_id) {
  case 1:
    secondary_base = new base::InverseMultiQuadratic(P, eps);
    break;

  case 2:
    secondary_base = new base::Gaussian(P, eps);
    break;

  default:
    break;
  }

  sens::_Sensor *ssor = nullptr;
  diff::_Diffusion *dif = nullptr;

  // For hybrid_dgsem, --nn_model drives the blending factor alpha directly and
  // is loaded further below as an equinox::AlphaNet.
  if (sensor_id == 1) {
    ssor = new sens::PerssonPeraire();
    dif = new diff::PerssonPeraire(trunc, s0, kappa, eps0);
  }

  // The Lax/Shu-Osher/Woodward-Colella cases define their own canonical
  // domain and initial state (the CLI L/x0/rho* arguments only apply to sod).
  mesh::Mesh *M = nullptr;
  if (test_case == "lax") {
    M = S1D::generateLax(primary_base, N_elem, delta);
  } else if (test_case == "shu-osher") {
    M = S1D::generateShuOsher(primary_base, N_elem, delta);
  } else if (test_case == "woodward-colella") {
    M = S1D::generateWoodwardColella(primary_base, N_elem, delta);
  } else {
    if (test_case != "sod")
      std::cerr << "Unknown --case '" << test_case << "', falling back to sod"
                << std::endl;
    M = S1D::generateMesh(primary_base, N_elem, L, rhoL, uL, pL, rhoR, uR, pR,
                          x0, delta);
  }

  // dt < 0 -> pick an automatic, stable time step from the CFL condition of the
  // initial field (see Mesh::computeCFLTimeStep for the formula and its source).
  if (dt < 0.0) {
    dt = M->computeCFLTimeStep(cfl);
    std::cout << "Auto CFL time step (cfl=" << cfl << "): dt=" << dt
              << std::endl;
  }

  const double s_shock = s0 - kappa;
  const double s_smooth = s0 - 2.0 * kappa;

  // Q == 0 selects "computed-nodes" export: each element is written at its own
  // P+1 quadrature (GLL) nodes at their true positions, i.e. the raw computed
  // solution rather than a uniform re-projection.
  const bool export_nodal = (Q == 0);

  if (solver_type == "hybrid_dgsem") {
    solver::HybridDGSEM *S = new solver::HybridDGSEM(M, Q);
    S->setIndicatorParams(alpha_max, /*alpha_min=*/0.001, /*diffuse=*/true);
    S->setVerbosity(verbose);
    S->getExporter().useComputedNodes(export_nodal);

    equinox::AlphaNet *anet = nullptr;
    if (!nn_model.empty()) {
      anet = new equinox::AlphaNet(nn_model);
      S->setAlphaNet(anet);
      std::cout << "hybrid_dgsem: alpha driven by exported equinox NN policy ("
                << nn_model << ")" << std::endl;
    }

    S1D::RunShockTube(S, T_final, dt, output, nullptr, nullptr);
    delete S;
    delete anet;
  } else {
    solver::RK4 *S = new solver::RK4(M, Q);
    S->setVerbosity(verbose);
    S->getExporter().useComputedNodes(export_nodal);

    if (secondary_base) {
      M->setAltBasis(secondary_base);
      S->enableBasisAdaptation(M->getAltBasis(), ssor, trunc, s_shock,
                               s_smooth);
    }

    S1D::RunShockTube(S, T_final, dt, output, dif, ssor);
    delete S;
  }
}
