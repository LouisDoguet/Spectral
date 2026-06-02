#include "lib/base/base.h"
#include "lib/diffusion/diffusion.h"
#include "lib/math/math.h"
#include "lib/space/mesh.h"
#include "lib/time/solver.h"
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
    ("L",     po::value<double>()->default_value(2.),   "Domain size")
    ("T",     po::value<double>()->default_value(0.2),  "Final time")
    ("dt",    po::value<double>()->default_value(5e-5),  "Timestep")
    ("eps",   po::value<double>()->default_value(0.0),  "Constant artificial viscosity (0 = disabled)")
    ("output",po::value<std::string>()->default_value("results/"), "Path to generated output ParaView files")
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

  const int    P        = vm["P"].as<int>();
  const int    N_elem   = vm["N"].as<int>();
  const int    Q        = vm["Q"].as<int>();
  const double L        = vm["L"].as<double>();
  const double eps      = vm["eps"].as<double>();
  const double T_final  = vm["T"].as<double>();
  const double dt       = vm["dt"].as<double>();
  const std::string snap_dir = vm["snap"].as<std::string>();
  const std::string model    = vm["model"].as<std::string>();
  std::string output   = vm["output"].as<std::string>();
  double delta               = vm["delta"].as<double>();
  const int    trunc         = vm["trunc"].as<int>();
  const double s0            = vm["s0"].as<double>();
  const double kappa         = vm["kappa"].as<double>();
  const double eps0          = vm["eps0"].as<double>();

  if (output != "results/") output = "results/" + output;
  else output = "results/spectral1D";

  const double dx   = L / N_elem;
  const double x0   = 0.5 * L;
  const double rhoL = 1.0,   uL = 0.0, pL = 1.0;
  const double rhoR = 0.125, uR = 0.0, pR = 0.1;

  mesh::Mesh* M = S1D::generateMesh(N_elem, P, L, rhoL, uL, pL, rhoR, uR, pR, x0, delta);
  solver::RK4* S = new solver::RK4(M, Q);
  diff::PerssonPeraire* diff_PP = new diff::PerssonPeraire(trunc, s0, kappa, eps0);
  sens::PerssonPeraire* sensor_PP = new sens::PerssonPeraire();

  // Per-element basis adaptation driven by the same PP indicator.
  //   - log10(Se) > s_shock  (= s0 - kappa)   -> Lagrange -> RBF
  //   - log10(Se) < s_smooth (= s0 - 2*kappa) -> RBF -> Lagrange
  // The hysteresis gap keeps an element pinned to one basis once it has
  // switched, avoiding flapping near the shock front.
  const double s_shock  = s0 - kappa;
  const double s_smooth = s0 - 2.0 * kappa;
  S->enableBasisAdaptation(M->getAltBasis(), sensor_PP, trunc, s_shock, s_smooth);

  // Opt-in extra exports — uncomment as needed:
  //   S->addSensorField("div_laplacian", sensor_PP);
  //   S->getExporter().addField(post::VTUExporter::fieldLapPressure());
  S1D::RunShockTube(S, diff_PP, sensor_PP, T_final, dt, output);
}