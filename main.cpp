#include "lib/base/gll.h"
#include "lib/diffusion/diffusion.h"
#include "lib/math/math.h"
#include "lib/space/mesh.h"
#include "lib/time/rk4.h"
#include "lib/S1D.h"
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
  double delta               = vm["delta"].as<double>();


  const double dx   = L / N_elem;
  const double x0   = 0.5 * L;
  const double rhoL = 1.0,   uL = 0.0, pL = 1.0;
  const double rhoR = 0.125, uR = 0.0, pR = 0.1;

  S1D::RunShockTube(N_elem, P, Q, L, T_final, dt, eps, rhoL, uL, pL, rhoR, uR, pR, x0, delta);

}