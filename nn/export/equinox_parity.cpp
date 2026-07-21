// Verification harness for the JAX -> C++ alpha-policy export.
//
// Builds the sharp Sod state on [0,1], N=32, P=4 (identical to the JAX reference
// in nn/export/parity_check.py), runs equinox::AlphaNet::fillAlpha, and dumps the
// state, the raw (pre-clip) alpha, and the two feature channels so a Python diff
// can confirm C++ reproduces the JAX NodalAlphaModel to machine precision.
//
//   build/equinox_parity <model.nnx> <out.txt>
//
// Not part of the solver; build on demand with:
//   cmake --build build --target equinox_parity
#include "../../lib/base/base.h"
#include "../../lib/equinox/alpha_net.h"
#include "../../lib/space/mesh.h"
#include "../../lib/test_cases.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 3) { std::cerr << "usage: equinox_parity <model.nnx> <out.txt>\n"; return 1; }
  const int P = 4, N = 32;
  base::_Basis* base_ = new base::Lagrange(P);
  // sharp Sod on [0,1], x0=0.5 (delta=0) -> matches the JAX reference exactly
  mesh::Mesh* M = S1D::generateMesh(base_, N, 1.0, 1.0, 0.0, 1.0,
                                    0.125, 0.0, 0.1, 0.5, 0.0);
  const int Nn = P + 1;

  equinox::AlphaNet net(argv[1]);
  std::vector<double> alpha;
  net.fillAlpha(M, alpha);                       // raw, pre-clip

  const double* u1 = M->getGlobalU1();
  const double* u2 = M->getGlobalU2();
  const double* u3 = M->getGlobalU3();

  std::ofstream out(argv[2]);
  out.precision(17);
  out << "STATE " << N << " " << Nn << "\n";
  for (int e = 0; e < N; ++e)
    for (int node = 0; node < Nn; ++node) {
      const int j = e * Nn + node;
      out << u1[j] << " " << u2[j] << " " << u3[j] << "\n";
    }

  out << "ALPHA " << N << " " << P << "\n";
  for (int e = 0; e < N; ++e)
    for (int i = 0; i < P; ++i)
      out << alpha[e * P + i] << "\n";

  // Feature channels (public Mesh API), for channel-level localization.
  std::vector<double> res;
  M->densityResidualDifference(res);             // size N*Nn
  double maxabs = 0.0;
  for (double v : res) maxabs = std::max(maxabs, std::fabs(v));
  const double nrm = maxabs + 1e-8;
  out << "RES " << N << " " << Nn << "\n";
  for (int j = 0; j < N * Nn; ++j) out << res[j] / nrm << "\n";

  std::vector<double> eind;
  M->perssonPeraireIndicator(eind);              // size N
  const double T = 0.5 * std::pow(10.0, -1.8 * std::pow(P + 1.0, 0.25));
  const double s = 9.21024;
  out << "ENERGY " << N << "\n";
  for (int e = 0; e < N; ++e)
    out << 1.0 / (1.0 + std::exp(-(s / T) * (eind[e] - T))) << "\n";

  std::cerr << "wrote " << argv[2] << "  (alpha size " << alpha.size() << ")\n";
  return 0;
}
