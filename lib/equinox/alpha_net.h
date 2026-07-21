#pragma once
#include <string>
#include <vector>
#include "conv1d.h"
#include "resblock.h"

namespace mesh { class Mesh; }

// C++ inference of the JAX/equinox NodalAlphaModel (nn/network/model.py),
// loading a `.nnx` file produced by nn/export/export.py. Attach to the hybrid
// solver via HybridDGSEM::setAlphaNet. Reads the model architecture (P, width,
// kernel, depth, n_data_channels) from the file header, so it adapts to the
// exported network without recompilation.
namespace equinox {

class AlphaNet {
public:
  explicit AlphaNet(const std::string& path);

  // Compute the blending factor per interior subcell interface from the current
  // mesh state. Fills alpha_iface with size n_elem * P (element-major
  // alpha[e*P + i]), matching the shape HybridDGSEM::computeHybridResidual wants.
  void fillAlpha(mesh::Mesh* mesh, std::vector<double>& alpha_iface);

  int order() const { return P_; }
  int dataChannels() const { return n_data_; }

private:
  void assembleFeatures(mesh::Mesh* mesh, int n_elem, int Nn, int L,
                        std::vector<double>& x);

  int P_ = 0, width_ = 0, kernel_ = 0, depth_ = 0, n_data_ = 0;
  double alpha_max_ = 1.0;
  Conv1d lift_;
  std::vector<ResBlock> blocks_;
  std::vector<double> proj_w_;
  double proj_b_ = 0.0;
};

}  // namespace equinox
