#include "alpha_net.h"
#include "../space/mesh.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace equinox {

static std::vector<double> read_f64(std::ifstream& f, size_t count,
                                    const std::string& what) {
  std::vector<double> v(count);
  f.read(reinterpret_cast<char*>(v.data()), count * sizeof(double));
  if (!f) throw std::runtime_error("equinox::AlphaNet: short read on " + what);
  return v;
}

static uint32_t read_u32(std::ifstream& f) {
  uint32_t x = 0;
  f.read(reinterpret_cast<char*>(&x), 4);
  return x;
}

AlphaNet::AlphaNet(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("equinox::AlphaNet: cannot open " + path);

  char magic[4];
  f.read(magic, 4);
  if (std::memcmp(magic, "EQXN", 4) != 0)
    throw std::runtime_error("equinox::AlphaNet: bad magic (not a .nnx) in " + path);
  read_u32(f);  // version (currently unused)
  P_ = static_cast<int>(read_u32(f));
  width_ = static_cast<int>(read_u32(f));
  kernel_ = static_cast<int>(read_u32(f));
  depth_ = static_cast<int>(read_u32(f));
  n_data_ = static_cast<int>(read_u32(f));
  f.read(reinterpret_cast<char*>(&alpha_max_), 8);
  if (!f) throw std::runtime_error("equinox::AlphaNet: truncated header in " + path);

  const int Nn = P_ + 1;
  const int in_ch = n_data_ + Nn;

  lift_ = Conv1d(in_ch, width_, kernel_);
  lift_.W = read_f64(f, static_cast<size_t>(width_) * in_ch * kernel_, "lift.W");
  lift_.b = read_f64(f, width_, "lift.b");

  blocks_.clear();
  for (int d = 0; d < depth_; ++d) {
    ResBlock blk(width_, kernel_);
    blk.conv1.W = read_f64(f, static_cast<size_t>(width_) * width_ * kernel_, "conv1.W");
    blk.conv1.b = read_f64(f, width_, "conv1.b");
    blk.conv2.W = read_f64(f, static_cast<size_t>(width_) * width_ * kernel_, "conv2.W");
    blk.conv2.b = read_f64(f, width_, "conv2.b");
    blocks_.push_back(std::move(blk));
  }

  proj_w_ = read_f64(f, width_, "proj_w");
  proj_b_ = read_f64(f, 1, "proj_b")[0];
}

// Build the (in_ch, L) network input, L = n_elem*Nn, global-node-order columns
// col = e*Nn + node. Mirrors nn/network/policy.py NODAL_DATA_CHANNELS + one-hot.
void AlphaNet::assembleFeatures(mesh::Mesh* mesh, int n_elem, int Nn, int L,
                                std::vector<double>& x) {
  const int in_ch = n_data_ + Nn;
  x.assign(static_cast<size_t>(in_ch) * L, 0.0);

  // data channel 0: DG-FV density residual, normalized by max|.| + 1e-8.
  std::vector<double> res;
  mesh->densityResidualDifference(res);  // size L
  double maxabs = 0.0;
  for (double v : res) maxabs = std::max(maxabs, std::fabs(v));
  const double norm = maxabs + 1e-8;
  for (int col = 0; col < L; ++col) x[col] = res[col] / norm;

  // data channel 1 (if present): PP decision sigmoid((s/T)(E_ind - T)), per
  // element, broadcast onto its nodes.
  if (n_data_ >= 2) {
    std::vector<double> eind;
    mesh->perssonPeraireIndicator(eind);  // size n_elem
    const double T = 0.5 * std::pow(10.0, -1.8 * std::pow(P_ + 1.0, 0.25));
    const double s = 9.21024;
    for (int e = 0; e < n_elem; ++e) {
      const double dec = 1.0 / (1.0 + std::exp(-(s / T) * (eind[e] - T)));
      for (int node = 0; node < Nn; ++node)
        x[static_cast<size_t>(1) * L + e * Nn + node] = dec;
    }
  }

  // one-hot position block: row (n_data_ + node) is 1 at that node.
  for (int e = 0; e < n_elem; ++e)
    for (int node = 0; node < Nn; ++node)
      x[static_cast<size_t>(n_data_ + node) * L + e * Nn + node] = 1.0;
}

static void relu(std::vector<double>& v) {
  for (double& x : v)
    if (x < 0.0) x = 0.0;
}

void AlphaNet::fillAlpha(mesh::Mesh* mesh, std::vector<double>& alpha_iface) {
  const int n_elem = mesh->getNumElements();
  const int Nn = P_ + 1;
  const int P = P_;          // interior subcell interfaces per element
  const int L = n_elem * Nn;

  std::vector<double> x;
  assembleFeatures(mesh, n_elem, Nn, L, x);

  std::vector<double> h;
  lift_.forward(x, L, h);
  relu(h);
  for (const ResBlock& blk : blocks_) {
    std::vector<double> h2;
    blk.forward(h, L, h2);
    h.swap(h2);
  }

  // Readout: h is (width, L). For each element e and interior interface i, take
  // the average of the two adjacent nodes' latent vectors, dot with proj_w,
  // add proj_b, sigmoid.  (nn/network/model.py NodalAlphaModel.__call__.)
  alpha_iface.assign(static_cast<size_t>(n_elem) * P, 0.0);
  for (int e = 0; e < n_elem; ++e) {
    for (int i = 0; i < P; ++i) {
      double logit = proj_b_;
      const size_t base = static_cast<size_t>(e) * Nn + i;
      for (int w = 0; w < width_; ++w) {
        const size_t row = static_cast<size_t>(w) * L;
        const double hval = 0.5 * (h[row + base] + h[row + base + 1]);
        logit += hval * proj_w_[w];
      }
      alpha_iface[static_cast<size_t>(e) * P + i] = 1.0 / (1.0 + std::exp(-logit));
    }
  }
}

}  // namespace equinox
