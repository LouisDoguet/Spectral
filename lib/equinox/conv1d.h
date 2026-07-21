#pragma once
#include <vector>

// Minimal inference-only mirror of equinox.nn.Conv1d (see nn/network/model.py).
// Cross-correlation (NO kernel flip), stride 1, zeros padding = kernel_size/2,
// so the spatial length is preserved. Data layout is channel-major:
// a tensor of shape (channels, L) is stored as v[c*L + t].
namespace equinox {

class Conv1d {
public:
  Conv1d() = default;
  Conv1d(int in_ch, int out_ch, int kernel_size);

  // x: (in_ch, L) -> y: (out_ch, L)
  void forward(const std::vector<double>& x, int L, std::vector<double>& y) const;

  int inChannels() const { return in_ch_; }
  int outChannels() const { return out_ch_; }
  int kernel() const { return k_; }

  std::vector<double> W;  // (out_ch, in_ch, k), row-major:  W[(o*in+c)*k + kk]
  std::vector<double> b;  // (out_ch)

private:
  int in_ch_ = 0, out_ch_ = 0, k_ = 0, pad_ = 0;
};

}  // namespace equinox
