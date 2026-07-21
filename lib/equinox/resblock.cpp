#include "resblock.h"

namespace equinox {

ResBlock::ResBlock(int width, int kernel_size)
    : conv1(width, width, kernel_size), conv2(width, width, kernel_size) {}

static void relu(std::vector<double>& v) {
  for (double& x : v)
    if (x < 0.0) x = 0.0;
}

void ResBlock::forward(const std::vector<double>& x, int L,
                       std::vector<double>& y) const {
  std::vector<double> t1, t2;
  conv1.forward(x, L, t1);
  relu(t1);
  conv2.forward(t1, L, t2);
  y.assign(x.size(), 0.0);
  for (size_t i = 0; i < x.size(); ++i) {
    const double v = x[i] + t2[i];
    y[i] = v < 0.0 ? 0.0 : v;
  }
}

}  // namespace equinox
