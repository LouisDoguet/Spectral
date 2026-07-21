#include "conv1d.h"

namespace equinox {

Conv1d::Conv1d(int in_ch, int out_ch, int kernel_size)
    : W(static_cast<size_t>(out_ch) * in_ch * kernel_size, 0.0),
      b(out_ch, 0.0),
      in_ch_(in_ch), out_ch_(out_ch), k_(kernel_size), pad_(kernel_size / 2) {}

void Conv1d::forward(const std::vector<double>& x, int L,
                     std::vector<double>& y) const {
  y.assign(static_cast<size_t>(out_ch_) * L, 0.0);
  for (int o = 0; o < out_ch_; ++o) {
    const double bo = b[o];
    for (int t = 0; t < L; ++t) {
      double acc = bo;
      for (int c = 0; c < in_ch_; ++c) {
        const double* wrow = &W[(static_cast<size_t>(o) * in_ch_ + c) * k_];
        const double* xrow = &x[static_cast<size_t>(c) * L];
        for (int kk = 0; kk < k_; ++kk) {
          const int pos = t + kk - pad_;  // matches equinox: window t..t+k-1 on
          if (pos >= 0 && pos < L)        // the input padded by pad_ each side
            acc += wrow[kk] * xrow[pos];
        }
      }
      y[static_cast<size_t>(o) * L + t] = acc;
    }
  }
}

}  // namespace equinox
