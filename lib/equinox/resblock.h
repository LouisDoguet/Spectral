#pragma once
#include <vector>
#include "conv1d.h"

// equinox ResBlock (nn/network/model.py): y = relu(x + conv2(relu(conv1(x)))).
namespace equinox {

class ResBlock {
public:
  ResBlock() = default;
  ResBlock(int width, int kernel_size);

  void forward(const std::vector<double>& x, int L,
               std::vector<double>& y) const;

  Conv1d conv1, conv2;
};

}  // namespace equinox
