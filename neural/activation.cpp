#include "activation.h"
#include "tensor.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <regex>

void ACTI::ReLU::apply(TENSOR::Tensor &tensor) {
  for (double &val : tensor.getData())
    val = std::max(0., val);
}

TENSOR::Tensor ACTI::ReLU::gradient(TENSOR::Tensor &tensor) {
  const size_t N = tensor.n_cols * tensor.n_rows;
  TENSOR::Tensor result(tensor.n_cols, tensor.n_rows);
  for (size_t i = 0; i < N; ++i) {
    double val = tensor.getData().data()[i];
    if (val > 0.)
      result.setData(i, 1.);
  }
  return result;
}

void ACTI::SoftMax::apply(TENSOR::Tensor &tensor) {
  for (size_t i = 0; i < tensor.n_rows; ++i) {
    // Find max for numerical stability (prevents exp overflow)
    double max_val =
        *std::max_element(tensor.getData().begin() + i * tensor.n_cols,
                          tensor.getData().begin() + (i + 1) * tensor.n_cols);

    // Compute exp(x - max) for each element in the row
    double sum = 0.0;
    for (size_t j = 0; j < tensor.n_cols; ++j) {
      double e = std::exp(tensor.getData()[i * tensor.n_cols + j] - max_val);
      tensor.setData(i * tensor.n_cols + j, e);
      sum += e;
    }

    // Normalise so the row sums to 1
    for (size_t j = 0; j < tensor.n_cols; ++j)
      tensor.setData(i * tensor.n_cols + j,
                     tensor.getData()[i * tensor.n_cols + j] / sum);
  }
}

TENSOR::Tensor ACTI::SoftMax::gradient(TENSOR::Tensor &tensor) {
  const size_t N = tensor.n_cols * tensor.n_rows;
  TENSOR::Tensor result(tensor.n_cols, tensor.n_rows);
  for (size_t b = 0; b < tensor.n_rows; ++b) {
    for (size_t i = 0; i < tensor.n_cols; ++i) {
      double ai = tensor.getData()[b * tensor.n_cols + i];
      result.setData(b * tensor.n_cols + i, ai * (1.0 - ai));
    }
  }
  return result;
}

void ACTI::Sigmoid::apply(TENSOR::Tensor &tensor) {
  for (double &val : tensor.getData())
    val = 1. / (1. + std::exp(-val));
}

TENSOR::Tensor ACTI::Sigmoid::gradient(TENSOR::Tensor &tensor) {
  TENSOR::Tensor result = tensor;
  for (size_t i = 0; i < tensor.n_rows * tensor.n_cols; ++i) {
    double a = tensor.getData()[i];
    result.setData(i, a * (1. - a));
  }
  return result;
}

void ACTI::HyperbolicTangent::apply(TENSOR::Tensor &tensor) {
  for (double &val : tensor.getData())
    val = std::tanh(val);
}

TENSOR::Tensor ACTI::HyperbolicTangent::gradient(TENSOR::Tensor &tensor) {
  TENSOR::Tensor result = tensor;
  for (size_t i = 0; i < tensor.n_rows * tensor.n_cols; ++i) {
    double a = tensor.getData()[i];
    result.setData(i, 1.0 - a * a);
  }
  return result;
}
