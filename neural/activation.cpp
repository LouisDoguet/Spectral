#include "activation.h"
#include "tensor.h"
#include <algorithm>
#include <cmath>

void ACTI::ReLU::apply(TENSOR::Tensor& tensor) {
    for (double& val : tensor.getData())
        val = std::max(0.,val);
}


void ACTI::SoftMax::apply(TENSOR::Tensor& tensor) {
    for (size_t i = 0; i < tensor.n_rows; ++i) {
        // Find max for numerical stability (prevents exp overflow)
        double max_val = *std::max_element(
            tensor.getData().begin() + i * tensor.n_cols,
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
            tensor.setData(i * tensor.n_cols + j, tensor.getData()[i * tensor.n_cols + j] / sum);
    }
}