/**
 * @file loss_function.cpp
 * @brief Loss function classes methods definitions
 */

#include "loss_function.h"
#include <algorithm>
#include <cmath>
#include <vector>

double LFUN::MSE::residuals(TENSOR::Tensor val, TENSOR::Tensor ref, TENSOR::Tensor& res) {
    size_t N = val.n_rows * val.n_cols;
    double L = 0;
    for (size_t i = 0; i < N; ++i) {
        double d = val.getData()[i] - ref.getData()[i];
        res.setData(i, d * d);
        L += d * d;
    }
    return L / N;
}

TENSOR::Tensor LFUN::MSE::gradient(TENSOR::Tensor val, TENSOR::Tensor ref) {
    size_t L = val.n_rows * val.n_cols;
    return (val - ref) * (2.0 / (double)L);
}

double LFUN::CrossEntropy::residuals(TENSOR::Tensor val, TENSOR::Tensor ref, TENSOR::Tensor& res) {
    size_t N = val.n_rows * val.n_cols;
    const double eps = 1e-7;
    double L = 0;
    for (size_t i = 0; i < N; ++i) {
        double y_hat = std::max(eps, std::min(1.0 - eps, val.getData()[i]));
        double y     = ref.getData()[i];
        double r     = -(y * std::log(y_hat) + (1.0 - y) * std::log(1.0 - y_hat));
        res.setData(i, r);
        L += r;
    }
    return L / N;
}

TENSOR::Tensor LFUN::CrossEntropy::gradient(TENSOR::Tensor val, TENSOR::Tensor ref) {
    size_t N = val.n_rows * val.n_cols;
    TENSOR::Tensor grad(val.n_rows, val.n_cols);
    const double eps = 1e-7;
    for (size_t i = 0; i < N; ++i) {
        double y_hat = std::max(eps, std::min(1.0 - eps, val.getData()[i]));
        double y     = ref.getData()[i];
        grad.setData(i, (y_hat - y) / (N * y_hat * (1.0 - y_hat)));
    }
    return grad;
}