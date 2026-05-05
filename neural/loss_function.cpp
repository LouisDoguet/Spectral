#include "loss_function.h"
#include <vector>

double LFUN::MSE::residuals(TENSOR::Tensor val, TENSOR::Tensor ref, TENSOR::Tensor& res) {
    res.setData( ((val-ref)*(val-ref)).getData() );
    double L = 0;
    for (double r : res.getData()) L+= r;
    return L/(val.n_cols * val.n_rows);
}

TENSOR::Tensor LFUN::MSE::gradient(TENSOR::Tensor val, TENSOR::Tensor ref) {
    size_t L = val.n_rows * val.n_cols;
    return (val - ref) * (2.0 / (double)L);
}

double LFUN::CrossEntropy::residuals(TENSOR::Tensor val, TENSOR::Tensor ref, TENSOR::Tensor& res) {
    res.setData( (ref * val.ln() + (ref + (-1.))*(val + (-1.)).ln()).getData() );
    double L = 0;
    for (double r : res.getData()) L+= r;
    return -L/(val.n_cols * val.n_rows);
}

TENSOR::Tensor LFUN::CrossEntropy::gradient(TENSOR::Tensor val, TENSOR::Tensor ref) {
    size_t N = val.n_rows * val.n_cols;
    TENSOR::Tensor grad(val.n_rows, val.n_cols);
    for (size_t i = 0; i < N; ++i) {
        double y_hat = val.getData()[i];
        double y     = ref.getData()[i];
        grad.setData(i, (y_hat - y) / (N * y_hat * (1.0 - y_hat)));
    }
    return grad;
}