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
    return (val - ref) * (2.0 / (double)L);  // note: val-ref, not ref-val
}