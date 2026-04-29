#pragma once
#include <cblas.h>
#include <stdexcept>
#include <vector>

namespace TENSOR {

class Tensor {
public:
    const size_t n_rows;
    const size_t n_cols;
    std::vector<double> array;

    Tensor(const size_t rows, const size_t cols) 
        : n_rows(rows), n_cols(cols), array(rows * cols, 0.0) {}

    double* getData() { return array.data(); }
    const double* getData() const { return array.data(); }

    Tensor operator*(const Tensor& other) const {
        if (n_cols != other.n_rows) {
            throw std::invalid_argument("Inner dimensions must match for GEMM.");
        }

        Tensor result(n_rows, other.n_cols);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    n_rows, other.n_cols, n_cols, 
                    1.0, array.data(), n_cols, 
                    other.array.data(), other.n_cols, 
                    0.0, result.array.data(), other.n_cols);
        
        return result;
    }

    Tensor operator+(const Tensor& bias) const {
        if (n_cols != bias.n_cols || bias.n_rows != 1) {
            throw std::invalid_argument("Bias must be a 1xN row vector matching tensor columns.");
        }

        Tensor result = *this; 
        for (size_t i = 0; i < n_rows; ++i) {
            cblas_daxpy(n_cols, 1.0, bias.array.data(), 1, result.array.data() + i * n_cols, 1);
        }
        
        return result;
    }
};

} // namespace TENSOR