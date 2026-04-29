#pragma once
#include <cblas.h>
#include <stdexcept>
#include <vector>

namespace TENSOR {

class Tensor {
public:
    size_t n_rows;
    size_t n_cols;
    std::vector<double> array;

    Tensor() : n_rows(0), n_cols(0) {}
    Tensor(const size_t rows, const size_t cols) 
        : n_rows(rows), n_cols(cols), array(rows * cols, 0.0) {}

    const std::vector<double>& getData() { return array; }
    void setData(const std::vector<double>& arr) {
        if (arr.size() != this->array.size()) throw std::invalid_argument("Size of the setter not matching (looser)");
        array = arr; 
    }

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

    /**
     * @brief Transpose tensor
     */
    Tensor T() const {
        Tensor result(n_cols, n_rows);
        for (size_t i = 0; i < n_rows; ++i) {
            for (size_t j = 0; j < n_cols; ++j) {
                result.array[j * n_rows + i] = array[i * n_cols + j];
            }
        }
        return result;
    }
};

} // namespace TENSOR