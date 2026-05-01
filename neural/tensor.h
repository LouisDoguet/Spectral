#pragma once
#include <cblas.h>
#include <stdexcept>
#include <vector>

namespace TENSOR {

class Tensor {
private:
    std::vector<double> array;
public:
    size_t n_rows;
    size_t n_cols;
    

    Tensor() : n_rows(0), n_cols(0) {}
    Tensor(const size_t n_rows, const size_t n_cols)
        : n_rows(n_rows), n_cols(n_cols), array(n_rows * n_cols, 0.0) {}

    std::vector<double>& getData() { return array; }
    const std::vector<double>& readData() const { return array; }
    void setData(const std::vector<double>& arr) {
        if (arr.size() != this->array.size()) throw std::invalid_argument("Size of the setter not matching (looser)");
        array = arr; 
    }
    void setData(size_t i, double val) {
        array.at(i) = val;
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

    Tensor operator*(const double val) const {
        Tensor result = *this;
        cblas_dscal(n_rows * n_cols, val, result.array.data(), 1);
        return result;
    }

    Tensor add_bias(const Tensor& bias) const {
        if (n_cols != bias.n_cols || bias.n_rows != 1) {
            throw std::invalid_argument("Bias must be a 1xN row vector matching tensor columns.");
        }

        Tensor result = *this; 
        for (size_t i = 0; i < n_rows; ++i) {
            cblas_daxpy(n_cols, 1.0, bias.array.data(), 1, result.array.data() + i * n_cols, 1);
        }
        
        return result;
    }

    Tensor operator+(const Tensor& other) const {
        if (n_rows != other.n_rows || n_cols != other.n_cols)
            throw std::invalid_argument("Tensor shapes must match for element-wise addition.");

        Tensor result = *this;
        cblas_daxpy(n_rows * n_cols, 1.0,
                    other.array.data(), 1,
                    result.array.data(), 1);
        return result;
    }

    Tensor operator-(const Tensor& other) const {
        if (n_rows != other.n_rows || n_cols != other.n_cols)
            throw std::invalid_argument("Tensor shapes must match for element-wise subtraction.");

        Tensor result = *this;
        cblas_daxpy(n_rows * n_cols, -1.0,
                    other.array.data(), 1,
                    result.array.data(), 1);
        return result;
    }


    /**
     * @brief Sum of the rows of the tensor
     * @return Tensor (1,out_features)
     */
    Tensor sum_rows() const {
        Tensor result(1, n_cols);
        for (size_t i = 0; i < n_rows; ++i) {
            cblas_daxpy(n_cols, 1.0, array.data() + i * n_cols, 1, result.array.data(), 1);
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