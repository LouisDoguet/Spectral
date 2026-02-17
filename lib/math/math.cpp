#include <iostream>
#include <iomanip>
#include <cmath>
#include <cblas.h>

namespace mat {
    void print(const double* vec, size_t n){
        std::cout << "[ ";
        for (size_t i=0; i<n; ++i){
            std::cout << std::fixed << vec[i] << "  ";
        }
        std::cout << " ]" << std::endl;
    }

    void print(const double* mat, size_t s_row, size_t s_col){
        std::cout << std::endl;
        for (size_t i=0; i<s_row; ++i){
            print(mat + i*s_col, s_col);
        }
        std::cout << std::endl;
    }
}

