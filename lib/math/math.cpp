#include <iostream>
#include <iomanip>
#include <cmath>
#include <cblas.h>

namespace mat {
    
    /**
     * @brief Helper to print vectors
     * @param vec Pointer to the vector
     * @param n size of the vector
     * @return void
     */
    void print(const double* vec, size_t n){
        std::cout << "[ ";
        for (size_t i=0; i<n; ++i){
            std::cout << std::fixed << vec[i] << "  ";
        }
        std::cout << " ]" << std::endl;
    }

    /**
     * @brief Helper to print matrices
     * @param mat Pointer to the matrix
     * @param s_row Size of the rows
     * @param s_col Size of the columns
     * @return void
     */
    void print(const double* mat, size_t s_row, size_t s_col){
        std::cout << std::endl;
        for (size_t i=0; i<s_row; ++i){
            print(mat + i*s_col, s_col);
        }
        std::cout << std::endl;
    }
}

