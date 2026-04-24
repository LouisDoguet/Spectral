#include <iostream>
#include <iomanip>
#include <cmath>
#include <cblas.h>

namespace mat {

    double legendre(int k, double xi) {
        if (k == 0) return 1.0;
        if (k == 1) return xi;
        double L_nm2 = 1.0, L_nm1 = xi, L_n = 0.0;
        for (int n = 2; n <= k; ++n) {
            L_n = ((2.0*n - 1.0) * xi * L_nm1 - (n - 1.0) * L_nm2) / n;
            L_nm2 = L_nm1;
            L_nm1 = L_n;
        }
        return L_n;
    }

    void computeLegendreCoeffs(double* c, const double* u, const double* quads,
                               const double* weights, int P) {
        for (int k = 0; k <= P; ++k) {
            double ck = 0.0;
            for (int j = 0; j <= P; ++j)
                ck += u[j] * legendre(k, quads[j]) * weights[j];
            c[k] = (2*k + 1) / 2.0 * ck;
        }
    }

    double evalLegendreExpansion(double s, const double* c, int P) {
        double result = 0.0;
        for (int k = 0; k <= P; ++k)
            result += c[k] * legendre(k, s);
        return result;
    }
    
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

