#ifndef MATH_H
#define MATH_H

#include <iostream>
#include <iomanip>
#include "../base/base.h"

namespace mat {
    void print(const double* vec, size_t n);
    void print(const double* mat, size_t s_row, size_t s_col);

    double legendre(int k, double xi);
    void computeLegendreCoeffs(double* c, const double* u, const double* quads,
                               const double* weights, int P);
    double evalLegendreExpansion(double s, const double* c, int P);
    void computeLaplacian(base::_Basis *basis, const double invJ, const double *array, double* result);

    double inner_product(const double* vec1, const double* vec2, size_t n);
}

#endif
