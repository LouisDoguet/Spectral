#ifndef MATH_H
#define MATH_H

#include <iostream>
#include <iomanip>
#include "../base/gll.h"

namespace mat {
    void print(const double* vec, size_t n);
    void print(const double* mat, size_t s_row, size_t s_col);

    double legendre(int k, double xi);
    void computeLegendreCoeffs(double* c, const double* u, const double* quads,
                               const double* weights, int P);
    double evalLegendreExpansion(double s, const double* c, int P);
    void computeLaplacian(gll::Basis *basis, const double invJ, const double *array, double* result);
}

#endif
