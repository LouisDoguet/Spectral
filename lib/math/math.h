#ifndef MATH_H
#define MATH_H

#include <iostream>
#include <iomanip>


namespace mat {
    void print(const double* vec, size_t n);
    void print(const double* mat, size_t s_row, size_t s_col);
}

#endif
