#include <cblas.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include "../base/gll.h"

namespace mat {

/**
 * @brief GLL Bonnet function to find the `k`-th order Legendre poly at point `xi` in the normalized base
 * @param k Order of the polynomial
 * @param xi Position in the normalized space `[-1,1]`
 * @return Legendre polynomial value
 */
double legendre(int k, double xi) {
  if (k == 0)
    return 1.0;
  if (k == 1)
    return xi;
  double L_nm2 = 1.0, L_nm1 = xi, L_n = 0.0;
  for (int n = 2; n <= k; ++n) {
    L_n = ((2.0 * n - 1.0) * xi * L_nm1 - (n - 1.0) * L_nm2) / n;
    L_nm2 = L_nm1;
    L_nm1 = L_n;
  }
  return L_n;
}

/**
 * @brief Computes the Legendre coefficients from a Lagrange polynomials set
 * @param c Legendre coefficients (modal)
 * @param u Polynomial solution value
 * @param quads Nodes position in the `xi` space `[-1,1]`
 * @param weights Lagrange coefficients (nodal)
 * @param P Polynomial order (Lagrange order or higher Legendre order)
 * @return None
 */
void computeLegendreCoeffs(double *c, const double *u, const double *quads,
                           const double *weights, int P) {
  for (int k = 0; k <= P; ++k) {
    double ck = 0.0;
    for (int j = 0; j <= P; ++j)
      ck += u[j] * legendre(k, quads[j]) * weights[j];
    c[k] = (2 * k + 1) / 2.0 * ck;
  }
}

/**
 * @brief Evaluates Legendre expansion at the position s in `[-1,1]`
 * @param s Position in `[-1,1]`
 * @param c Legendre coefficients set
 * @param P Higher order of the Legendre expansion
 * @return Evaluation of polynomial at the position `s`
 */
double evalLegendreExpansion(double s, const double *c, int P) {
  double result = 0.0;
  for (int k = 0; k <= P; ++k)
    result += c[k] * legendre(k, s);
  return result;
}

/**
 * @brief Use CBLAS to compute `d2/dx2` using a defined GLL basis derivative matrix
 * @param basis Pointer to the basis object
 * @param invJ Inverse of the Jacobian
 * @param array Pointer to the array to differentiate
 * @param result Overwritten object
 * @return None
 */
void computeLaplacian(gll::Basis *basis, const double invJ, const double *array, double* result){
  const double *D = basis->getD();
  int n = basis->getOrder() + 1;
  double *tmp = new double[n];
  cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, 1., D, n, array, 1,
              0., tmp, 1);
  cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, invJ*invJ, D, n, tmp, 1,
              0., result, 1);
  delete[] tmp;
}

/**
 * @brief Helper to print vectors
 * @param vec Pointer to the vector
 * @param n size of the vector
 * @return void
 */
void print(const double *vec, size_t n) {
  std::cout << "[ ";
  for (size_t i = 0; i < n; ++i) {
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
void print(const double *mat, size_t s_row, size_t s_col) {
  std::cout << std::endl;
  for (size_t i = 0; i < s_row; ++i) {
    print(mat + i * s_col, s_col);
  }
  std::cout << std::endl;
}
} // namespace mat
