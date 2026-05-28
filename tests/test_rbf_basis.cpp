#include "../lib/base/base.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

namespace {

int g_failures = 0;

void check_near(double got, double expected, double tol, const std::string& name) {
    const double diff = std::abs(got - expected);
    std::cout << std::left << std::setw(48) << ("  " + name)
              << "  got=" << std::setw(14) << got
              << "  expected=" << std::setw(14) << expected
              << "  diff=" << std::setw(12) << diff
              << "  tol=" << tol;
    if (diff <= tol) {
        std::cout << "   ok\n";
    } else {
        std::cout << "   FAIL\n";
        ++g_failures;
    }
}

// Multiplies y = A x for a row-major NxN matrix A.
void mat_vec(const double* A, const double* x, double* y, int N) {
    for (int i = 0; i < N; ++i) {
        double s = 0.0;
        for (int j = 0; j < N; ++j) s += A[i * N + j] * x[j];
        y[i] = s;
    }
}

double dot_w(const double* w, const double* f, int N) {
    double s = 0.0;
    for (int i = 0; i < N; ++i) s += w[i] * f[i];
    return s;
}

void test_imq(int P, double eps) {
    std::cout << "\n=== IMQ basis: P=" << P << ", eps=" << eps << " ===\n";
    base::InverseMultiQuadratic basis(P, eps);
    const int N = P + 1;
    const double* w = basis.getWeights();
    const double* x = basis.getQuads();
    const double* D = basis.getD();

    // -------- node layout --------
    check_near(x[0],    -1.0, 1e-14, "node[0] = -1");
    check_near(x[P],     1.0, 1e-14, "node[P] = +1");

    // -------- quadrature: monomials --------
    double m0 = 0.0, m1 = 0.0, m2 = 0.0, m3 = 0.0, m4 = 0.0;
    for (int i = 0; i < N; ++i) {
        const double xi = x[i];
        m0 += w[i];
        m1 += w[i] * xi;
        m2 += w[i] * xi * xi;
        m3 += w[i] * xi * xi * xi;
        m4 += w[i] * xi * xi * xi * xi;
    }
    check_near(m0,  2.0,        1e-3,  "int 1   dx = 2");
    check_near(m1,  0.0,        1e-12, "int x   dx = 0   (symmetry)");
    check_near(m2,  2.0/3.0,    1e-2,  "int x^2 dx = 2/3");
    check_near(m3,  0.0,        1e-12, "int x^3 dx = 0   (symmetry)");
    check_near(m4,  2.0/5.0,    5e-2,  "int x^4 dx = 2/5");

    // -------- quadrature: a smooth function with known integral --------
    // int_{-1}^{1} cos(pi*xi/2) dxi = 4/pi
    double Scos = 0.0;
    for (int i = 0; i < N; ++i) Scos += w[i] * std::cos(M_PI * x[i] / 2.0);
    check_near(Scos, 4.0 / M_PI, 5e-3, "int cos(pi x/2) dx = 4/pi");

    // -------- weights symmetry  (LAPACK pivoting limits this to ~1e-10) --------
    double sym_err = 0.0;
    for (int i = 0; i < N; ++i)
        sym_err = std::max(sym_err, std::abs(w[i] - w[P - i]));
    check_near(sym_err, 0.0, 1e-10, "weights symmetric  w[i] = w[P-i]");

    // -------- derivative on a smooth non-polynomial: D sin(x) ~ cos(x) --------
    // This is the meaningful convergence test for the derivative operator.
    double* u  = new double[N];
    double* du = new double[N];
    for (int i = 0; i < N; ++i) u[i] = std::sin(x[i]);
    mat_vec(D, u, du, N);
    double max_interior = 0.0;
    for (int i = 1; i < N - 1; ++i)
        max_interior = std::max(max_interior, std::abs(du[i] - std::cos(x[i])));
    check_near(max_interior, 0.0, 5e-2, "max |D sin(x) - cos(x)|  (interior)");

    // -------- diagnostics: unaugmented IMQ does NOT exactly reproduce 1 or x.
    //   We bound the error rather than expect zero.
    for (int i = 0; i < N; ++i) u[i] = 1.0;
    mat_vec(D, u, du, N);
    double max_row_sum = 0.0;
    for (int i = 0; i < N; ++i) max_row_sum = std::max(max_row_sum, std::abs(du[i]));
    check_near(max_row_sum, 0.0, 1e-1, "max |D * 1|       (RBF reproduction err)");

    for (int i = 0; i < N; ++i) u[i] = x[i];
    mat_vec(D, u, du, N);
    double max_lin_err = 0.0;
    for (int i = 0; i < N; ++i) max_lin_err = std::max(max_lin_err, std::abs(du[i] - 1.0));
    check_near(max_lin_err, 0.0, 3e-1, "max |D * x - 1|   (RBF reproduction err)");

    delete[] u;
    delete[] du;
}

}  // namespace

int main() {
    std::cout << "RBF (Inverse Multi Quadratic) basis test\n";
    test_imq(6,  1.0);
    test_imq(8,  1.0);
    test_imq(10, 1.5);

    std::cout << "\n--------------------------------------------------\n";
    if (g_failures == 0) {
        std::cout << "ALL CHECKS PASSED\n";
        return 0;
    }
    std::cout << g_failures << " CHECK(S) FAILED\n";
    return 1;
}
