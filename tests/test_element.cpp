#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <iomanip>

#include "lib/space/element.h"
#include "lib/base/gll.h"

// Simple test framework
#define ASSERT_NEAR(val1, val2, tol) \
    if (std::abs((val1) - (val2)) > (tol)) { \
        std::cerr << "Assertion failed: |" << (val1) << " - " << (val2) << "| = " << std::abs((val1) - (val2)) \
                  << " > " << (tol) << " at line " << __LINE__ << std::endl; \
        return 1; \
    }

int main() {
    std::cout << "Running Element Test..." << std::endl;

    const int P = 20; // Order of basis
    const int N = P + 1;
    gll::Basis* B = new gll::Basis(P);

    // Setup constant state
    // rho = 1.0
    // u = 1.0 -> rhou = 1.0
    // p = 1.0, gamma = 1.4
    // p = (gamma - 1) * (e - 0.5 * rho * u^2)
    // 1.0 = 0.4 * (e - 0.5 * 1.0 * 1.0)
    // 2.5 = e - 0.5
    // e = 3.0
    
    // We allocate these arrays because Element (constructor 2) takes ownership and deletes them.
    double* rho = new double[N];
    double* rhou = new double[N];
    double* e = new double[N];

    for (int i = 0; i < N; ++i) {
        rho[i] = 1.0;
        rhou[i] = 1.0;
        e[i] = 3.0;
    }

    // Element from -1 to 1. dx = 2. J = 1. invJ = 1.
    elem::Element* E = new elem::Element(1, B, -1.0, 1.0, *rho, *rhou, *e);

    // 1. Test Construction and Print
    std::cout << *E;

    // 2. Test setF()
    // Expected Fluxes:
    // F1 = rhou = 1.0
    // F2 = rhou*u + p = 1.0*1.0 + 1.0 = 2.0
    // F3 = u*(e + p) = 1.0*(3.0 + 1.0) = 4.0
    E->setFlux();

    const double* F1 = E->getF1();
    const double* F2 = E->getF2();
    const double* F3 = E->getF3();

    for (int i = 0; i < N; ++i) {
        ASSERT_NEAR(F1[i], 1.0, 1e-12);
        ASSERT_NEAR(F2[i], 2.0, 1e-12);
        ASSERT_NEAR(F3[i], 4.0, 1e-12);
    }
    std::cout << "Flux calculation: PASSED" << std::endl;

    // 3. Test computeDivFlux()
    // Derivative of constants should be 0.
    E->computeDivFlux();

    const double* divF1 = E->getDivF1();
    const double* divF2 = E->getDivF2();
    const double* divF3 = E->getDivF3();

    for (int i = 0; i < N; ++i) {
        ASSERT_NEAR(divF1[i], 0.0, 1e-12);
        ASSERT_NEAR(divF2[i], 0.0, 1e-12);
        ASSERT_NEAR(divF3[i], 0.0, 1e-12);
    }
    std::cout << "Divergence calculation (Constant State): PASSED" << std::endl;

    delete E;
    delete B;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
