#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <iomanip>

#include "lib/space/element.h"
#include "lib/base/gll.h"
#include "lib/space/mesh.h"
#include "lib/math/math.h"
#include "lib/phy/physics.h"
#include "lib/time/rk4.h"

// Simple test framework
#define TEST_SECTION(name) std::cout << "\n=== Testing " << name << " ===" << std::endl;
#define ASSERT_NEAR(val1, val2, tol, msg) \
    if (std::abs((val1) - (val2)) > (tol)) { \
        std::cerr << "Assertion failed: " << msg << " | " << (val1) << " vs " << (val2) << " (diff: " << std::abs((val1) - (val2)) << ")" << std::endl; \
        exit(1); \
    }

void test_mesh_generation() {
    TEST_SECTION("Mesh Generation");
    const int P = 10;
    const int N_elem = 2;
    gll::Basis B(P);
    mesh::Mesh M(N_elem, &B, 0.0, 10.0);
    const elem::Element* e0 = M.getElem(0);
    const elem::Element* e1 = M.getElem(1);
    ASSERT_NEAR(*e0->getInvJ(), 0.4, 1e-12, "Element 0 InvJ");
    ASSERT_NEAR(*e1->getInvJ(), 0.4, 1e-12, "Element 1 InvJ");
    std::cout << "  Mesh generation tests passed!" << std::endl;
}

void test_physics_flux() {
    TEST_SECTION("Physics Flux");
    double rho = 1.0, rhou = 2.0, e = 4.5, p_calc;
    phy::getP(&p_calc, &rho, &rhou, &e, 1);
    ASSERT_NEAR(p_calc, 1.0, 1e-12, "Pressure calculation");
    double f1, f2, f3;
    phy::computeFlux(&f1, &f2, &f3, &rho, &rhou, &e, &p_calc, 1);
    ASSERT_NEAR(f1, 2.0, 1e-12, "Flux F1 (rho*u)");
    ASSERT_NEAR(f2, 5.0, 1e-12, "Flux F2 (rho*u^2 + p)"); 
    ASSERT_NEAR(f3, 11.0, 1e-12, "Flux F3 (u*(E+p))");
    std::cout << "  Physics flux tests passed!" << std::endl;
}

void test_divergence() {
    TEST_SECTION("Divergence Calculation");
    const int P = 4;
    gll::Basis B(P);
    elem::Element E(0, &B, 0.0, 2.0);
    const double* quads = B.getQuads();
    double* F1 = const_cast<double*>(E.getF1());
    for (int i = 0; i <= P; ++i) {
        F1[i] = quads[i] + 1.0; 
    }
    E.computeDivFlux();
    const double* divF1 = E.getDivF1();
    for (int i = 0; i <= P; ++i) {
        ASSERT_NEAR(divF1[i], 1.0, 1e-12, "Divergence of linear function");
    }
    std::cout << "  Divergence calculation tests passed!" << std::endl;
}

void test_riemann_interface() {
    TEST_SECTION("Riemann Interface Correction");
    const int P = 2; 
    gll::Basis B(P);
    mesh::Mesh M(2, &B, 0.0, 2.0); // 2 elements, dx=1.0, J=0.5, invJ=2.0
    
    for (int i=0; i<=P; ++i) {
        *const_cast<double*>(M.getElem(0)->getU1(i)) = 1.0;
        *const_cast<double*>(M.getElem(0)->getU2(i)) = 0.0;
        *const_cast<double*>(M.getElem(0)->getU3(i)) = 2.5;
        
        *const_cast<double*>(M.getElem(1)->getU1(i)) = 0.125;
        *const_cast<double*>(M.getElem(1)->getU2(i)) = 0.0;
        *const_cast<double*>(M.getElem(1)->getU3(i)) = 0.25;
    }
    
    const_cast<elem::Element*>(M.getElem(0))->setFlux();
    const_cast<elem::Element*>(M.getElem(1))->setFlux();
    const_cast<elem::Element*>(M.getElem(0))->computeDivFlux();
    const_cast<elem::Element*>(M.getElem(1))->computeDivFlux();
    
    M.computeInterfaces();
    
    // Correct physics scaling: invWJ = invJ / w = 2.0 / (1/3) = 6.0
    // Correction = 6.0 * (0.55 - 1.0) = -2.7
    double divF2_L_P = *M.getElem(0)->getDivF2(P);
    ASSERT_NEAR(divF2_L_P, -2.7, 1e-12, "Interface Riemann correction for Momentum");
    
    std::cout << "  Riemann interface correction tests passed!" << std::endl;
}

void test_boundary_conditions() {
    TEST_SECTION("Boundary Conditions (Dirichlet)");
    const int P = 2;
    gll::Basis B(P);
    mesh::Mesh M(1, &B, 0.0, 1.0); // 1 element, dx=1, J=0.5, invJ=2
    
    for (int i=0; i<=P; ++i) {
        *const_cast<double*>(M.getElem(0)->getU1(i)) = 1.0;
        *const_cast<double*>(M.getElem(0)->getU2(i)) = 0.0;
        *const_cast<double*>(M.getElem(0)->getU3(i)) = 2.5;
    }
    const_cast<elem::Element*>(M.getElem(0))->setFlux();
    const_cast<elem::Element*>(M.getElem(0))->computeDivFlux();
    
    M.applyDirichlet();
    
    ASSERT_NEAR(*M.getElem(0)->getDivF2(0), 0.0, 1e-12, "Boundary correction with matching state");

    // Right boundary correction: invWJ = 6.0. f* - f_int = 0.55 - 1.0 = -0.45.
    // 6.0 * -0.45 = -2.7
    ASSERT_NEAR(*M.getElem(0)->getDivF2(P), -2.7, 1e-12, "Boundary correction with mismatching state");

    std::cout << "  Boundary condition tests passed!" << std::endl;
}

void test_rk4_step() {
    TEST_SECTION("RK4 Time Stepping");
    const int P = 2;
    gll::Basis B(P);
    mesh::Mesh M(1, &B, 0.0, 1.0);
    
    // Set initial condition: high pressure side state
    for (int i=0; i<=P; ++i) {
        *const_cast<double*>(M.getElem(0)->getU1(i)) = 1.0;
        *const_cast<double*>(M.getElem(0)->getU2(i)) = 0.0;
        *const_cast<double*>(M.getElem(0)->getU3(i)) = 2.5;
    }
    
    solver::RK4 S(&M);
    double dt = 0.001;
    S.step(dt);
    
    // At node P, divF2 was -2.7 (due to mismatch at boundary)
    // dU2/dt = -divF2 = 2.7
    // U2_new = 0 + 2.7 * 0.001 = 0.0027 (approx)
    double u2_new = *M.getElem(0)->getU2(P);
    ASSERT_NEAR(u2_new, 0.0027, 1e-5, "RK4 step for Momentum at boundary");
    
    std::cout << "  RK4 time stepping tests passed!" << std::endl;
}

int main() {
    std::cout << "Starting Spectral1D Test Suite" << std::endl;
    test_mesh_generation();
    test_physics_flux();
    test_divergence();
    test_riemann_interface();
    test_boundary_conditions();
    test_rk4_step();
    std::cout << "\nAll tests finished successfully!" << std::endl;
    return 0;
}
