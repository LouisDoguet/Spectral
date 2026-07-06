#include "test_cases.h"
#include "space/mesh.h"

namespace S1D {

// Lax shock tube (Lax 1954). A classic Riemann problem with a stronger
// contact discontinuity than Sod and a non-zero left velocity, which makes
// it a good stress test for the contact resolution of the scheme.
//
//   Left  state (x < 0.5): rho=0.445, u=0.698, p=3.528
//   Right state (x > 0.5): rho=0.5,   u=0.0,   p=0.571
//   Domain [0,1], gamma=1.4, recommended final time T=0.16.
mesh::Mesh* generateLax(base::_Basis* basis, int N_elem, double delta) {
    const double L  = 1.0;
    const double x0 = 0.5;

    return generateMesh(basis, N_elem, L,
                        /* left  */ 0.445, 0.698, 3.528,
                        /* right */ 0.5,   0.0,   0.571,
                        x0, delta);
}

} // namespace S1D
